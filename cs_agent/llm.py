"""大模型客户端（OpenAI 兼容协议）。

为什么不用 openai SDK / langchain：这个项目里对模型的需求很窄——
**一次对话补全 + 流式增量**。用标准库 ``urllib`` 直接打 HTTP 能让
依赖保持在 ``fastapi + uvicorn`` 两三个包，也便于把超时、重试、
错误分类这些真正要讲清楚的地方写明白，而不是埋进框架里。

离线模式：没有 Key（或 ``LLM_MODE=offline``）时不发任何网络请求，
改用确定性的模板应答，保证接口、流式协议、持久化行为完全一致，测试可离线跑。
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.client import HTTPException
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

from .config import Settings, get_settings

logger = logging.getLogger(__name__)

#: 可重试的 HTTP 状态码（限流与临时故障）
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """模型调用失败。``retryable`` 表示是否值得重试。"""

    def __init__(self, message: str, *, retryable: bool = False, status: int = 0) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status = status


@dataclass
class Usage:
    """一次调用的用量（部分供应商不返回 usage，则按字符估算）。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class ChatResult:
    """一次非流式补全的结果。"""

    text: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: int = 0
    model: str = ""
    offline: bool = False


def _approx_tokens(text: str) -> int:
    chinese = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return chinese + max((len(text) - chinese) // 4, 0)


class LLMClient:
    """OpenAI 兼容的对话客户端。"""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    @property
    def online(self) -> bool:
        return self.settings.online

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> ChatResult:
        """一次性补全（不带流式）。"""
        if not self.online:
            return self._offline_result(messages)

        payload = self._build_payload(messages, temperature, max_tokens, json_mode, stream=False)
        started = time.perf_counter()
        body = self._post(payload)
        latency_ms = int((time.perf_counter() - started) * 1000)

        try:
            choice = body["choices"][0]
            text = (choice.get("message") or {}).get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:  # pragma: no cover - 防御性分支
            raise LLMError(f"模型返回结构异常：{exc}") from exc

        usage_raw = body.get("usage") or {}
        usage = Usage(
            prompt_tokens=int(usage_raw.get("prompt_tokens") or 0),
            completion_tokens=int(usage_raw.get("completion_tokens") or 0),
            total_tokens=int(usage_raw.get("total_tokens") or 0),
        )
        if not usage.total_tokens:
            usage = Usage(
                prompt_tokens=sum(_approx_tokens(m.get("content", "")) for m in messages),
                completion_tokens=_approx_tokens(text),
                total_tokens=0,
            )
            usage.total_tokens = usage.prompt_tokens + usage.completion_tokens

        return ChatResult(
            text=text,
            usage=usage,
            latency_ms=latency_ms,
            model=body.get("model") or self.settings.model,
        )

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """流式补全，逐段产出增量文本。

        失败重试策略：**只在还没吐出任何内容时重试**。
        已经产出部分文本后再重试，会让用户看到重复内容，不如直接抛错让上层收尾。
        """
        if not self.online:
            yield from self._offline_stream(messages)
            return

        payload = self._build_payload(messages, temperature, max_tokens, False, stream=True)
        attempts = 0
        while True:
            emitted = False
            try:
                for delta in self._post_stream(payload):
                    emitted = True
                    yield delta
                return
            except LLMError as exc:
                attempts += 1
                if emitted or not exc.retryable or attempts > 2:
                    raise
                logger.warning("流式调用失败，重试第 %s 次：%s", attempts, exc)
                time.sleep(0.4 * attempts)

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------
    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        json_mode: bool,
        stream: bool,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": (
                self.settings.llm_temperature if temperature is None else temperature
            ),
            "max_tokens": self.settings.llm_max_tokens if max_tokens is None else max_tokens,
            "stream": stream,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _request(self, payload: Dict[str, Any]) -> urllib.request.Request:
        url = self.settings.base_url.rstrip("/") + "/chat/completions"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.settings.api_key}",
                "Accept": "application/json" if not payload.get("stream") else "text/event-stream",
            },
            method="POST",
        )

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        request = self._request(payload)
        try:
            with urllib.request.urlopen(request, timeout=self.settings.llm_timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")[:300]
            except Exception:  # pragma: no cover - 防御性分支
                pass
            raise LLMError(
                f"模型接口返回 {exc.code}：{detail}",
                retryable=exc.code in RETRYABLE_STATUS,
                status=exc.code,
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LLMError(f"模型接口连接失败：{exc}", retryable=True) from exc
        # RemoteDisconnected / IncompleteRead 等属于 http.client.HTTPException，
        # **不是** URLError 的子类。不单独捕获的话，一次网络抖动会直接冒泡成
        # "未预期错误"，既不会重试、也拿不到可读原因（实测长会话中途遇到过）。
        except HTTPException as exc:
            raise LLMError(f"模型连接中断：{type(exc).__name__}: {exc}", retryable=True) from exc
        except json.JSONDecodeError as exc:  # pragma: no cover - 防御性分支
            raise LLMError(f"模型返回不是合法 JSON：{exc}") from exc

    def _post_stream(self, payload: Dict[str, Any]) -> Iterator[str]:
        request = self._request(payload)
        try:
            with urllib.request.urlopen(request, timeout=self.settings.llm_timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        yield piece
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")[:300]
            except Exception:  # pragma: no cover - 防御性分支
                pass
            raise LLMError(
                f"模型接口返回 {exc.code}：{detail}",
                retryable=exc.code in RETRYABLE_STATUS,
                status=exc.code,
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LLMError(f"模型接口连接失败：{exc}", retryable=True) from exc
        # 流式读取过程中对端断开（RemoteDisconnected 等）也属于可重试的瞬时故障
        except HTTPException as exc:
            raise LLMError(f"模型连接中断：{type(exc).__name__}: {exc}", retryable=True) from exc

    # ------------------------------------------------------------------
    # 离线模式
    # ------------------------------------------------------------------
    def _offline_result(self, messages: List[Dict[str, str]]) -> ChatResult:
        text = "".join(self._offline_stream(messages))
        return ChatResult(
            text=text,
            usage=Usage(
                prompt_tokens=sum(_approx_tokens(m.get("content", "")) for m in messages),
                completion_tokens=_approx_tokens(text),
                total_tokens=0,
            ),
            latency_ms=0,
            model="offline-template",
            offline=True,
        )

    def _offline_stream(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        """离线应答：确定性地说明"当前是离线模式"，不假装自己是大模型。

        这样做有两个好处：测试可预期；演示时不会让人误以为模型真的答对了。
        """
        last_user = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), ""
        )
        turns = sum(1 for m in messages if m.get("role") == "user")
        text = (
            f"（离线模式）我收到了你的第 {turns} 条消息：「{last_user[:60]}」。"
            "当前未配置可用的大模型 API Key，因此这条回复来自本地模板，"
            "不是模型生成。配置 DEEPSEEK_API_KEY 后可获得真实应答。"
        )
        for piece in _chunk_text(text, 12):
            yield piece


def _chunk_text(text: str, size: int) -> Iterator[str]:
    for index in range(0, len(text), size):
        yield text[index : index + size]
