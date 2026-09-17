"""P2：摘要生成策略的单测（用可编程假模型，不联网）。

盯三件在真实模型上踩过的事：

1. 摘要超标时要**改写自己**，而不是直接丢掉 LLM 摘要退化成抽取式；
2. 模型偶发空响应/连接中断要**重试**，不能一次抖动就降级；
3. 抽取式兜底**不能递归复制旧摘要**（否则每压缩一次套一层，会指数膨胀）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from cs_agent.memory import MemoryManager
from cs_agent.storage import ROLE_ASSISTANT, ROLE_USER, SessionStore


class _ScriptedLLM:
    """按脚本返回内容的假模型。

    ``responses`` 里的每一项：字符串表示正常返回，``None`` 表示返回空内容，
    ``Exception`` 表示抛出该异常。
    """

    def __init__(self, responses: List[Any], online: bool = True) -> None:
        self.responses = list(responses)
        self.online = online
        self.calls: List[List[Dict[str, str]]] = []

    def chat(self, messages, **kwargs):  # noqa: ANN001
        self.calls.append(messages)
        if not self.responses:
            payload: Any = "（脚本已用尽）"
        else:
            payload = self.responses.pop(0)
        if isinstance(payload, Exception):
            raise payload

        class _Result:
            def __init__(self, text: str) -> None:
                self.text = text

        return _Result(payload or "")


def _history(store: SessionStore, sid: str, turns: int, text: str = "请帮我看下订单") -> None:
    for index in range(turns):
        store.append_message(sid, ROLE_USER, f"{text}（第 {index + 1} 次）", turn=index + 1)
        store.append_message(sid, ROLE_ASSISTANT, "好的，我帮您看看。", turn=index + 1)


def _manager(settings, store: SessionStore, llm) -> MemoryManager:
    return MemoryManager(store, settings, llm)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 摘要生成
# ---------------------------------------------------------------------------
def test_summary_uses_llm_result_within_budget(settings, store) -> None:
    sid = store.create_session()["id"]
    _history(store, sid, 6)
    llm = _ScriptedLLM(["【用户身份】张先生\n【已确认信息】订单 SO1"])
    manager = _manager(settings, store, llm)

    text, source = manager._build_summary("", store.list_messages(sid, roles=[ROLE_USER, ROLE_ASSISTANT]))

    assert source == "llm"
    assert "张先生" in text
    assert len(llm.calls) == 1


def test_oversized_summary_is_tightened_not_discarded(settings, store) -> None:
    """超标时先尝试"改写自己"，成功则保留 LLM 摘要（标记 llm+compressed）。"""
    sid = store.create_session()["id"]
    _history(store, sid, 6)
    oversized = "【用户身份】张先生\n" + "很长的内容。" * 200
    llm = _ScriptedLLM([oversized, "【用户身份】张先生\n【已确认信息】订单 SO1"])
    manager = _manager(settings, store, llm)

    messages = store.list_messages(sid, roles=[ROLE_USER, ROLE_ASSISTANT])
    text, source = manager._build_summary("", messages)

    assert source == "llm+compressed"
    assert len(text) <= manager._summary_budget(sum(len(m["content"]) for m in messages))
    assert len(llm.calls) == 2, "第二次调用应是改写自己的二次压缩"


def test_tighten_failure_falls_back_to_extractive(settings, store) -> None:
    """改写也失败才退化为抽取式，并如实标注来源。"""
    sid = store.create_session()["id"]
    _history(store, sid, 6)
    oversized = "长摘要。" * 300
    llm = _ScriptedLLM([oversized, "还是太长。" * 300])
    manager = _manager(settings, store, llm)

    messages = store.list_messages(sid, roles=[ROLE_USER, ROLE_ASSISTANT])
    text, source = manager._build_summary("", messages)

    assert source == "extractive"
    assert "抽取式" in text


def test_empty_response_is_retried(settings, store) -> None:
    """供应商偶发空响应必须重试，不能一次就降级。"""
    sid = store.create_session()["id"]
    _history(store, sid, 6)
    llm = _ScriptedLLM([None, "【用户身份】张先生"])
    manager = _manager(settings, store, llm)

    messages = store.list_messages(sid, roles=[ROLE_USER, ROLE_ASSISTANT])
    text, source = manager._build_summary("", messages)

    assert source == "llm"
    assert "张先生" in text
    assert len(llm.calls) == 2


def test_retryable_error_is_retried(settings, store) -> None:
    from cs_agent.llm import LLMError

    sid = store.create_session()["id"]
    _history(store, sid, 6)
    llm = _ScriptedLLM([LLMError("连接中断", retryable=True), "【用户身份】李女士"])
    manager = _manager(settings, store, llm)

    messages = store.list_messages(sid, roles=[ROLE_USER, ROLE_ASSISTANT])
    text, source = manager._build_summary("", messages)

    assert source == "llm"
    assert "李女士" in text


def test_non_retryable_error_gives_up_quickly(settings, store) -> None:
    """不可重试的错误（如鉴权失败）不该白白重试三次。"""
    from cs_agent.llm import LLMError

    sid = store.create_session()["id"]
    _history(store, sid, 6)
    llm = _ScriptedLLM([LLMError("401 未授权", retryable=False)] * 5)
    manager = _manager(settings, store, llm)

    messages = store.list_messages(sid, roles=[ROLE_USER, ROLE_ASSISTANT])
    _, source = manager._build_summary("", messages)

    assert source == "extractive"
    assert len(llm.calls) == 1, "不可重试的错误应立即放弃"


def test_offline_never_calls_model(settings, store) -> None:
    sid = store.create_session()["id"]
    _history(store, sid, 6)
    llm = _ScriptedLLM([], online=False)
    manager = _manager(settings, store, llm)

    messages = store.list_messages(sid, roles=[ROLE_USER, ROLE_ASSISTANT])
    _, source = manager._build_summary("", messages)

    assert source == "extractive"
    assert llm.calls == []


# ---------------------------------------------------------------------------
# 抽取式摘要
# ---------------------------------------------------------------------------
def test_extractive_summary_does_not_recurse(settings, store) -> None:
    """关键回归：反复压缩不能让旧摘要层层套娃（曾导致摘要指数膨胀）。"""
    sid = store.create_session()["id"]
    _history(store, sid, 4)
    manager = _manager(settings, store, _ScriptedLLM([], online=False))
    messages = store.list_messages(sid, roles=[ROLE_USER, ROLE_ASSISTANT])

    summary = ""
    for _ in range(6):
        summary = manager._extractive_summary(summary, messages, budget=600)

    assert summary.count("【摘要来源】") == 1
    assert summary.count("【此前要点】") <= 1
    assert len(summary) <= 600


def test_extractive_summary_respects_budget(settings, store) -> None:
    sid = store.create_session()["id"]
    _history(store, sid, 8, text="这是一段比较长的用户描述，用来把摘要撑大")
    manager = _manager(settings, store, _ScriptedLLM([], online=False))
    messages = store.list_messages(sid, roles=[ROLE_USER, ROLE_ASSISTANT])

    summary = manager._extractive_summary("", messages, budget=200)
    assert len(summary) <= 200 + len("…（已截断）")


def test_summary_budget_has_sane_floor_and_ceiling(settings, store) -> None:
    manager = _manager(settings, store, _ScriptedLLM([], online=False))
    assert manager._summary_budget(100) == 400, "预算有下限，太小会导致模型必然超标"
    assert manager._summary_budget(10**6) <= settings.history_compress_chars * 0.5 + 1
