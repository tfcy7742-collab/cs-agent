"""记忆管理器：分层记忆的调度中心。

职责
----
1. **事实抽取与版本化**（每轮，用户消息落库后）：正则先行、LLM 补漏，写进会话事实；
2. **滚动摘要**（超阈值时）：把最老的一批原文压成结构化摘要，并标记原文已覆盖；
3. **跨会话画像**（摘要完成后）：把稳定事实与历史工单摘要沉淀到用户级画像。

一条重要原则：**摘要与事实是两条独立通道**。
摘要会丢细节（也有损），所以订单号这类关键事实必须结构化单独存——
"订单号只活在摘要里"是长会话系统最常见的翻车方式。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import Settings, get_settings
from .facts import (
    SESSION_ONLY_KEYS,
    extract_by_rules,
    extract_with_llm,
    merge_extractions,
    render_facts,
    select_profile_updates,
)
from .llm import LLMClient, LLMError
from .storage import (
    ROLE_ASSISTANT,
    ROLE_HUMAN,
    ROLE_SUMMARY,
    ROLE_USER,
    SessionStore,
    estimate_tokens,
)

logger = logging.getLogger(__name__)

#: 画像里保留的历史会话摘要条数
PROFILE_HISTORY_LIMIT = 5

SUMMARY_SYSTEM_PROMPT = """你是客服会话的摘要器。把给出的对话压缩成一段**事实准确的**摘要，供后续继续服务使用。

必须遵守：
1. 订单号、金额、时间、地址等**具体信息必须原样保留**，不许改写、不许模糊化；
2. 用户明确说过的诉求与已确认的处理方式必须保留；
3. 用户前后改过口的信息，写成"X（此前为 Y）"；
4. 没解决、待跟进的事项单独列出；
5. 只做压缩，不许补充对话里没有的信息，不许推测用户意图。

输出格式（纯文本，不要 Markdown）：
【用户身份】…
【已确认信息】…
【诉求与进展】…
【待办/未决】…
"""


@dataclass
class MemoryFlags:
    """一轮记忆操作的产出（进轨迹，便于观测与测试）。"""

    facts: Dict[str, Any] = field(default_factory=dict)
    fact_keys: List[str] = field(default_factory=list)
    fact_changes: List[Dict[str, Any]] = field(default_factory=list)
    summary_updated: bool = False
    summary_source: str = ""
    compressed_messages: int = 0
    compressed_chars: int = 0
    summary_chars: int = 0
    compression_ratio: float = 0.0
    profile_updated: bool = False
    extract_sources: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        data = {
            "fact_keys": self.fact_keys,
            "extract_sources": self.extract_sources,
            "summary_updated": self.summary_updated,
            "summary_source": self.summary_source,
            "compressed_messages": self.compressed_messages,
            "compressed_chars": self.compressed_chars,
            "summary_chars": self.summary_chars,
            "compression_ratio": round(self.compression_ratio, 3),
            "profile_updated": self.profile_updated,
        }
        if self.fact_changes:
            data["fact_changes"] = self.fact_changes
        return data


class MemoryManager:
    """分层记忆的读写。"""

    def __init__(
        self,
        store: SessionStore,
        settings: Optional[Settings] = None,
        llm: Optional[LLMClient] = None,
    ) -> None:
        self.store = store
        self.settings = settings or get_settings()
        self.llm = llm or LLMClient(self.settings)

    # ------------------------------------------------------------------
    # 读：给本轮组装工作记忆
    # ------------------------------------------------------------------
    def working_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """工作记忆里的原文：**未被摘要覆盖**的最近若干轮。

        包含人工坐席的回复（``human_agent``）：转人工之后用户接着说时，
        模型必须知道"人工已经答复过什么"，否则会重复人工说过的话。
        """
        return self.store.list_messages(
            session_id,
            limit=self.settings.working_recent_turns * 2,
            roles=[ROLE_USER, ROLE_ASSISTANT, ROLE_HUMAN],
            only_uncompressed=True,
        )

    def context(self, session_id: str) -> Dict[str, Any]:
        """组装工作记忆需要的三层内容。"""
        summary = self.store.latest_summary(session_id)
        facts = self.store.get_facts(session_id)
        session = self.store.get_session(session_id) or {}
        profile = self.store.get_profile(session.get("user_id", "anonymous"))
        return {
            "summary": (summary or {}).get("content") or None,
            "state": render_facts(facts),
            "profile": self._render_profile(profile),
            "facts": facts,
        }

    @staticmethod
    def _render_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
        """画像渲染：只给稳定属性与历史工单摘要，避免把过期细节塞进上下文。"""
        rendered: Dict[str, Any] = {}
        attributes = profile.get("attributes") or {}
        if attributes:
            rendered.update(attributes)
        history = profile.get("history") or []
        if history:
            recent = history[-2:]
            rendered["近期记录"] = " / ".join(
                str(item.get("summary", ""))[:80] for item in recent if item.get("summary")
            )
        stats = profile.get("stats") or {}
        if stats.get("sessions"):
            rendered["历史会话数"] = stats["sessions"]
        return {k: v for k, v in rendered.items() if v}

    # ------------------------------------------------------------------
    # 写：每轮抽取事实
    # ------------------------------------------------------------------
    def ingest_user_message(
        self,
        session_id: str,
        text: str,
        turn: int,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> MemoryFlags:
        """抽取本轮用户消息里的事实并并入会话事实（值变化时保留旧值）。"""
        flags = MemoryFlags()
        if not (text or "").strip():
            return flags

        rules = extract_by_rules(text)
        semantic: Dict[str, Dict[str, Any]] = {}
        # 离线下跳过 LLM 抽取：正则是确定性的，仍然可用
        if self.llm.online:
            semantic = extract_with_llm(text, self.llm, history)
        merged = merge_extractions(rules, semantic)

        if merged:
            changes = self.store.merge_facts(session_id, merged, turn=turn)
            for change in changes:
                self.store.mark_fact_changed(session_id, change["key"], turn)
                change["message"] = self._describe_change(change)
            flags.fact_changes = changes

        sources: Dict[str, int] = {}
        for payload in merged.values():
            origin = str(payload.get("source") or "unknown")
            sources[origin] = sources.get(origin, 0) + 1
        flags.extract_sources = sources
        flags.facts = self.store.get_facts(session_id)
        flags.fact_keys = sorted(flags.facts.keys())
        return flags

    @staticmethod
    def _describe_change(change: Dict[str, Any]) -> str:
        return f"{change['key']}：{change.get('old')} → {change.get('new')}"

    # ------------------------------------------------------------------
    # 写：滚动摘要
    # ------------------------------------------------------------------
    def needs_compression(self, session_id: str) -> bool:
        """是否需要压缩。

        两个触发条件（任一满足）：
        * 未压缩原文条数超过"工作记忆窗口 ×2"——保证窗口里始终是最近的原文；
        * 未压缩原文总字符数超过 ``history_compress_chars``。
        """
        stats = self.store.uncompressed_stats(session_id)
        return (
            stats["count"] > self.settings.working_recent_turns * 2
            or stats["chars"] > self.settings.history_compress_chars
        )

    def compress_if_needed(self, session_id: str, turn: int = 0) -> MemoryFlags:
        """超阈值时滚动压缩最老的一批原文。"""
        flags = MemoryFlags()
        if not self.needs_compression(session_id):
            return flags
        return self.compress(session_id, turn=turn)

    def compress(self, session_id: str, turn: int = 0) -> MemoryFlags:
        """把"最近窗口之外"的原文压进摘要。

        返回的 ``compression_ratio`` 是"摘要长度 / 被压缩原文长度"，
        这是"有损程度"的量化口径——我们会把它如实展示，而不是宣称无损。
        """
        flags = MemoryFlags()
        keep = self.settings.working_recent_turns * 2
        all_uncompressed = self.store.list_messages(
            session_id, roles=[ROLE_USER, ROLE_ASSISTANT], only_uncompressed=True
        )
        if len(all_uncompressed) <= keep:
            return flags

        to_fold = all_uncompressed[:-keep]
        if not to_fold:
            return flags

        previous = self.store.latest_summary(session_id)
        previous_text = (previous or {}).get("content") or ""

        summary_text, source = self._build_summary(previous_text, to_fold)
        compressed_chars = sum(len(m["content"]) for m in to_fold)

        last_id = int(to_fold[-1]["id"])
        last_turn = int(to_fold[-1]["turn"])
        marked = self.store.compress_up_to(session_id, last_id)
        self.store.save_summary(
            session_id,
            summary_text,
            covers_until_message_id=last_id,
            covers_until_turn=last_turn,
            stats={
                "source": source,
                "folded_messages": len(to_fold),
                "folded_chars": compressed_chars,
                "summary_chars": len(summary_text),
            },
        )

        flags.summary_updated = True
        flags.summary_source = source
        flags.compressed_messages = marked
        flags.compressed_chars = compressed_chars
        flags.summary_chars = len(summary_text)
        flags.compression_ratio = (
            len(summary_text) / compressed_chars if compressed_chars else 0.0
        )
        self.sync_profile(session_id)
        flags.profile_updated = True
        logger.info(
            "会话 %s 压缩完成：%s 条 -> %s 字摘要（压缩比 %.2f，来源 %s）",
            session_id,
            marked,
            len(summary_text),
            flags.compression_ratio,
            source,
        )
        return flags

    def _llm_call_with_retry(
        self,
        messages: List[Dict[str, str]],
        *,
        attempts: int = 3,
        temperature: float = 0.1,
        max_tokens: int = 900,
    ):
        """带重试的 LLM 调用。

        摘要与事实抽取都是**非流式**调用，重试是安全幂等的；
        而摘要一旦生成就会长期影响后续所有轮次，因此值得多试两次
        （实测供应商侧会偶发 RemoteDisconnected 与空响应）。
        """
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                result = self.llm.chat(
                    messages, temperature=temperature, max_tokens=max_tokens
                )
                if (result.text or "").strip():
                    return result
                last_error = LLMError("模型返回空内容")
            except LLMError as exc:
                last_error = exc
                if not exc.retryable:
                    break
            except Exception as exc:  # pragma: no cover - 防御性分支
                last_error = exc
            if attempt < attempts:
                time.sleep(0.5 * attempt)
        logger.warning("LLM 调用重试 %s 次后仍未成功：%s", attempts, last_error)
        return None

    def _summary_budget(self, fold_chars: int) -> int:
        """摘要的字符预算。

        上限取"被折叠原文的 40%"与"历史阈值的一半"里的较小值——
        否则摘要会随会话无限增长，"压缩"就变成了换一种方式膨胀。
        下限 400 字：摘要要同时容纳身份、已确认信息、进展与待办，
        预算压得太低（实测设 120 时）模型几乎必然超标、只能一路退化成抽取式。
        """
        return max(
            int(min(fold_chars * 0.4, self.settings.history_compress_chars * 0.5)), 400
        )

    def _build_summary(
        self, previous_summary: str, messages: List[Dict[str, Any]]
    ) -> tuple:
        """生成摘要文本。

        优先用模型做真正的语义压缩；不可用（离线/调用失败）时退化为**抽取式**——
        保留事实与用户原话的可读片段。宁可摘要长一点，也不能凭空编造。
        """
        transcript = self._render_transcript(messages)
        fold_chars = sum(len(m["content"]) for m in messages)
        budget = self._summary_budget(fold_chars)

        if self.llm.online:
            prompt_parts = []
            if previous_summary:
                prompt_parts.append(
                    "【此前已有的摘要（请与新内容**合并**，不要丢失其中信息，"
                    "也不要把这段摘要本身再抄一遍）】\n" + previous_summary
                )
            prompt_parts.append("【需要并入摘要的新对话】\n" + transcript)
            # 给模型的额度留出余量：模型对"字数"的控制本来就不精确，
            # 卡着硬预算要求它，几乎必然超标（实测设 400 时稳定产出 500+）。
            prompt_parts.append(
                f"直接输出合并后的摘要正文（不要开场白、不要复述指令），"
                f"总长度控制在 {int(budget * 0.6)} 字以内。"
            )
            try:
                result = self._llm_call_with_retry(
                    [
                        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                        {"role": "user", "content": "\n\n".join(prompt_parts)},
                    ]
                )
                text = (result.text or "").strip() if result else ""
                if not text:
                    logger.warning("摘要多次重试后仍无内容，退化为抽取式")
                elif len(text) > budget * 1.25:
                    # 超标就让它**改写自己**（真正的二次压缩），而不是直接丢掉 LLM 摘要——
                    # 否则每超标一次就退回抽取式，摘要质量会反复掉档。
                    tightened = self._tighten_summary(text, budget)
                    if tightened:
                        return tightened, "llm+compressed"
                    logger.warning(
                        "摘要改写未成功（%s > %s），退化为抽取式", len(text), budget
                    )
                else:
                    return text, "llm"
            except Exception as exc:  # pragma: no cover - 防御性分支
                logger.warning("摘要出现未预期错误，退化为抽取式：%s", exc)

        return self._extractive_summary(previous_summary, messages, budget), "extractive"

    def _tighten_summary(self, text: str, budget: int) -> str:
        """二次压缩：把已超标的摘要改写到预算内，要求只压缩、不新增。"""
        result = self._llm_call_with_retry(
            [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "下面这段摘要太长了，请在不丢失任何关键信息"
                        "（尤其是订单号、金额、时间、地址、已确认处理方式、未决事项）"
                        f"的前提下压缩到 {int(budget * 0.85)} 字以内。"
                        "只做压缩，不许新增或推测信息。直接输出压缩后的摘要正文。\n\n"
                        + text
                    ),
                },
            ]
        )
        tightened = (result.text or "").strip() if result else ""
        if tightened and len(tightened) <= budget:
            return tightened
        return ""

    @staticmethod
    def _render_transcript(messages: List[Dict[str, Any]]) -> str:
        lines = []
        for item in messages:
            speaker = "用户" if item["role"] == ROLE_USER else "客服"
            lines.append(f"{speaker}：{item['content'].strip()}")
        return "\n".join(lines)

    @staticmethod
    def _extractive_summary(
        previous_summary: str, messages: List[Dict[str, Any]], budget: int = 600
    ) -> str:
        """抽取式兜底摘要。

        保留：用户原话（截断）+ 客服答复的**结论句**。
        明确标注这是抽取式，避免读者误以为经过了语义压缩。

        两个刻意的约束：
        * **只继承旧摘要里的"要点"部分**（不是整段照抄）——否则每压缩一次就套一层，
          摘要会指数级膨胀（实测踩过）；
        * 总长度受 ``budget`` 限制，保证"压缩"始终是压缩。
        """
        user_lines: List[str] = []
        assistant_points: List[str] = []

        for item in messages:
            content = item["content"].strip().replace("\n", " ")
            if item["role"] == ROLE_USER:
                user_lines.append(content[:100])
            else:
                # 取每条答复的前两句，通常就是结论
                sentences = [
                    s
                    for s in content.replace("！", "。").replace("？", "。").split("。")
                    if s.strip()
                ]
                assistant_points.extend(s.strip()[:60] for s in sentences[:2])

        fact_keys: List[str] = []
        for text in user_lines:
            fact_keys.extend(extract_by_rules(text).keys())

        parts = ["【摘要来源】抽取式（未启用模型语义压缩）"]

        inherited = MemoryManager._inherited_points(previous_summary)
        if inherited:
            parts.append("【此前要点】" + inherited[: max(budget // 3, 60)])
        parts.append("【用户说过的】" + "；".join(user_lines[-8:]))
        if assistant_points:
            parts.append("【已答复的要点】" + "；".join(dict.fromkeys(assistant_points)))
        parts.append("【涉及字段】" + ("、".join(sorted(set(fact_keys))) or "无"))

        text = "\n".join(parts)
        if len(text) > budget:
            text = MemoryManager._truncate_at_boundary(text, budget)
        return text

    @staticmethod
    def _truncate_at_boundary(text: str, budget: int) -> str:
        """按句子边界截断，避免出现"半句话"的摘要。"""
        head = text[:budget]
        for mark in ("。", "；", "，", "\n"):
            index = head.rfind(mark)
            if index > budget * 0.6:
                return head[: index + 1] + "…（已截断）"
        return head + "…（已截断）"

    @staticmethod
    def _inherited_points(previous_summary: str) -> str:
        """从旧摘要里取出可继承的"要点"，避免整段递归复制。"""
        if not previous_summary:
            return ""
        collected: List[str] = []
        for line in previous_summary.splitlines():
            if line.startswith(("【此前要点】", "【已答复的要点】", "【涉及字段】")):
                collected.append(line.split("】", 1)[-1].strip())
        return "；".join(part for part in collected if part)

    # ------------------------------------------------------------------
    # 写：跨会话画像
    # ------------------------------------------------------------------
    def sync_profile(self, session_id: str) -> Dict[str, Any]:
        """把会话里的稳定事实与摘要沉淀到用户级画像。

        注意只取白名单字段：订单号这类单会话信息**绝不**进画像，
        否则用户开新会话会被上一单的信息污染。
        """
        session = self.store.get_session(session_id)
        if not session:
            return {}
        user_id = session.get("user_id") or "anonymous"

        profile = self.store.get_profile(user_id)
        attributes = dict(profile.get("attributes") or {})

        facts = self.store.get_facts(session_id)
        for key, value in select_profile_updates(facts).items():
            if key in SESSION_ONLY_KEYS:  # 双保险
                continue
            attributes[key] = value
        # 防御：历史遗留的会话级字段也从画像里剔除
        for key in SESSION_ONLY_KEYS:
            attributes.pop(key, None)

        history = list(profile.get("history") or [])
        summary = self.store.latest_summary(session_id)
        if summary and summary.get("content"):
            entry = {
                "session_id": session_id,
                "summary": summary["content"][:300],
                "updated_at": summary["created_at"],
            }
            history = [h for h in history if h.get("session_id") != session_id]
            history.append(entry)
            history = history[-PROFILE_HISTORY_LIMIT:]

        stats = dict(profile.get("stats") or {})
        stats["sessions"] = len({h.get("session_id") for h in history} | {session_id})
        stats["last_session_id"] = session_id
        stats["last_seen"] = session.get("updated_at") or ""

        profile.update(
            {
                "user_id": user_id,
                "attributes": attributes,
                "history": history,
                "stats": stats,
            }
        )
        self.store.save_profile(user_id, profile)
        return profile

    # ------------------------------------------------------------------
    # 观测
    # ------------------------------------------------------------------
    def memory_report(self, session_id: str) -> Dict[str, Any]:
        """长会话记忆的体检数据（P6 的评测与前端都会用）。"""
        session = self.store.get_session(session_id) or {}
        summary = self.store.latest_summary(session_id)
        facts = self.store.get_facts(session_id)
        stats = self.store.uncompressed_stats(session_id)
        all_messages = self.store.list_messages(
            session_id, roles=[ROLE_USER, ROLE_ASSISTANT]
        )
        total_chars = sum(len(m["content"]) for m in all_messages)
        summary_chars = len((summary or {}).get("content") or "")
        working = self.working_messages(session_id)
        working_chars = sum(len(m["content"]) for m in working)
        profile = self.store.get_profile(session.get("user_id", "anonymous"))

        return {
            "session_id": session_id,
            "turns": int(session.get("turn_count") or 0),
            "messages_total": len(all_messages),
            "messages_compressed": stats["count"] >= 0
            and len(all_messages) - stats["count"],
            "messages_uncompressed": stats["count"],
            "total_chars": total_chars,
            "uncompressed_chars": stats["chars"],
            "working_messages": len(working),
            "working_chars": working_chars,
            "has_summary": summary is not None,
            "summary_chars": summary_chars,
            "compression_ratio": (
                round(summary_chars / total_chars, 3) if total_chars else 0.0
            ),
            "facts": render_facts(facts),
            "fact_count": len(facts),
            "profile_attributes": (profile.get("attributes") or {}),
            "profile_sessions": (profile.get("stats") or {}).get("sessions", 0),
            "working_token_estimate": sum(
                estimate_tokens(m["content"]) for m in working
            )
            + estimate_tokens(summary_chars and (summary or {}).get("content") or ""),
        }
