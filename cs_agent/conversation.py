"""会话服务：P1 的核心——一轮对话的完整流程。

一轮对话做四件事：

1. 落库用户消息（**先落库再调模型**：模型调用失败也不能丢用户的话）；
2. 组装工作记忆（最近 N 轮 + 摘要/状态/画像的占位）；
3. 调模型流式产出，边产出边通过事件流推给前端；
4. 落库助手消息 + 更新会话元信息 + 记录本轮轨迹。

事件流协议（SSE）：
    event: session  data: {"session_id": ..., "turn": ..., "created": bool}
    event: start    data: {"turn": ..., "history_messages": N, "prompt_chars": N,
                           "prompt_token_estimate": N, "compressed": bool, "offline": bool}
    event: delta    data: {"text": "增量"}
    event: done     data: {"turn": ..., "content": "...", "latency_ms": N,
                           "first_token_ms": N, "usage": {...}, "offline": bool}
    event: error    data: {"message": "...", "retryable": bool}
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from .config import Settings, get_settings
from .handoff import HandoffResult, HandoffService
from .handoff_rules import (
    EscalationEngine,
    EscalationVerdict,
    TurnObservation,
    analyse_sentiment,
)
from .llm import LLMClient, LLMError
from .memory import MemoryFlags, MemoryManager
from .planner import Plan, SessionPlanner
from .prompt import build_chat_messages
from .state import StateSnapshot
from .storage import (
    ROLE_ASSISTANT,
    ROLE_HUMAN,
    ROLE_SUMMARY,
    ROLE_USER,
    SESSION_ACTIVE,
    SESSION_WAITING_HUMAN,
    SessionStore,
    estimate_tokens,
)
from .tasks import task_for_intent
from .tools.executor import SkillExecutor, ToolPlan

logger = logging.getLogger(__name__)


@dataclass
class TurnPrep:
    """一轮请求的准备工作结果（便于测试与观测）。"""

    session: Dict[str, Any]
    turn: int
    history_messages: int
    prompt_chars: int
    prompt_token_estimate: int
    compressed: bool
    offline: bool
    title: str = ""
    memory: Optional[MemoryFlags] = None
    plan: Optional[Plan] = None
    state: Dict[str, Any] = field(default_factory=dict)
    tools: Optional[ToolPlan] = None
    #: 本轮真正发给模型的消息（含工具结果等临时上下文）
    messages: List[Dict[str, str]] = field(default_factory=list)
    #: 转人工判定结果与本轮工单
    escalation: Optional[EscalationVerdict] = None
    handoff: Optional[HandoffResult] = None


def _merge_flags(first: MemoryFlags, second: MemoryFlags) -> MemoryFlags:
    """合并"事实抽取"与"滚动压缩"两批记忆操作的产出。"""
    merged = MemoryFlags(
        facts=second.facts or first.facts,
        fact_keys=sorted(set(first.fact_keys) | set(second.fact_keys)),
        fact_changes=[*first.fact_changes, *second.fact_changes],
        summary_updated=first.summary_updated or second.summary_updated,
        summary_source=second.summary_source or first.summary_source,
        compressed_messages=first.compressed_messages + second.compressed_messages,
        compressed_chars=first.compressed_chars + second.compressed_chars,
        summary_chars=second.summary_chars or first.summary_chars,
        compression_ratio=second.compression_ratio or first.compression_ratio,
        profile_updated=first.profile_updated or second.profile_updated,
    )
    sources: Dict[str, int] = {}
    for source, count in {**first.extract_sources}.items():
        sources[source] = sources.get(source, 0) + count
    for source, count in {**second.extract_sources}.items():
        sources[source] = sources.get(source, 0) + count
    merged.extract_sources = sources
    return merged


@dataclass
class TurnTrace:
    """一轮对话的可观测信息。"""

    turn: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_chars: int = 0
    history_messages: int = 0
    compressed: bool = False
    latency_ms: int = 0
    first_token_ms: int = 0
    offline: bool = False
    model: str = ""
    error: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        data = {
            "turn": self.turn,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "prompt_chars": self.prompt_chars,
            "history_messages": self.history_messages,
            "compressed": self.compressed,
            "latency_ms": self.latency_ms,
            "first_token_ms": self.first_token_ms,
            "offline": self.offline,
            "model": self.model,
        }
        if self.error:
            data["error"] = self.error
        data.update(self.extra)
        return data


class ConversationService:
    """把存储、记忆、模型串成一轮对话。"""

    def __init__(
        self,
        store: SessionStore,
        settings: Optional[Settings] = None,
        llm: Optional[LLMClient] = None,
        memory: Optional[MemoryManager] = None,
        planner: Optional[SessionPlanner] = None,
        skills: Optional[SkillExecutor] = None,
        escalation: Optional[EscalationEngine] = None,
        handoffs: Optional[HandoffService] = None,
    ) -> None:
        self.store = store
        self.settings = settings or get_settings()
        self.llm = llm or LLMClient(self.settings)
        self.memory = memory or MemoryManager(self.store, self.settings, self.llm)
        self.planner = planner or SessionPlanner()
        self.skills = skills or SkillExecutor()
        self.escalation = escalation or EscalationEngine()
        self.handoffs = handoffs or HandoffService(self.store, self.settings)

    # ------------------------------------------------------------------
    # 会话
    # ------------------------------------------------------------------
    def open_session(self, user_id: str = "anonymous", title: str = "") -> Dict[str, Any]:
        return self.store.create_session(user_id=user_id, title=title)

    def history(self, session_id: str) -> Dict[str, Any]:
        """返回可展示的会话历史（不含摘要类虚拟消息）。"""
        messages = self.store.list_messages(
            session_id, roles=[ROLE_USER, ROLE_ASSISTANT]
        )
        return {
            "session": self.store.get_session(session_id),
            "messages": messages,
        }

    # ------------------------------------------------------------------
    # 一轮对话的准备（同步部分，便于测试）
    # ------------------------------------------------------------------
    def prepare_turn(self, session_id: str, user_text: str) -> TurnPrep:
        """落库用户消息、抽取事实、组装本轮工作记忆。"""
        session = self.store.get_session(session_id)
        if session is None:
            raise KeyError(f"会话不存在：{session_id}")

        user_text = (user_text or "").strip()
        if not user_text:
            raise ValueError("消息内容不能为空")
        if len(user_text) > self.settings.max_message_chars:
            raise ValueError(
                f"消息过长（{len(user_text)} 字），请压缩到 {self.settings.max_message_chars} 字以内"
            )

        turn = int(session["turn_count"]) + 1
        self.store.append_message(session_id, ROLE_USER, user_text, turn=turn)

        # 首条用户消息直接作为会话标题，便于会话列表识别
        title = ""
        if turn == 1 and not session["title"]:
            title = user_text[:24]
            self.store.update_session(session_id, title=title)

        # ---- 记忆：抽取本轮事实（正则高置信 + LLM 补漏）----
        memory_flags = self.memory.ingest_user_message(session_id, user_text, turn)

        # ---- 记忆：超阈值时滚动压缩历史 ----
        compression_flags = self.memory.compress_if_needed(session_id, turn)

        # ---- 状态：登记待办、灌槽位、解析指代、决定是否追问 ----
        snapshot = StateSnapshot.load(self.store, session_id)
        facts = self.store.get_facts(session_id)
        plan = self.planner.plan(snapshot, facts, user_text, turn=turn)
        # 本轮新抽到的槽位（订单号等）要落回状态，供下一轮复用
        snapshot.fill_from_facts(facts)

        # ---- 技能：执行工具（含写操作确认）----
        state_data = self.store.get_state(session_id)
        pending_write = state_data.get("pending_write") or None
        tool_plan = self.skills.execute(plan, snapshot, user_text, pending_write=pending_write)
        # 新的待确认写操作要挂起；已处理完的（确认执行 / 用户取消）要清掉
        if tool_plan.confirmation_prompt:
            state_data["pending_write"] = self._extract_pending(tool_plan)
        elif pending_write and tool_plan.cleared_pending:
            state_data.pop("pending_write", None)

        # ---- 转人工：按规则判定（模型不参与"要不要转"的决策）----
        verdict = self._evaluate_escalation(session_id, user_text, plan, tool_plan, state_data)
        # 先落库再建工单：交接单要带上本轮刚登记的待办与槽位，
        # 否则坐席看到的是"上一轮为止"的状态，会漏掉最新诉求。
        state_data["sentiment_score"] = verdict.sentiment.score
        state_data["escalation"] = verdict.as_dict()
        self.store.save_state(session_id, state_data)
        snapshot.save(self.store, session_id)

        handoff: Optional[HandoffResult] = None
        if verdict.should_handoff:
            handoff = self.handoffs.create(session_id, verdict)

        recent = self.memory.working_messages(session_id)
        context = self.memory.context(session_id)
        prompt_state = {**(context["state"] or {}), **plan.filled_slots}

        messages = self._compose_messages(recent, context, prompt_state, plan, tool_plan, handoff)
        prompt_chars = sum(len(m["content"]) for m in messages)

        return TurnPrep(
            session=self.store.get_session(session_id) or session,
            turn=turn,
            history_messages=len(recent),
            prompt_chars=prompt_chars,
            prompt_token_estimate=sum(estimate_tokens(m["content"]) for m in messages),
            compressed=context["summary"] is not None,
            offline=not self.llm.online,
            title=title,
            memory=_merge_flags(memory_flags, compression_flags),
            plan=plan,
            state=snapshot.summary(),
            tools=tool_plan,
            messages=messages,
            escalation=verdict,
            handoff=handoff,
        )

    def _evaluate_escalation(
        self,
        session_id: str,
        user_text: str,
        plan: Plan,
        tool_plan: ToolPlan,
        state_data: Dict[str, Any],
    ) -> EscalationVerdict:
        """按规则判定是否转人工。

        历史（意图序列与工具成功记录）从**每轮轨迹**里重建——
        轨迹本来就是落库的，不需要额外维护一份统计，也避免两处口径不一致。
        """
        previous = int(state_data.get("sentiment_score") or 0)
        history_intents: List[List[str]] = []
        history_tools: List[bool] = []
        for message in self.store.list_messages(session_id, roles=[ROLE_ASSISTANT]):
            trace = (message.get("meta") or {}).get("trace") or {}
            plan_data = trace.get("plan") or {}
            intents = list(plan_data.get("intents") or [])
            history_intents.append(intents)
            runs = (trace.get("tools") or {}).get("runs") or []
            history_tools.append(any(run.get("ok") for run in runs))

        observation = TurnObservation(
            turn=int((self.store.get_session(session_id) or {}).get("turn_count") or 0),
            intents=list(plan.intents),
            tool_ok=any(run.ok for run in tool_plan.runs),
            asked_question=plan.action == "ask",
            intent_history=history_intents,
            tool_ok_history=history_tools,
        )
        return self.escalation.evaluate(user_text, observation, cumulative_sentiment=previous)

    def _compose_messages(
        self,
        recent: List[Dict[str, Any]],
        context: Dict[str, Any],
        prompt_state: Dict[str, Any],
        plan: Optional[Plan],
        tool_plan: Optional[ToolPlan],
        handoff: Optional[HandoffResult] = None,
    ) -> List[Dict[str, str]]:
        """唯一的工作记忆组装入口。

        流式与调试（``build_messages``）都走这里——避免出现
        "调试时看到的提示词和真正发给模型的不一样"这类两套逻辑不一致的问题。
        """
        extra_parts: List[str] = []
        if plan is not None:
            hint = self._plan_hint(plan)
            if hint:
                extra_parts.append(hint)
        if tool_plan is not None:
            hint = self._tool_hint(tool_plan)
            if hint:
                extra_parts.append(hint)
        if handoff is not None:
            extra_parts.append(self._handoff_hint(handoff))

        return build_chat_messages(
            recent,
            summary=context.get("summary"),
            state=prompt_state,
            profile=context.get("profile"),
            extra_system="\n\n".join(extra_parts) if extra_parts else None,
        )

    @staticmethod
    def _handoff_hint(handoff: HandoffResult) -> str:
        """转人工时给模型的指令：把转接这件事说清楚，且**不许自称能解决**。"""
        lines = [
            "【本轮已转人工】",
            f"- 工单号：{handoff.handoff_id}",
            f"- 转接原因：{handoff.reason}",
            "请据此回复用户：",
            "1. 明确告知已转接人工，并给出工单号；",
            "2. 说明大致原因（如超出客服权限 / 需要人工核实），不要甩锅、不要含糊；",
            "3. 告诉用户已确认过的信息都会带给人工客服，不需要重复提供；",
            "4. 不要承诺你自己能解决，也不要给出超出权限的补偿或时效承诺。",
        ]
        return "\n".join(lines)

    @staticmethod
    def _extract_pending(tool_plan: ToolPlan) -> Dict[str, Any]:
        """从本轮结果里取回待确认写操作的参数，用于下一轮续办。"""
        for run in tool_plan.runs:
            if run.needs_confirmation:
                return {"tool": run.tool, "params": run.params}
        return {}

    @staticmethod
    def _tool_hint(tool_plan: ToolPlan) -> str:
        """把工具事实与"必须如实告知"的失败拼进系统提示。"""
        parts: List[str] = []
        if tool_plan.context:
            parts.append(tool_plan.context)
        if tool_plan.confirmation_prompt:
            parts.append(
                "【需要用户确认的操作】\n"
                f"{tool_plan.confirmation_prompt}\n"
                "请把这件事讲清楚并请用户确认，不要自行执行，也不要说已经办好了。"
            )
        if tool_plan.honest_failures:
            failures = "\n".join(f"- {item}" for item in tool_plan.honest_failures)
            parts.append(
                "【必须如实告知的情况（不要掩饰、不要编造替代答案）】\n" + failures
            )
        if tool_plan.policy_citations:
            cites = "\n".join(f"- {item}" for item in tool_plan.policy_citations)
            parts.append("【回答政策问题时的依据来源】\n" + cites)
        return "\n\n".join(parts)

    @staticmethod
    def _plan_hint(plan: Plan) -> str:
        """把本轮的"现状"告诉模型：正在办什么、待办有哪些、是否在等补充信息。

        这不是让模型做决策（决策在 ``SessionPlanner`` 里已经做完了），
        而是让它的措辞与当前状态一致——例如别在用户等答复时突然反问一件无关的事。
        """
        lines: List[str] = []
        if plan.task_kind:
            task_def = task_for_intent(plan.task_kind)
            name = task_def.name if task_def else plan.task_kind
            lines.append(f"用户当前正在处理：{name}")
        if plan.pending_tasks:
            names = []
            for kind in plan.pending_tasks:
                task_def = task_for_intent(kind)
                names.append(task_def.name if task_def else kind)
            lines.append("还有待处理的诉求：" + "、".join(names) + "（办完当前这件事后再回应）")
        if plan.ask is not None:
            lines.append(
                "本轮系统已向用户提问，请围绕这个提问来组织回复，不要另外再问别的问题。"
            )
        if plan.clarification:
            lines.append("本轮需要先请用户澄清指代，请围绕这一点回复。")
        if not lines:
            return ""
        return "【本轮会话状态】\n" + "\n".join(f"- {line}" for line in lines)

    def build_messages(self, session_id: str) -> List[Dict[str, str]]:
        """组装当前工作记忆（不发请求），用于调试与测试。

        只包含"状态层"的内容；工具结果属于**本轮**的临时上下文，
        由 :meth:`prepare_turn` 产出并随 ``TurnPrep.messages`` 一起使用。
        """
        session = self.store.get_session(session_id)
        if session is None:
            raise KeyError(f"会话不存在：{session_id}")
        recent = self.memory.working_messages(session_id)
        context = self.memory.context(session_id)
        snapshot = StateSnapshot.load(self.store, session_id)
        prompt_state = {**(context["state"] or {}), **snapshot.slots}
        return self._compose_messages(recent, context, prompt_state, None, None)

    # ------------------------------------------------------------------
    # 事件流
    # ------------------------------------------------------------------
    def stream_turn(self, session_id: str, user_text: str) -> Iterator[str]:
        """执行一轮对话并产出 SSE 格式的文本行。"""
        started = time.perf_counter()
        trace = TurnTrace()

        try:
            prep = self.prepare_turn(session_id, user_text)
        except (KeyError, ValueError) as exc:
            yield _sse("error", {"message": str(exc), "retryable": False})
            return

        trace.turn = prep.turn
        trace.history_messages = prep.history_messages
        trace.prompt_chars = prep.prompt_chars
        trace.prompt_tokens = prep.prompt_token_estimate
        trace.compressed = prep.compressed
        trace.offline = prep.offline
        trace.model = "offline-template" if prep.offline else self.settings.model
        if prep.memory is not None:
            trace.extra["memory"] = prep.memory.as_dict()
        if prep.plan is not None:
            trace.extra["plan"] = prep.plan.as_dict()
        if prep.state:
            trace.extra["state"] = prep.state
        if prep.tools is not None and prep.tools.called:
            trace.extra["tools"] = prep.tools.as_dict()
        if prep.escalation is not None:
            trace.extra["escalation"] = prep.escalation.as_dict()
        if prep.handoff is not None:
            trace.extra["handoff"] = {
                "handoff_id": prep.handoff.handoff_id,
                "reason": prep.handoff.reason,
                "already_waiting": prep.handoff.already_waiting,
            }

        # 本轮真正发给模型的系统提示（含工具结果等临时上下文）——
        # 暴露出来才能观测"模型到底看到了什么"，否则调试只能靠猜。
        system_prompt = prep.messages[0]["content"] if prep.messages else ""
        # 完整消息列表也一并暴露：只看 system prompt 会漏掉人工坐席消息、
        # 历史原文这些**独立消息**，容易误判成"信息丢了"。
        prompt_messages = [
            {"role": item["role"], "content": item["content"][:4000]}
            for item in prep.messages
        ]

        yield _sse(
            "session",
            {
                "session_id": session_id,
                "turn": prep.turn,
                "created": False,
                "title": prep.session.get("title", ""),
            },
        )
        yield _sse(
            "start",
            {
                "turn": prep.turn,
                "history_messages": prep.history_messages,
                "prompt_chars": prep.prompt_chars,
                "prompt_token_estimate": prep.prompt_token_estimate,
                "compressed": prep.compressed,
                "offline": prep.offline,
                "model": trace.model,
                "memory": (prep.memory.as_dict() if prep.memory is not None else {}),
                "plan": (prep.plan.as_dict() if prep.plan is not None else {}),
                "state": prep.state,
                "tools": (prep.tools.as_dict() if prep.tools is not None else {}),
                "escalation": (prep.escalation.as_dict() if prep.escalation is not None else {}),
                "handoff": (
                    {
                        "handoff_id": prep.handoff.handoff_id,
                        "reason": prep.handoff.reason,
                        "already_waiting": prep.handoff.already_waiting,
                        "notice": self.handoffs.user_notice(prep.handoff),
                    }
                    if prep.handoff is not None
                    else {}
                ),
                "system_prompt": system_prompt,
                "messages": prompt_messages,
            },
        )

        messages = prep.messages or self.build_messages(session_id)
        pieces: List[str] = []
        first_token_ms = 0
        error: Optional[str] = None

        try:
            for piece in self.llm.stream_chat(messages):
                if not pieces:
                    first_token_ms = int((time.perf_counter() - started) * 1000)
                pieces.append(piece)
                yield _sse("delta", {"text": piece})
        except LLMError as exc:
            error = str(exc)
            logger.warning("第 %s 轮模型调用失败：%s", prep.turn, exc)
            trace.error = error
            yield _sse("error", {"message": error, "retryable": exc.retryable})
        except Exception as exc:  # pragma: no cover - 防御性分支
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("第 %s 轮出现未预期错误", prep.turn)
            trace.error = error
            yield _sse("error", {"message": error, "retryable": False})

        content = "".join(pieces).strip()
        trace.first_token_ms = first_token_ms
        trace.latency_ms = int((time.perf_counter() - started) * 1000)
        trace.completion_tokens = estimate_tokens(content)
        trace.total_tokens = trace.prompt_tokens + trace.completion_tokens

        # 即使失败也要落库已产出的部分：用户看到的内容与历史必须一致
        if content or not error:
            self.store.append_message(
                session_id,
                ROLE_ASSISTANT,
                content,
                turn=prep.turn,
                meta={"trace": trace.as_dict(), "partial": bool(error)},
            )
            self.store.update_session(session_id, bump_turn=True)

        yield _sse(
            "done",
            {
                "turn": prep.turn,
                "content": content,
                "latency_ms": trace.latency_ms,
                "first_token_ms": trace.first_token_ms,
                "usage": {
                    "prompt_tokens": trace.prompt_tokens,
                    "completion_tokens": trace.completion_tokens,
                    "total_tokens": trace.total_tokens,
                },
                "offline": trace.offline,
                "model": trace.model,
                "error": error or "",
                "trace": trace.as_dict(),
            },
        )

    def chat_once(self, session_id: str, user_text: str) -> Dict[str, Any]:
        """非流式跑一轮（测试与命令行用），返回最终结果。

        注意错误判定用 ``is not None`` 而不是真值判断：``done`` 事件在**成功**时
        ``error`` 字段是空字符串，用真值判断会把成功结果当成失败提前返回。
        """
        final: Optional[Dict[str, Any]] = None
        failure: Optional[str] = None

        for frame in self.stream_turn(session_id, user_text):
            event, payload = _parse_frame(frame)
            if event == "done":
                final = payload
                break
            if event == "error":
                failure = str(payload.get("message") or "")

        if final is not None:
            return final
        return {"error": failure or "本轮对话未能完成"}

    # ------------------------------------------------------------------
    # 轨迹
    # ------------------------------------------------------------------
    def latest_trace(self, session_id: str) -> Optional[Dict[str, Any]]:
        messages = self.store.list_messages(session_id, roles=[ROLE_ASSISTANT])
        if not messages:
            return None
        return (messages[-1].get("meta") or {}).get("trace")

    def session_stats(self, session_id: str) -> Dict[str, Any]:
        """会话级统计：轮数、字符数、token 估算、记忆规模与最近一次轨迹。

        P6 会用它画"长会话的 token 增长曲线"，所以口径在这里定死：
        **按落库内容估算**，而不是按调用时的 prompt（后者重叠计算，会虚高）。
        """
        messages = self.store.list_messages(session_id, roles=[ROLE_USER, ROLE_ASSISTANT])
        total_chars = sum(len(m["content"]) for m in messages)
        summary = self.store.latest_summary(session_id)
        working = self.memory.working_messages(session_id)
        return {
            "session_id": session_id,
            "messages": len(messages),
            "turns": len(messages) // 2,
            "total_chars": total_chars,
            "token_estimate": sum(m["token_estimate"] for m in messages),
            "has_summary": summary is not None,
            "summary_chars": len(summary["content"]) if summary else 0,
            # 长会话关心的不是"总共多大"，而是"每次真正送进模型的有多少"
            "working_messages": len(working),
            "working_chars": sum(len(m["content"]) for m in working)
            + (len(summary["content"]) if summary else 0),
            "latest_trace": self.latest_trace(session_id),
        }


def _sse(event: str, data: Dict[str, Any]) -> str:
    """把事件编码为 SSE 帧（``event:`` + 单行 ``data:`` + 空行结束）。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _parse_frame(frame: str) -> tuple:
    """解析单个 SSE 帧，返回 ``(event, payload)``。

    与 ``_sse`` 互为逆操作，供 ``chat_once`` 复用同一条执行路径——
    非流式与流式走同一生成器，就不会出现"两种模式行为不一致"。
    """
    event = ""
    data_lines: List[str] = []
    for line in frame.splitlines():
        if line.startswith("event: "):
            event = line[7:].strip()
        elif line.startswith("data: "):
            data_lines.append(line[6:])
    if not data_lines:
        return event, {}
    try:
        return event, json.loads("\n".join(data_lines))
    except json.JSONDecodeError:  # pragma: no cover - 防御性分支
        return event, {}
