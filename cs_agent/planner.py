"""会话规划器：把"意图 + 槽位 + 待办 + 指代"合成本轮该做什么。

一轮的规划顺序（顺序本身就是设计）：

1. **登记待办**：本轮提到的意图入栈，已有任务不重建；
2. **灌槽位**：把版本化事实（跨任务共享）与新抽到的事实写进槽位库；
3. **解析指代**：用户说"那这个能退吗"时把"这个"绑到订单；**绑不唯一就澄清，绝不猜**；
4. **决定追问**：只在"继续办这件事确实需要"时才问，且同一槽位不重复问；
5. **给出结论**：``action`` 为 ``ask``（先问再办）/ ``answer``（直接答）/ ``tool``（P4 接工具）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .facts import PROFILE_ALLOWED_KEYS, SESSION_ONLY_KEYS
from .state import (
    AskDecision,
    StateSnapshot,
    decide_ask,
    is_consult_only,
    known_slot_digest,
    resolve_order_reference,
)
from .tasks import (
    CONSULT_INTENTS,
    SLOT_ADDRESS,
    SLOT_EXPECTATION,
    SLOT_INVOICE_TITLE,
    SLOT_ITEM,
    SLOT_ORDER_ID,
    SLOT_QUESTION,
    SLOT_REASON,
    is_consultation,
    split_intents,
    task_for_intent,
)

#: 哪些事实可以当槽位用（事实键与槽位名同名的直接映射；不同名的在这里对齐）
FACT_TO_SLOT: Dict[str, str] = {
    SLOT_ORDER_ID: SLOT_ORDER_ID,
    SLOT_REASON: SLOT_REASON,
    SLOT_ITEM: SLOT_ITEM,
    SLOT_ADDRESS: SLOT_ADDRESS,
    SLOT_INVOICE_TITLE: SLOT_INVOICE_TITLE,
    SLOT_EXPECTATION: SLOT_EXPECTATION,
}

#: 不是槽位、但值得记住的会话事实（仅用于展示与画像）
EXTRA_FACT_KEYS = ("称呼", "城市", "联系方式", "会员等级", "沟通偏好", "常用收货地址", "涉及金额")

#: 明确表示"放弃当前任务"的说法
CANCEL_MARKERS = ("不用了", "算了", "不退了", "取消了", "不办了", "没事了")


@dataclass
class Plan:
    """本轮的行动计划（进轨迹，也决定对话侧怎么走）。"""

    action: str = "answer"  # ask | answer | tool
    intents: List[str] = field(default_factory=list)
    task_kind: str = ""
    ask: Optional[AskDecision] = None
    #: 需要先向用户澄清的指代（例如"这个"指哪一单）
    clarification: str = ""
    new_tasks: List[str] = field(default_factory=list)
    completed_tasks: List[str] = field(default_factory=list)
    pending_tasks: List[str] = field(default_factory=list)
    filled_slots: Dict[str, str] = field(default_factory=dict)
    consulted: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "action": self.action,
            "intents": self.intents,
            "task_kind": self.task_kind,
            "new_tasks": self.new_tasks,
            "completed_tasks": self.completed_tasks,
            "pending_tasks": self.pending_tasks,
            "filled_slots": self.filled_slots,
        }
        if self.ask is not None:
            data["ask"] = {"kind": self.ask.kind, "target": self.ask.target}
        if self.clarification:
            data["clarification"] = self.clarification
        if self.consulted:
            data["consulted"] = self.consulted
        return data


class SessionPlanner:
    """把一个会话的结构化状态与当前消息合成行动计划。"""

    def plan(
        self,
        state: StateSnapshot,
        facts: Dict[str, Dict[str, Any]],
        user_text: str,
        *,
        turn: int = 0,
    ) -> Plan:
        plan = Plan()

        # ---- 1. 意图与待办 ----
        intents = self._intents(facts)
        plan.intents = intents
        consulted = [i for i in intents if i in CONSULT_INTENTS or task_for_intent(i) is None]
        plan.consulted = consulted

        if self._is_cancel(user_text):
            active = state.active_task
            if active is not None:
                nxt = state.cancel(active, turn=turn)
                plan.completed_tasks.append(active.kind)
                if nxt is not None:
                    plan.pending_tasks.append(nxt.kind)

        for intent in intents:
            task_def = task_for_intent(intent)
            if task_def is None:
                continue
            existed = any(
                t.kind == task_def.key and t.status != "done" for t in state.tasks
            )
            task = state.push(task_def, turn=turn)
            if not existed:
                plan.new_tasks.append(task.kind)

        # ---- 2. 槽位：事实是共享知识，回灌进槽位库 ----
        state.fill_from_facts(facts)
        plan.filled_slots = known_slot_digest(state)

        active = state.active_task
        plan.task_kind = active.kind if active else ""
        plan.pending_tasks = [t.kind for t in state.open_tasks() if t is not active]

        # ---- 3. 指代解析（不唯一就澄清）----
        clarification = self._clarify_if_ambiguous(state, facts, user_text, intents)
        if clarification:
            plan.action = "ask"
            plan.clarification = clarification
            state.mark_clarification("order_reference")
            return plan

        # ---- 4. 是否需要追问 ----
        decision = self._decide(state, active, intents, plan, user_text)
        if decision is not None:
            plan.action = "ask"
            plan.ask = decision
            state.mark_asked(decision.target)
            return plan

        # ---- 5. 无需追问 ----
        plan.action = "answer"
        return plan

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    @staticmethod
    def _intents(facts: Dict[str, Dict[str, Any]]) -> List[str]:
        raw = str(((facts or {}).get("诉求") or {}).get("value") or "")
        return split_intents(raw)

    @staticmethod
    def _is_cancel(text: str) -> bool:
        return any(marker in (text or "") for marker in CANCEL_MARKERS)

    @staticmethod
    def _clarify_if_ambiguous(
        state: StateSnapshot,
        facts: Dict[str, Dict[str, Any]],
        user_text: str,
        intents: List[str],
    ) -> str:
        """用户用了指代词但上下文绑不唯一时，要求澄清（这是纠错成本最高的一步）。"""
        if not user_text:
            return ""
        if state.has_asked_clarification("order_reference"):
            return ""
        markers = ("这个", "这单", "这笔", "那单", "那笔", "该订单", "此订单", "刚才那", "上面那")
        if not any(marker in user_text for marker in markers):
            return ""
        if state.known(SLOT_ORDER_ID):
            return ""
        # 咨询类问题不强行要订单号（"这个政策是怎么规定的"里的"这个"指政策本身）
        if is_consultation(user_text):
            return ""
        if intents and is_consult_only(intents):
            return ""
        if resolve_order_reference(state, user_text, facts):
            return ""
        return (
            "您说的「这个」我这边对不上具体订单，麻烦把订单号发我一下，"
            "我确认好再帮您处理，避免办错单子。"
        )

    @staticmethod
    def _same_topic(intents: List[str], task_def: Any) -> bool:
        """本轮意图是否仍与这个任务同属一件事。"""
        for intent in intents:
            other = task_for_intent(intent)
            if other is not None and other.key == task_def.key:
                return True
        return False

    @classmethod
    def _decide(
        cls,
        state: StateSnapshot,
        active: Optional[Any],
        intents: List[str],
        plan: Plan,
        user_text: str = "",
    ) -> Optional[AskDecision]:
        """决定是否追问；重复追问会被抑制。

        核心规则：**只有"继续办这件事确实需要"才问**。
        缺订单号这类硬槽位会问一次；问过还没给，就不再纠缠——
        用户可以只想知道规则，不该被追着要单号。
        """
        if active is None:
            return None

        task_def = task_for_intent(active.kind)
        if task_def is None:
            return None

        # 问过一次仍未提供：不再追问
        decision = decide_ask(task_def, state)
        if decision is None or decision.kind == "reask":
            return None

        # 规则咨询不索要单号。线上实测过这个反例：用户问"退货运费一般谁承担？"，
        # 意图抽取把"退货"也抽了出来，于是系统去要订单号——用规则问题拦住用户是硬伤。
        # 只在**确实缺**订单号时才让位给咨询判断（已有单号就正常往下办）。
        if decision.target == SLOT_ORDER_ID and is_consultation(user_text):
            return None

        # 用户已经换话题（本轮意图与该任务无关）→ 不打断，等回到这件事再问
        if intents and not cls._same_topic(intents, task_def):
            return None

        return decision


def slot_view(state: StateSnapshot) -> Dict[str, str]:
    """给提示词用的"已知信息"（槽位 + 可复用事实）。"""
    view = known_slot_digest(state)
    return {k: v for k, v in view.items() if k not in PROFILE_ALLOWED_KEYS or k in SESSION_ONLY_KEYS}
