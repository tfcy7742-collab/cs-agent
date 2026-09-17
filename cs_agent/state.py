"""结构化会话状态：待办栈 + 槽位库 + 追问/澄清决策。

为什么要有这一层（长会话里最容易翻车的三件事）：

1. **任务丢失**：用户先说退货、又问物流、再回到退货——纯靠模型记，任务会飘。
   这里用**待办栈**显式记账，办完一个接着办下一个。
2. **重复追问**：同一个槽位问过就不该再问。``asked_slots`` 记录"问过什么"，
   第二次改用不同措辞并给出"跳过"的出口。
3. **乱猜指代**："那这个能退吗"里的"这个"如果指向不唯一，**必须澄清**，
   猜错会直接办错单子——这是客服场景里代价最高的错误。

状态存在 ``session_state['tasks']`` / ``['slots']``，与版本化事实（``['facts']``）分开：
事实回答"我知道什么"，状态回答"我正在办什么事、还缺什么"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .storage import SessionStore
from .tasks import (
    CONSULT_INTENTS,
    SLOT_ADDRESS,
    SLOT_ITEM,
    SLOT_ORDER_ID,
    TaskDef,
    missing_required,
    task_for_intent,
)

#: 任务状态
STATUS_PENDING = "pending"
STATUS_ACTIVE = "active"
STATUS_WAITING = "waiting"
STATUS_DONE = "done"


@dataclass
class TaskState:
    """一个待办任务。"""

    kind: str
    name: str = ""
    status: str = STATUS_PENDING
    filled: Dict[str, str] = field(default_factory=dict)
    opened_turn: int = 0
    updated_turn: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "status": self.status,
            "filled": self.filled,
            "opened_turn": self.opened_turn,
            "updated_turn": self.updated_turn,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TaskState":
        return TaskState(
            kind=str(data.get("kind") or ""),
            name=str(data.get("name") or ""),
            status=str(data.get("status") or STATUS_PENDING),
            filled={str(k): str(v) for k, v in (data.get("filled") or {}).items()},
            opened_turn=int(data.get("opened_turn") or 0),
            updated_turn=int(data.get("updated_turn") or 0),
        )


@dataclass
class StateSnapshot:
    """整个会话状态的快照。"""

    tasks: List[TaskState] = field(default_factory=list)
    #: 槽位库：跨任务共享的已知值（订单号通常只用记一次）
    slots: Dict[str, str] = field(default_factory=dict)
    #: 已经问过的槽位（避免重复追问）
    asked_slots: List[str] = field(default_factory=list)
    #: 已经问过的澄清（避免反复问"你指的是哪一单"）
    asked_clarifications: List[str] = field(default_factory=list)

    # ---- 持久化 --------------------------------------------------------
    @staticmethod
    def load(store: SessionStore, session_id: str) -> "StateSnapshot":
        state = store.get_state(session_id)
        return StateSnapshot(
            tasks=[TaskState.from_dict(item) for item in (state.get("tasks") or [])],
            slots={str(k): str(v) for k, v in (state.get("slots") or {}).items()},
            asked_slots=[str(s) for s in (state.get("asked_slots") or [])],
            asked_clarifications=[
                str(s) for s in (state.get("asked_clarifications") or [])
            ],
        )

    def save(self, store: SessionStore, session_id: str) -> None:
        state = store.get_state(session_id)
        state["tasks"] = [task.as_dict() for task in self.tasks]
        state["slots"] = dict(self.slots)
        state["asked_slots"] = list(self.asked_slots)
        state["asked_clarifications"] = list(self.asked_clarifications)
        store.save_state(session_id, state)

    # ---- 任务栈 --------------------------------------------------------
    @property
    def active_task(self) -> Optional[TaskState]:
        for task in self.tasks:
            if task.status in (STATUS_ACTIVE, STATUS_WAITING):
                return task
        return None

    @property
    def pending_tasks(self) -> List[TaskState]:
        return [t for t in self.tasks if t.status == STATUS_PENDING]

    def open_tasks(self) -> List[TaskState]:
        """还没办完的任务（未完成、未取消）。"""
        return [t for t in self.tasks if t.status != STATUS_DONE]

    def push(self, task_def: TaskDef, turn: int = 0) -> TaskState:
        """登记一个任务。

        同一个 kind 不重复建：长会话里用户会反复提同一件事，
        每次都新建任务会让待办列表迅速变成垃圾场。
        """
        for task in self.tasks:
            if task.kind == task_def.key and task.status != STATUS_DONE:
                task.updated_turn = turn or task.updated_turn
                return task

        # 已有正在办的任务时，新任务排队（这就是"多意图不丢"的机制）
        current = self.active_task
        status = STATUS_PENDING if current else STATUS_ACTIVE
        task = TaskState(
            kind=task_def.key,
            name=task_def.name,
            status=status,
            opened_turn=turn,
            updated_turn=turn,
        )
        self.tasks.append(task)
        return task

    def complete(self, task: TaskState, turn: int = 0) -> Optional[TaskState]:
        """把任务标记为完成，并**自动激活下一个待办**。"""
        task.status = STATUS_DONE
        task.updated_turn = turn or task.updated_turn
        nxt = self.pending_tasks
        if nxt and self.active_task is None:
            nxt[0].status = STATUS_ACTIVE
            return nxt[0]
        return None

    def cancel(self, task: TaskState, turn: int = 0) -> Optional[TaskState]:
        """用户明确说不办了：从待办里移除，而不是假装完成。"""
        task.status = STATUS_DONE
        task.updated_turn = turn or task.updated_turn
        remaining = [t for t in self.tasks if t.status != STATUS_DONE]
        if remaining and self.active_task is None:
            remaining[0].status = STATUS_ACTIVE
            return remaining[0]
        return None

    # ---- 槽位 ----------------------------------------------------------
    def fill(self, slot: str, value: str) -> bool:
        """"写入槽位库，并同步给所有未完成任务的同名槽位。

        同步很重要：用户换个任务后重说一次订单号，不该要求他再说一遍。
        """
        value = (value or "").strip()
        if not slot or not value:
            return False
        changed = self.slots.get(slot) != value
        self.slots[slot] = value
        for task in self.tasks:
            if task.status != STATUS_DONE and slot in self._task_slots(task):
                task.filled[slot] = value
        return changed

    def fill_from_facts(self, facts: Dict[str, Dict[str, Any]]) -> None:
        """把版本化事实里的值灌进槽位库（事实是跨任务的共享知识）。"""
        for key, payload in (facts or {}).items():
            value = payload.get("value") if isinstance(payload, dict) else payload
            if value:
                self.fill(str(key), str(value))

    def _task_slots(self, task: TaskState) -> List[str]:
        task_def = task_for_intent(task.kind)
        if task_def is None:
            return []
        return list(task_def.required) + list(task_def.optional)

    def known(self, slot: str) -> Optional[str]:
        value = self.slots.get(slot)
        return value or None

    def mark_asked(self, slot: str) -> None:
        if slot not in self.asked_slots:
            self.asked_slots.append(slot)

    def has_asked(self, slot: str) -> bool:
        return slot in self.asked_slots

    def mark_clarification(self, key: str) -> None:
        if key not in self.asked_clarifications:
            self.asked_clarifications.append(key)

    def has_asked_clarification(self, key: str) -> bool:
        return key in self.asked_clarifications

    # ---- 观测 ----------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        """给轨迹与前端用的精简视图。"""
        return {
            "slots": dict(self.slots),
            "tasks": [
                {
                    "kind": t.kind,
                    "name": t.name,
                    "status": t.status,
                    "filled": t.filled,
                }
                for t in self.open_tasks()
            ],
            "active_task": self.active_task.kind if self.active_task else "",
            "pending_count": len(self.pending_tasks),
            "asked_slots": list(self.asked_slots),
        }


# ---------------------------------------------------------------------------
# 决策：下一步该问什么 / 该做什么
# ---------------------------------------------------------------------------
@dataclass
class AskDecision:
    """一次追问或澄清。"""

    kind: str  # "slot" | "clarify" | "reask"
    target: str  # 槽位名或澄清键
    question: str
    task_kind: str = ""


def decide_ask(task_def: Optional[TaskDef], state: StateSnapshot) -> Optional[AskDecision]:
    """决定是否需要追问。

    规则（按优先级）：

    1. 全局缺订单号 → 问订单号（几乎所有售后任务都依赖它）；
    2. 当前任务自己的硬槽位还缺 → 问最靠前的那个；
    3. 问过的槽位不再问——但**升级处理**：换措辞并给出跳过出口，
       而不是沉默或反复重复同一句。
    """
    if task_def is None:
        return None

    order_required = SLOT_ORDER_ID in task_def.required and not state.known(SLOT_ORDER_ID)
    if order_required:
        return _build_ask(SLOT_ORDER_ID, task_def, state)

    for slot in missing_required(task_def, state.slots):
        return _build_ask(slot, task_def, state)

    return None


def _build_ask(slot: str, task_def: TaskDef, state: StateSnapshot) -> AskDecision:
    base = task_def.ask_for(slot)
    if state.has_asked(slot):
        # 已经问过一次还没拿到：换措辞 + 给出口，避免变成"复读机"
        return AskDecision(
            kind="reask",
            target=slot,
            task_kind=task_def.key,
            question=(
                f"还是需要{slot}才能继续～您可以在「我的订单」里找到它。"
                f"如果暂时不方便提供，我也可以先为您说明{task_def.name}的一般规则。"
            ),
        )
    return AskDecision(kind="slot", target=slot, task_kind=task_def.key, question=base)


def is_consult_only(intents: List[str]) -> bool:
    """这次消息是否只是**规则咨询**（不涉及具体单据）。

    咨询类不该被"缺订单号"拦住——这是体验底线。
    """
    if not intents:
        return False
    return all(
        intent in CONSULT_INTENTS or task_for_intent(intent) is None for intent in intents
    )


def resolve_order_reference(
    state: StateSnapshot,
    text: str,
    facts: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    """解析"这个/那笔/这单"指向哪个订单。

    Returns:
        订单号；无法确定时返回 ``None``（调用方必须先澄清，不能猜）。
    """
    if not text:
        return None
    markers = ("这个", "这单", "这笔", "那单", "那笔", "该订单", "此订单", "刚才那", "上面那")
    if not any(marker in text for marker in markers):
        return None

    known = state.known(SLOT_ORDER_ID)
    if known:
        return known

    # 只能从事实里取；有多个订单号时**不能猜**
    order_fact = (facts or {}).get(SLOT_ORDER_ID) or {}
    value = str(order_fact.get("value") or "").strip()
    if not value:
        return None
    # 版本化事实里 previous 是用户改口前的旧单号，不能拿来当"当前这一单"
    return value


def resolve_item_reference(state: StateSnapshot, text: str) -> Optional[str]:
    """解析"那件商品"指向哪件商品（同样：不唯一就不猜）。"""
    if not text or "商品" not in text:
        return None
    return state.known(SLOT_ITEM)


def known_slot_digest(state: StateSnapshot) -> Dict[str, str]:
    """渲染成可放进提示词的"已知信息"（跳过空值）。"""
    return {k: v for k, v in state.slots.items() if str(v).strip()}


__all__ = [
    "AskDecision",
    "StateSnapshot",
    "TaskState",
    "STATUS_ACTIVE",
    "STATUS_DONE",
    "STATUS_PENDING",
    "STATUS_WAITING",
    "decide_ask",
    "is_consult_only",
    "known_slot_digest",
    "resolve_item_reference",
    "resolve_order_reference",
    "SLOT_ADDRESS",
]
