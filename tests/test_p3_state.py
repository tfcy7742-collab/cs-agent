"""P3：结构化会话状态测试（待办栈 / 槽位 / 追问策略）。

这一阶段盯的是长会话里最伤体验的三件事：
1. 任务丢失——多意图必须排队，办完一个接着办下一个；
2. 重复追问——同一个槽位问过就不再问；
3. 乱猜指代——"那这个能退吗"绑不唯一时必须澄清。
"""

from __future__ import annotations

import pytest

from cs_agent.planner import SessionPlanner
from cs_agent.state import (
    STATUS_ACTIVE,
    STATUS_DONE,
    STATUS_PENDING,
    StateSnapshot,
    decide_ask,
    is_consult_only,
    resolve_order_reference,
)
from cs_agent.storage import SessionStore
from cs_agent.tasks import (
    SLOT_ADDRESS,
    SLOT_ITEM,
    SLOT_ORDER_ID,
    SLOT_REASON,
    TASKS,
    missing_required,
    split_intents,
    task_for_intent,
)


def _facts(intent: str = "", order: str = "", **extra: str) -> dict:
    facts: dict = {}
    if intent:
        facts["诉求"] = {"value": intent, "source": "regex"}
    if order:
        facts[SLOT_ORDER_ID] = {"value": order, "source": "regex"}
    for key, value in extra.items():
        facts[key] = {"value": value, "source": "regex"}
    return facts


# ---------------------------------------------------------------------------
# 任务表
# ---------------------------------------------------------------------------
def test_every_task_has_guide_for_required_slots() -> None:
    """硬槽位必须有追问话术，否则系统会问出"麻烦提供一下XX"这种机器话。"""
    for task in TASKS.values():
        for slot in task.required:
            assert slot in task.guide, f"{task.key} 缺少 {slot} 的追问话术"


def test_consult_intents_are_not_tasks() -> None:
    """咨询类问题不该被登记成任务，也就不该被追问拦下。"""
    assert task_for_intent("运费咨询") is None
    assert task_for_intent("优惠咨询") is None


def test_task_matching_by_alias_and_substring() -> None:
    assert task_for_intent("退货").key == "return_goods"
    assert task_for_intent("我要退货").key == "return_goods"
    assert task_for_intent("query_logistics").key == "query_logistics"
    assert task_for_intent("完全无关的说法") is None


def test_split_intents_keeps_order_and_dedupes() -> None:
    assert split_intents("退货、查询物流") == ["退货", "查询物流"]
    assert split_intents("退货，查询物流和退货") == ["退货", "查询物流"]
    assert split_intents("") == []


def test_missing_required_follows_definition_order() -> None:
    """追问顺序 = 任务定义里的槽位顺序（订单号永远先问）。"""
    task = TASKS["change_address"]
    assert missing_required(task, {}) == [SLOT_ORDER_ID, SLOT_ADDRESS]
    assert missing_required(task, {SLOT_ORDER_ID: "SO1"}) == [SLOT_ADDRESS]
    assert missing_required(task, {SLOT_ORDER_ID: "SO1", SLOT_ADDRESS: "北京"}) == []


# ---------------------------------------------------------------------------
# 待办栈
# ---------------------------------------------------------------------------
def test_first_task_becomes_active(store: SessionStore) -> None:
    state = StateSnapshot()
    task = state.push(TASKS["return_goods"], turn=1)
    assert task.status == STATUS_ACTIVE
    assert state.active_task is task


def test_second_intent_is_queued_not_lost(store: SessionStore) -> None:
    """多意图：第二个任务排队等待，而不是被覆盖。"""
    state = StateSnapshot()
    state.push(TASKS["return_goods"], turn=1)
    second = state.push(TASKS["query_logistics"], turn=1)

    assert second.status == STATUS_PENDING
    assert state.active_task.kind == "return_goods"
    assert [t.kind for t in state.open_tasks()] == ["return_goods", "query_logistics"]


def test_completing_task_activates_next(store: SessionStore) -> None:
    """办完一件事自动接着办下一件——这是"多意图不丢"的关键行为。"""
    state = StateSnapshot()
    first = state.push(TASKS["return_goods"], turn=1)
    state.push(TASKS["query_logistics"], turn=1)

    nxt = state.complete(first, turn=3)
    assert first.status == STATUS_DONE
    assert nxt is not None and nxt.kind == "query_logistics"
    assert state.active_task.kind == "query_logistics"


def test_same_kind_is_not_duplicated(store: SessionStore) -> None:
    """用户反复提同一件事，不能把待办列表堆成垃圾场。"""
    state = StateSnapshot()
    first = state.push(TASKS["return_goods"], turn=1)
    again = state.push(TASKS["return_goods"], turn=5)
    assert again is first
    assert len(state.open_tasks()) == 1


def test_done_task_can_be_reopened(store: SessionStore) -> None:
    state = StateSnapshot()
    task = state.push(TASKS["return_goods"], turn=1)
    state.complete(task, turn=2)
    reopened = state.push(TASKS["return_goods"], turn=6)
    assert reopened is not task
    assert reopened.status == STATUS_ACTIVE


def test_cancel_removes_from_todo(store: SessionStore) -> None:
    """"不用了"是真的不办了：不能假装完成，也不能继续追问。"""
    state = StateSnapshot()
    first = state.push(TASKS["return_goods"], turn=1)
    state.push(TASKS["query_logistics"], turn=1)

    nxt = state.cancel(first, turn=2)
    assert first.status == STATUS_DONE
    assert nxt is not None and nxt.kind == "query_logistics"


def test_state_roundtrip_through_store(store: SessionStore) -> None:
    sid = store.create_session()["id"]
    state = StateSnapshot()
    state.push(TASKS["return_goods"], turn=1)
    state.fill(SLOT_ORDER_ID, "SO20260101")
    state.mark_asked(SLOT_ORDER_ID)
    state.save(store, sid)

    loaded = StateSnapshot.load(store, sid)
    assert [t.kind for t in loaded.tasks] == ["return_goods"]
    assert loaded.known(SLOT_ORDER_ID) == "SO20260101"
    assert loaded.has_asked(SLOT_ORDER_ID) is True
    assert loaded.active_task is not None


# ---------------------------------------------------------------------------
# 槽位库
# ---------------------------------------------------------------------------
def test_fill_propagates_to_open_tasks(store: SessionStore) -> None:
    """用户给过一次订单号，不该在新任务里再被问一遍。"""
    state = StateSnapshot()
    state.push(TASKS["return_goods"], turn=1)
    state.push(TASKS["query_logistics"], turn=1)

    state.fill(SLOT_ORDER_ID, "SO20260101")
    assert all(t.filled.get(SLOT_ORDER_ID) == "SO20260101" for t in state.open_tasks())


def test_fill_from_facts_ignores_empty(store: SessionStore) -> None:
    state = StateSnapshot()
    state.fill_from_facts({"订单号": {"value": ""}, "称呼": {"value": "张先生"}})
    assert state.known(SLOT_ORDER_ID) is None
    assert state.known("称呼") == "张先生"


def test_fill_reports_change(store: SessionStore) -> None:
    state = StateSnapshot()
    assert state.fill(SLOT_ORDER_ID, "SO1") is True
    assert state.fill(SLOT_ORDER_ID, "SO1") is False
    assert state.fill(SLOT_ORDER_ID, "SO2") is True


# ---------------------------------------------------------------------------
# 追问策略
# ---------------------------------------------------------------------------
def test_ask_when_required_slot_missing(store: SessionStore) -> None:
    state = StateSnapshot()
    state.push(TASKS["return_goods"], turn=1)
    decision = decide_ask(TASKS["return_goods"], state)
    assert decision is not None
    assert decision.kind == "slot"
    assert decision.target == SLOT_ORDER_ID
    assert "订单号" in decision.question


def test_no_ask_when_slot_present(store: SessionStore) -> None:
    state = StateSnapshot()
    state.push(TASKS["return_goods"], turn=1)
    state.fill(SLOT_ORDER_ID, "SO20260101")
    assert decide_ask(TASKS["return_goods"], state) is None


def test_reask_changes_wording_and_offers_exit(store: SessionStore) -> None:
    """已问过一次还缺：换措辞并给出"我可以先讲规则"的出口，不当复读机。"""
    state = StateSnapshot()
    state.push(TASKS["return_goods"], turn=1)
    state.mark_asked(SLOT_ORDER_ID)
    decision = decide_ask(TASKS["return_goods"], state)
    assert decision is not None
    assert decision.kind == "reask"
    assert "一般规则" in decision.question


def test_optional_slot_never_blocks(store: SessionStore) -> None:
    """软槽位缺失绝不能拦人——"退货运费谁出"这类问题不该被要商品名。"""
    state = StateSnapshot()
    state.push(TASKS["return_goods"], turn=1)
    state.fill(SLOT_ORDER_ID, "SO20260101")
    # 少了 退货原因 / 商品名称 这两个软槽位，仍然不该追问
    assert decide_ask(TASKS["return_goods"], state) is None


def test_consult_only_detection() -> None:
    assert is_consult_only(["运费咨询"]) is True
    assert is_consult_only(["退货"]) is False
    assert is_consult_only([]) is False


@pytest.mark.parametrize(
    "text,expected",
    [
        ("退货运费一般谁承担？", True),
        ("退款多久到账", True),
        ("包装拆开了还能退吗", True),
        ("你们支持上门取件吗", True),
        ("我要退货", False),
        ("帮我退一下这单", False),
        ("订单号是 SO1，我要退款", False),
    ],
)
def test_consultation_detection(text: str, expected: bool) -> None:
    """区分"问规则"与"要办事"——这条判据直接决定该不该索要订单号。"""
    from cs_agent.tasks import is_consultation

    assert is_consultation(text) is expected


def test_policy_question_does_not_get_blocked_by_missing_order(
    store: SessionStore,
) -> None:
    """线上实测过的反例：意图抽取把"退货"也抽出来了，结果用规则问题拦住用户。

    用户问"退货运费一般谁承担？"→ 即使 intents 里同时有"退货"和"运费咨询"，
    也不该去索要订单号。
    """
    planner = SessionPlanner()
    state = StateSnapshot()
    plan = planner.plan(
        state, _facts("退货、运费咨询"), "退货运费一般谁承担？", turn=1
    )
    assert plan.action == "answer"
    assert plan.ask is None


def test_explicit_request_still_asks(store: SessionStore) -> None:
    """反过来：明确说"我要退货"时该问还得问。"""
    planner = SessionPlanner()
    state = StateSnapshot()
    plan = planner.plan(state, _facts("退货"), "我要退货", turn=1)
    assert plan.action == "ask"
    assert plan.ask is not None and plan.ask.target == SLOT_ORDER_ID


# ---------------------------------------------------------------------------
# 指代消解
# ---------------------------------------------------------------------------
def test_reference_resolves_to_known_order(store: SessionStore) -> None:
    state = StateSnapshot()
    state.fill(SLOT_ORDER_ID, "SO20260101")
    assert resolve_order_reference(state, "那这个能退吗", {}) == "SO20260101"


def test_reference_returns_none_when_unknown(store: SessionStore) -> None:
    """绑不上就返回 None，调用方必须澄清——绝不能猜一个订单号。"""
    state = StateSnapshot()
    assert resolve_order_reference(state, "那这个能退吗", {}) is None


def test_reference_only_triggers_on_markers(store: SessionStore) -> None:
    state = StateSnapshot()
    state.fill(SLOT_ORDER_ID, "SO1")
    assert resolve_order_reference(state, "我要退款", {}) is None
    assert resolve_order_reference(state, "这笔订单怎么了", {}) == "SO1"


def test_reference_never_uses_previous_value(store: SessionStore) -> None:
    """用户改过单号后，previous 是旧单号——不能拿它当"这一单"。"""
    state = StateSnapshot()
    facts = {SLOT_ORDER_ID: {"value": "SO-NEW", "previous": "SO-OLD"}}
    assert resolve_order_reference(state, "这个订单", facts) == "SO-NEW"


# ---------------------------------------------------------------------------
# 规划器端到端
# ---------------------------------------------------------------------------
def test_planner_asks_order_id_for_return(store: SessionStore) -> None:
    planner = SessionPlanner()
    state = StateSnapshot()
    plan = planner.plan(state, _facts("退货"), "我要退货", turn=1)

    assert plan.action == "ask"
    assert plan.ask is not None and plan.ask.target == SLOT_ORDER_ID
    assert plan.new_tasks == ["return_goods"]
    assert state.has_asked(SLOT_ORDER_ID) is True


def test_planner_answers_consult_question_without_asking(store: SessionStore) -> None:
    """纯咨询：不能因为"没订单号"就把用户拦在门外。"""
    planner = SessionPlanner()
    state = StateSnapshot()
    plan = planner.plan(state, _facts("运费咨询"), "退货运费谁承担？", turn=1)

    assert plan.action == "answer"
    assert plan.ask is None
    assert plan.consulted == ["运费咨询"]


def test_planner_does_not_ask_twice(store: SessionStore) -> None:
    planner = SessionPlanner()
    state = StateSnapshot()

    first = planner.plan(state, _facts("退货"), "我要退货", turn=1)
    first.ask and state.mark_asked(first.ask.target)
    second = planner.plan(state, _facts("退货"), "到底能不能退", turn=2)

    assert first.action == "ask"
    assert second.action == "answer", "问过一次还缺，就不该再追问"


def test_planner_queues_second_intent(store: SessionStore) -> None:
    """「我要退货，另外帮我查下物流」：两件事都要进待办。"""
    planner = SessionPlanner()
    state = StateSnapshot()
    plan = planner.plan(state, _facts("退货、查询物流", order="SO1"), "退货，另外查物流", turn=1)

    assert set(plan.new_tasks) == {"return_goods", "query_logistics"}
    assert state.active_task.kind == "return_goods"
    assert plan.pending_tasks == ["query_logistics"]


def test_planner_completes_task_on_cancel(store: SessionStore) -> None:
    planner = SessionPlanner()
    state = StateSnapshot()
    planner.plan(state, _facts("退货"), "我要退货", turn=1)
    state.push(TASKS["query_logistics"], turn=1)

    plan = planner.plan(state, _facts("退货"), "算了不用了", turn=2)
    assert "return_goods" in plan.completed_tasks
    assert state.active_task.kind == "query_logistics"


def test_planner_clarifies_ambiguous_reference(store: SessionStore) -> None:
    """指代不明必须先澄清，且澄清只问一次。"""
    planner = SessionPlanner()
    state = StateSnapshot()
    plan = planner.plan(state, _facts("退货"), "那这个能退吗", turn=1)

    assert plan.action == "ask"
    assert plan.clarification
    assert "订单号" in plan.clarification

    again = planner.plan(state, _facts("退货"), "那这个到底行不行", turn=2)
    assert again.clarification == "", "同一件事不该反复要求澄清"


def test_planner_does_not_interrupt_when_topic_changed(store: SessionStore) -> None:
    """用户换话题问规则时，不追着要上一个任务的订单号。"""
    planner = SessionPlanner()
    state = StateSnapshot()
    planner.plan(state, _facts("退货"), "我要退货", turn=1)  # 已问过订单号

    plan = planner.plan(state, _facts("运费咨询"), "顺便问下退货运费谁出", turn=2)
    assert plan.action == "answer"
    assert plan.ask is None


def test_planner_hint_mentions_pending_tasks(store: SessionStore) -> None:
    from cs_agent.conversation import ConversationService

    service = ConversationService(store, None, None)  # type: ignore[arg-type]
    planner = SessionPlanner()
    state = StateSnapshot()
    plan = planner.plan(state, _facts("退货、查询物流", order="SO1"), "退货，另外查物流", turn=1)
    hint = service._plan_hint(plan)
    assert "正在处理" in hint
    assert "查询物流" in hint


def test_state_persisted_across_turns_in_conversation(service, store: SessionStore) -> None:
    """端到端：一轮对话之后，槽位与待办必须落库（否则长会话会失忆）。"""
    sid = service.open_session()["id"]
    service.chat_once(sid, "我的订单号是 SO20260101，我要退货")

    snapshot = StateSnapshot.load(store, sid)
    assert snapshot.known(SLOT_ORDER_ID) == "SO20260101"
    assert any(t.kind == "return_goods" for t in snapshot.open_tasks())


def test_conversation_trace_carries_plan_and_state(service, store: SessionStore) -> None:
    from cs_agent.conversation import _parse_frame

    sid = service.open_session()["id"]
    frames = list(service.stream_turn(sid, "我要退货"))
    start = next(
        payload for event, payload in (_parse_frame(f) for f in frames) if event == "start"
    )
    assert start["plan"]["action"] == "ask"
    assert start["plan"]["new_tasks"] == ["return_goods"]
    assert "asked_slots" in start["state"]


def test_slot_item_reference(store: SessionStore) -> None:
    from cs_agent.state import resolve_item_reference

    state = StateSnapshot()
    state.fill(SLOT_ITEM, "蓝色卫衣")
    assert resolve_item_reference(state, "那件商品能退吗") == "蓝色卫衣"
    assert resolve_item_reference(state, "我想退货") is None


def test_reason_slot_is_optional_and_filled_when_given(service, store: SessionStore) -> None:
    """软槽位给了就记下（用于判断运费谁承担），没给也不追问。"""
    sid = service.open_session()["id"]
    service.chat_once(sid, "订单号 SO20260101，质量有问题我想退货")

    snapshot = StateSnapshot.load(store, sid)
    assert snapshot.known(SLOT_REASON) or snapshot.known(SLOT_ITEM)
    assert snapshot.known(SLOT_ORDER_ID) == "SO20260101"
