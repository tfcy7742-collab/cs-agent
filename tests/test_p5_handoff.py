"""P5：交接单、挂起与坐席接续测试。

核心要证的三件事：
1. **交接单不让坐席从零开始**：诉求、已核实信息、已尝试动作、卡点都在；
2. **挂起是持久化的**：服务重启后坐席仍能接手（数据在库里，不是内存）；
3. **接续不重跑**：坐席回复后会话继续，已确认的槽位还在，不会重复追问。
"""

from __future__ import annotations

import uuid

import pytest

from cs_agent.conversation import ConversationService, _parse_frame
from cs_agent.handoff import HandoffService
from cs_agent.handoff_rules import EscalationEngine, EscalationThresholds
from cs_agent.storage import (
    HANDOFF_ACCEPTED,
    HANDOFF_RESOLVED,
    HANDOFF_WAITING,
    ROLE_HUMAN,
    SESSION_ACTIVE,
    SESSION_WAITING_HUMAN,
    SessionStore,
)
from cs_agent.tools.mock_data import reset_repository


@pytest.fixture(autouse=True)
def _clean_data():
    reset_repository()
    yield
    reset_repository()


def run_turn(service: ConversationService, sid: str, text: str) -> dict:
    """跑一轮。

    ``prompt`` 是**整份上下文**（system 提示 + 各轮消息）拼起来的文本：
    人工坐席消息与历史原文是作为独立消息进入上下文的，不在 system prompt 里，
    只检查 system prompt 会误判成"信息丢了"（这个坑实测踩过）。
    """
    start: dict = {}
    done: dict = {}
    for frame in service.stream_turn(sid, text):
        event, payload = _parse_frame(frame)
        if event == "start":
            start = payload
        elif event == "done":
            done = payload
    messages = start.get("messages") or [
        {"role": "system", "content": start.get("system_prompt", "")}
    ]
    prompt = "\n".join(f"{item['role']}: {item['content']}" for item in messages)
    return {"start": start, "reply": done.get("content", ""), "prompt": prompt}


def make_service(store: SessionStore, settings, **kwargs) -> ConversationService:
    from cs_agent.llm import LLMClient
    from cs_agent.memory import MemoryManager

    llm = LLMClient(settings)
    return ConversationService(
        store,
        settings,
        llm,
        MemoryManager(store, settings, llm),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 端到端：触发 → 建单 → 挂起
# ---------------------------------------------------------------------------
def test_explicit_request_creates_handoff(service, store) -> None:
    sid = service.open_session(user_id="u1")["id"]
    result = run_turn(service, sid, "订单号 SO20260101，你们服务太差了，我要转人工")

    handoff = result["start"]["handoff"]
    assert handoff["handoff_id"].startswith("HO")
    assert handoff["reason"]

    session = store.get_session(sid)
    assert session["status"] == SESSION_WAITING_HUMAN

    stored = store.latest_handoff(sid)
    assert stored["status"] == HANDOFF_WAITING
    assert stored["packet"]["交接说明"]


def test_handoff_notice_is_in_system_prompt(service, store) -> None:
    """转人工后模型必须知道"已转接、别自称能解决"。"""
    sid = service.open_session()["id"]
    result = run_turn(service, sid, "我要投诉，转人工")
    prompt = result["start"]["system_prompt"]
    assert "本轮已转人工" in prompt
    assert "工单号" in prompt
    assert "不要承诺你自己能解决" in prompt


def test_out_of_scope_handoff_reason_is_explicit(service, store) -> None:
    sid = service.open_session()["id"]
    result = run_turn(service, sid, "你们质量太差了，得赔我 200 块")
    handoff = result["start"]["handoff"]
    assert "赔偿" in handoff["reason"] or "权限" in handoff["reason"]


def test_normal_turn_does_not_create_handoff(service, store) -> None:
    sid = service.open_session()["id"]
    result = run_turn(service, sid, "订单号 SO20260101，帮我查下物流")
    assert result["start"]["handoff"] == {}
    assert store.get_session(sid)["status"] == SESSION_ACTIVE
    assert store.latest_handoff(sid) is None


# ---------------------------------------------------------------------------
# 交接单内容
# ---------------------------------------------------------------------------
def test_packet_contains_what_agent_needs(service, store) -> None:
    """交接单必须回答：是谁、要什么、已核实什么、试过什么、卡在哪。"""
    sid = service.open_session(user_id="u-zhang")["id"]
    run_turn(service, sid, "我叫张先生，订单号 SO20260101，帮我查下物流")
    run_turn(service, sid, "还是没收到，我要转人工")

    handoff = store.latest_handoff(sid)
    packet = handoff["packet"]

    assert packet["用户"] == "u-zhang"
    assert packet["已核实信息"]["订单号"] == "SO20260101"
    assert packet["已核实信息"].get("称呼") == "张先生"
    assert packet["轮次数"] >= 2
    assert packet["触发规则"], "必须记录是哪条规则触发的"
    assert packet["系统已尝试"], "必须列出系统试过什么"
    assert packet["交接说明"].startswith("【交接单")

    text = packet["交接说明"]
    assert "SO20260101" in text
    assert "已核实信息" in text


def test_packet_lists_unfinished_tasks_with_missing_slots(service, store) -> None:
    sid = service.open_session()["id"]
    run_turn(service, sid, "我要改收货地址，另外我要转人工")
    packet = store.latest_handoff(sid)["packet"]
    tasks = packet["未办结事项"]
    assert tasks, "未办结任务要列出来"
    assert any("仍缺" in item for item in tasks), "要写明还缺什么信息"


def test_packet_records_attempted_tools(service, store) -> None:
    """坐席要知道系统查过什么、结果如何，避免重复劳动。"""
    sid = service.open_session()["id"]
    run_turn(service, sid, "订单号 SO99999999 帮我查物流")
    run_turn(service, sid, "算了，转人工吧")
    packet = store.latest_handoff(sid)["packet"]
    assert any("track_logistics" in item for item in packet["系统已尝试"])


def test_packet_marks_already_waiting(service, store) -> None:
    """已在等人工时再触发：不重复建单，但新信息要补进同一张工单。"""
    sid = service.open_session()["id"]
    first = run_turn(service, sid, "我要转人工")
    handoff_id = first["start"]["handoff"]["handoff_id"]

    second = run_turn(service, sid, "订单号 SO20260101")
    again = run_turn(service, sid, "我要转人工")
    assert again["start"]["handoff"]["handoff_id"] == handoff_id
    assert again["start"]["handoff"]["already_waiting"] is True
    assert len(store.list_handoffs()) == 1, "同一个会话不应重复建单"


# ---------------------------------------------------------------------------
# 坐席工作台
# ---------------------------------------------------------------------------
def test_workspace_lists_waiting_first(service, store) -> None:
    s1 = service.open_session(user_id="u1")["id"]
    s2 = service.open_session(user_id="u2")["id"]
    run_turn(service, s1, "我要转人工")
    run_turn(service, s2, "我要转人工")
    resolved = store.latest_handoff(s2)
    service.handoffs.resolve(resolved["id"], note="已处理")

    items = service.handoffs.workspace()
    assert len(items) == 2
    assert items[0]["status"] == HANDOFF_WAITING
    assert items[-1]["status"] == HANDOFF_RESOLVED
    assert items[0]["demand"]
    assert items[0]["blocker"]


def test_workspace_filter_by_status(service, store) -> None:
    sid = service.open_session()["id"]
    run_turn(service, sid, "我要转人工")
    assert len(service.handoffs.workspace(status=HANDOFF_WAITING)) == 1
    assert service.handoffs.workspace(status=HANDOFF_RESOLVED) == []


def test_detail_returns_messages_and_state(service, store) -> None:
    sid = service.open_session()["id"]
    run_turn(service, sid, "订单号 SO20260101，我要转人工")
    handoff = store.latest_handoff(sid)

    detail = service.handoffs.detail(handoff["id"])
    assert detail["packet_text"]
    assert len(detail["messages"]) >= 2
    assert detail["state"]["slots"]["订单号"] == "SO20260101"


def test_missing_handoff_returns_none(service) -> None:
    assert service.handoffs.detail("HO-NOT-EXIST") is None
    assert service.handoffs.accept("HO-NOT-EXIST") is None
    assert service.handoffs.reply("HO-NOT-EXIST", "你好") is None


# ---------------------------------------------------------------------------
# 坐席接续
# ---------------------------------------------------------------------------
def test_agent_reply_appends_message_and_resumes(service, store) -> None:
    sid = service.open_session()["id"]
    run_turn(service, sid, "订单号 SO20260101，我要转人工")
    handoff = store.latest_handoff(sid)

    service.handoffs.reply(handoff["id"], "您好，我是人工客服小王，已经帮您加急处理。", agent="小王")

    messages = store.list_messages(sid, roles=[ROLE_HUMAN])
    assert len(messages) == 1
    assert "人工客服小王" in messages[0]["content"]
    assert store.get_session(sid)["status"] == SESSION_ACTIVE


def test_resume_keeps_confirmed_slots(service, store) -> None:
    """接续不重跑：人工处理完，用户接着说时不会被重复追问订单号。"""
    sid = service.open_session()["id"]
    run_turn(service, sid, "订单号 SO20260101，帮我查物流")
    run_turn(service, sid, "还是没动静，转人工")
    handoff = store.latest_handoff(sid)
    service.handoffs.reply(handoff["id"], "已为您催件，请留意物流更新。")

    result = run_turn(service, sid, "那大概什么时候能到？")
    assert result["start"]["plan"]["action"] != "ask", "不该再追问订单号"
    assert "SO20260101" in str(result["start"]["state"].get("slots"))

    assert "[人工客服]" in result["prompt"], "人工消息必须以客服身份进入上下文"
    assert "已为您催件" in result["prompt"]


def test_human_message_enters_working_memory(service, store) -> None:
    sid = service.open_session()["id"]
    run_turn(service, sid, "我要转人工")
    handoff = store.latest_handoff(sid)
    service.handoffs.reply(handoff["id"], "人工处理结果：已为您补发。")

    working = service.memory.working_messages(sid)
    assert any(message["role"] == ROLE_HUMAN for message in working)


def test_resolve_closes_handoff_and_reopens_session(service, store) -> None:
    sid = service.open_session()["id"]
    run_turn(service, sid, "我要转人工")
    handoff = store.latest_handoff(sid)

    service.handoffs.resolve(handoff["id"], note="已补发商品")
    stored = store.get_handoff(handoff["id"])
    assert stored["status"] == HANDOFF_RESOLVED
    assert stored["closed_at"]
    assert stored["packet"]["坐席备注"][-1]["note"] == "已补发商品"
    assert store.get_session(sid)["status"] == SESSION_ACTIVE


def test_accept_marks_handoff(service, store) -> None:
    sid = service.open_session()["id"]
    run_turn(service, sid, "我要转人工")
    handoff = store.latest_handoff(sid)

    accepted = service.handoffs.accept(handoff["id"], agent="小李")
    assert accepted["status"] == HANDOFF_ACCEPTED
    assert accepted["packet"]["坐席备注"][-1]["note"].startswith("小李")


def test_waiting_handoff_lookup(service, store) -> None:
    sid = service.open_session()["id"]
    assert service.handoffs.waiting_handoff(sid) is None
    run_turn(service, sid, "我要转人工")
    assert service.handoffs.waiting_handoff(sid) is not None
    service.handoffs.resolve(store.latest_handoff(sid)["id"])
    assert service.handoffs.waiting_handoff(sid) is None


# ---------------------------------------------------------------------------
# 挂起持久化（服务重启）
# ---------------------------------------------------------------------------
def test_handoff_survives_restart(settings, tmp_path) -> None:
    """挂起状态必须落库：换一个 store 实例（模拟重启）仍然能接手。"""
    db = tmp_path / "handoff.db"
    first_store = SessionStore(db)
    svc1 = make_service(first_store, settings)
    sid = svc1.open_session(user_id="u-restart")["id"]
    run_turn(svc1, sid, "订单号 SO20260102，我要转人工")
    handoff_id = first_store.latest_handoff(sid)["id"]
    first_store.close()

    second_store = SessionStore(db)
    try:
        handoff = second_store.get_handoff(handoff_id)
        assert handoff is not None
        assert handoff["status"] == HANDOFF_WAITING
        assert handoff["packet"]["已核实信息"]["订单号"] == "SO20260102"
        assert second_store.get_session(sid)["status"] == SESSION_WAITING_HUMAN

        svc2 = make_service(second_store, settings)
        assert svc2.handoffs.reply(handoff_id, "人工已接手") is not None
        assert second_store.get_session(sid)["status"] == SESSION_ACTIVE
    finally:
        second_store.close()


def test_cascade_delete_removes_handoffs(store) -> None:
    service = HandoffService(store)
    sid = store.create_session()["id"]
    store.create_handoff(sid, "u1", "测试", [], {})
    assert len(store.list_handoffs()) == 1
    store.delete_session(sid)
    assert store.list_handoffs() == []


# ---------------------------------------------------------------------------
# 阈值可调（便于演示"不轻易转人工"）
# ---------------------------------------------------------------------------
def test_high_threshold_keeps_chatting(store, settings) -> None:
    engine = EscalationEngine(EscalationThresholds(sentiment_score=99, repeat_turns=9))
    from cs_agent.llm import LLMClient
    from cs_agent.memory import MemoryManager

    llm = LLMClient(settings)
    service = ConversationService(
        store,
        settings,
        llm,
        MemoryManager(store, settings, llm),
        escalation=engine,
    )
    sid = service.open_session()["id"]
    result = run_turn(service, sid, "怎么这么慢啊，都几天了")
    assert result["start"]["handoff"] == {}, "阈值放宽后不该转人工"
    assert result["start"]["escalation"]["sentiment"]["score"] > 0, "但情绪分要如实记录"
