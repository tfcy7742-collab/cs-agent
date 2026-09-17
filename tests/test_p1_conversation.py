"""P1：会话流程与流式协议测试。

用离线模式跑，验证的是**流程与协议**本身：事件顺序、持久化一致性、
提示词组装、错误处理。真实模型行为由 ``scripts/check_llm.py`` 单独验证。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from cs_agent.conversation import ConversationService
from cs_agent.llm import LLMClient
from cs_agent.prompt import STATE_HEADER, SUMMARY_HEADER, build_chat_messages
from cs_agent.storage import ROLE_ASSISTANT, ROLE_USER, SessionStore


def parse_sse(frames: List[str]) -> List[Dict[str, Any]]:
    """把 SSE 帧解析成 ``[{event, data}, ...]``。

    SSE 规定：一个帧以空行结束，``event:`` 与 ``data:`` 各占一行。
    这里把每个产出元素按行展开后统一解析——不能"见到 data 就收尾"，
    因为载荷里的转义换行展开后会变成多行，误当帧边界会把内容截断。
    """
    lines: List[str] = []
    for frame in frames:
        lines.extend(frame.splitlines())

    events: List[Dict[str, Any]] = []
    event = ""
    data_lines: List[str] = []

    for line in lines:
        if not line.strip():
            if data_lines:
                try:
                    events.append(
                        {"event": event, "data": json.loads("\n".join(data_lines))}
                    )
                except json.JSONDecodeError:  # pragma: no cover - 防御性分支
                    pass
            event = ""
            data_lines = []
        elif line.startswith("event: "):
            event = line[7:].strip()
        elif line.startswith("data: "):
            data_lines.append(line[6:])

    if data_lines:
        try:
            events.append({"event": event, "data": json.loads("\n".join(data_lines))})
        except json.JSONDecodeError:  # pragma: no cover - 防御性分支
            pass
    return events


def run_turn(service: ConversationService, session_id: str, text: str) -> List[Dict[str, Any]]:
    return parse_sse(list(service.stream_turn(session_id, text)))


# ---------------------------------------------------------------------------
# 提示词组装
# ---------------------------------------------------------------------------
def test_build_chat_messages_has_system_first() -> None:
    messages = build_chat_messages([{"role": "user", "content": "你好"}])
    assert messages[0]["role"] == "system"
    assert "星尘商城" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "你好"}


def test_system_prompt_carries_the_red_lines() -> None:
    """红线必须在系统提示里——这是"不编造/不越权"的第一道保障。"""
    system = build_chat_messages([])[0]["content"]
    for keyword in ("不编造事实", "不越权承诺", "一次只问最关键的那一个", "不重复追问"):
        assert keyword in system


def test_build_chat_messages_injects_summary_state_profile() -> None:
    messages = build_chat_messages(
        [{"role": "user", "content": "那这个能退吗"}],
        summary="用户之前询问过订单 SO20260101 的物流。",
        state={"order_id": "SO20260101", "诉求": "退货"},
        profile={"称呼": "张先生"},
    )
    system = messages[0]["content"]
    assert SUMMARY_HEADER in system
    assert "SO20260101" in system
    assert STATE_HEADER in system
    assert "张先生" in system


def test_build_chat_messages_skips_summary_role_and_empty_content() -> None:
    messages = build_chat_messages(
        [
            {"role": "user", "content": "问题"},
            {"role": "summary", "content": "这段不该进原文"},
            {"role": "assistant", "content": "   "},
        ]
    )
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user"]


# ---------------------------------------------------------------------------
# 一轮对话
# ---------------------------------------------------------------------------
def test_first_turn_creates_history(service: ConversationService, store: SessionStore) -> None:
    session = service.open_session(user_id="u1")
    sid = session["id"]

    events = run_turn(service, sid, "你好，我的快递还没到")
    names = [e["event"] for e in events]
    # 离线模板会切成多个 delta，所以只约束首尾与必需事件
    assert names[0] == "session"
    assert names[1] == "start"
    assert names[-1] == "done"
    assert "delta" in names
    assert "error" not in names

    messages = store.list_messages(sid)
    assert [m["role"] for m in messages] == [ROLE_USER, ROLE_ASSISTANT]
    assert messages[0]["content"] == "你好，我的快递还没到"
    assert messages[1]["content"].strip()
    assert store.get_session(sid)["turn_count"] == 1


def test_first_message_becomes_title(service: ConversationService, store: SessionStore) -> None:
    sid = service.open_session()["id"]
    run_turn(service, sid, "我要查询订单物流状态，订单号是 SO20260101")
    assert store.get_session(sid)["title"] == "我要查询订单物流状态，订单号是 SO202601"[:24]


def test_multi_turn_history_grows(service: ConversationService, store: SessionStore) -> None:
    sid = service.open_session()["id"]
    for index in range(3):
        run_turn(service, sid, f"第 {index + 1} 个问题")

    messages = store.list_messages(sid)
    assert len(messages) == 6
    assert store.get_session(sid)["turn_count"] == 3
    # 助手回复不能为空，否则历史里会出现空洞
    assert all(m["content"].strip() for m in messages)


def test_ten_turns_stay_consistent(service: ConversationService, store: SessionStore) -> None:
    """P1 验收：连续聊 10 轮，历史与轮次计数完全一致。"""
    sid = service.open_session()["id"]
    for index in range(10):
        result = service.chat_once(sid, f"连续对话第 {index + 1} 轮")
        assert result["turn"] == index + 1
        assert result["content"].strip()

    stats = service.session_stats(sid)
    assert stats["turns"] == 10
    assert stats["messages"] == 20
    assert stats["token_estimate"] > 0


def test_done_event_carries_trace_and_usage(service: ConversationService) -> None:
    sid = service.open_session()["id"]
    events = run_turn(service, sid, "你好")
    done = next(e["data"] for e in events if e["event"] == "done")

    assert done["turn"] == 1
    assert done["offline"] is True
    assert done["usage"]["prompt_tokens"] > 0
    assert done["usage"]["completion_tokens"] > 0
    assert done["trace"]["prompt_chars"] > 0
    assert done["trace"]["model"] == "offline-template"


def test_start_event_reports_working_memory_size(service: ConversationService) -> None:
    sid = service.open_session()["id"]
    run_turn(service, sid, "第一轮")
    events = run_turn(service, sid, "第二轮")
    start = next(e["data"] for e in events if e["event"] == "start")

    # 第二轮的工作记忆里应包含第一轮的问与答
    assert start["history_messages"] == 3
    assert start["compressed"] is False
    assert start["prompt_token_estimate"] > 0


def test_recent_turns_window_is_bounded(service: ConversationService, store: SessionStore) -> None:
    """工作记忆只保留最近 N 轮原文，不能无限增长（P2 起还会把更早的压成摘要）。"""
    sid = service.open_session()["id"]
    for index in range(6):
        run_turn(service, sid, f"第 {index + 1} 轮")
    messages = service.build_messages(sid)
    # 配置里 WORKING_RECENT_TURNS=4 → 最多 8 条原文 + 1 条 system
    assert len(messages) <= 9
    assert messages[0]["role"] == "system"
    # 但磁盘上的对话历史必须完整保留（摘要消息不计入对话轮次）
    dialogue = store.list_messages(sid, roles=[ROLE_USER, ROLE_ASSISTANT])
    assert len(dialogue) == 12


# ---------------------------------------------------------------------------
# 错误处理
# ---------------------------------------------------------------------------
def test_unknown_session_emits_error_event(service: ConversationService) -> None:
    events = parse_sse(list(service.stream_turn("not-exist", "你好")))
    assert events[0]["event"] == "error"
    assert "会话不存在" in events[0]["data"]["message"]


def test_empty_message_is_rejected(service: ConversationService) -> None:
    sid = service.open_session()["id"]
    events = parse_sse(list(service.stream_turn(sid, "   ")))
    assert events[0]["event"] == "error"
    assert "不能为空" in events[0]["data"]["message"]


def test_too_long_message_is_rejected(service: ConversationService, settings) -> None:
    sid = service.open_session()["id"]
    too_long = "啊" * (settings.max_message_chars + 1)
    events = parse_sse(list(service.stream_turn(sid, too_long)))
    assert events[0]["event"] == "error"
    assert "消息过长" in events[0]["data"]["message"]


def test_user_message_is_persisted_even_if_model_fails(
    service: ConversationService, store: SessionStore, monkeypatch
) -> None:
    """模型挂了也不能丢用户的话——先落库再调模型的意义就在这。"""
    from cs_agent.llm import LLMError

    def broken_stream(self, messages, **kwargs):  # noqa: ANN001
        raise LLMError("模拟接口故障", retryable=True)
        yield  # pragma: no cover

    monkeypatch.setattr(LLMClient, "stream_chat", broken_stream)
    sid = service.open_session()["id"]
    events = run_turn(service, sid, "这句話必須留下")

    assert [e["event"] for e in events][-1] == "done"
    error_event = next(e for e in events if e["event"] == "error")
    assert "模拟接口故障" in error_event["data"]["message"]
    assert error_event["data"]["retryable"] is True

    messages = store.list_messages(sid)
    assert messages[0]["content"] == "这句話必須留下"
    done = next(e["data"] for e in events if e["event"] == "done")
    assert done["error"]


def test_partial_content_is_persisted_on_failure(
    service: ConversationService, store: SessionStore, monkeypatch
) -> None:
    """中断前已产出的内容要落库，否则用户看到的内容与历史不一致。"""
    from cs_agent.llm import LLMError

    def partial_stream(self, messages, **kwargs):  # noqa: ANN001
        yield "已经说了一半"
        raise LLMError("连接中断", retryable=True)

    monkeypatch.setattr(LLMClient, "stream_chat", partial_stream)
    sid = service.open_session()["id"]
    events = run_turn(service, sid, "在吗")

    done = next(e["data"] for e in events if e["event"] == "done")
    assert done["content"] == "已经说了一半"
    assistant = store.list_messages(sid, roles=[ROLE_ASSISTANT])[0]
    assert assistant["content"] == "已经说了一半"
    assert assistant["meta"]["partial"] is True


# ---------------------------------------------------------------------------
# 离线模式
# ---------------------------------------------------------------------------
def test_offline_mode_discloses_itself(service: ConversationService) -> None:
    """离线时不假装自己是大模型——演示与测试都需要这个诚实性。"""
    sid = service.open_session()["id"]
    result = service.chat_once(sid, "你好")
    assert result["offline"] is True
    assert "离线模式" in result["content"]


def test_session_stats_shape(service: ConversationService) -> None:
    sid = service.open_session()["id"]
    run_turn(service, sid, "你好")
    stats = service.session_stats(sid)
    for key in ("messages", "turns", "total_chars", "token_estimate", "has_summary"):
        assert key in stats
    assert stats["has_summary"] is False
