"""P1：HTTP 接口测试。

覆盖：健康检查、会话 CRUD、非流式对话、SSE 流式对话、错误码与中文提示。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from fastapi.testclient import TestClient


def parse_sse(response: Any) -> List[Dict[str, Any]]:
    """解析 SSE 响应。

    SSE 规定一个帧以一个空行结束，而 ``data:`` 与 ``event:`` 是**分别**的行。
    所以必须按帧解析（累积到空行为止），不能"见到 data 就收尾"——
    本项目的事件载荷里含有真实换行符的转义形式，误当帧结束会把消息截断。
    """
    raw = getattr(response, "text", None)
    if raw is None:  # pragma: no cover - 防御性分支
        content = getattr(response, "content", b"")
        raw = content.decode("utf-8") if isinstance(content, bytes) else str(content)

    events: List[Dict[str, Any]] = []
    event = ""
    data_lines: List[str] = []

    def flush() -> None:
        nonlocal event, data_lines
        if data_lines:
            try:
                events.append({"event": event, "data": json.loads("\n".join(data_lines))})
            except json.JSONDecodeError:  # pragma: no cover - 防御性分支
                pass
        event = ""
        data_lines = []

    for line in str(raw).splitlines():
        if not line.strip():
            flush()
        elif line.startswith("event: "):
            event = line[7:].strip()
        elif line.startswith("data: "):
            data_lines.append(line[6:])
    flush()
    return events


# ---------------------------------------------------------------------------
# 基础
# ---------------------------------------------------------------------------
def test_root_and_manifest(client: TestClient) -> None:
    root = client.get("/")
    assert root.status_code == 200
    assert root.json()["service"] == "cs-agent"

    # 浏览器会自动请求，必须存在，否则日志里全是 404
    assert client.get("/manifest.json").status_code == 200


def test_health_reports_offline_mode(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["llm"]["online"] is False
    assert body["llm"]["mode"] == "offline"
    # 健康检查只回显脱敏 Key：保留头尾、中间省略，长度远小于原值
    key = body["llm"]["api_key"]
    assert key == "" or ("..." in key and len(key) <= 20)
    assert "db_path" in body["storage"]
    assert body["conversation"]["working_recent_turns"] == 4


def test_health_masks_api_key(client: TestClient, monkeypatch) -> None:
    """健康检查可以暴露"配没配 Key"，但绝不能回显完整 Key。"""
    from cs_agent.config import get_settings

    settings = get_settings(reload=True)
    settings.api_key = "sk-1234567890abcdef"
    assert settings.masked_key == "sk-123...cdef"
    assert "1234567890" not in settings.masked_key


# ---------------------------------------------------------------------------
# 会话 CRUD
# ---------------------------------------------------------------------------
def test_create_and_fetch_session(client: TestClient) -> None:
    created = client.post("/api/sessions", json={"user_id": "u1", "title": "我的会话"})
    assert created.status_code == 200
    session = created.json()["session"]
    assert session["user_id"] == "u1"

    detail = client.get(f"/api/sessions/{session['id']}").json()
    assert detail["session"]["id"] == session["id"]
    assert detail["messages"] == []
    assert detail["stats"]["turns"] == 0


def test_list_sessions(client: TestClient) -> None:
    for index in range(3):
        client.post("/api/sessions", json={"user_id": "u1", "title": f"s{index}"})
    client.post("/api/sessions", json={"user_id": "u2"})

    body = client.get("/api/sessions", params={"user_id": "u1"}).json()
    assert body["total"] == 3
    assert all(s["user_id"] == "u1" for s in body["sessions"])
    assert client.get("/api/sessions").json()["total"] == 4


def test_get_missing_session_returns_404_with_chinese_detail(client: TestClient) -> None:
    response = client.get("/api/sessions/not-exist")
    assert response.status_code == 404
    assert "会话不存在" in response.json()["detail"]


def test_delete_session(client: TestClient) -> None:
    sid = client.post("/api/sessions", json={}).json()["session"]["id"]
    assert client.delete(f"/api/sessions/{sid}").json()["deleted"] is True
    assert client.get(f"/api/sessions/{sid}").status_code == 404
    assert client.delete(f"/api/sessions/{sid}").status_code == 404


# ---------------------------------------------------------------------------
# 对话（非流式）
# ---------------------------------------------------------------------------
def test_chat_non_stream_creates_session_and_replies(client: TestClient) -> None:
    """不传 session_id 时应自动建会话，并在响应里带回 ID。"""
    response = client.post("/api/chat", json={"message": "你好", "stream": False})
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"]
    assert body["turn"] == 1
    assert body["content"].strip()
    assert body["offline"] is True
    assert body["usage"]["prompt_tokens"] > 0

    detail = client.get(f"/api/sessions/{body['session_id']}").json()
    assert len(detail["messages"]) == 2


def test_chat_continues_existing_session(client: TestClient) -> None:
    first = client.post("/api/chat", json={"message": "第一轮", "stream": False}).json()
    sid = first["session_id"]
    second = client.post(
        "/api/chat", json={"message": "第二轮", "session_id": sid, "stream": False}
    ).json()
    assert second["session_id"] == sid
    assert second["turn"] == 2
    assert client.get(f"/api/sessions/{sid}").json()["stats"]["turns"] == 2


def test_chat_with_unknown_session_returns_404(client: TestClient) -> None:
    response = client.post(
        "/api/chat", json={"message": "你好", "session_id": "nope", "stream": False}
    )
    assert response.status_code == 404
    assert "会话不存在" in response.json()["detail"]


def test_chat_rejects_empty_message(client: TestClient) -> None:
    response = client.post("/api/chat", json={"message": "", "stream": False})
    # pydantic 的 min_length 校验会拦下来
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 对话（SSE）
# ---------------------------------------------------------------------------
def test_chat_stream_returns_sse_frames(client: TestClient) -> None:
    response = client.post("/api/chat", json={"message": "你好", "stream": True})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["X-Session-Id"]

    events = parse_sse(response)
    names = [e["event"] for e in events]
    assert names[0] == "session"
    assert names[1] == "start"
    assert "delta" in names
    assert names[-1] == "done"

    done = events[-1]["data"]
    assert done["content"].strip()
    assert done["turn"] == 1


def test_sse_deltas_concatenate_to_final_content(client: TestClient) -> None:
    """流式协议的自洽性：所有 delta 拼起来必须等于最终内容。"""
    response = client.post("/api/chat", json={"message": "帮我查个订单", "stream": True})
    events = parse_sse(response)
    joined = "".join(e["data"]["text"] for e in events if e["event"] == "delta")
    done = next(e["data"] for e in events if e["event"] == "done")
    assert joined == done["content"]


def test_chat_stream_get_endpoint(client: TestClient) -> None:
    """EventSource 只能发 GET，因此必须有 GET 入口。"""
    response = client.get("/api/chat/stream", params={"message": "你好"})
    assert response.status_code == 200
    events = parse_sse(response)
    assert events[-1]["event"] == "done"


def test_stream_error_for_unknown_session(client: TestClient) -> None:
    response = client.get("/api/chat/stream", params={"message": "你好", "session_id": "nope"})
    assert response.status_code == 404
    assert "会话不存在" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 会话恢复（长会话的关键场景）
# ---------------------------------------------------------------------------
def test_session_can_be_resumed_after_restart(client: TestClient, settings) -> None:
    """关掉整个应用再打开，仍能接着聊——长会话必须支持这个场景。"""
    from cs_agent import app as app_module
    from cs_agent.api import chat as chat_module

    first = client.post(
        "/api/chat", json={"message": "记住：订单 SO20260101", "stream": False}
    ).json()
    sid = first["session_id"]

    # 模拟重启：丢掉全局 store 与 service，重新建一个应用
    app_module.reset_store()
    chat_module.reset_service()
    from fastapi.testclient import TestClient as FreshClient

    with FreshClient(app_module.create_app()) as fresh:
        detail = fresh.get(f"/api/sessions/{sid}").json()
        assert detail["stats"]["turns"] == 1
        assert "SO20260101" in detail["messages"][0]["content"]

        second = fresh.post(
            "/api/chat", json={"message": "接着聊", "session_id": sid, "stream": False}
        ).json()
        assert second["turn"] == 2

    app_module.reset_store()
    chat_module.reset_service()


def test_session_stats_endpoint_reports_growth(client: TestClient) -> None:
    """P6 的"长会话 token 曲线"依赖这个口径：按落库内容估算。"""
    sid = client.post("/api/chat", json={"message": "第一轮", "stream": False}).json()[
        "session_id"
    ]
    before = client.get(f"/api/sessions/{sid}").json()["stats"]["token_estimate"]
    client.post("/api/chat", json={"message": "第二轮", "session_id": sid, "stream": False})
    after = client.get(f"/api/sessions/{sid}").json()["stats"]["token_estimate"]
    assert after > before
