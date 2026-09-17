"""P1：持久化层测试。

重点验证三件事：
1. 会话与消息能正确落库、按序读回；
2. **重开连接（模拟服务重启）后数据仍在**——这是长会话的前提；
3. 级联删除、状态与画像的读写不串会话。
"""

from __future__ import annotations

import pytest

from cs_agent.storage import (
    ROLE_ASSISTANT,
    ROLE_SUMMARY,
    ROLE_USER,
    SessionStore,
    estimate_tokens,
)


# ---------------------------------------------------------------------------
# 基础读写
# ---------------------------------------------------------------------------
def test_create_and_get_session(store: SessionStore) -> None:
    session = store.create_session(user_id="u1", title="测试会话")
    assert session["id"]
    assert session["user_id"] == "u1"
    assert session["title"] == "测试会话"
    assert session["status"] == "active"
    assert session["turn_count"] == 0
    assert store.get_session(session["id"])["id"] == session["id"]


def test_get_missing_session_returns_none(store: SessionStore) -> None:
    assert store.get_session("does-not-exist") is None


def test_append_and_list_messages_in_order(store: SessionStore) -> None:
    session = store.create_session()
    sid = session["id"]
    store.append_message(sid, ROLE_USER, "你好", turn=1)
    store.append_message(sid, ROLE_ASSISTANT, "您好，请问有什么可以帮您？", turn=1)
    store.append_message(sid, ROLE_USER, "我的订单还没发货", turn=2)

    messages = store.list_messages(sid)
    assert [m["role"] for m in messages] == [ROLE_USER, ROLE_ASSISTANT, ROLE_USER]
    assert [m["content"] for m in messages] == [
        "你好",
        "您好，请问有什么可以帮您？",
        "我的订单还没发货",
    ]
    assert [m["turn"] for m in messages] == [1, 1, 2]
    assert messages[0]["token_estimate"] > 0


def test_list_messages_limit_returns_latest_in_order(store: SessionStore) -> None:
    """``limit`` 取"最近 N 条"，但返回时仍按时间正序（顺序错了会让模型读反对话）。"""
    session = store.create_session()
    sid = session["id"]
    for index in range(6):
        store.append_message(sid, ROLE_USER, f"第{index}条", turn=index + 1)

    recent = store.list_messages(sid, limit=3)
    assert [m["content"] for m in recent] == ["第3条", "第4条", "第5条"]


def test_list_messages_filter_by_role(store: SessionStore) -> None:
    session = store.create_session()
    sid = session["id"]
    store.append_message(sid, ROLE_USER, "问题")
    store.append_message(sid, ROLE_ASSISTANT, "回答")
    store.append_message(sid, ROLE_SUMMARY, "摘要内容")

    only_user = store.list_messages(sid, roles=[ROLE_USER])
    assert [m["content"] for m in only_user] == ["问题"]
    only_summary = store.list_messages(sid, roles=[ROLE_SUMMARY])
    assert [m["content"] for m in only_summary] == ["摘要内容"]
    assert store.latest_summary(sid)["content"] == "摘要内容"


def test_message_meta_roundtrip(store: SessionStore) -> None:
    session = store.create_session()
    sid = session["id"]
    meta = {"trace": {"turn": 1, "latency_ms": 120}, "note": "中文备注"}
    store.append_message(sid, ROLE_ASSISTANT, "答复", turn=1, meta=meta)
    read_back = store.list_messages(sid)[0]["meta"]
    assert read_back["trace"]["latency_ms"] == 120
    assert read_back["note"] == "中文备注"


# ---------------------------------------------------------------------------
# 持久性（长会话的核心前提）
# ---------------------------------------------------------------------------
def test_messages_survive_reconnect(settings) -> None:
    """模拟"服务重启"：关掉连接重新打开，历史必须还在。"""
    db_path = settings.db_path
    first = SessionStore(db_path)
    session = first.create_session(user_id="u9")
    sid = session["id"]
    first.append_message(sid, ROLE_USER, "记住这句话：我的订单号是 SO20260101", turn=1)
    first.append_message(sid, ROLE_ASSISTANT, "好的，已记录", turn=1)
    first.close()

    second = SessionStore(db_path)
    try:
        messages = second.list_messages(sid)
        assert len(messages) == 2
        assert "SO20260101" in messages[0]["content"]
        assert second.get_session(sid)["user_id"] == "u9"
    finally:
        second.close()


def test_delete_session_cascades_messages(store: SessionStore) -> None:
    sid = store.create_session()["id"]
    store.append_message(sid, ROLE_USER, "问题")
    store.save_state(sid, {"order_id": "SO1"})

    assert store.delete_session(sid) is True
    assert store.get_session(sid) is None
    assert store.list_messages(sid) == []
    assert store.get_state(sid) == {}
    assert store.delete_session(sid) is False


# ---------------------------------------------------------------------------
# 老库升级路径（回归：这一条漏测过，导致服务无法启动）
# ---------------------------------------------------------------------------
def test_upgrade_from_old_database_without_compressed_column(tmp_path) -> None:
    """模拟 P1 时期的老库：messages 表没有 compressed 列。

    老库升级时曾经直接崩在启动阶段——依赖新列的 ``CREATE INDEX``
    在迁移补列之前执行，抛 "no such column: compressed"。
    这条测试专门守住升级路径（新建库的测试是发现不了这个问题的）。
    """
    import sqlite3

    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL DEFAULT 'anonymous',
            title TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'active',
            turn_count INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
            role TEXT NOT NULL, content TEXT NOT NULL, turn INTEGER NOT NULL DEFAULT 0,
            token_estimate INTEGER NOT NULL DEFAULT 0, meta TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        INSERT INTO sessions (id, user_id, title, status, turn_count, created_at, updated_at)
        VALUES ('old-1', 'u-old', '历史会话', 'active', 1, '2026-01-01 00:00:00', '2026-01-01 00:00:00');
        INSERT INTO messages (session_id, role, content, turn, created_at)
        VALUES ('old-1', 'user', '老库里的历史消息', 1, '2026-01-01 00:00:00');
        """
    )
    conn.commit()
    conn.close()

    # 关键：打开老库不应抛异常，且历史数据要能读出来
    upgraded = SessionStore(db)
    try:
        columns = {
            row["name"]
            for row in upgraded._conn.execute("PRAGMA table_info(messages)").fetchall()
        }
        assert "compressed" in columns, "迁移应补上 compressed 列"
        messages = upgraded.list_messages("old-1")
        assert messages and messages[0]["content"] == "老库里的历史消息"
        assert messages[0]["compressed"] == 0
        # 新功能在老库上可用
        assert upgraded.compress_up_to("old-1", messages[0]["id"]) == 1
        assert upgraded.uncompressed_stats("old-1")["count"] == 0
    finally:
        upgraded.close()


# ---------------------------------------------------------------------------
# 会话状态与画像
# ---------------------------------------------------------------------------
def test_state_is_per_session(store: SessionStore) -> None:
    a = store.create_session()["id"]
    b = store.create_session()["id"]
    store.save_state(a, {"order_id": "SO-A"})
    store.save_state(b, {"order_id": "SO-B"})

    assert store.get_state(a) == {"order_id": "SO-A"}
    assert store.get_state(b) == {"order_id": "SO-B"}
    assert store.get_state("none") == {}


def test_state_upsert_overwrites(store: SessionStore) -> None:
    """状态是"当前值"语义，重复保存必须覆盖而不是追加。"""
    sid = store.create_session()["id"]
    store.save_state(sid, {"order_id": "SO-OLD", "city": "北京"})
    store.save_state(sid, {"order_id": "SO-NEW"})
    assert store.get_state(sid) == {"order_id": "SO-NEW"}


def test_profile_is_per_user(store: SessionStore) -> None:
    store.save_profile("u1", {"称呼": "张先生", "常用地址": "北京市朝阳区"})
    store.save_profile("u2", {"称呼": "李女士"})
    assert store.get_profile("u1")["称呼"] == "张先生"
    assert store.get_profile("u2")["称呼"] == "李女士"
    assert store.get_profile("unknown") == {}


def test_update_session_fields_and_turn_bump(store: SessionStore) -> None:
    sid = store.create_session()["id"]
    store.update_session(sid, title="新标题", bump_turn=True)
    store.update_session(sid, bump_turn=True)
    session = store.get_session(sid)
    assert session["title"] == "新标题"
    assert session["turn_count"] == 2
    store.update_session(sid, status="closed")
    assert store.get_session(sid)["status"] == "closed"


def test_list_sessions_orders_by_recent(store: SessionStore) -> None:
    first = store.create_session(user_id="u1")["id"]
    second = store.create_session(user_id="u1")["id"]
    store.create_session(user_id="u2")
    store.update_session(first, title="最近更新过")

    sessions = store.list_sessions(user_id="u1")
    assert [s["id"] for s in sessions] == [first, second]
    assert len(store.list_sessions()) == 3


def test_find_stale_sessions(store: SessionStore) -> None:
    sid = store.create_session()["id"]
    assert store.find_stale_sessions(idle_minutes=30) == []
    assert [s["id"] for s in store.find_stale_sessions(idle_minutes=-1)] == [sid]


# ---------------------------------------------------------------------------
# token 估算
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected_min",
    [("", 0), ("你好", 2), ("hello world", 1), ("订单 SO123 已发货", 7)],
)
def test_estimate_tokens_is_reasonable(text: str, expected_min: int) -> None:
    value = estimate_tokens(text)
    assert value >= expected_min
    if not text:
        assert value == 0


def test_estimate_tokens_chinese_is_denser_than_english() -> None:
    """中文按 1 字 ≈ 1 token，英文按 4 字符 ≈ 1 token，同长度下中文估算应更高。"""
    chinese = estimate_tokens("我" * 40)
    english = estimate_tokens("a" * 40)
    assert chinese > english
