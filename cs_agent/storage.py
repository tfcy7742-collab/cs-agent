"""SQLite 持久化层。

为什么客服 Agent 必须有持久化（而不是像普通 demo 那样放内存）：

* **长会话**：用户随时可能关掉页面再回来，会话必须能恢复；
* **跨会话记忆**（P2）：用户画像要活过一次会话；
* **可审计**：客服场景的每一次答复、每一次工具调用都应该能追溯。

并发模型：每线程一个连接（``threading.local``）。SQLite 默认的
``check_same_thread`` 限制与 FastAPI 的线程池模型冲突，用线程本地连接 + WAL
是最简单且不会踩坑的做法。所有写操作串行化在 ``self._write_lock`` 下。
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

#: 消息角色
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_SYSTEM = "system"
ROLE_SUMMARY = "summary"  # 滚动摘要（虚拟消息，不参与展示）
ROLE_HUMAN = "human_agent"  # 人工坐席的回复（P5）

#: 会话状态
SESSION_ACTIVE = "active"
SESSION_WAITING_HUMAN = "waiting_human"
SESSION_CLOSED = "closed"

#: 工单状态
HANDOFF_WAITING = "waiting"
HANDOFF_ACCEPTED = "accepted"
HANDOFF_RESOLVED = "resolved"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL DEFAULT 'anonymous',
    title           TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'active',
    turn_count      INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    turn            INTEGER NOT NULL DEFAULT 0,
    token_estimate  INTEGER NOT NULL DEFAULT 0,
    meta            TEXT NOT NULL DEFAULT '{}',
    -- 是否已被滚动摘要覆盖：被覆盖的原文不再进入工作记忆（但永久保留在库里）
    compressed      INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
-- 注意：涉及 compressed 列的索引**不在这里建**。老库没有该列时，
-- executescript 会在迁移补列之前就执行 CREATE INDEX 并直接报
-- "no such column: compressed"，导致服务起不来。见 _migrate()。

CREATE TABLE IF NOT EXISTS session_state (
    session_id      TEXT PRIMARY KEY,
    state           TEXT NOT NULL DEFAULT '{}',
    updated_at      TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS user_profiles (
    user_id         TEXT PRIMARY KEY,
    profile         TEXT NOT NULL DEFAULT '{}',
    updated_at      TEXT NOT NULL
);

-- 转人工工单：挂起状态与交接单都持久化，服务重启后坐席仍能接手
CREATE TABLE IF NOT EXISTS handoffs (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    user_id         TEXT NOT NULL DEFAULT 'anonymous',
    status          TEXT NOT NULL DEFAULT 'waiting',
    reason          TEXT NOT NULL DEFAULT '',
    triggers        TEXT NOT NULL DEFAULT '[]',
    packet          TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    closed_at       TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_handoffs_status ON handoffs(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_handoffs_session ON handoffs(session_id, created_at DESC);
"""


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数。

    中文按 1 字 ≈ 1 token、英文按 4 字符 ≈ 1 token 近似。
    只用于"是否需要压缩上下文"的决策与展示，不用于计费。
    """
    if not text:
        return 0
    chinese = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    others = len(text) - chinese
    return chinese + max(others // 4, 0) + 1


class SessionStore:
    """会话与消息的读写。"""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.RLock()
        self._init_schema()

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------
    @property
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path), timeout=15.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """写事务：串行化 + 自动提交/回滚。"""
        with self._write_lock:
            conn = self._conn
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def _init_schema(self) -> None:
        with self._write() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """轻量迁移：给老库补上后加的列与依赖这些列的索引。

        P2 引入了 ``messages.compressed``。已有数据库不会因为
        ``CREATE TABLE IF NOT EXISTS`` 而自动加列，所以这里显式检查。

        依赖新列的索引必须**在补列之后**创建：否则老库升级时
        ``CREATE INDEX`` 会先执行并抛 "no such column"（实测导致服务无法启动）。
        """
        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(messages)").fetchall()
        }
        if "compressed" not in columns:
            conn.execute(
                "ALTER TABLE messages ADD COLUMN compressed INTEGER NOT NULL DEFAULT 0"
            )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_uncompressed"
            " ON messages(session_id, compressed, id)"
        )

    # ------------------------------------------------------------------
    # 会话
    # ------------------------------------------------------------------
    def create_session(self, user_id: str = "anonymous", title: str = "") -> Dict[str, Any]:
        session_id = uuid.uuid4().hex[:16]
        stamp = _now()
        with self._write() as conn:
            conn.execute(
                "INSERT INTO sessions (id, user_id, title, status, turn_count, created_at, updated_at)"
                " VALUES (?, ?, ?, 'active', 0, ?, ?)",
                (session_id, user_id, title, stamp, stamp),
            )
        return self.get_session(session_id)  # type: ignore[return-value]

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_sessions(
        self, user_id: Optional[str] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        if user_id:
            rows = self._conn.execute(
                "SELECT * FROM sessions WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def update_session(
        self,
        session_id: str,
        title: Optional[str] = None,
        status: Optional[str] = None,
        bump_turn: bool = False,
    ) -> None:
        sets = ["updated_at = ?"]
        params: List[Any] = [_now()]
        if title is not None:
            sets.append("title = ?")
            params.append(title)
        if status is not None:
            sets.append("status = ?")
            params.append(status)
        if bump_turn:
            sets.append("turn_count = turn_count + 1")
        params.append(session_id)
        with self._write() as conn:
            conn.execute(f"UPDATE sessions SET {', '.join(sets)} WHERE id = ?", params)

    def delete_session(self, session_id: str) -> bool:
        with self._write() as conn:
            cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return cursor.rowcount > 0

    def find_stale_sessions(self, idle_minutes: int = 30) -> List[Dict[str, Any]]:
        """找出长时间没有新消息的活跃会话（供后台收尾，例如自动关闭）。"""
        cutoff = (datetime.now() - timedelta(minutes=idle_minutes)).strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )[:-3]
        rows = self._conn.execute(
            "SELECT * FROM sessions WHERE status = 'active' AND updated_at < ?",
            (cutoff,),
        ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # 消息
    # ------------------------------------------------------------------
    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        turn: int = 0,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        import json

        stamp = _now()
        payload = json.dumps(meta or {}, ensure_ascii=False)
        with self._write() as conn:
            cursor = conn.execute(
                "INSERT INTO messages (session_id, role, content, turn, token_estimate, meta, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, role, content, turn, estimate_tokens(content), payload, stamp),
            )
            message_id = int(cursor.lastrowid)
        return {
            "id": message_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "turn": turn,
            "token_estimate": estimate_tokens(content),
            "meta": meta or {},
            "created_at": stamp,
        }

    def list_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
        roles: Optional[List[str]] = None,
        only_uncompressed: bool = False,
        after_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """按写入顺序返回消息；``limit`` 表示"取最近 N 条"（仍按正序返回）。

        Args:
            only_uncompressed: 只要还没被摘要覆盖的消息（工作记忆用）。
            after_id: 只取 id 大于该值的消息（增量压缩用）。
        """
        import json

        sql = "SELECT * FROM messages WHERE session_id = ?"
        params: List[Any] = [session_id]
        if roles:
            sql += f" AND role IN ({','.join('?' * len(roles))})"
            params.extend(roles)
        if only_uncompressed:
            sql += " AND compressed = 0"
        if after_id is not None:
            sql += " AND id > ?"
            params.append(after_id)
        if limit is not None:
            sql += " ORDER BY id DESC LIMIT ?"
            params.append(limit)
            rows = self._conn.execute(sql, params).fetchall()
            rows = list(reversed(rows))
        else:
            sql += " ORDER BY id ASC"
            rows = self._conn.execute(sql, params).fetchall()

        result = []
        for row in rows:
            item = dict(row)
            try:
                item["meta"] = json.loads(item.get("meta") or "{}")
            except json.JSONDecodeError:
                item["meta"] = {}
            result.append(item)
        return result

    def uncompressed_stats(self, session_id: str) -> Dict[str, int]:
        """未压缩原文的条数与字符数（压缩决策用）。"""
        row = self._conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(LENGTH(content)), 0) AS chars"
            " FROM messages WHERE session_id = ? AND compressed = 0",
            (session_id,),
        ).fetchone()
        return {"count": int(row["n"]), "chars": int(row["chars"])}

    def compress_up_to(self, session_id: str, max_id: int) -> int:
        """把 id <= ``max_id`` 的原文标记为"已被摘要覆盖"。

        注意是**标记**而不是删除：长会话的出问题排查、以及"用户改口"时回溯，
        都依赖原始对话还在库里。

        **人工坐席的回复永不压缩**：它是权威信息（承诺了什么、怎么处理的），
        被摘要糊掉会让后续对话失去依据——这类信息必须原样留在工作记忆里。
        """
        with self._write() as conn:
            cursor = conn.execute(
                "UPDATE messages SET compressed = 1 WHERE session_id = ? AND id <= ?"
                " AND compressed = 0 AND role != ?",
                (session_id, max_id, ROLE_HUMAN),
            )
        return int(cursor.rowcount)

    def latest_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """取最近一条滚动摘要。"""
        messages = self.list_messages(session_id, roles=[ROLE_SUMMARY])
        return messages[-1] if messages else None

    # ------------------------------------------------------------------
    # 版本化事实（P2）
    # ------------------------------------------------------------------
    # 事实存在 session_state 的 "facts" 键下，结构：
    #   {"订单号": {"value": "SO20260101", "previous": "SO20251231",
    #               "changed_at": "2026-01-02 10:00:00", "source": "regex",
    #               "evidence": "订单号是 SO20260101", "updated_turn": 3}}
    # 为什么单独维护 "previous"：用户改口后，若把旧值直接丢掉，
    # 模型就无法回答"我改之前填的是哪个地址"；若新旧并列，模型又会随机挑一个。
    def get_facts(self, session_id: str) -> Dict[str, Dict[str, Any]]:
        return dict(self.get_state(session_id).get("facts") or {})

    def set_facts(self, session_id: str, facts: Dict[str, Dict[str, Any]]) -> None:
        state = self.get_state(session_id)
        state["facts"] = facts
        self.save_state(session_id, state)

    def merge_facts(
        self,
        session_id: str,
        updates: Dict[str, Dict[str, Any]],
        turn: int = 0,
    ) -> List[Dict[str, Any]]:
        """把新抽取到的事实并入会话事实，值变化时保留旧值。

        Returns:
            本次发生"值变化"的事实列表（供轨迹与前端展示"用户改了什么"）。
        """
        facts = self.get_facts(session_id)
        changes: List[Dict[str, Any]] = []

        for key, payload in updates.items():
            value = payload.get("value")
            if value in (None, ""):
                continue
            record = dict(payload)
            record.setdefault("updated_turn", turn)

            old = facts.get(key)
            if old and old.get("value") not in (None, "") and old.get("value") != value:
                record["previous"] = old.get("value")
                changes.append(
                    {"key": key, "old": old.get("value"), "new": value, "turn": turn}
                )
            elif old and old.get("value") == value:
                # 值没变：沿用旧的 previous / changed_at，避免把历史洗掉
                record.setdefault("previous", old.get("previous"))
                record.setdefault("changed_at", old.get("changed_at"))

            facts[key] = {k: v for k, v in record.items() if v not in (None, "")}

        self.set_facts(session_id, facts)
        return changes

    def mark_fact_changed(self, session_id: str, key: str, turn: int) -> None:
        """给某条事实打上"刚刚变更"的时间戳（值变化时调用）。"""
        facts = self.get_facts(session_id)
        if key not in facts:
            return
        facts[key]["changed_at"] = _now()
        facts[key]["updated_turn"] = turn
        self.set_facts(session_id, facts)

    # ------------------------------------------------------------------
    # 滚动摘要（P2）
    # ------------------------------------------------------------------
    def save_summary(
        self,
        session_id: str,
        content: str,
        *,
        covers_until_message_id: int,
        covers_until_turn: int,
        stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """写入滚动摘要。

        摘要采用**就地更新**：摘要本身是"截至目前的压缩视图"，同一会话保留一条即可，
        避免摘要层层叠加又变成新的上下文负担。被它覆盖的**原文仍然完整保留在库里**。
        """
        with self._write() as conn:
            conn.execute(
                "DELETE FROM messages WHERE session_id = ? AND role = ?",
                (session_id, ROLE_SUMMARY),
            )
        return self.append_message(
            session_id,
            ROLE_SUMMARY,
            content,
            turn=covers_until_turn,
            meta={
                "covers_until_message_id": covers_until_message_id,
                "covers_until_turn": covers_until_turn,
                "stats": stats or {},
            },
        )

    def count_messages(self, session_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()
        return int(row["n"]) if row else 0

    # ------------------------------------------------------------------
    # 会话状态（P3 使用；P1 先建好读写通道）
    # ------------------------------------------------------------------
    def get_state(self, session_id: str) -> Dict[str, Any]:
        import json

        row = self._conn.execute(
            "SELECT state FROM session_state WHERE session_id = ?", (session_id,)
        ).fetchone()
        if not row:
            return {}
        try:
            return json.loads(row["state"] or "{}")
        except json.JSONDecodeError:
            return {}

    def save_state(self, session_id: str, state: Dict[str, Any]) -> None:
        import json

        payload = json.dumps(state, ensure_ascii=False)
        with self._write() as conn:
            conn.execute(
                "INSERT INTO session_state (session_id, state, updated_at) VALUES (?, ?, ?)"
                " ON CONFLICT(session_id) DO UPDATE SET state = excluded.state,"
                " updated_at = excluded.updated_at",
                (session_id, payload, _now()),
            )

    # ------------------------------------------------------------------
    # 用户画像（P2 使用）
    # ------------------------------------------------------------------
    def get_profile(self, user_id: str) -> Dict[str, Any]:
        import json

        row = self._conn.execute(
            "SELECT profile FROM user_profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
        if not row:
            return {}
        try:
            return json.loads(row["profile"] or "{}")
        except json.JSONDecodeError:
            return {}

    def save_profile(self, user_id: str, profile: Dict[str, Any]) -> None:
        import json

        payload = json.dumps(profile, ensure_ascii=False)
        with self._write() as conn:
            conn.execute(
                "INSERT INTO user_profiles (user_id, profile, updated_at) VALUES (?, ?, ?)"
                " ON CONFLICT(user_id) DO UPDATE SET profile = excluded.profile,"
                " updated_at = excluded.updated_at",
                (user_id, payload, _now()),
            )

    # ------------------------------------------------------------------
    # 转人工工单（P5）
    # ------------------------------------------------------------------
    def create_handoff(
        self,
        session_id: str,
        user_id: str,
        reason: str,
        triggers: List[Dict[str, Any]],
        packet: Dict[str, Any],
    ) -> Dict[str, Any]:
        """创建工单（交接单随工单一起落库）。"""
        import json

        handoff_id = f"HO{uuid.uuid4().hex[:10].upper()}"
        stamp = _now()
        with self._write() as conn:
            conn.execute(
                "INSERT INTO handoffs (id, session_id, user_id, status, reason, triggers,"
                " packet, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    handoff_id,
                    session_id,
                    user_id,
                    HANDOFF_WAITING,
                    reason,
                    json.dumps(triggers, ensure_ascii=False),
                    json.dumps(packet, ensure_ascii=False),
                    stamp,
                    stamp,
                ),
            )
        return self.get_handoff(handoff_id)  # type: ignore[return-value]

    def get_handoff(self, handoff_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM handoffs WHERE id = ?", (handoff_id,)
        ).fetchone()
        return self._decode_handoff(dict(row)) if row else None

    def latest_handoff(self, session_id: str) -> Optional[Dict[str, Any]]:
        """取会话最近一张工单（判断是否仍在等人工）。"""
        row = self._conn.execute(
            "SELECT * FROM handoffs WHERE session_id = ? ORDER BY created_at DESC, id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        return self._decode_handoff(dict(row)) if row else None

    def list_handoffs(
        self, status: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """坐席工作台用：按状态列出工单（等待中的排在最前）。"""
        if status:
            rows = self._conn.execute(
                "SELECT * FROM handoffs WHERE status = ? ORDER BY created_at ASC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM handoffs ORDER BY"
                " CASE status WHEN 'waiting' THEN 0 WHEN 'accepted' THEN 1 ELSE 2 END,"
                " created_at ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._decode_handoff(dict(row)) for row in rows]

    def update_handoff(
        self,
        handoff_id: str,
        status: Optional[str] = None,
        note: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """更新工单状态；传 ``note`` 时追加一条坐席备注。"""
        handoff = self.get_handoff(handoff_id)
        if handoff is None:
            return None

        sets = ["updated_at = ?"]
        params: List[Any] = [_now()]
        if status is not None:
            sets.append("status = ?")
            params.append(status)
            if status in (HANDOFF_RESOLVED,):
                sets.append("closed_at = ?")
                params.append(_now())

        packet = dict(handoff.get("packet") or {})
        if note:
            notes = list(packet.get("坐席备注") or [])
            notes.append({"at": _now(), "note": note})
            packet["坐席备注"] = notes
            sets.append("packet = ?")
            import json

            params.append(json.dumps(packet, ensure_ascii=False))

        params.append(handoff_id)
        with self._write() as conn:
            conn.execute(f"UPDATE handoffs SET {', '.join(sets)} WHERE id = ?", params)
        return self.get_handoff(handoff_id)

    def update_handoff_packet(self, handoff_id: str, packet: Dict[str, Any]) -> None:
        import json

        with self._write() as conn:
            conn.execute(
                "UPDATE handoffs SET packet = ?, updated_at = ? WHERE id = ?",
                (json.dumps(packet, ensure_ascii=False), _now(), handoff_id),
            )

    @staticmethod
    def _decode_handoff(row: Dict[str, Any]) -> Dict[str, Any]:
        import json

        for key in ("triggers", "packet"):
            try:
                row[key] = json.loads(row.get(key) or ("[]" if key == "triggers" else "{}"))
            except json.JSONDecodeError:
                row[key] = [] if key == "triggers" else {}
        return row
