"""转人工服务：交接单生成、挂起、以及坐席侧的处理闭环。

交接单的价值在于**不让坐席从零开始**。它必须结构化回答四个问题：

1. 用户是谁、要办什么（诉求）；
2. 已经确认了哪些信息（订单号、原因…）；
3. 系统已经试过什么（查了哪些工具、结果如何）；
4. 现在卡在哪里（缺什么信息 / 超权限 / 情绪升级）。

另外两件在长会话里特别重要的事：

* **挂起必须持久化**：会话状态与工单都落库，服务重启后坐席仍能接手；
* **接续不重跑**：坐席回复后会话继续，已确认的槽位与事实都还在，
  不会出现"人工接手后又问一遍订单号"。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import Settings, get_settings
from .facts import render_facts
from .handoff_rules import EscalationVerdict
from .state import StateSnapshot
from .storage import (
    HANDOFF_ACCEPTED,
    HANDOFF_RESOLVED,
    HANDOFF_WAITING,
    ROLE_HUMAN,
    ROLE_USER,
    SESSION_ACTIVE,
    SESSION_WAITING_HUMAN,
    SessionStore,
)
from .tasks import missing_required, task_for_intent

logger = logging.getLogger(__name__)

#: 交接单里"已核实信息"最多列几条（避免长会话把整份状态都倒给坐席）
MAX_PACKET_ITEMS = 12


@dataclass
class HandoffResult:
    """一次转人工的结果。"""

    handoff_id: str = ""
    session_id: str = ""
    reason: str = ""
    reply: str = ""
    packet: Dict[str, Any] = field(default_factory=dict)
    already_waiting: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "handoff_id": self.handoff_id,
            "session_id": self.session_id,
            "reason": self.reason,
            "reply": self.reply,
            "already_waiting": self.already_waiting,
            "packet": self.packet,
        }


class HandoffService:
    """工单的创建、查询与坐席操作。"""

    def __init__(
        self, store: SessionStore, settings: Optional[Settings] = None
    ) -> None:
        self.store = store
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------
    # 交接单
    # ------------------------------------------------------------------
    def build_packet(
        self,
        session_id: str,
        verdict: Optional[EscalationVerdict] = None,
        *,
        extra_notes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """生成结构化交接单。"""
        session = self.store.get_session(session_id) or {}
        snapshot = StateSnapshot.load(self.store, session_id)
        facts = render_facts(self.store.get_facts(session_id))
        summary = self.store.latest_summary(session_id)
        trace = self._latest_trace(session_id)

        # 已核实信息 = 槽位 + 关键事实（去重）
        confirmed: Dict[str, str] = {}
        for key, value in {**facts, **snapshot.slots}.items():
            if value and key not in confirmed:
                confirmed[key] = str(value)

        open_tasks: List[Dict[str, Any]] = []
        for task in snapshot.open_tasks():
            task_def = task_for_intent(task.kind)
            item: Dict[str, Any] = {
                "任务": task_def.name if task_def else task.kind,
                "状态": task.status,
            }
            if task_def is not None:
                missing = missing_required(task_def, {**snapshot.slots, **task.filled})
                if missing:
                    item["仍缺"] = missing
            open_tasks.append(item)

        attempted = self._attempted_actions(trace)

        packet: Dict[str, Any] = {
            "工单号": "",  # 创建后回填
            "会话ID": session_id,
            "用户": session.get("user_id", "anonymous"),
            "诉求": facts.get("诉求") or "（未识别出明确诉求）",
            "已核实信息": dict(list(confirmed.items())[:MAX_PACKET_ITEMS]),
            "未办结事项": open_tasks,
            "系统已尝试": attempted,
            "当前卡点": (verdict.reason if verdict else "") or "需人工判断",
            "触发规则": [item.as_dict() for item in (verdict.triggers if verdict else [])],
            "用户情绪": (
                verdict.sentiment.as_dict() if verdict else {"score": 0, "words": []}
            ),
            "历史摘要": (summary or {}).get("content", ""),
            "轮次数": self._current_turn(session_id, session),
        }
        if extra_notes:
            packet["补充说明"] = extra_notes
        packet["交接说明"] = self.render_text(packet)
        return packet

    @staticmethod
    def _attempted_actions(trace: Optional[Dict[str, Any]]) -> List[str]:
        """从最近一轮轨迹里提取"已经试过什么"。"""
        if not trace:
            return []
        items: List[str] = []
        tools = (trace.get("tools") or {}).get("runs") or []
        for run in tools:
            name = run.get("tool", "")
            if run.get("ok"):
                items.append(f"调用 {name}：成功")
            else:
                items.append(
                    f"调用 {name}：失败（{run.get('error_code') or run.get('error') or '未知原因'}）"
                )
        plan = trace.get("plan") or {}
        if plan.get("action") == "ask":
            ask = plan.get("ask") or {}
            items.append(f"向用户追问：{ask.get('target') or '补充信息'}")
        if plan.get("clarification"):
            items.append("请用户澄清指代（订单不唯一）")
        memory = trace.get("memory") or {}
        if memory.get("fact_changes"):
            for change in memory["fact_changes"]:
                items.append(f"用户改口：{change.get('message', '')}")
        return items[:MAX_PACKET_ITEMS]

    def _current_turn(self, session_id: str, session: Dict[str, Any]) -> int:
        """本轮的真实轮次数。

        ``turn_count`` 是"**已完成**的轮数"：转人工发生在助手回复之前，
        直接用它会少算一轮（交接单写 1 轮、实际用户已经说了 2 次）。
        这里按已落库的用户消息条数计算，更贴近坐席看到的对话长度。
        """
        user_messages = self.store.list_messages(session_id, roles=[ROLE_USER])
        return max(len(user_messages), int(session.get("turn_count") or 0), 1)

    def _latest_trace(self, session_id: str) -> Optional[Dict[str, Any]]:
        messages = self.store.list_messages(session_id, roles=["assistant"])
        if not messages:
            return None
        return (messages[-1].get("meta") or {}).get("trace")

    @staticmethod
    def render_text(packet: Dict[str, Any]) -> str:
        """把交接单渲染成坐席一眼能扫完的文本。"""
        lines: List[str] = []
        header = packet.get("工单号") or "（待生成）"
        lines.append(f"【交接单 {header}】")
        lines.append(f"用户：{packet.get('用户')}｜轮次：{packet.get('轮次数')}")

        confirmed = packet.get("已核实信息") or {}
        if confirmed:
            items = "；".join(f"{k}={v}" for k, v in confirmed.items())
            lines.append(f"已核实信息：{items}")
        else:
            lines.append("已核实信息：（无）")

        tasks = packet.get("未办结事项") or []
        if tasks:
            parts = []
            for item in tasks:
                text = f"{item.get('任务')}（{item.get('状态')}）"
                if item.get("仍缺"):
                    text += f"仍缺：{'、'.join(item['仍缺'])}"
                parts.append(text)
            lines.append("未办结事项：" + "；".join(parts))

        attempted = packet.get("系统已尝试") or []
        if attempted:
            lines.append("系统已尝试：" + "；".join(attempted))

        lines.append(f"当前卡点：{packet.get('当前卡点', '')}")

        sentiment = packet.get("用户情绪") or {}
        if sentiment.get("score"):
            words = "、".join(sentiment.get("words") or [])
            lines.append(f"用户情绪：{sentiment.get('score')} 分（{words}）")

        if packet.get("历史摘要"):
            lines.append("历史摘要：" + str(packet["历史摘要"]).replace("\n", " ")[:300])
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 转人工
    # ------------------------------------------------------------------
    def create(
        self,
        session_id: str,
        verdict: EscalationVerdict,
        *,
        reply: str = "",
        extra_notes: Optional[List[str]] = None,
    ) -> HandoffResult:
        """创建工单并把会话挂起。"""
        existing = self.store.latest_handoff(session_id)
        if existing and existing["status"] in (HANDOFF_WAITING, HANDOFF_ACCEPTED):
            # 已经在等人工了：不重复建单，只把新信息补进交接单
            packet = existing.get("packet") or {}
            fresh = self.build_packet(session_id, verdict, extra_notes=extra_notes)
            fresh["工单号"] = existing["id"]
            fresh["历史摘要"] = packet.get("历史摘要") or fresh.get("历史摘要", "")
            fresh["交接说明"] = self.render_text(fresh)
            self.store.update_handoff_packet(existing["id"], fresh)
            self.store.update_session(session_id, status=SESSION_WAITING_HUMAN)
            return HandoffResult(
                handoff_id=existing["id"],
                session_id=session_id,
                reason=existing["reason"],
                reply=reply,
                packet=fresh,
                already_waiting=True,
            )

        packet = self.build_packet(session_id, verdict, extra_notes=extra_notes)
        session = self.store.get_session(session_id) or {}
        handoff = self.store.create_handoff(
            session_id=session_id,
            user_id=session.get("user_id", "anonymous"),
            reason=verdict.reason or "转人工",
            triggers=[item.as_dict() for item in verdict.triggers],
            packet=packet,
        )
        packet["工单号"] = handoff["id"]
        packet["交接说明"] = self.render_text(packet)
        self.store.update_handoff_packet(handoff["id"], packet)
        self.store.update_session(session_id, status=SESSION_WAITING_HUMAN)

        logger.info(
            "会话 %s 转人工：工单 %s（%s）", session_id, handoff["id"], verdict.reason
        )
        return HandoffResult(
            handoff_id=handoff["id"],
            session_id=session_id,
            reason=verdict.reason,
            reply=reply,
            packet=packet,
        )

    # ------------------------------------------------------------------
    # 坐席侧
    # ------------------------------------------------------------------
    def workspace(self, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """坐席工作台列表：等待中的排最前，带交接单摘要。"""
        items = self.store.list_handoffs(status=status, limit=limit)
        result: List[Dict[str, Any]] = []
        for item in items:
            packet = item.get("packet") or {}
            session = self.store.get_session(item["session_id"]) or {}
            result.append(
                {
                    "handoff_id": item["id"],
                    "session_id": item["session_id"],
                    "user_id": item["user_id"],
                    "status": item["status"],
                    "reason": item["reason"],
                    "created_at": item["created_at"],
                    "turns": packet.get("轮次数", 0),
                    "demand": packet.get("诉求", ""),
                    "blocker": packet.get("当前卡点", ""),
                    "sentiment": (packet.get("用户情绪") or {}).get("score", 0),
                    "session_title": session.get("title", ""),
                }
            )
        return result

    def detail(self, handoff_id: str) -> Optional[Dict[str, Any]]:
        handoff = self.store.get_handoff(handoff_id)
        if handoff is None:
            return None
        session_id = handoff["session_id"]
        return {
            "handoff": handoff,
            "packet_text": (handoff.get("packet") or {}).get("交接说明", ""),
            "messages": self.store.list_messages(session_id),
            "state": StateSnapshot.load(self.store, session_id).summary(),
        }

    def accept(self, handoff_id: str, agent: str = "坐席") -> Optional[Dict[str, Any]]:
        """坐席认领工单。"""
        return self.store.update_handoff(
            handoff_id, status=HANDOFF_ACCEPTED, note=f"{agent} 已认领"
        )

    def reply(self, handoff_id: str, text: str, agent: str = "人工客服") -> Optional[Dict[str, Any]]:
        """坐席回复：写入会话历史，会话**回到活跃**，用户可继续对话。

        关键点：只追加一条人工消息，**不重建会话状态**——
        已确认的槽位与事实原样保留，人工处理完用户接着说也不会被重复追问。
        """
        handoff = self.store.get_handoff(handoff_id)
        if handoff is None:
            return None
        session_id = handoff["session_id"]
        # 人工回复只在"有实际内容"时才占用一个对话轮次
        session = self.store.get_session(session_id) or {}
        turn = int(session.get("turn_count") or 0)
        self.store.append_message(
            session_id,
            ROLE_HUMAN,
            text,
            turn=turn,
            meta={"handoff_id": handoff_id, "agent": agent},
        )
        self.store.update_session(session_id, status=SESSION_ACTIVE)
        self.store.update_handoff(handoff_id, status=HANDOFF_ACCEPTED, note=f"{agent}：{text[:80]}")
        logger.info("工单 %s 收到坐席回复，会话 %s 恢复活跃", handoff_id, session_id)
        return self.store.get_handoff(handoff_id)

    def resolve(self, handoff_id: str, note: str = "") -> Optional[Dict[str, Any]]:
        """办结工单：会话回到活跃（用户可继续追问）。"""
        handoff = self.store.update_handoff(
            handoff_id, status=HANDOFF_RESOLVED, note=note or "已办结"
        )
        if handoff is None:
            return None
        self.store.update_session(handoff["session_id"], status=SESSION_ACTIVE)
        return handoff

    # ------------------------------------------------------------------
    def waiting_handoff(self, session_id: str) -> Optional[Dict[str, Any]]:
        """会话当前是否在等人工（有未办结工单）。"""
        handoff = self.store.latest_handoff(session_id)
        if handoff and handoff["status"] in (HANDOFF_WAITING, HANDOFF_ACCEPTED):
            return handoff
        return None

    @staticmethod
    def user_notice(handoff: HandoffResult) -> str:
        """给用户看的转接说明（解释原因，不甩锅、不含糊）。"""
        if handoff.already_waiting:
            return (
                f"您的诉求我已经记录在工单 {handoff.handoff_id} 里，"
                "人工客服会尽快接手。这期间您可以继续补充信息，我会一并转过去。"
            )
        return (
            f"这件事我已经帮您转接人工客服处理，工单号 {handoff.handoff_id}。\n"
            f"转接原因：{handoff.reason}。\n"
            "人工客服会看到前面我们已经核对过的信息，您不需要再重复一遍。"
        )
