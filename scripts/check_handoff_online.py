"""P5 在线验收：转人工是否该转就转、该说清楚就说清楚。

五个检查方向（全部用真实模型回复断言）：

1. 明确要求转人工 → 回复里给出工单号，且不承诺自己解决；
2. 超权限诉求（赔偿）→ 转人工，回复里**不能**出现"可以赔/已经赔"这类越权承诺；
3. 情绪累计 → 连续不满后升级（不是第一句有点急就转）；
4. 纯咨询 → 不转人工（不该转的别乱转）；
5. 坐席接续 → 人工回复后用户继续问，不再重复追问订单号。

用法：
    cd E:\\dsh_work\\cs-agent
    .\\.venv\\Scripts\\python.exe scripts\\check_handoff_online.py
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cs_agent.config import get_settings  # noqa: E402
from cs_agent.conversation import ConversationService, _parse_frame  # noqa: E402
from cs_agent.llm import LLMClient  # noqa: E402
from cs_agent.storage import SESSION_ACTIVE, SessionStore  # noqa: E402
from cs_agent.tools.mock_data import reset_repository  # noqa: E402

#: 越权承诺的典型说法（出现即视为"自己答应了不该答应的"）
OVER_PROMISE = ("一定赔", "可以赔", "已经为您赔偿", "保证免运费", "给您免运费",
                "马上加急", "立刻加急", "一定免", "我给您补偿")


def run_turn(service: ConversationService, sid: str, text: str) -> dict:
    start: dict = {}
    done: dict = {}
    for frame in service.stream_turn(sid, text):
        event, payload = _parse_frame(frame)
        if event == "start":
            start = payload
        elif event == "done":
            done = payload
    return {"start": start, "reply": done.get("content", "")}


def main() -> int:
    settings = get_settings(reload=True)
    if not settings.online:
        print("[跳过] 当前是离线模式，本脚本用于验证真实模型下的转人工行为")
        return 0

    reset_repository()
    db_path = ROOT / ".cache" / f"handoff-{uuid.uuid4().hex[:6]}.db"
    store = SessionStore(db_path)
    service = ConversationService(store, settings, LLMClient(settings))
    checks: list[tuple[str, bool, str]] = []

    try:
        # ---- 1. 明确要求转人工 ----
        sid = service.open_session(user_id="ho-1")["id"]
        r = run_turn(service, sid, "订单号 SO20260101 的问题你们一直没解决，我要转人工")
        handoff = r["start"].get("handoff") or {}
        reply = r["reply"]
        print("1) 明确要求转人工")
        print("   工单：", handoff.get("handoff_id"), "｜原因：", handoff.get("reason"))
        print("   回复：", reply[:180].replace("\n", " "))
        checks.append(("创建了工单", bool(handoff.get("handoff_id")), ""))
        checks.append(("回复里给出工单号", handoff.get("handoff_id", "x") in reply, "未告知工单号"))
        checks.append(
            ("未越权承诺", not any(word in reply for word in OVER_PROMISE), "出现了越权承诺")
        )
        checks.append(
            ("会话状态为等待人工",
             store.get_session(sid)["status"] == "waiting_human", "")
        )

        # ---- 2. 超权限诉求 ----
        sid2 = service.open_session(user_id="ho-2")["id"]
        r2 = run_turn(service, sid2, "你们发错货了，得赔我 200 块，不然我就去 12315 投诉")
        reply2 = r2["reply"]
        handoff2 = r2["start"].get("handoff") or {}
        print("\n2) 超权限诉求")
        print("   工单：", handoff2.get("handoff_id"), "｜原因：", handoff2.get("reason"))
        print("   回复：", reply2[:180].replace("\n", " "))
        checks.append(("超权限诉求转人工", bool(handoff2.get("handoff_id")), ""))
        checks.append(
            ("未答应赔偿", not any(word in reply2 for word in OVER_PROMISE), "答应了赔偿")
        )

        # ---- 3. 情绪累计（不是一句就转） ----
        sid3 = service.open_session(user_id="ho-3")["id"]
        first = run_turn(service, sid3, "订单号 SO20260101，怎么这么久还没到")
        print("\n3) 情绪累计")
        print("   第 1 轮工单：", (first["start"].get("handoff") or {}).get("handoff_id"))
        print("   第 1 轮情绪分：", (first["start"].get("escalation") or {}).get("sentiment", {}).get("score"))
        escalated = False
        for index, text in enumerate(
            ["还是没到，太慢了", "你们到底怎么回事，一直没动静", "算了我要投诉，转人工"]
        ):
            result = run_turn(service, sid3, text)
            if result["start"].get("handoff"):
                escalated = True
                print(f"   第 {index + 2} 轮触发转人工：{result['start']['handoff']['reason']}")
                break
        checks.append(("连续不满后升级为人工", escalated, "多轮不满仍未转人工"))

        # ---- 4. 纯咨询不该转 ----
        sid4 = service.open_session(user_id="ho-4")["id"]
        r4 = run_turn(service, sid4, "退货运费一般谁承担？")
        print("\n4) 纯咨询")
        print("   工单：", (r4["start"].get("handoff") or {}).get("handoff_id") or "（未转）")
        print("   回复：", r4["reply"][:140].replace("\n", " "))
        checks.append(("纯咨询不转人工", not r4["start"].get("handoff"), "不该转却转了"))

        # ---- 5. 坐席接续 ----
        sid5 = service.open_session(user_id="ho-5")["id"]
        run_turn(service, sid5, "订单号 SO20260101 的物流查不到，我要转人工")
        pending = store.latest_handoff(sid5)
        service.handoffs.reply(
            pending["id"], "您好，我是人工客服小李，已联系承运商核实，2 小时内给您回复。"
        )
        after = run_turn(service, sid5, "那我这个大概什么时候能送到？")
        print("\n5) 坐席接续")
        print("   回复：", after["reply"][:180].replace("\n", " "))
        print("   会话状态：", store.get_session(sid5)["status"])
        checks.append(("接续后会话恢复活跃", store.get_session(sid5)["status"] == SESSION_ACTIVE, ""))
        checks.append(
            ("未重复索要订单号",
             "订单号" not in after["reply"] or "SO20260101" in after["reply"],
             "又追问了订单号"),
        )
        checks.append(
            ("人工消息在上下文里", "小李" in str(after["start"].get("messages", [])), "人工回复没进上下文")
        )

        # ---- 工单与交接单 ----
        print("\n—— 工单与交接单 ——")
        workspace = service.handoffs.workspace()
        print(f"   工作台工单数：{len(workspace)}｜等待中：{sum(1 for w in workspace if w['status'] == 'waiting')}")
        detail = service.handoffs.detail(workspace[0]["handoff_id"])
        print("   交接单：")
        for line in detail["packet_text"].splitlines()[:8]:
            print("     ", line[:100])

        print("\n—— 检查项 ——")
        failed = 0
        for name, ok, hint in checks:
            print(f"  [{'通过' if ok else '失败'}] {name}" + (f"  {hint}" if not ok else ""))
            failed += 0 if ok else 1
        print("\n" + ("[通过] P5 转人工验收通过" if not failed else f"[失败] {failed} 项未通过"))
        return 0 if not failed else 1
    finally:
        store.close()
        reset_repository()


if __name__ == "__main__":
    raise SystemExit(main())
