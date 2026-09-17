"""P3 在线验收：结构化会话状态的行为是否符合预期。

脚本化的多轮对话，逐步检查：
1. 只说"我要退货"（缺订单号）→ 应主动追问订单号；
2. 补上订单号后 → 不再重复要订单号；
3. 同时提两件事 → 第二件进待办，不被丢掉；
4. 纯规则咨询（"退货运费谁承担"）→ **不该**被要订单号；
5. 指代不清（"那这个能退吗"且无单号）→ 应要求澄清而不是猜。

用法：
    cd E:\\dsh_work\\cs-agent
    .\\.venv\\Scripts\\python.exe scripts\\check_session_state.py
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
from cs_agent.state import StateSnapshot  # noqa: E402
from cs_agent.storage import SessionStore  # noqa: E402


def run_turn(service: ConversationService, sid: str, text: str) -> dict:
    """跑一轮并取出 start 事件里的 plan / state，以及最终回复。"""
    plan: dict = {}
    state: dict = {}
    done: dict = {}
    for frame in service.stream_turn(sid, text):
        event, payload = _parse_frame(frame)
        if event == "start":
            plan = payload.get("plan") or {}
            state = payload.get("state") or {}
        elif event == "done":
            done = payload
    return {"plan": plan, "state": state, "reply": done.get("content", "")}


def main() -> int:
    settings = get_settings(reload=True)
    if not settings.online:
        print("[跳过] 当前是离线模式，本脚本用于验证真实模型下的行为")
        return 0

    db_path = ROOT / ".cache" / f"state-{uuid.uuid4().hex[:6]}.db"
    store = SessionStore(db_path)
    service = ConversationService(store, settings, LLMClient(settings))
    checks: list[tuple[str, bool, str]] = []

    try:
        sid = service.open_session(user_id="state-checker")["id"]

        # ---- 1. 缺订单号 → 应追问 ----
        r1 = run_turn(service, sid, "我要退货")
        checks.append(("缺订单号时主动追问", r1["plan"].get("action") == "ask", str(r1["plan"])))
        checks.append(
            ("追问目标是订单号", (r1["plan"].get("ask") or {}).get("target") == "订单号", "")
        )
        checks.append(("回复里确实在要订单号", "订单号" in r1["reply"], r1["reply"][:60]))
        print("1) 用户：我要退货")
        print("   plan:", r1["plan"])
        print("   回复：", r1["reply"][:100])

        # ---- 2. 补上订单号 → 不该再问 ----
        r2 = run_turn(service, sid, "订单号是 SO20260101")
        checks.append(("补上订单号后不再追问", r2["plan"].get("action") == "answer", str(r2["plan"])))
        print("\n2) 用户：订单号是 SO20260101")
        print("   plan:", r2["plan"])
        print("   槽位：", r2["state"].get("slots"))
        print("   回复：", r2["reply"][:100])

        # ---- 3. 多意图：第二件事进待办 ----
        r3 = run_turn(service, sid, "另外帮我查一下物流到哪了")
        open_tasks = [t["kind"] for t in r3["state"].get("tasks", [])]
        checks.append(("多意图进入待办", len(open_tasks) >= 2, str(open_tasks)))
        print("\n3) 用户：另外帮我查一下物流到哪了")
        print("   待办：", open_tasks, "｜active:", r3["state"].get("active_task"))
        print("   回复：", r3["reply"][:100])

        # ---- 4. 纯咨询不该被拦 ----
        sid2 = service.open_session(user_id="state-checker")["id"]
        r4 = run_turn(service, sid2, "退货运费一般谁承担？")
        checks.append(("纯咨询不索要订单号", r4["plan"].get("action") == "answer", str(r4["plan"])))
        checks.append(("识别为咨询类", "consulted" in r4["plan"], str(r4["plan"].get("consulted"))))
        print("\n4) 新会话用户：退货运费一般谁承担？")
        print("   plan:", r4["plan"])
        print("   回复：", r4["reply"][:120])

        # ---- 5. 指代不清 → 澄清 ----
        sid3 = service.open_session(user_id="state-checker")["id"]
        r5 = run_turn(service, sid3, "那这个能退吗？")
        checks.append(("指代不清要求澄清", bool(r5["plan"].get("clarification")), str(r5["plan"])))
        print("\n5) 新会话用户：那这个能退吗？")
        print("   plan:", r5["plan"])
        print("   回复：", r5["reply"][:120])

        # ---- 汇总 ----
        snapshot = StateSnapshot.load(store, sid)
        print("\n—— 会话最终状态 ——")
        print("  槽位：", snapshot.slots)
        print("  待办：", snapshot.summary()["tasks"])
        print("  已问过的槽位：", snapshot.asked_slots)

        print("\n—— 检查项 ——")
        failed = 0
        for name, ok, detail in checks:
            print(f"  [{'通过' if ok else '失败'}] {name}" + (f"  {detail}" if not ok else ""))
            failed += 0 if ok else 1

        print("\n" + ("[通过] P3 会话状态验收通过" if not failed else f"[失败] {failed} 项未通过"))
        return 0 if not failed else 1
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
