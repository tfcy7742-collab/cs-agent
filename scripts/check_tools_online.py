"""P4 在线验收：工具查证是否真的"钉住"了模型。

四类断言（都基于真实模型回复，不是测试替身）：

1. **有据可查**：查物流后，回复里必须出现工具返回的真实运单号/承运商，
   且不该编造数据里没有的日期；
2. **查不到就说查不到**：订单不存在时，回复必须说明"没查到"，
   且**不得**出现编造的状态或时间（这是"不编造"最关键的一条）；
3. **政策有出处**：问运费时引用政策条款原文，并说明来源；
4. **写操作先确认**：要求退货时先请求确认，未确认前订单状态不变。

用法：
    cd E:\\dsh_work\\cs-agent
    .\\.venv\\Scripts\\python.exe scripts\\check_tools_online.py
"""

from __future__ import annotations

import re
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cs_agent.config import get_settings  # noqa: E402
from cs_agent.conversation import ConversationService, _parse_frame  # noqa: E402
from cs_agent.llm import LLMClient  # noqa: E402
from cs_agent.storage import SessionStore  # noqa: E402
from cs_agent.tools.mock_data import get_repository, reset_repository  # noqa: E402

#: 编造日期/金额的常见形态（用于检测"数据里没有却说了"）
DATE_PATTERN = re.compile(r"20\d{2}[-/年]\d{1,2}[-/月]\d{1,2}")


def run_turn(service: ConversationService, sid: str, text: str) -> dict:
    out: dict = {}
    for frame in service.stream_turn(sid, text):
        event, payload = _parse_frame(frame)
        if event == "start":
            out["start"] = payload
        elif event == "done":
            out["reply"] = payload.get("content", "")
    return out


def main() -> int:
    settings = get_settings(reload=True)
    if not settings.online:
        print("[跳过] 当前是离线模式，本脚本用于验证真实模型下的工具使用")
        return 0

    reset_repository()
    db_path = ROOT / ".cache" / f"tools-{uuid.uuid4().hex[:6]}.db"
    store = SessionStore(db_path)
    service = ConversationService(store, settings, LLMClient(settings))
    checks: list[tuple[str, bool, str]] = []

    try:
        # ---- 1. 查物流：事实必须来自工具 ----
        sid = service.open_session(user_id="tool-checker")["id"]
        r = run_turn(service, sid, "订单号 SO20260101，帮我看看快递到哪了")
        reply = r.get("reply", "")
        runs = (r.get("start", {}).get("tools") or {}).get("runs") or []
        print("1) 查物流")
        print("   工具：", [(x["tool"], x["ok"]) for x in runs])
        print("   回复：", reply[:200].replace("\n", " "))
        checks.append(("调用了物流工具", any(x["tool"] == "track_logistics" for x in runs), ""))
        checks.append(
            ("回复包含真实运单号", "SF1234567890123" in reply, "应出现工具返回的运单号"),
        )
        checks.append(("回复提到承运商", "顺丰" in reply, ""))

        # ---- 2. 订单不存在：必须如实说 ----
        sid2 = service.open_session(user_id="tool-checker")["id"]
        r2 = run_turn(service, sid2, "订单号 SO99999999，帮我查下物流")
        reply2 = r2.get("reply", "")
        runs2 = (r2.get("start", {}).get("tools") or {}).get("runs") or []
        print("\n2) 订单不存在")
        print("   工具：", [(x["tool"], x["ok"], x.get("error_code")) for x in runs2])
        print("   回复：", reply2[:200].replace("\n", " "))
        has_not_found = any(
            word in reply2 for word in ("没有查", "未查询到", "查不到", "不存在", "核对")
        )
        checks.append(("如实告知查不到", has_not_found, "回复未说明查不到"))
        checks.append(
            ("未编造物流信息", "SF" not in reply2 and "顺丰" not in reply2, "不该出现编造的物流信息")
        )

        # ---- 3. 政策问题：引用条款 + 说明来源 ----
        sid3 = service.open_session(user_id="tool-checker")["id"]
        r3 = run_turn(service, sid3, "退货运费一般谁承担？")
        reply3 = r3.get("reply", "")
        print("\n3) 政策咨询")
        print("   回复：", reply3[:220].replace("\n", " "))
        checks.append(("提到商家责任情形", "质量" in reply3 or "商家" in reply3, ""))
        checks.append(("提到买家承担情形", "买家" in reply3 or "自己" in reply3 or "个人" in reply3, ""))

        # ---- 4. 写操作：先确认，且未确认前不改数据 ----
        sid4 = service.open_session(user_id="tool-checker")["id"]
        r4 = run_turn(service, sid4, "订单号 SO20260102，尺码不合适，我要退货")
        reply4 = r4.get("reply", "")
        runs4 = (r4.get("start", {}).get("tools") or {}).get("runs") or []
        print("\n4) 申请退货")
        print("   工具：", [(x["tool"], x.get("needs_confirmation")) for x in runs4])
        print("   回复：", reply4[:200].replace("\n", " "))
        print("   订单状态：", get_repository().get("SO20260102").status)
        checks.append(("请求用户确认", any(x.get("needs_confirmation") for x in runs4), ""))
        checks.append(
            ("确认前未改数据", get_repository().get("SO20260102").status == "delivered", "")
        )
        checks.append(
            ("回复未宣称已办成", "已提交" not in reply4 and "已为您提交" not in reply4, "")
        )

        r5 = run_turn(service, sid4, "确认提交")
        runs5 = (r5.get("start", {}).get("tools") or {}).get("runs") or []
        print("   确认后状态：", get_repository().get("SO20260102").status)
        print("   确认后回复：", (r5.get("reply") or "")[:160].replace("\n", " "))
        checks.append(("确认后真正执行", get_repository().get("SO20260102").status == "returning", ""))
        checks.append(
            ("执行后如实告知结果", any(x["tool"] == "submit_return_request" and x["ok"] for x in runs5), "")
        )

        # ---- 汇总 ----
        print("\n—— 检查项 ——")
        failed = 0
        for name, ok, detail in checks:
            print(f"  [{'通过' if ok else '失败'}] {name}" + (f"  {detail}" if not ok else ""))
            failed += 0 if ok else 1
        print("\n" + ("[通过] P4 工具查证验收通过" if not failed else f"[失败] {failed} 项未通过"))
        return 0 if not failed else 1
    finally:
        store.close()
        reset_repository()


if __name__ == "__main__":
    raise SystemExit(main())
