"""P2 在线验收：用真实模型跑长会话，检验分层内存的两个承诺。

承诺 1：聊到几十轮，真正送进模型的工作记忆**不随时长线性增长**；
承诺 2：第 1 轮给出的订单号/称呼，在几十轮之后**仍然可见**（结构化事实独立于摘要）。

用法：
    cd E:\\dsh_work\\cs-agent
    .\\.venv\\Scripts\\python.exe scripts\\check_long_conversation.py [轮数]
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cs_agent.config import get_settings  # noqa: E402
from cs_agent.conversation import ConversationService  # noqa: E402
from cs_agent.llm import LLMClient  # noqa: E402
from cs_agent.storage import SessionStore  # noqa: E402

#: 真实的长会话脚本：先给关键信息，再持续闲聊，最后要求复述
OPENING = "你好，我叫张先生，我的订单号是 SO20260101，我在北京，想咨询一下退货政策。"

FOLLOW_UPS = [
    "这个订单什么时候能到？",
    "如果我要退，运费谁承担？",
    "好的，那退款多久到账？",
    "我还有一件别的商品想一起退可以吗？",
    "包装拆开了还能退吗？",
    "你们支持上门取件吗？",
    "发票怎么开？",
    "有没有优惠券可以用？",
]


def main() -> int:
    turns_wanted = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    settings = get_settings(reload=True)
    print("配置：", {k: settings.describe()[k] for k in ("llm_mode", "online", "model")})

    db_path = ROOT / ".cache" / f"long-{uuid.uuid4().hex[:6]}.db"
    store = SessionStore(db_path)
    service = ConversationService(store, settings, LLMClient(settings))

    try:
        sid = service.open_session(user_id="long-runner")["id"]
        samples = []

        print(f"\n第 1 轮（给出关键信息）：{OPENING}")
        first = service.chat_once(sid, OPENING)
        print("  回复：", (first.get("content") or "")[:80].replace("\n", " "))
        print("  抽出事实：", {k: v["value"] for k, v in store.get_facts(sid).items()})

        for index in range(1, turns_wanted):
            text = FOLLOW_UPS[index % len(FOLLOW_UPS)] if index < 40 else "请继续说明。"
            result = service.chat_once(sid, text)
            if result.get("error") is not None and "turn" not in result:
                print(f"  第 {index + 1} 轮失败：{result.get('error')}")
                break
            if (index + 1) % 5 == 0 or index + 1 == turns_wanted:
                stats = service.session_stats(sid)
                report = service.memory.memory_report(sid)
                samples.append(stats["working_chars"])
                print(
                    f"  第 {index + 1:>3} 轮：工作记忆 {stats['working_chars']:>5} 字 / "
                    f"{stats['working_messages']} 条｜累计 {report['total_chars']:>6} 字｜"
                    f"已压缩 {report['messages_compressed']:>2} 条｜摘要 {report['summary_chars']:>4} 字"
                )

        report = service.memory.memory_report(sid)
        print("\n—— 记忆报告 ——")
        print(f"  轮数：{report['turns']}｜总字符：{report['total_chars']}｜"
              f"未压缩：{report['messages_uncompressed']} 条 {report['uncompressed_chars']} 字")
        print(f"  摘要：{report['summary_chars']} 字｜压缩比：{report['compression_ratio']}")
        print(f"  事实：{report['facts']}")
        print(f"  画像：{report['profile_attributes']}")

        summary = store.latest_summary(sid)
        if summary:
            print("\n—— 摘要内容 ——")
            print("  来源：", summary["meta"]["stats"].get("source"))
            print(summary["content"][:500])

        # 承诺 2：几十轮后仍能报出关键信息
        messages = service.build_messages(sid)
        system = messages[0]["content"]
        ok_fact = "SO20260101" in system
        ok_name = "张先生" in system
        print(f"\n关键信息仍在提示中：订单号={ok_fact} 称呼={ok_name}")

        # 承诺 1：工作记忆不随时长线性增长
        # 硬口径：配置上限（working_recent_turns*2 条原文）+ 摘要预算；
        # 软口径：后期采样相对中期不应持续攀升。
        cap = settings.history_compress_chars * 0.5 + settings.working_recent_turns * 2 * 200
        bounded_by_cap = all(sample <= cap for sample in samples)
        tail_stable = len(samples) < 3 or samples[-1] <= max(samples[:-1]) * 1.25
        print(f"工作记忆采样：{samples}")
        print(f"  上限口径（≤{int(cap)} 字）：{bounded_by_cap}｜后期是否稳定：{tail_stable}")

        passed = ok_fact and ok_name and bounded_by_cap and tail_stable and report["has_summary"]
        print("\n[通过] P2 长会话验收通过" if passed else "\n[失败] 未满足预期，见上文")
        return 0 if passed else 1
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
