"""评测命令行：跑评测集、打印指标、导出报告。

用法::

    # 离线（默认）：只验工具路由/状态/规则，不评回答质量，CI 可用
    .venv\\Scripts\\python.exe scripts\\run_eval.py

    # 在线：真实模型，评回答里是否引用工具事实、是否编造、是否重复追问
    .venv\\Scripts\\python.exe scripts\\run_eval.py --online

    # 常用参数
    --rounds 30        长会话场景的轮数（默认 24）
    --only 安全,对抗    只跑带这些标签的场景
    --out report.json  导出完整 JSON 报告
    --verbose          列出每个场景的结果
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cs_agent.config import get_settings  # noqa: E402
from cs_agent.conversation import ConversationService  # noqa: E402
from cs_agent.eval_scenarios import all_scenarios  # noqa: E402
from cs_agent.evaluation import Evaluator  # noqa: E402
from cs_agent.llm import LLMClient  # noqa: E402
from cs_agent.memory import MemoryManager  # noqa: E402
from cs_agent.storage import SessionStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="长会话客服 Agent 评测")
    parser.add_argument("--online", action="store_true", help="使用真实模型评测")
    parser.add_argument("--rounds", type=int, default=24, help="长会话轮数")
    parser.add_argument("--only", default="", help="只跑带这些标签的场景（逗号分隔）")
    parser.add_argument("--out", default="", help="导出 JSON 报告路径")
    parser.add_argument("--verbose", action="store_true", help="列出每个场景")
    args = parser.parse_args()

    settings = get_settings(reload=True)
    if args.online and not settings.online:
        print("[失败] --online 需要可用的 API Key（当前为离线模式）")
        return 2

    scenarios = all_scenarios(rounds=args.rounds)
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        scenarios = [item for item in scenarios if wanted & set(item.tags)]
        if not scenarios:
            print(f"[失败] 没有匹配标签 {wanted} 的场景")
            return 2

    db_path = ROOT / ".cache" / "eval.db"
    store = SessionStore(db_path)
    llm = LLMClient(settings)
    service = ConversationService(
        store, settings, llm, MemoryManager(store, settings, llm)
    )
    evaluator = Evaluator(service, store, online=args.online)

    print(f"共 {len(scenarios)} 个场景，模式：{'在线（真实模型）' if args.online else '离线'}")
    report = evaluator.run(
        scenarios,
        progress=lambda result: print(
            f"  [{'通过' if result.ok else '失败'}] {result.scenario.key} "
            f"（{result.scenario.name}）"
            + ("" if result.ok else f"  失败 {len(result.failures)} 项")
        ),
    )
    print()
    print(Evaluator.format_report(report, verbose=args.verbose))

    if args.out:
        evaluator.dump_report(report, args.out)
        print(f"\n报告已写入：{args.out}")

    store.close()
    return 0 if report["metrics"]["passed"] == report["metrics"]["scenarios"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
