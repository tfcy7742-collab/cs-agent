"""在线模式冒烟测试：确认真的能调到 DeepSeek，而不是走了离线模板。

用法：
    cd E:\\dsh_work\\cs-agent
    .\\.venv\\Scripts\\python.exe scripts\\check_llm.py

它会：
1. 打印当前配置（模式 / 模型 / Key 是否配置）；
2. 发一轮真实对话，检查回复里**不包含**"离线模式"字样；
3. 验证多轮上下文是否被带上（第二轮问"我刚才说了什么"）。
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if os.environ.get("LLM_MODE", "").lower() == "offline":
    print("[跳过] LLM_MODE=offline，本脚本用于验证在线调用")
    raise SystemExit(0)

from cs_agent.config import get_settings  # noqa: E402
from cs_agent.conversation import ConversationService  # noqa: E402
from cs_agent.llm import LLMClient, LLMError  # noqa: E402
from cs_agent.storage import SessionStore  # noqa: E402


def main() -> int:
    settings = get_settings(reload=True)
    print("配置：", settings.describe())

    if not settings.online:
        print("\n[失败] 当前不是在线模式，无法验证真实模型调用")
        return 1

    db_path = ROOT / ".cache" / f"llm-smoke-{uuid.uuid4().hex[:6]}.db"
    store = SessionStore(db_path)
    service = ConversationService(store, settings, LLMClient(settings))

    try:
        # ---- 单轮 ----
        sid = service.open_session(user_id="smoke")["id"]
        try:
            first = service.chat_once(sid, "你好，我想问一下退货运费谁承担？")
        except LLMError as exc:
            print(f"\n[失败] 模型调用出错：{exc}")
            return 1

        print("\n第一轮回复：", first.get("content", "")[:200])
        usage = first.get("usage") or {}
        print("用量：", usage, "| 耗时：", first.get("latency_ms"), "ms")
        print("首字延迟：", (first.get("trace") or {}).get("first_token_ms"), "ms")

        if first.get("offline"):
            print("\n[失败] 回复被标记为 offline，说明并未真正调用模型")
            return 1
        if "离线模式" in (first.get("content") or ""):
            print("\n[失败] 回复里出现离线模板字样，说明并未真正调用模型")
            return 1

        # ---- 多轮：验证上下文确实带上了 ----
        second = service.chat_once(sid, "我刚才问的是什么问题？请原样复述。")
        content = second.get("content") or ""
        print("\n第二轮回复：", content[:200])
        if "退货" in content and "运费" in content:
            print("\n[通过] 模型正确复述了上一轮的问题，多轮上下文生效")
        else:
            print("\n[警告] 模型未能复述上一轮问题，请检查上下文组装")

        print("\n[通过] 在线模型调用正常")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
