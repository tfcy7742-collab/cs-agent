"""端到端检查：直接打运行中的服务，确认中文与持久化都正确。

用法（服务需已启动）：
    .\\.venv\\Scripts\\python.exe scripts\\check_server.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8001"


def call(path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload else None
    request = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    health = call("/api/health")
    print(f"健康检查：status={health['status']} online={health['llm']['online']} "
          f"model={health['llm']['model']}")

    first = call("/api/chat", {"message": "你好，我想查一下我的订单", "stream": False})
    print(f"\n第 {first['turn']} 轮：offline={first['offline']} "
          f"latency={first['latency_ms']}ms tokens={first['usage'].get('total_tokens')}")
    print("回复：", first["content"][:150])
    assert not first["offline"], "应为在线模式"
    assert first["content"].strip(), "回复不能为空"

    sid = first["session_id"]
    second = call("/api/chat", {"message": "我要退货，怎么操作？", "session_id": sid, "stream": False})
    print(f"\n第 {second['turn']} 轮回复：", second["content"][:150])

    detail = call(f"/api/sessions/{sid}")
    stats = detail["stats"]
    print(f"\n会话统计：messages={stats['messages']} turns={stats['turns']} "
          f"tokens={stats['token_estimate']} chars={stats['total_chars']}")
    assert stats["turns"] == 2, "轮次应为 2"
    assert stats["messages"] == 4, "消息应为 4 条"

    print("\n最近一轮轨迹：", json.dumps(stats["latest_trace"], ensure_ascii=False))
    print("\n[通过] 服务端到端正常：中文、多轮、持久化均正确")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
