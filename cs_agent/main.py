"""启动入口。

用法::

    python -m cs_agent.main          # 或
    .venv\\Scripts\\python.exe -m cs_agent.main

绑定的端口被占用时会给出**可操作的提示**（谁占了、怎么处理），
而不是直接抛一个 ``[Errno 10048]``。
"""

from __future__ import annotations

import logging
import sys

import uvicorn

from .app import create_app, port_owner
from .config import get_settings


def main() -> int:
    settings = get_settings()

    if port_owner(settings.port):
        print(
            f"\n[启动失败] 端口 {settings.port} 已被占用。\n"
            f"  查占用：Get-NetTCPConnection -LocalPort {settings.port} -State Listen\n"
            f"  结束它：Stop-Process -Id <OwningProcess> -Force\n"
            f"  或改端口：设置环境变量 PORT=其他端口\n",
            file=sys.stderr,
        )
        return 1

    print(
        f"\n长会话客服 Agent\n"
        f"  地址：http://{settings.host}:{settings.port}\n"
        f"  文档：http://{settings.host}:{settings.port}/docs\n"
        f"  模型：{settings.model}（{'在线' if settings.online else '离线模板模式'}）\n"
        f"  数据：{settings.db_path}\n"
    )
    logging.getLogger(__name__).info("以 %s:%s 启动", settings.host, settings.port)
    uvicorn.run(create_app(), host=settings.host, port=settings.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
