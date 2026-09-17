"""FastAPI 应用装配。

P1 只挂一个 chat 路由，但应用骨架（健康检查、CORS、异常兜底、
端口占用预检）一次做好，后面几个阶段只往上加路由。
"""

from __future__ import annotations

import json
import logging
import socket
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import get_settings
from .storage import SessionStore

logger = logging.getLogger(__name__)

#: 全局单例（SQLite 用的是线程本地连接，可以跨线程共享）
_store: SessionStore | None = None


def get_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore(get_settings().db_path)
    return _store


def reset_store() -> None:
    """测试用：丢弃全局 store，让下一次调用重新建。"""
    global _store
    if _store is not None:
        _store.close()
    _store = None


def port_owner(port: int) -> bool:
    """预检端口是否已被占用（用于给出友好提示，而不是抛 10048）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return True
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    store = get_store()
    logger.info(
        "cs-agent 启动：port=%s online=%s model=%s db=%s",
        settings.port,
        settings.online,
        settings.model,
        store.db_path,
    )
    yield
    store.close()


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    app = FastAPI(
        title="长会话客服 Agent",
        description=(
            "电商售后场景的客服 Agent：长会话记忆、结构化会话状态、工具查证、"
            "人工兜底。订单与物流为本地模拟数据。"
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("未处理异常：%s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": f"服务内部错误：{type(exc).__name__}"},
        )

    @app.get("/", include_in_schema=False)
    async def root() -> Dict[str, Any]:
        return {
            "service": "cs-agent",
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/api/health",
            "chat": "POST /api/chat",
        }

    @app.get("/manifest.json", include_in_schema=False)
    async def manifest() -> Dict[str, Any]:
        """浏览器会自动请求它，返回一个最小清单避免 404 噪声。"""
        return {"name": "长会话客服 Agent", "short_name": "cs-agent", "start_url": "/ui/"}

    mount_frontend(app, settings)

    @app.get("/api/health")
    async def health() -> Dict[str, Any]:
        store = get_store()
        return {
            "status": "ok",
            "version": "0.1.0",
            "llm": {
                "mode": settings.llm_mode,
                "online": settings.online,
                "model": settings.model,
                "base_url": settings.base_url,
                "api_key": settings.masked_key,
            },
            "storage": {
                "db_path": str(store.db_path),
                "sessions": len(store.list_sessions(limit=1000)),
            },
            "conversation": {
                "working_recent_turns": settings.working_recent_turns,
                "history_compress_chars": settings.history_compress_chars,
            },
        }

    from .api.chat import router as chat_router

    app.include_router(chat_router)

    return app


#: 前端构建产物目录（`frontend/dist`）
FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"

UI_MOUNT_PATH = "/ui"


class SPAStaticFiles(StaticFiles):
    """静态文件 + SPA 回退。

    前端是单页应用：`/ui/workspace` 这类路径磁盘上并不存在对应文件，
    必须回退到 ``index.html`` 交给前端路由，否则刷新页面就 404。
    """

    async def get_response(self, path: str, scope):  # type: ignore[override]
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404:
                return await super().get_response("index.html", scope)
            raise


def mount_frontend(app: FastAPI, settings: Any) -> bool:
    """把前端构建产物挂到 ``/ui``。

    没构建过就**不挂载**，只在启动日志里提示一句——
    保证"只跑后端"也能正常用（接口与 /docs 都在），不会因为缺 dist 起不来。
    """
    if not (FRONTEND_DIST / "index.html").exists():
        logger.warning(
            "未找到前端构建产物（%s）。接口可用，界面请先构建："
            "cd frontend && npm install && npm run build",
            FRONTEND_DIST,
        )
        return False

    app.mount(
        UI_MOUNT_PATH,
        SPAStaticFiles(directory=str(FRONTEND_DIST), html=True),
        name="ui",
    )
    logger.info("前端界面已挂载：http://%s:%s%s", settings.host, settings.port, UI_MOUNT_PATH)
    return True
