"""pytest 公共夹具。

原则：**测试永不联网**。默认强制 ``LLM_MODE=offline``，
需要验证真实模型行为的场景请另建脚本（``scripts/check_llm.py``），
不要在单元测试里依赖网络。
"""

from __future__ import annotations

import os
import shutil
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 必须在导入 cs_agent.* 之前设好，config 在导入期就会读环境变量
os.environ["LLM_MODE"] = "offline"
os.environ["LOG_LEVEL"] = "WARNING"


# ---------------------------------------------------------------------------
# 自己实现 tmp_path
# ---------------------------------------------------------------------------
# 为什么不用 pytest 内置的：它默认把目录建在系统临时目录，本项目运行的
# 受限环境里既写不进、也没权限在收尾时扫描（PermissionError: WinError 5）。
# 自建一份放在仓库内的 .cache/tmp，并在每个测试结束后清理。
_TEMP_ROOT = ROOT / ".cache" / "tmp"


@pytest.fixture(scope="session")
def workspace_tmp_root() -> Path:
    _TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    return _TEMP_ROOT


@pytest.fixture()
def tmp_path(workspace_tmp_root: Path):
    """替代 pytest 内置实现：返回本测试专属目录，测试后删除。"""
    path = workspace_tmp_root / f"t-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)



@pytest.fixture()
def settings(tmp_path, monkeypatch):
    """每个测试一份独立配置：临时库 + 强制离线。"""
    monkeypatch.setenv("LLM_MODE", "offline")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("WORKING_RECENT_TURNS", "4")

    from cs_agent.config import get_settings

    return get_settings(reload=True)


@pytest.fixture()
def store(settings):
    from cs_agent.storage import SessionStore

    instance = SessionStore(settings.db_path)
    yield instance
    instance.close()


@pytest.fixture()
def service(store, settings):
    from cs_agent.conversation import ConversationService
    from cs_agent.llm import LLMClient

    return ConversationService(store, settings, LLMClient(settings))


@pytest.fixture()
def client(settings, tmp_path, monkeypatch):
    """FastAPI 测试客户端（同一测试内共享一个 store）。"""
    from fastapi.testclient import TestClient

    from cs_agent import app as app_module
    from cs_agent.api import chat as chat_module

    app_module.reset_store()
    chat_module.reset_service()

    application = app_module.create_app()
    with TestClient(application) as test_client:
        yield test_client

    app_module.reset_store()
    chat_module.reset_service()
