"""P7：前端挂载与 SPA 回退测试。

界面已构建时应该能正常访问；未构建时后端也必须能起（只跑接口）——
"缺 dist 就启动失败"是很常见的部署事故，这里两条都测。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cs_agent.app import FRONTEND_DIST, UI_MOUNT_PATH, create_app


def test_frontend_mounted_when_built(client: TestClient) -> None:
    """构建产物存在时，/ui 应返回 index.html。"""
    if not (FRONTEND_DIST / "index.html").exists():
        pytest.skip("前端尚未构建（cd frontend && npm run build）")

    response = client.get(f"{UI_MOUNT_PATH}/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "root" in response.text  # index.html 里的挂载点


def test_spa_fallback_serves_index_for_client_routes(client: TestClient) -> None:
    """SPA 深链接必须回退到 index.html，否则刷新页面 404。"""
    if not (FRONTEND_DIST / "index.html").exists():
        pytest.skip("前端尚未构建")

    for path in (f"{UI_MOUNT_PATH}/workspace", f"{UI_MOUNT_PATH}/eval", f"{UI_MOUNT_PATH}/whatever"):
        response = client.get(path)
        assert response.status_code == 200, f"{path} 未回退到 index.html"
        assert "text/html" in response.headers["content-type"]


def test_static_asset_is_served(client: TestClient) -> None:
    if not (FRONTEND_DIST / "index.html").exists():
        pytest.skip("前端尚未构建")

    assets = list((FRONTEND_DIST / "assets").glob("*.js"))
    assert assets, "构建产物里应当有 js 资源"
    response = client.get(f"{UI_MOUNT_PATH}/assets/{assets[0].name}")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]


def test_api_still_works_alongside_ui(client: TestClient) -> None:
    """挂载静态资源不能影响接口路由。"""
    assert client.get("/api/health").status_code == 200
    assert client.get("/").json()["service"] == "cs-agent"
    assert client.get("/manifest.json").json()["start_url"] == f"{UI_MOUNT_PATH}/"


def test_app_starts_without_frontend_build(monkeypatch, tmp_path) -> None:
    """没有构建产物时后端仍要能启动（只跑接口），不能因为缺 dist 就崩。"""
    import cs_agent.app as app_module

    missing = tmp_path / "no-such-dist"
    monkeypatch.setattr(app_module, "FRONTEND_DIST", missing)

    application = app_module.create_app()
    with TestClient(application) as client:
        assert client.get("/api/health").status_code == 200
        # 未挂载 /ui
        assert client.get(f"{UI_MOUNT_PATH}/").status_code == 404


def test_mount_returns_false_without_build(monkeypatch, tmp_path) -> None:
    import cs_agent.app as app_module

    monkeypatch.setattr(app_module, "FRONTEND_DIST", tmp_path / "missing")
    from fastapi import FastAPI

    assert app_module.mount_frontend(FastAPI(), app_module.get_settings()) is False


def test_build_output_shape() -> None:
    """构建产物结构自检：index.html + assets 目录（缺了会导致界面白屏）。"""
    if not FRONTEND_DIST.exists():
        pytest.skip("前端尚未构建")
    assert (FRONTEND_DIST / "index.html").exists()
    assert (FRONTEND_DIST / "assets").is_dir()
    assert any((FRONTEND_DIST / "assets").iterdir())
