"""项目配置：全部来自环境变量 / .env，任何一项都有合理默认值。

设计原则与项目定位一致——**没有 Key 也能跑通全链路**：
``LLM_MODE=auto`` 时若检测不到可用 Key，会自动进入离线模式（模板应答），
接口行为、流式协议、持久化逻辑完全一致，方便测试与 CI。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

BASE_DIR = Path(__file__).resolve().parents[1]


def _load_env_file(path: Path) -> None:
    """极简 .env 解析：只支持 KEY=VALUE 与 # 注释，不覆盖已存在的环境变量。"""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file(BASE_DIR / ".env")


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: List[str]) -> List[str]:
    value = os.environ.get(name)
    if not value:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class Settings:
    """运行期配置。"""

    # ---- LLM ----
    llm_mode: str = field(default_factory=lambda: _env_str("LLM_MODE", "auto"))
    api_key: str = field(default_factory=lambda: _env_str("DEEPSEEK_API_KEY", ""))
    base_url: str = field(
        default_factory=lambda: _env_str("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    model: str = field(default_factory=lambda: _env_str("DEEPSEEK_MODEL", "deepseek-chat"))
    llm_timeout: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT", 60.0))
    llm_temperature: float = field(
        default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.3)
    )
    llm_max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 2048))

    # ---- 会话 ----
    #: 工作时记忆里保留的最近原文轮数（一轮 = 一问一答）
    working_recent_turns: int = field(
        default_factory=lambda: _env_int("WORKING_RECENT_TURNS", 8)
    )
    #: 会话历史总字符数超过该值时触发摘要压缩（P2 使用）
    history_compress_chars: int = field(
        default_factory=lambda: _env_int("HISTORY_COMPRESS_CHARS", 6000)
    )
    #: 单条用户消息最大长度（防滥用）
    max_message_chars: int = field(
        default_factory=lambda: _env_int("MAX_MESSAGE_CHARS", 4000)
    )

    # ---- 服务 ----
    host: str = field(default_factory=lambda: _env_str("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8001))
    cors_origins: List[str] = field(
        default_factory=lambda: _env_list(
            "CORS_ORIGINS", ["http://localhost:5173", "http://127.0.0.1:5173"]
        )
    )
    log_level: str = field(default_factory=lambda: _env_str("LOG_LEVEL", "INFO"))

    # ---- 存储 ----
    db_path: Path = field(
        default_factory=lambda: BASE_DIR / _env_str("DB_PATH", "./data/cs_agent.db")
        if not Path(_env_str("DB_PATH", "./data/cs_agent.db")).is_absolute()
        else Path(_env_str("DB_PATH", "./data/cs_agent.db"))
    )

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key.strip())

    @property
    def online(self) -> bool:
        """是否真的会去调用大模型。

        - ``offline``：强制离线（CI 用）；
        - ``online``：强制联网（没 Key 时会在调用处报错，便于暴露配置问题）；
        - ``auto``：有 Key 就联网。
        """
        mode = self.llm_mode.strip().lower()
        if mode == "offline":
            return False
        if mode == "online":
            return True
        return self.has_api_key

    @property
    def masked_key(self) -> str:
        """脱敏后的 Key，用于健康检查展示。"""
        key = self.api_key.strip()
        if len(key) <= 8:
            return "***" if key else ""
        return f"{key[:6]}...{key[-4:]}"

    def describe(self) -> dict:
        return {
            "llm_mode": self.llm_mode,
            "online": self.online,
            "model": self.model,
            "base_url": self.base_url,
            "api_key": self.masked_key,
            "port": self.port,
            "db_path": str(self.db_path),
            "working_recent_turns": self.working_recent_turns,
            "history_compress_chars": self.history_compress_chars,
        }


_settings: Optional[Settings] = None


def get_settings(reload: bool = False) -> Settings:
    """获取全局配置（默认缓存，测试可用 ``reload=True`` 重新读取）。"""
    global _settings
    if _settings is None or reload:
        _settings = Settings()
    return _settings
