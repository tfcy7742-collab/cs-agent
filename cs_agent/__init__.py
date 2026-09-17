"""长会话客服 Agent（电商售后场景）。

分层：

* ``config``       运行期配置（环境变量 / .env）
* ``storage``      SQLite 持久化（会话 / 消息 / 状态 / 画像）
* ``llm``          OpenAI 兼容的大模型客户端（含离线模板模式）
* ``prompt``       分层记忆 → 工作记忆的组装
* ``conversation`` 一轮对话的完整流程（含 SSE 事件流）
* ``api``          HTTP 接口
* ``app``          FastAPI 装配
"""

__version__ = "0.1.0"
