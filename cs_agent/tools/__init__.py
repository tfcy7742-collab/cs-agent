"""客服技能工具：协议、注册表、六个具体工具。

⚠️ 工具依赖的订单、物流、退款数据都是**本地模拟数据**（``mock_data.py``），
政策条文是**演示用示例文本**（``policy.py``），都不是真实系统对接。
"""

from .base import BaseTool, ToolError, ToolParamError, ToolResult
from .registry import ExecutionOutcome, ToolRegistry

__all__ = [
    "BaseTool",
    "ExecutionOutcome",
    "ToolError",
    "ToolParamError",
    "ToolRegistry",
    "ToolResult",
]
