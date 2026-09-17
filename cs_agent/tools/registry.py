"""工具注册表与执行编排。

编排层要回答三个问题，这个模块就是答案：

1. **有哪些工具**：``specs()`` 输出带治理信息的清单（给模型或前端）；
2. **该不该直接执行**：标了 ``requires_confirmation`` 的写操作不直接跑，
   而是返回"需要确认"，由对话层挂起等用户点头（P5 会把挂起做到持久化）；
3. **失败怎么办**：只对瞬时故障重试（``ToolError.retryable``），
   业务性失败（订单不存在、状态不允许）不重试——重试一万次也还是不存在。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import BaseTool, ToolResult
from .cs_tools import build_tools

logger = logging.getLogger(__name__)


@dataclass
class ExecutionOutcome:
    """一次工具执行的结果 + 编排层补充的信息。"""

    result: ToolResult
    #: 是否需要用户先确认（写操作）
    needs_confirmation: bool = False
    #: 需要确认时给用户看的说明
    confirmation_prompt: str = ""
    #: 被跳过的原因（未执行时）
    skipped_reason: str = ""

    @property
    def ok(self) -> bool:
        return self.result.ok

    def as_dict(self) -> Dict[str, Any]:
        payload = self.result.as_dict()
        if self.needs_confirmation:
            payload["needs_confirmation"] = True
            payload["confirmation_prompt"] = self.confirmation_prompt
        if self.skipped_reason:
            payload["skipped_reason"] = self.skipped_reason
        return payload


class ToolRegistry:
    """工具注册与执行。"""

    def __init__(self, tools: Optional[List[BaseTool]] = None) -> None:
        self._tools: Dict[str, BaseTool] = {}
        for tool in tools or build_tools():
            self.register(tool)

    # ------------------------------------------------------------------
    def register(self, tool: BaseTool) -> None:
        if not tool.name:
            raise ValueError("工具必须有 name")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools)

    def all(self) -> List[BaseTool]:
        return list(self._tools.values())

    def specs(self) -> List[Dict[str, Any]]:
        """工具清单（含成本/延迟/是否需确认），按成本与延迟排序。"""
        return [
            tool.spec()
            for tool in sorted(
                self._tools.values(), key=lambda t: (t.est_cost, t.est_latency_ms)
            )
        ]

    # ------------------------------------------------------------------
    def call(self, name: str, params: Optional[Dict[str, Any]] = None) -> ToolResult:
        """直接执行（不做确认检查）。内部与测试用。"""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.failure(name, f"未注册的工具：{name}", code="unknown_tool")
        return tool.call(params or {})

    def execute(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        confirmed: bool = False,
    ) -> ExecutionOutcome:
        """受治理的执行入口。

        Args:
            confirmed: 用户是否已经确认过这次写操作。
        """
        tool = self._tools.get(name)
        if tool is None:
            return ExecutionOutcome(
                result=ToolResult.failure(name, f"未注册的工具：{name}", code="unknown_tool")
            )

        if tool.requires_confirmation and not confirmed:
            prompt = self._confirmation_prompt(tool, params or {})
            return ExecutionOutcome(
                result=ToolResult(
                    ok=False,
                    tool=name,
                    error="该操作会改变订单状态，需要用户确认后执行",
                    error_code="needs_confirmation",
                    meta={"retryable": False},
                ),
                needs_confirmation=True,
                confirmation_prompt=prompt,
                skipped_reason="等待用户确认",
            )

        result = tool.call(params or {})
        logger.info(
            "工具 %s 执行%s：ok=%s latency=%sms",
            name,
            "（已确认）" if tool.requires_confirmation else "",
            result.ok,
            result.latency_ms,
        )
        return ExecutionOutcome(result=result)

    # ------------------------------------------------------------------
    @staticmethod
    def _confirmation_prompt(tool: BaseTool, params: Dict[str, Any]) -> str:
        """把写操作的参数讲清楚再问——确认必须让用户知道"到底要做什么"。"""
        if tool.name == "submit_return_request":
            order_id = params.get("order_id", "该订单")
            reason = params.get("reason", "未填写原因")
            return (
                f"即将为订单 {order_id} 提交退货申请（原因：{reason}）。"
                "提交后订单会进入退货处理中，且不可撤销。确认提交吗？"
            )
        ordered = "、".join(f"{k}={v}" for k, v in params.items())
        return f"即将执行「{tool.description}」（{ordered}），确认继续吗？"
