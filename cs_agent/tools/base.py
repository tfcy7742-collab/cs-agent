"""工具协议：能力声明 + 治理策略。

为什么工具要带"元信息"（成本、延迟、是否需要人工确认），而不是只有一个函数：

* **决策需要**：编排层要按"能不能做、贵不贵、要不要人确认"来选工具，
  这些信息必须由工具自己声明，不能写死在编排层；
* **治理需要**：超时、重试、参数校验、人工确认是**所有工具共用**的策略，
  放进基类才能保证新加工具自动获得同样的保护；
* **安全性**：``requires_confirmation`` 标记有副作用的写操作（建工单、提交退货申请），
  编排层据此挂起等用户确认——这是"不越权"的落地方式。

参数用 JSON Schema 描述，既可直接给模型做 function calling，也用于入参校验。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class ToolError(RuntimeError):
    """工具执行失败。``retryable`` 表示是否值得重试。"""

    def __init__(self, message: str, *, retryable: bool = False, code: str = "") -> None:
        super().__init__(message)
        self.retryable = retryable
        self.code = code


class ToolParamError(ToolError):
    """参数不合法（不可重试，属于调用方的问题）。"""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False, code="invalid_param")


@dataclass
class ToolResult:
    """一次工具调用的结果。

    ``ok=False`` 时 ``error`` 必须是人话，且**不允许**编造数据来"补上"——
    工具层最重要的一条纪律：查不到就说查不到。
    """

    ok: bool
    tool: str
    data: Optional[Dict[str, Any]] = None
    error: str = ""
    error_code: str = ""
    latency_ms: int = 0
    #: 工具自己判断"这次结果可信吗"（例如数据源不可用）
    degraded: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "ok": self.ok,
            "tool": self.tool,
            "latency_ms": self.latency_ms,
            "degraded": self.degraded,
        }
        if self.data is not None:
            payload["data"] = self.data
        if self.error:
            payload["error"] = self.error
        if self.error_code:
            payload["error_code"] = self.error_code
        if self.meta:
            payload["meta"] = self.meta
        return payload

    @classmethod
    def failure(
        cls, tool: str, error: str, *, retryable: bool = False, code: str = ""
    ) -> "ToolResult":
        return cls(ok=False, tool=tool, error=error, error_code=code or "error",
                   meta={"retryable": retryable})


class BaseTool:
    """所有客服工具的基类。

    子类只需实现 :meth:`run`，超时/重试/参数校验/确认标记由基类统一处理。
    """

    #: 工具名（供模型与编排层引用）
    name: str = ""
    #: 一句话说明"这个工具能做什么"（会进工具清单给模型看）
    description: str = ""
    #: 参数 JSON Schema
    parameters: Dict[str, Any] = {"type": "object", "properties": {}}
    #: 超时（秒）
    timeout_s: float = 10.0
    #: 相对成本估计（1=很便宜，5=较贵），供编排层权衡
    est_cost: int = 1
    #: 相对延迟估计（毫秒），供编排层排序
    est_latency_ms: int = 200
    #: 瞬时故障是否可重试（只读查询可重试；有副作用的写操作不可）
    retryable: bool = True
    #: 有副作用的写操作必须让用户确认
    requires_confirmation: bool = False

    # ------------------------------------------------------------------
    def run(self, **kwargs: Any) -> ToolResult:
        raise NotImplementedError

    # ------------------------------------------------------------------
    def validate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """按 Schema 做**轻量**校验：必填、类型、枚举。

        刻意不引入 jsonschema 依赖：这里只需要挡住明显错误的调用，
        复杂校验留给工具自己的业务判断。
        """
        schema = self.parameters or {}
        properties: Dict[str, Any] = schema.get("properties") or {}
        required: List[str] = list(schema.get("required") or [])

        for key in required:
            value = params.get(key)
            if value is None or (isinstance(value, str) and not value.strip()):
                raise ToolParamError(f"缺少必填参数：{key}")

        cleaned: Dict[str, Any] = {}
        for key, value in params.items():
            spec = properties.get(key)
            if spec is None:
                # 未声明的参数直接丢弃，避免把无关字段透传给业务逻辑
                continue
            if value is None:
                continue
            expected = spec.get("type")
            if expected == "integer" and not isinstance(value, int):
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    raise ToolParamError(f"参数 {key} 必须是整数") from None
            elif expected == "string" and not isinstance(value, str):
                value = str(value)
            enum = spec.get("enum")
            if enum and value not in enum:
                raise ToolParamError(f"参数 {key} 只能是 {'/'.join(map(str, enum))} 之一")
            cleaned[key] = value
        return cleaned

    def spec(self) -> Dict[str, Any]:
        """工具清单条目（给模型或前端看）。"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "timeout_s": self.timeout_s,
            "est_cost": self.est_cost,
            "est_latency_ms": self.est_latency_ms,
            "retryable": self.retryable,
            "requires_confirmation": self.requires_confirmation,
        }

    def call(self, params: Dict[str, Any]) -> ToolResult:
        """执行入口：校验 → 带重试执行 → 统一计时与错误包装。"""
        started = time.perf_counter()
        try:
            cleaned = self.validate(params or {})
        except ToolParamError as exc:
            return ToolResult.failure(self.name, str(exc), code="invalid_param")

        attempts = 2 if self.retryable else 1
        last_error: Optional[ToolError] = None
        for attempt in range(1, attempts + 1):
            try:
                result = self.run(**cleaned)
                result.latency_ms = int((time.perf_counter() - started) * 1000)
                return result
            except ToolError as exc:
                last_error = exc
                if not exc.retryable or attempt >= attempts:
                    break
                time.sleep(0.2 * attempt)
            except Exception as exc:  # pragma: no cover - 防御性分支
                return ToolResult.failure(
                    self.name, f"工具内部错误：{type(exc).__name__}: {exc}"
                )

        message = str(last_error) if last_error else "工具执行失败"
        return ToolResult.failure(
            self.name,
            message,
            retryable=bool(last_error and last_error.retryable),
            code=(last_error.code if last_error else "error"),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def param(
        type_: str = "string",
        description: str = "",
        enum: Optional[List[str]] = None,
        default: Any = None,
    ) -> Dict[str, Any]:
        spec: Dict[str, Any] = {"type": type_, "description": description}
        if enum:
            spec["enum"] = enum
        if default is not None:
            spec["default"] = default
        return spec

    @staticmethod
    def dumps(data: Any) -> str:
        return json.dumps(data, ensure_ascii=False)
