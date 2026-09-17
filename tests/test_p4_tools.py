"""P4：六个客服工具与执行编排测试。

重点：
1. **不编造**：订单不存在、无物流、无退款记录都要如实返回；
2. **治理生效**：参数校验、只读工具重试、写操作必须确认；
3. **业务判断正确**：退货结论要结合窗口期与商品类型，并给出依据条款。
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from cs_agent.tools import ToolRegistry
from cs_agent.tools.base import BaseTool, ToolError, ToolResult
from cs_agent.tools.mock_data import get_repository, reset_repository


@pytest.fixture()
def registry() -> ToolRegistry:
    reset_repository()  # 每个测试都从干净的模拟数据开始（写操作会改数据）
    return ToolRegistry()


# ---------------------------------------------------------------------------
# 工具清单与治理元信息
# ---------------------------------------------------------------------------
def test_all_six_tools_registered(registry: ToolRegistry) -> None:
    assert set(registry.names()) == {
        "query_order",
        "track_logistics",
        "query_refund",
        "check_returnable",
        "search_policy",
        "submit_return_request",
    }


def test_specs_expose_governance_metadata(registry: ToolRegistry) -> None:
    """清单必须带成本/延迟/是否需确认——编排层要靠它做决策。"""
    specs = {spec["name"]: spec for spec in registry.specs()}
    for name, spec in specs.items():
        assert spec["description"], f"{name} 缺少说明"
        assert spec["parameters"].get("type") == "object"
        assert isinstance(spec["est_cost"], int)
        assert isinstance(spec["est_latency_ms"], int)
    # 写操作必须标记需要确认，且不重试
    submit = specs["submit_return_request"]
    assert submit["requires_confirmation"] is True
    assert submit["retryable"] is False
    # 只读查询可重试
    assert specs["query_order"]["retryable"] is True


def test_specs_sorted_by_cost_then_latency(registry: ToolRegistry) -> None:
    specs = registry.specs()
    keys = [(spec["est_cost"], spec["est_latency_ms"]) for spec in specs]
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# 参数校验
# ---------------------------------------------------------------------------
def test_missing_required_param_is_rejected(registry: ToolRegistry) -> None:
    result = registry.call("query_order", {})
    assert result.ok is False
    assert result.error_code == "invalid_param"
    assert "order_id" in result.error


def test_blank_param_is_rejected(registry: ToolRegistry) -> None:
    result = registry.call("query_order", {"order_id": "   "})
    assert result.ok is False
    assert result.error_code == "invalid_param"


def test_unknown_params_are_dropped(registry: ToolRegistry) -> None:
    """未声明的参数应被丢弃，不该透传进业务逻辑。"""
    result = registry.call("query_order", {"order_id": "SO20260101", "恶意字段": "x"})
    assert result.ok is True
    assert "恶意字段" not in result.data


def test_integer_coercion(registry: ToolRegistry) -> None:
    ok = registry.call("search_policy", {"query": "退货运费", "top_k": "2"})
    assert ok.ok is True
    assert len(ok.data["hits"]) <= 2


def test_enum_validation() -> None:
    class _Echo(BaseTool):
        name = "echo"
        description = "测试用"
        parameters = {
            "type": "object",
            "properties": {"mode": BaseTool.param("string", "模式", enum=["a", "b"])},
            "required": ["mode"],
        }

        def run(self, mode: str = "", **_: Any) -> ToolResult:
            return ToolResult(ok=True, tool=self.name, data={"mode": mode})

    tool = _Echo()
    assert tool.call({"mode": "a"}).ok is True
    bad = tool.call({"mode": "c"})
    assert bad.ok is False
    assert "只能是" in bad.error


# ---------------------------------------------------------------------------
# 查询类：不编造
# ---------------------------------------------------------------------------
def test_query_order_success(registry: ToolRegistry) -> None:
    result = registry.call("query_order", {"order_id": "SO20260101"})
    assert result.ok is True
    assert result.data["订单号"] == "SO20260101"
    assert result.data["状态"] == "已发货运输中"
    assert result.meta["data_source"].startswith("本地模拟")


def test_query_order_normalises_case(registry: ToolRegistry) -> None:
    assert registry.call("query_order", {"order_id": "so20260101"}).ok is True


def test_query_order_not_found_is_honest(registry: ToolRegistry) -> None:
    result = registry.call("query_order", {"order_id": "SO99999999"})
    assert result.ok is False
    assert result.error_code == "not_found"
    assert "没有查询到" in result.error
    assert result.data is None, "查不到时绝不能返回编造的数据"


def test_track_logistics_returns_traces_in_time_order(registry: ToolRegistry) -> None:
    result = registry.call("track_logistics", {"order_id": "SO20260101"})
    assert result.ok is True
    times = [item["time"] for item in result.data["轨迹"]]
    assert times == sorted(times), "轨迹必须按时间正序展示"
    assert result.data["承运商"] == "顺丰速运"
    assert result.data["最新动态"]


def test_track_logistics_without_tracking_number(registry: ToolRegistry) -> None:
    """未发货订单没有运单号：要如实说明，而不是编一条轨迹。"""
    repo = get_repository()
    order = repo.get("SO20260101")
    assert order is not None
    order.tracking_no = ""
    order.status = "paid"

    result = registry.call("track_logistics", {"order_id": "SO20260101"})
    assert result.ok is False
    assert result.error_code == "no_logistics"
    assert "还没有物流单号" in result.error


def test_query_refund_without_record(registry: ToolRegistry) -> None:
    result = registry.call("query_refund", {"order_id": "SO20260101"})
    assert result.ok is True
    assert result.data["退款状态"] == "该订单暂无退款记录"
    assert result.meta["has_refund"] is False


def test_query_refund_with_record(registry: ToolRegistry) -> None:
    result = registry.call("query_refund", {"order_id": "SO20260104"})
    assert result.ok is True
    assert "退货运输中" in result.data["退款状态"]
    assert result.data["退款金额"] == 199.0
    assert result.meta["has_refund"] is True


# ---------------------------------------------------------------------------
# 退货判断：结合窗口期 + 商品类型，给出依据
# ---------------------------------------------------------------------------
def test_check_returnable_within_window(registry: ToolRegistry) -> None:
    """SO20260102 签收在窗口外（12 天前）→ 只支持质量问题退。"""
    result = registry.call("check_returnable", {"order_id": "SO20260102"})
    assert result.ok is True
    assert result.data["结论"]["verdict"] == "quality_only"
    assert result.data["判断依据"]
    assert result.data["依据条款"], "结论必须给依据条款，不能只给判断"


def test_check_returnable_oversized_window(registry: ToolRegistry) -> None:
    """SO20260104 签收 7 天内 → 可无理由退货。"""
    result = registry.call("check_returnable", {"order_id": "SO20260104"})
    assert result.data["结论"]["verdict"] == "returnable"


def test_check_returnable_customised_item(registry: ToolRegistry) -> None:
    """定制商品不支持无理由退货（政策 POL-004）。"""
    result = registry.call("check_returnable", {"order_id": "SO20260103"})
    assert result.data["结论"]["verdict"] == "quality_only"
    assert "定制" in result.data["不可退原因"] if "不可退原因" in result.data else True
    assert any(
        cite["doc_id"] == "POL-004" for cite in result.data["依据条款"]
    ), "定制商品应引用不可退换条款"


def test_check_returnable_freight_by_reason(registry: ToolRegistry) -> None:
    seller = registry.call(
        "check_returnable", {"order_id": "SO20260104", "reason": "质量问题，收到就破了"}
    )
    buyer = registry.call(
        "check_returnable", {"order_id": "SO20260104", "reason": "不喜欢，买错了"}
    )
    assert "商家" in seller.data["运费承担"]
    assert "买家" in buyer.data["运费承担"]
    assert any(cite["doc_id"] == "POL-002" for cite in seller.data["依据条款"])


def test_check_returnable_without_reason_asks_for_it(registry: ToolRegistry) -> None:
    result = registry.call("check_returnable", {"order_id": "SO20260104"})
    assert "补充退货原因" in result.data["运费承担"]


# ---------------------------------------------------------------------------
# 政策检索工具
# ---------------------------------------------------------------------------
def test_search_policy_tool_success(registry: ToolRegistry) -> None:
    result = registry.call("search_policy", {"query": "退货运费谁承担"})
    assert result.ok is True
    assert result.data["hits"][0]["doc_id"] == "POL-002"
    assert result.data["top_source"]
    assert "非语义检索" in result.meta["retrieval"]


def test_search_policy_tool_no_match_is_failure(registry: ToolRegistry) -> None:
    """检索不到必须失败，不能返回一条不相干的政策充数。"""
    result = registry.call("search_policy", {"query": "帮我写一首诗"})
    assert result.ok is False
    assert result.error_code == "no_policy_match"
    assert result.data["hits"] == []


# ---------------------------------------------------------------------------
# 写操作：必须确认 + 状态校验
# ---------------------------------------------------------------------------
def test_write_tool_requires_confirmation(registry: ToolRegistry) -> None:
    outcome = registry.execute(
        "submit_return_request", {"order_id": "SO20260104", "reason": "尺码不合适"}
    )
    assert outcome.needs_confirmation is True
    assert outcome.ok is False
    assert "不可撤销" in outcome.confirmation_prompt
    assert "SO20260104" in outcome.confirmation_prompt
    # 未确认时数据不能被改动
    assert get_repository().get("SO20260104").refund_status.startswith("退货运输中")


def test_confirmed_write_updates_state(registry: ToolRegistry) -> None:
    outcome = registry.execute(
        "submit_return_request",
        {"order_id": "SO20260102", "reason": "尺码不合适"},
        confirmed=True,
    )
    assert outcome.ok is True
    assert outcome.result.data["新状态"] == "退货处理中"
    assert get_repository().get("SO20260102").status == "returning"


def test_write_tool_rejects_not_delivered_order(registry: ToolRegistry) -> None:
    outcome = registry.execute(
        "submit_return_request", {"order_id": "SO20260101", "reason": "不想要了"}, confirmed=True
    )
    assert outcome.ok is False
    assert outcome.result.error_code == "not_delivered"
    assert "尚未签收" in outcome.result.error


def test_write_tool_rejects_duplicate_request(registry: ToolRegistry) -> None:
    outcome = registry.execute(
        "submit_return_request", {"order_id": "SO20260104", "reason": "尺码不合适"}, confirmed=True
    )
    assert outcome.ok is False
    assert outcome.result.error_code == "already_returning"


def test_unknown_tool(registry: ToolRegistry) -> None:
    result = registry.call("不存在的工具", {})
    assert result.ok is False
    assert result.error_code == "unknown_tool"


# ---------------------------------------------------------------------------
# 重试策略
# ---------------------------------------------------------------------------
def test_retryable_tool_is_retried() -> None:
    """瞬时故障重试，第二次成功。"""
    calls = {"n": 0}

    class _Flaky(BaseTool):
        name = "flaky"
        description = "第一次失败、第二次成功"
        retryable = True

        def run(self, **_: Any) -> ToolResult:
            calls["n"] += 1
            if calls["n"] == 1:
                raise ToolError("临时故障", retryable=True)
            return ToolResult(ok=True, tool=self.name, data={"n": calls["n"]})

    result = _Flaky().call({})
    assert result.ok is True
    assert calls["n"] == 2


def test_non_retryable_tool_is_not_retried() -> None:
    """业务性失败（订单不存在）重试一万次也还是不存在。"""
    calls = {"n": 0}

    class _Business(BaseTool):
        name = "business"
        description = "业务失败"
        retryable = True  # 即使工具声明可重试，ToolError 说不可重试就不重试

        def run(self, **_: Any) -> ToolResult:
            calls["n"] += 1
            raise ToolError("订单不存在", retryable=False, code="not_found")

    result = _Business().call({})
    assert result.ok is False
    assert result.error_code == "not_found"
    assert calls["n"] == 1


def test_internal_exception_is_wrapped(registry: ToolRegistry) -> None:
    class _Boom(BaseTool):
        name = "boom"
        description = "抛异常"
        retryable = False

        def run(self, **_: Any) -> ToolResult:
            raise ValueError("内部炸了")

    result = _Boom().call({})
    assert result.ok is False
    assert "工具内部错误" in result.error


def test_latency_is_measured(registry: ToolRegistry) -> None:
    result = registry.call("query_order", {"order_id": "SO20260101"})
    assert result.latency_ms >= 0
    assert "latency_ms" in result.as_dict()
