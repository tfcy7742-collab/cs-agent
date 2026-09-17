"""六个客服技能工具。

工具清单与"它解决什么"：

| 工具 | 能力 | 关键纪律 |
| --- | --- | --- |
| ``query_order`` | 查订单详情 | 查不到就说查不到，不猜 |
| ``track_logistics`` | 查物流轨迹 | 无轨迹时明确说明未更新 |
| ``check_returnable`` | 判断能否退货 | 结合政策窗口 + 商品类型，给出**依据条款** |
| ``query_refund`` | 查退款进度 | 未申请退款 vs 已申请要分清 |
| ``search_policy`` | 检索平台政策 | 命中给原文与来源；**不命中就返回空** |
| ``submit_return_request`` | 提交退货申请 | **有副作用**，必须用户确认后才执行 |

贯穿的一条纪律：工具只能**如实返回**它查到的东西。查不到数据时返回 ``ok=False`` 或
``data`` 里显式的 not_found，绝不编一个"看起来合理"的答案——上层据此如实告知用户。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..policy import get_index
from .base import BaseTool, ToolError, ToolResult
from .mock_data import (
    STATUS_TEXT,
    OrderRecord,
    get_repository,
    in_return_window,
)


class _OrderTool(BaseTool):
    """订单类工具的公共部分：取订单 + 统一的"查不到"话术。"""

    est_cost = 1
    est_latency_ms = 120
    retryable = True

    @staticmethod
    def _fetch(order_id: str) -> OrderRecord:
        order = get_repository().get(order_id)
        if order is None:
            # 明确区分"参数非法"和"查不到"：前者是调用方错误，后者是业务事实
            raise ToolError(
                f"没有查询到订单 {order_id}，请确认订单号是否正确"
                "（如核对无误，可能是该订单不属于当前账号或已删除）",
                retryable=False,
                code="not_found",
            )
        return order


class QueryOrderTool(_OrderTool):
    name = "query_order"
    description = "根据订单号查询订单详情：商品、金额、订单状态、下单时间、收货信息、是否可退。"
    parameters = {
        "type": "object",
        "properties": {
            "order_id": BaseTool.param("string", "订单号，例如 SO20260101"),
        },
        "required": ["order_id"],
    }

    def run(self, order_id: str = "", **_: Any) -> ToolResult:
        order = self._fetch(order_id)
        return ToolResult(
            ok=True,
            tool=self.name,
            data=order.as_dict(),
            meta={"data_source": "本地模拟订单库（非真实电商系统）"},
        )


class TrackLogisticsTool(_OrderTool):
    name = "track_logistics"
    description = "根据订单号查询物流轨迹与预计送达时间。"
    parameters = {
        "type": "object",
        "properties": {
            "order_id": BaseTool.param("string", "订单号"),
        },
        "required": ["order_id"],
    }

    def run(self, order_id: str = "", **_: Any) -> ToolResult:
        order = self._fetch(order_id)
        if not order.tracking_no:
            return ToolResult(
                ok=False,
                tool=self.name,
                error=(
                    f"订单 {order.order_id} 当前状态为「{STATUS_TEXT.get(order.status, order.status)}」，"
                    "还没有物流单号，暂时无法查询轨迹"
                ),
                error_code="no_logistics",
                data={"订单号": order.order_id, "状态": STATUS_TEXT.get(order.status, order.status)},
            )

        ordered = sorted(order.traces, key=lambda item: item["time"])
        latest = ordered[-1]["desc"] if ordered else ""
        return ToolResult(
            ok=True,
            tool=self.name,
            data={
                "订单号": order.order_id,
                "承运商": order.carrier,
                "运单号": order.tracking_no,
                "当前状态": STATUS_TEXT.get(order.status, order.status),
                "最新动态": latest,
                "预计送达": order.eta or "暂无明确时间，以物流实际更新为准",
                "轨迹": ordered,
            },
            meta={"data_source": "本地模拟物流库（非真实承运商接口）"},
        )


class CheckReturnableTool(_OrderTool):
    name = "check_returnable"
    description = (
        "判断某订单是否可以退货，并给出依据条款。会综合商品类型、签收时间与平台政策。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "order_id": BaseTool.param("string", "订单号"),
            "reason": BaseTool.param("string", "退货原因（可选，用于判断运费承担方）"),
        },
        "required": ["order_id"],
    }

    def run(self, order_id: str = "", reason: str = "", **_: Any) -> ToolResult:
        order = self._fetch(order_id)
        index = get_index()

        within_window = in_return_window(order)
        reasons: List[str] = []
        citations: List[Dict[str, Any]] = []

        policy_quality = index.by_id("POL-001")
        policy_special = index.by_id("POL-004")
        policy_freight = index.by_id("POL-002")

        # 1) 商品本身是否属于不可无理由退货的范围
        if not order.returnable:
            reasons.append(order.not_returnable_reason or "该商品不支持七天无理由退货")
            if policy_special:
                citations.append(policy_special.as_dict())
            verdict = "quality_only"
        # 2) 是否还在无理由窗口内
        elif within_window:
            reasons.append("在签收后 7 个自然日内，符合七天无理由退货条件")
            if policy_quality:
                citations.append(policy_quality.as_dict())
            verdict = "returnable"
        else:
            reasons.append("已超过签收后 7 个自然日，不再支持无理由退货；若为质量问题仍可申请")
            if policy_quality:
                citations.append(policy_quality.as_dict())
            verdict = "quality_only"

        # 3) 运费承担（有原因时给出判断依据）
        freight = ""
        if reason:
            seller_faults = ("质量", "破损", "坏了", "发错", "漏发", "少件", "不符")
            if any(word in reason for word in seller_faults):
                freight = "按您描述的原因属于商家责任，运费一般由商家承担"
            else:
                freight = "按您描述的原因属于个人原因，运费一般由买家承担（运费险可另行理赔）"
            if policy_freight:
                citations.append(policy_freight.as_dict())

        return ToolResult(
            ok=True,
            tool=self.name,
            data={
                "订单号": order.order_id,
                "商品": order.item,
                "类目": order.category,
                "订单状态": STATUS_TEXT.get(order.status, order.status),
                "签收时间": order.delivered_at or "尚未签收",
                "结论": {
                    "returnable": "可以申请退货" if verdict == "returnable" else "仅质量问题可退",
                    "verdict": verdict,
                },
                "判断依据": reasons,
                "运费承担": freight or "请补充退货原因，以便判断运费由谁承担",
                "依据条款": citations,
            },
            meta={
                "data_source": "本地模拟订单库 + 示例政策库",
                "within_window": within_window,
            },
        )


class QueryRefundTool(_OrderTool):
    name = "query_refund"
    description = "根据订单号查询退款进度与预计到账时间。"
    parameters = {
        "type": "object",
        "properties": {
            "order_id": BaseTool.param("string", "订单号"),
        },
        "required": ["order_id"],
    }

    def run(self, order_id: str = "", **_: Any) -> ToolResult:
        order = self._fetch(order_id)
        if not order.refund_status:
            return ToolResult(
                ok=True,
                tool=self.name,
                data={
                    "订单号": order.order_id,
                    "退款状态": "该订单暂无退款记录",
                    "说明": "如需申请退货退款，可以先提交退货申请",
                },
                meta={"has_refund": False},
            )
        return ToolResult(
            ok=True,
            tool=self.name,
            data={
                "订单号": order.order_id,
                "商品": order.item,
                "退款状态": order.refund_status,
                "退款金额": order.refund_amount,
                "预计到账": order.refund_eta or "以实际到账时间为准",
            },
            meta={"has_refund": True, "data_source": "本地模拟退款库（非真实支付系统）"},
        )


class SearchPolicyTool(BaseTool):
    name = "search_policy"
    description = (
        "检索平台售后政策（退货时限、运费承担、退款时效、发票、价保、保修等）。"
        "回答规则类问题前应先调用它，并依据返回的原文作答。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": BaseTool.param("string", "用户的问题或关键词"),
            "top_k": BaseTool.param("integer", "返回条数，默认 3", default=3),
        },
        "required": ["query"],
    }
    est_cost = 1
    est_latency_ms = 60
    retryable = True

    def run(self, query: str = "", top_k: int = 3, **_: Any) -> ToolResult:
        index = get_index()
        top_k = max(1, min(int(top_k or 3), 6))
        hits = index.search(query, top_k=top_k)

        if not hits:
            # 检索不到就如实返回空——绝不能拿一条不相关的政策凑数
            return ToolResult(
                ok=False,
                tool=self.name,
                error=f"政策库里没有找到与「{query}」相关的条款，无法据此回答",
                error_code="no_policy_match",
                data={"query": query, "hits": []},
                meta={"index_size": index.count()},
            )

        return ToolResult(
            ok=True,
            tool=self.name,
            data={
                "query": query,
                "hits": [hit.as_dict() for hit in hits],
                "top_source": hits[0].doc.title,
            },
            meta={
                "index_size": index.count(),
                "data_source": "示例政策库（条文为演示编写，非真实平台规则）",
                "retrieval": "关键词 + 中文二元组 IDF（可解释，非语义检索）",
            },
        )


class SubmitReturnRequestTool(_OrderTool):
    name = "submit_return_request"
    description = (
        "为用户提交退货申请（有副作用）。需要订单号与退货原因；"
        "提交后订单进入退货处理中，不可撤销。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "order_id": BaseTool.param("string", "订单号"),
            "reason": BaseTool.param("string", "退货原因"),
        },
        "required": ["order_id", "reason"],
    }
    # 写操作：不重试、必须用户确认
    retryable = False
    requires_confirmation = True

    def run(self, order_id: str = "", reason: str = "", **_: Any) -> ToolResult:
        order = self._fetch(order_id)

        if order.status in {"returning", "refunded"}:
            return ToolResult(
                ok=False,
                tool=self.name,
                error=(
                    f"订单 {order.order_id} 当前状态为「{STATUS_TEXT.get(order.status, order.status)}」，"
                    "无需重复提交退货申请"
                ),
                error_code="already_returning",
            )
        if order.status == "cancelled":
            return ToolResult(
                ok=False,
                tool=self.name,
                error="该订单已取消，无需退货",
                error_code="cancelled",
            )
        if order.status in {"paid", "shipped"}:
            return ToolResult(
                ok=False,
                tool=self.name,
                error=(
                    f"订单 {order.order_id} 尚未签收（{STATUS_TEXT.get(order.status, order.status)}），"
                    "建议先拒收或等待签收后再申请退货"
                ),
                error_code="not_delivered",
            )

        # 真正的写操作：更新模拟数据源
        order.status = "returning"
        order.refund_status = f"退货申请已提交，原因：{reason}。等待仓库签收质检"
        order.refund_amount = order.amount
        order.refund_eta = "仓库签收后 1-3 个工作日审核，审核通过后 1-7 个工作日到账"
        get_repository().update(order)

        return ToolResult(
            ok=True,
            tool=self.name,
            data={
                "订单号": order.order_id,
                "商品": order.item,
                "退货原因": reason,
                "新状态": STATUS_TEXT["returning"],
                "退款金额": order.refund_amount,
                "后续流程": order.refund_eta,
                "提示": "请按页面提示寄回商品并保留运单号",
            },
            meta={
                "side_effect": True,
                "data_source": "本地模拟订单库（已就地修改演示数据）",
            },
        )


#: 工具清单：顺序即"推荐优先级"（先查证、后写操作）
TOOL_CLASSES = (
    QueryOrderTool,
    TrackLogisticsTool,
    QueryRefundTool,
    CheckReturnableTool,
    SearchPolicyTool,
    SubmitReturnRequestTool,
)


def build_tools() -> List[BaseTool]:
    """构造全部工具实例。"""
    return [cls() for cls in TOOL_CLASSES]
