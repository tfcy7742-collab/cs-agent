"""技能编排：决定调哪个工具、怎么处理结果、写操作怎么确认。

三条纪律，都是这一层的存在理由：

1. **没参数就别假装能办**：缺订单号时不去猜、也不空调用工具，而是先问（P3 已做）；
2. **工具说没有就是没有**：查不到订单、检索不到政策，都把原话如实交给模型，
   并在提示里明说"不允许编造"——这是"不编造"从工具层传到生成层的最后一公里；
3. **写操作必须用户点头**：``submit_return_request`` 这种有副作用的操作，
   第一次只返回"待确认"，等用户明确同意后才真正执行。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..state import StateSnapshot
from ..tasks import SLOT_ITEM, SLOT_ORDER_ID, SLOT_REASON, is_policy_question
from .registry import ExecutionOutcome, ToolRegistry

#: 意图 → 工具。只读查询，按"先查证"的顺序排列。
INTENT_TOOLS: Dict[str, str] = {
    "查询订单": "query_order",
    "查询物流": "track_logistics",
    "查询到货时间": "track_logistics",
    "催发货": "track_logistics",
    "查询退款进度": "query_refund",
    "退款": "query_refund",
    "退货": "check_returnable",
    "退换货": "check_returnable",
    "换货": "check_returnable",
    "运费咨询": "search_policy",
    "优惠咨询": "search_policy",
    "保修咨询": "search_policy",
    "政策咨询": "search_policy",
    "支付咨询": "search_policy",
    "修改收货地址": "",  # 没有对应工具：只能告知政策，说明需要人工
    "开具发票": "",  # 同上
    "转人工工单": "",  # P5 才做
}

#: 需要订单号才能查的工具
NEEDS_ORDER = {"query_order", "track_logistics", "query_refund", "check_returnable"}

#: 用户表示同意的说法（用于确认写操作）
AFFIRMATIVE_MARKERS = (
    "确认", "同意", "可以", "好的", "好呀", "行", "提交吧", "办吧", "退吧",
    "是的", "对", "嗯", "ok", "OK", "Ok", "麻烦你", "请办", "就这样",
)

#: 用户表示否定的说法。
#: 刻意不用单字"不"——"不用了""不想要了"都是否定，但"不要发票"里的"不"会误伤；
#: 更危险的是把"不用了"当成"我要退货"的确认。这里只收录明确的否定短语。
NEGATIVE_MARKERS = (
    "不用了", "不用", "不要了", "不要", "取消", "算了", "先不", "别办", "否", "no",
)


@dataclass
class ToolRun:
    """一次工具调用的记录（进轨迹）。"""

    tool: str
    params: Dict[str, Any]
    ok: bool
    latency_ms: int = 0
    error: str = ""
    error_code: str = ""
    needs_confirmation: bool = False
    degraded: bool = False

    def as_dict(self) -> Dict[str, Any]:
        data = {
            "tool": self.tool,
            "params": self.params,
            "ok": self.ok,
            "latency_ms": self.latency_ms,
        }
        if self.error:
            data["error"] = self.error
            data["error_code"] = self.error_code
        if self.needs_confirmation:
            data["needs_confirmation"] = True
        if self.degraded:
            data["degraded"] = True
        return data


@dataclass
class ToolPlan:
    """本轮的工具执行结果。"""

    runs: List[ToolRun] = field(default_factory=list)
    #: 写操作待确认时的提示语（直接讲给用户听）
    confirmation_prompt: str = ""
    #: 是否已处理完挂起的写操作（确认执行 / 用户取消都要清掉挂起状态）
    cleared_pending: bool = False
    #: 注入系统提示的工具事实（不允许模型改写）
    context: str = ""
    #: 需要如实告知用户的失败说明
    honest_failures: List[str] = field(default_factory=list)
    policy_citations: List[str] = field(default_factory=list)

    @property
    def called(self) -> bool:
        return bool(self.runs)

    def as_dict(self) -> Dict[str, Any]:
        data = {"runs": [run.as_dict() for run in self.runs]}
        if self.confirmation_prompt:
            data["confirmation_prompt"] = self.confirmation_prompt
        if self.honest_failures:
            data["honest_failures"] = self.honest_failures
        if self.policy_citations:
            data["policy_citations"] = self.policy_citations
        return data


class SkillExecutor:
    """把（意图 + 槽位 + 用户话术）变成一次受治理的工具调用。"""

    def __init__(self, registry: Optional[ToolRegistry] = None) -> None:
        self.registry = registry or ToolRegistry()

    # ------------------------------------------------------------------
    def execute(
        self,
        plan: Any,
        state: StateSnapshot,
        user_text: str,
        *,
        pending_write: Optional[Dict[str, Any]] = None,
    ) -> ToolPlan:
        """按计划执行工具。

        Args:
            plan: ``planner.Plan``（提供 intents / action）。
            state: 当前会话状态（提供槽位）。
            user_text: 用户本轮原话（判断是否同意、是否在问政策）。
            pending_write: 上一轮挂起的写操作（等待用户确认）。
        """
        result = ToolPlan()

        # ---- 0. 挂起的写操作：先判断用户是否同意 ----
        if pending_write:
            tool_name = str(pending_write.get("tool") or "")
            params = dict(pending_write.get("params") or {})
            if self._is_negative(user_text):
                result.honest_failures.append("好的，已取消这次操作，订单状态没有变化")
                result.cleared_pending = True
                return result
            if self._is_affirmative(user_text):
                outcome = self.registry.execute(tool_name, params, confirmed=True)
                result.runs.append(self._record(tool_name, params, outcome))
                result.context = self._render_context(tool_name, outcome)
                result.cleared_pending = True
                if not outcome.ok:
                    result.honest_failures.append(outcome.result.error)
                return result
            # 既不同意也不否定：把确认再问一次，不要擅自执行
            outcome = self.registry.execute(tool_name, params, confirmed=False)
            result.confirmation_prompt = outcome.confirmation_prompt
            return result

        intents: List[str] = list(getattr(plan, "intents", []) or [])

        # ---- 计划本身就是"先追问"时，不执行任何工具 ----
        # 缺参数就该先问清楚（"帮我查下物流"没有订单号），
        # 这时跑去检索政策只会让回复偏题。
        if getattr(plan, "action", "") == "ask":
            return result

        # ---- 1. 明确要办退货：走"确认后才提交"的写流程 ----
        if "退货" in intents and self._is_explicit_return(user_text):
            return self._plan_return_submission(state, result)

        # ---- 2. 只读查询：按意图依次尝试，第一个能办的执行 ----
        for intent in intents:
            tool_name = INTENT_TOOLS.get(intent)
            if not tool_name:
                continue
            if tool_name in NEEDS_ORDER and not state.known(SLOT_ORDER_ID):
                continue  # 缺参数：交给追问逻辑，不空调用
            params = self._params_for(tool_name, state, user_text)
            if not params:
                continue
            return self._run_readonly(tool_name, params, result)

        # ---- 3. 兜底：有诉求但没匹配到工具时，先去政策库找依据 ----
        # 这比"什么都不查直接回答"安全得多：检索到就按原文答，检索不到就如实说没有。
        # 另外，没抽到意图但在问政策（如"会员积分怎么兑换"）也要去查一次——
        # 抽不到意图不等于没有依据可查。
        if intents or is_policy_question(user_text):
            return self._run_readonly(
                "search_policy",
                {"query": (user_text or "").strip()[:200]},
                result,
            )

        return result

    def _run_readonly(
        self, tool_name: str, params: Dict[str, Any], result: ToolPlan
    ) -> ToolPlan:
        """执行一个只读工具，并把结果/失败如实填进提示上下文。"""
        outcome = self.registry.execute(tool_name, params)
        result.runs.append(self._record(tool_name, params, outcome))

        if not outcome.ok:
            result.honest_failures.append(outcome.result.error)
            # 失败时也要给出提示上下文：否则模型只看到"没有工具结果"，
            # 很容易自己编一个答案——这正是要防的情况。
            result.context = (
                "【工具执行结果（唯一事实来源，禁止编造）】\n"
                f"- 工具：{tool_name}\n"
                f"- 结果：查询失败\n"
                f"- 原因：{outcome.result.error}\n"
                "请如实把上面的原因告诉用户，不要猜测或编造数据；"
                "如果需要补充信息才能查，请说明需要什么。"
            )
        else:
            result.context = self._render_context(tool_name, outcome)
            result.honest_failures.extend(self._soft_failures(tool_name, outcome))
        self._collect_citations(outcome, result)
        return result

    # ------------------------------------------------------------------
    # 参数与话术
    # ------------------------------------------------------------------
    @staticmethod
    def _params_for(tool_name: str, state: StateSnapshot, user_text: str) -> Dict[str, Any]:
        if tool_name == "search_policy":
            return {"query": (user_text or "").strip()[:200]}
        order_id = state.known(SLOT_ORDER_ID)
        if not order_id:
            return {}
        params: Dict[str, Any] = {"order_id": order_id}
        if tool_name == "check_returnable":
            reason = state.known(SLOT_REASON)
            if reason:
                params["reason"] = reason
        return params

    @staticmethod
    def _is_explicit_return(user_text: str) -> bool:
        """用户是否**明确要求提交退货**（而不是问"能不能退"）。"""
        text = user_text or ""
        markers = ("我要退货", "帮我退", "申请退", "我要退", "退了吧", "那就退", "提交退货", "办理退货")
        return any(marker in text for marker in markers)

    @staticmethod
    def _is_affirmative(user_text: str) -> bool:
        """用户是否在**确认**这件事。

        三个条件缺一不可：短、不是问句、以确认词开头或以纯确认词回应。
        只判断"包含可以/好的"会把"这个运费谁出啊"这种答非所问也当成同意——
        写操作一旦误执行就无法撤销，这里必须保守。
        """
        text = (user_text or "").strip()
        if not text or len(text) > 15:
            return False
        if any(mark in text for mark in ("?", "？", "吗", "呢", "怎么", "为什么", "谁")):
            return False
        return any(
            text == marker or text.startswith(marker) for marker in AFFIRMATIVE_MARKERS
        )

    @staticmethod
    def _is_negative(user_text: str) -> bool:
        text = (user_text or "").strip()
        return any(marker in text for marker in NEGATIVE_MARKERS)

    # ------------------------------------------------------------------
    # 写流程
    # ------------------------------------------------------------------
    def _plan_return_submission(self, state: StateSnapshot, result: ToolPlan) -> ToolPlan:
        """提交退货申请：先做资格核对，再挂起等确认。"""
        order_id = state.known(SLOT_ORDER_ID)
        reason = state.known(SLOT_REASON) or (state.known(SLOT_ITEM) and "商品问题") or ""
        if not order_id:
            return result

        params = {"order_id": order_id, "reason": reason or "用户申请退货"}
        outcome = self.registry.execute("submit_return_request", params)
        result.runs.append(self._record("submit_return_request", params, outcome))

        if outcome.needs_confirmation:
            result.confirmation_prompt = outcome.confirmation_prompt
        elif not outcome.ok:
            result.honest_failures.append(outcome.result.error)
        else:
            result.context = self._render_context("submit_return_request", outcome)
        return result

    # ------------------------------------------------------------------
    # 注入提示
    # ------------------------------------------------------------------
    @staticmethod
    def _record(tool: str, params: Dict[str, Any], outcome: ExecutionOutcome) -> ToolRun:
        return ToolRun(
            tool=tool,
            params=params,
            ok=outcome.ok,
            latency_ms=outcome.result.latency_ms,
            error=outcome.result.error,
            error_code=outcome.result.error_code,
            needs_confirmation=outcome.needs_confirmation,
            degraded=outcome.result.degraded,
        )

    @staticmethod
    def _soft_failures(tool_name: str, outcome: ExecutionOutcome) -> List[str]:
        """成功但内容"实际为空"的情况，需要如实告知（例如没有退款记录）。"""
        data = outcome.result.data or {}
        if data.get("退款状态") == "该订单暂无退款记录":
            return ["该订单没有退款记录"]
        return []

    @staticmethod
    def _collect_citations(outcome: ExecutionOutcome, result: ToolPlan) -> None:
        data = outcome.result.data or {}
        for cite in data.get("依据条款") or []:
            source = cite.get("source")
            if source and source not in result.policy_citations:
                result.policy_citations.append(source)
        for hit in data.get("hits") or []:
            source = hit.get("source")
            if source and source not in result.policy_citations:
                result.policy_citations.append(source)

    @staticmethod
    def _render_context(tool_name: str, outcome: ExecutionOutcome) -> str:
        """把工具结果渲染成系统提示片段。

        这里刻意把"这是唯一事实来源"写进提示——模型最常见的失败是
        工具已经明确说"查不到"，它还是编一个答案出来。
        """
        import json

        if not outcome.ok:
            return (
                "【工具执行结果（唯一事实来源，禁止编造）】\n"
                f"- 工具：{tool_name}\n"
                f"- 结果：查询失败\n"
                f"- 原因：{outcome.result.error}\n"
                "请如实把上面的原因告诉用户，不要猜测或编造数据；"
                "如果需要补充信息才能查，请说明需要什么。"
            )

        payload = json.dumps(outcome.result.data, ensure_ascii=False, indent=2)
        lines = [
            "【工具执行结果（唯一事实来源，禁止编造、禁止改写数字）】",
            f"- 工具：{tool_name}",
            f"- 结果：成功",
            "- 数据：",
            payload,
        ]
        if outcome.result.meta.get("data_source"):
            lines.append(f"- 数据来源：{outcome.result.meta['data_source']}（演示数据）")
        lines.append(
            "请严格依据上面的数据回答；数据里没有的信息不要补充，"
            "涉及金额与时效时说明「以实际到账 / 实际物流为准」。"
        )
        return "\n".join(lines)
