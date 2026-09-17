"""客服任务定义：意图 → 必需槽位 → 追问话术。

这是 P3 的"单一事实来源"。三个关键设计：

1. **``required``（硬槽位）**：缺了就**必须追问**才能继续办事。例如退货没有订单号，
   系统无法定位单据——这时追问是对的，编一个更糟。
2. **``optional``（软槽位）**：缺了照样能办，只是办得不够精准。**绝不能因为缺软槽位就拦着用户**，
   否则会出现"我只是想问退货运费谁出，它却一直追着我要订单号"这种最烦人的体验。
3. **``guide``（追问话术）**：每个槽位给一句人话，一次只问一句；
   追问过的槽位会被记录，避免反复问同一件事。

后续阶段（P4）会把 ``required`` 里的槽位直接喂给工具参数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TaskDef:
    """一类客服任务。"""

    key: str
    name: str
    #: 硬槽位：缺失时必须先问清楚
    required: tuple = ()
    #: 软槽位：缺失不影响继续，只是精度下降
    optional: tuple = ()
    #: 每个槽位的追问话术（一次只问最靠前的那一个）
    guide: Dict[str, str] = field(default_factory=dict)
    #: 同义说法，用于把用户表述归一到任务 key
    aliases: tuple = ()

    def ask_for(self, slot: str) -> str:
        return self.guide.get(slot) or f"麻烦提供一下{slot}，我好继续为您处理。"


#: 槽位名（统一在这里定义，避免各处拼字符串拼错）
SLOT_ORDER_ID = "订单号"
SLOT_REASON = "退货原因"
SLOT_ITEM = "商品名称"
SLOT_EXPECTATION = "期望处理方式"
SLOT_ADDRESS = "新地址"
SLOT_INVOICE_TITLE = "发票抬头"
SLOT_QUESTION = "咨询内容"

TASKS: Dict[str, TaskDef] = {
    "query_logistics": TaskDef(
        key="query_logistics",
        name="查询物流",
        required=(SLOT_ORDER_ID,),
        optional=(SLOT_ITEM,),
        guide={
            SLOT_ORDER_ID: "麻烦提供一下订单号，我帮您查物流进度。",
            SLOT_ITEM: "方便的话也告诉我商品名称，我可以定位到具体包裹。",
        },
        aliases=("查询物流", "物流查询", "催发货", "查询到货时间"),
    ),
    "query_order": TaskDef(
        key="query_order",
        name="查询订单",
        required=(SLOT_ORDER_ID,),
        guide={SLOT_ORDER_ID: "请提供订单号，我帮您查这笔订单的详情。"},
        aliases=("查询订单", "查订单"),
    ),
    "return_goods": TaskDef(
        key="return_goods",
        name="退货",
        required=(SLOT_ORDER_ID,),
        optional=(SLOT_REASON, SLOT_ITEM),
        guide={
            SLOT_ORDER_ID: "请把订单号发我，我帮您看这笔订单能不能退。",
            SLOT_REASON: "方便说下退货原因吗？这会影响运费由谁承担。",
            SLOT_ITEM: "要退的是哪件商品呢？",
        },
        aliases=("退货", "退换货", "换货"),
    ),
    "refund_status": TaskDef(
        key="refund_status",
        name="查询退款进度",
        required=(SLOT_ORDER_ID,),
        guide={SLOT_ORDER_ID: "请提供订单号，我帮您查退款到账进度。"},
        aliases=("退款", "退款进度", "退钱"),
    ),
    "change_address": TaskDef(
        key="change_address",
        name="修改收货地址",
        required=(SLOT_ORDER_ID, SLOT_ADDRESS),
        guide={
            SLOT_ORDER_ID: "请提供要修改的订单号。",
            SLOT_ADDRESS: "请把新的收货地址发我，我帮您提交修改。",
        },
        aliases=("修改地址", "改地址", "换地址"),
    ),
    "invoice": TaskDef(
        key="invoice",
        name="开具发票",
        required=(SLOT_ORDER_ID,),
        optional=(SLOT_INVOICE_TITLE,),
        guide={
            SLOT_ORDER_ID: "请提供订单号，我帮您申请开票。",
            SLOT_INVOICE_TITLE: "抬头开个人还是单位？单位需要提供名称和税号。",
        },
        aliases=("开具发票", "开发票", "要发票"),
    ),
    "create_ticket": TaskDef(
        key="create_ticket",
        name="转人工工单",
        required=(SLOT_QUESTION,),
        guide={SLOT_QUESTION: "请简单描述一下遇到的问题，我帮您记录并转给人工客服。"},
        aliases=("投诉", "转人工", "人工"),
    ),
}

#: 咨询类意图 → 走**政策问答**，不要求任何槽位。
#: 这类问题按规矩不该拦着用户要订单号（"退货运费谁承担"本身不依赖具体订单）。
CONSULT_INTENTS: Dict[str, str] = {
    "运费咨询": "退换货运费如何承担",
    "优惠咨询": "优惠券使用规则",
    "保修咨询": "保修与售后范围",
    "政策咨询": "平台售后政策",
    "支付咨询": "支付与发票问题",
}


#: 明确表示"我要办这件事"的动作词
ACTION_MARKERS = (
    "我要", "我想", "帮我退", "帮我查", "帮我改", "帮我开", "给我退",
    "申请退", "申请换", "申请开", "麻烦退", "麻烦帮", "需要退货", "要退货",
    "退货吧", "我要退", "帮我处理",
)

#: 明显是在问"规则/政策"的说法。
#: 注意判据是"问句信号 **且** 没有动作词"，所以哪怕单独一个"吗"也不会误伤——
#: "我要退货吗"这种怪句子若真出现，动作词优先，仍然会去办事。
POLICY_MARKERS = (
    "谁承担", "谁出", "谁付", "怎么算", "怎么收", "怎么", "多久", "几天", "多长时间",
    "一般", "通常", "可以吗", "能吗", "行不行", "能不能", "是否", "规定",
    "政策", "规则", "支持吗", "支持么", "支持不", "流程", "标准", "条件",
    "能不能退", "还能退", "可以退吗", "吗", "呢", "兑换", "怎么弄",
)


def task_for_intent(intent: str) -> Optional[TaskDef]:
    """把归一化后的意图映射到任务定义。

    匹配顺序：任务 key → 别名精确匹配 → 别名的子串包含。
    意图可能形如"退货、查询物流"（多意图），这里只取**第一个能匹配上的**，
    多意图的拆分由 ``state.py`` 的待办栈负责。
    """
    if not intent:
        return None
    text = intent.strip()
    if text in TASKS:
        return TASKS[text]
    for task in TASKS.values():
        if text == task.name or text in task.aliases:
            return task
    for task in TASKS.values():
        for alias in task.aliases:
            if alias and alias in text:
                return task
    return None


#: 指代词：出现"这个/那笔"这类说法时，用户问的是**具体这一单**，不是在问政策。
REFERENCE_MARKERS = (
    "这个", "这单", "这笔", "那单", "那笔", "该订单", "此订单", "刚才那", "上面那", "这件",
)


def has_action_marker(text: str) -> bool:
    """用户是否明确表达了"要办这件事"（而不是问规则）。"""
    return any(marker in (text or "") for marker in ACTION_MARKERS)


def has_reference_marker(text: str) -> bool:
    """用户是否用了指代词（说明问的是具体单据，不是政策）。"""
    return any(marker in (text or "") for marker in REFERENCE_MARKERS)


def is_policy_question(text: str) -> bool:
    """用户是否在问**规则/政策**。

    为什么要单独判断：意图抽取很容易把"退货运费谁承担"也抽成"退货"，
    如果只看意图就会去索要订单号——用规则问题拦住用户是体验上的硬伤。
    """
    return any(marker in (text or "") for marker in POLICY_MARKERS)


def is_consultation(text: str) -> bool:
    """规则咨询 = 在问政策 **且** 没有要求办事 **且** 没有指代具体单据。

    三重条件缺一不可：
    * "退货运费一般谁承担？" → 纯咨询，只回答，不索要单号；
    * "我要退货" → 明确要办事，追问订单号；
    * "那这个能退吗？" → 有指代，必须澄清指哪一单（不能当成问政策放过）。
    """
    return (
        is_policy_question(text)
        and not has_action_marker(text)
        and not has_reference_marker(text)
    )


def split_intents(intent_value: str) -> List[str]:
    """把"退货、查询物流"这类复合诉求拆成单个意图，保持出现顺序并去重。"""
    if not intent_value:
        return []
    parts: List[str] = []
    for chunk in intent_value.replace("，", "、").replace(",", "、").replace("和", "、").split("、"):
        item = chunk.strip()
        if item and item not in parts:
            parts.append(item)
    return parts


def required_slots(task: TaskDef) -> List[str]:
    return list(task.required)


def missing_required(task: TaskDef, filled: Dict[str, object]) -> List[str]:
    """返回该任务**还没拿到值**的硬槽位，保持任务定义里的顺序（顺序即追问优先级）。"""
    return [
        slot
        for slot in task.required
        if not str(filled.get(slot) or "").strip()
    ]
