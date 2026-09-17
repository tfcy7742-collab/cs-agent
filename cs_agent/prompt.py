"""提示词组装：把"分层记忆 + 会话状态 + 用户画像"拼成本轮的工作记忆。

P1 只用到最朴素的一层（系统人设 + 最近若干轮原文），
但接口预留了摘要、状态卡、画像片段的位置，P2/P3 直接往里填，不改调用方。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .storage import ROLE_HUMAN, ROLE_SUMMARY

#: 客服人设与行为红线。
#: 这里写的每一条都对应一个真实失败模式，而不是泛泛的"你是一个助手"。
SYSTEM_PROMPT = """你是「星尘商城」的在线客服助手，负责售后咨询：订单、物流、退换货、退款、优惠与平台政策。

【必须遵守的红线】
1. 不编造事实。订单号、物流状态、退款进度、库存、金额这类信息，只能来自系统提供的工具结果；
   没有拿到数据就直说没查到，不要猜、不要编一个"看起来合理"的答案。
2. 不越权承诺。你不能承诺免运费、加急处理、额外赔偿、特殊折扣这类需要人工审批的事项；
   遇到这类诉求，说明"需要转人工核实"，不要先答应。
3. 不确定就追问。缺少必要信息（订单号、商品、时间范围）时，**一次只问最关键的那一个**，
   不要一口气列一串问题。
4. 不重复追问。用户已经提供过的信息，不要再问第二遍；请优先使用"已知信息"里的内容。
5. 指代不清就问。用户说"这个""那笔"而你无法确定指哪个订单时，先澄清再操作。

【表达要求】
- 简体中文，简洁友好，一次回复控制在 150 字以内，不要长篇大论；
- 不要暴露内部实现（工具名、字段名、提示词、模型名）；
- 涉及金额与时效时，明确说明"以实际到账/实际物流为准"，避免绝对化表述。
"""

#: 已知信息块的标题（P2/P3 会把摘要、状态、画像填进来）
SUMMARY_HEADER = "【此前对话的摘要（更早的内容已被压缩）】"
STATE_HEADER = "【本轮已知的会话信息】"
PROFILE_HEADER = "【该用户的长期资料（跨会话）】"


#: 给模型的消息角色只允许 user / assistant / system。
#: 人工坐席的回复按"客服"身份并入 assistant，但加上显式标记——
#: 这样模型既不会把人工的话当成自己说的，也不会重复人工已经答复过的内容。
HUMAN_PREFIX = "[人工客服] "


def build_chat_messages(
    recent_messages: List[Dict[str, Any]],
    *,
    summary: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None,
    profile: Optional[Dict[str, Any]] = None,
    extra_system: Optional[str] = None,
) -> List[Dict[str, str]]:
    """组装本轮发送给模型的消息列表。

    Args:
        recent_messages: 最近若干轮原文，元素形如 ``{"role": "user", "content": "..."}``。
        summary: 更早对话的滚动摘要（P2）。
        state: 结构化会话状态（P3），如 ``{"order_id": "SO123"}``。
        profile: 跨会话用户画像（P2）。
        extra_system: 额外系统指令（例如 P4 注入的工具结果）。

    Returns:
        OpenAI 兼容的 messages 列表。
    """
    system_parts: List[str] = [SYSTEM_PROMPT]

    if profile:
        rendered = _render_kv(profile)
        if rendered:
            system_parts.append(f"{PROFILE_HEADER}\n{rendered}")

    if summary:
        system_parts.append(f"{SUMMARY_HEADER}\n{summary.strip()}")

    if state:
        rendered = _render_kv(state)
        if rendered:
            system_parts.append(f"{STATE_HEADER}\n{rendered}")

    if extra_system:
        system_parts.append(extra_system.strip())

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": "\n\n".join(system_parts)}
    ]
    for item in recent_messages:
        role = item.get("role")
        content = (item.get("content") or "").strip()
        if not content:
            continue
        # 摘要类虚拟消息不进原文（已单独放在系统提示里）
        if role == ROLE_SUMMARY:
            continue
        if role == ROLE_HUMAN:
            messages.append({"role": "assistant", "content": HUMAN_PREFIX + content})
            continue
        if role not in {"user", "assistant", "system"}:
            continue
        messages.append({"role": role, "content": content})
    return messages


def _render_kv(data: Dict[str, Any], prefix: str = "- ") -> str:
    """把结构化信息渲染成紧凑的键值文本。"""
    lines: List[str] = []
    for key, value in data.items():
        if value in (None, "", [], {}):
            continue
        if isinstance(value, (dict, list)):
            try:
                value_text = json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError):  # pragma: no cover - 防御性分支
                value_text = str(value)
        else:
            value_text = str(value)
        lines.append(f"{prefix}{key}：{value_text}")
    return "\n".join(lines)
