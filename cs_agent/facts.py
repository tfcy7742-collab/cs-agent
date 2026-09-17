"""事实抽取：从用户消息里提取"可复用的事实"，并区分哪些能跨会话。

设计要点（这是 P2 里最容易被做歪的地方）：

1. **正则先行、LLM 补漏**。订单号、手机号这类事实有明显格式，用正则抽取**不会幻觉**，
   而且离线也能用；LLM 只负责正则覆盖不到的语义信息（"我叫张先生"、"我要退货"）。
   两者产出的每条事实都带 ``source``（regex / llm），便于追溯是谁抽的。
2. **版本化**。同一条事实被用户改口后，新值**覆盖**旧值但保留 ``previous``。
   新旧并列会让模型随机挑一个（长会话里最隐蔽的 bug）。
3. **跨会话白名单**。"称呼""城市"这类稳定信息才进长期画像；
   订单号、诉求属于**单会话**信息，绝不能带到下一次会话——
   否则用户开新会话问"我的订单状态"，模型会拿上一单的订单号去查。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .llm import LLMClient, LLMError

#: 只进会话、不进长期画像的事实（都属于"这一单/这一次"）
SESSION_ONLY_KEYS = {"订单号", "诉求", "退货原因", "期望处理方式", "涉及金额"}

#: 允许沉淀为跨会话画像的事实（稳定、且对下次服务有用）
PROFILE_ALLOWED_KEYS = {"称呼", "城市", "联系方式", "会员等级", "沟通偏好", "常用收货地址"}

#: 事实在提示词里的展示顺序（先身份、再诉求、最后单号）
FACT_ORDER = [
    "称呼",
    "城市",
    "联系方式",
    "会员等级",
    "沟通偏好",
    "常用收货地址",
    "订单号",
    "诉求",
    "退货原因",
    "涉及金额",
    "期望处理方式",
]

# ---------------------------------------------------------------------------
# 正则规则：只放"格式明确、几乎不会误判"的
# ---------------------------------------------------------------------------
_PATTERNS: List[Tuple[str, re.Pattern]] = [
    # 订单号：显式前缀 + 数字；或用户明确说"订单号是 xxx"
    (
        "订单号",
        re.compile(
            r"(?:订单号|订单编号|单号|订单)\s*(?:是|为|：|:|no\.?|#)?\s*"
            r"([A-Za-z]{0,4}\d{6,20})",
            re.IGNORECASE,
        ),
    ),
    # 手机号（中国大陆）
    ("联系方式", re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")),
    # 称呼
    (
        "称呼",
        re.compile(r"(?:我叫|我是|我姓|你可以叫我)\s*([\u4e00-\u9fff]{1,4}(?:先生|女士|小姐|老师)?)"),
    ),
    # 会员等级
    ("会员等级", re.compile(r"(黑卡|金卡|银卡|钻石|白金|黄金|普通会员|VIP\d?)")),
    # 沟通偏好
    ("沟通偏好", re.compile(r"(?:请|希望|麻烦)?\s*(?:用|以)?\s*(简洁|简短|详细|正式|口语化|别啰嗦)\s*(?:一点|些|的)?(?:方式|语气|说|回复)")),
]

#: 退货原因：影响"运费由谁承担"，是软槽位里最有价值的一个。
#: 用关键词表而不是泛化正则——宁可漏抽，也不能把用户随口一句话当成退货原因。
REASON_KEYWORDS: List[str] = [
    "质量问题",
    "质量有问题",
    "有质量问题",
    "有瑕疵",
    "瑕疵",
    "破损",
    "坏了",
    "用不了",
    "发错",
    "漏发",
    "少件",
    "不符",
    "与描述不符",
    "色差",
    "尺码不合适",
    "尺码不对",
    "尺寸不对",
    "太小",
    "太大",
    "不喜欢",
    "不想要了",
    "七天无理由",
    "无理由",
    "买错了",
    "多买了",
]

#: 商品名称：只认"买的/退的/购入的 XXX"这类明确表达。
#: 写法是"先贪心取一段，再回退到最近的终止符"——比 lookahead 更直观，
#: 也能正确处理"要退订单里的运动鞋"（保留"订单里的"）与
#: "我买的蓝色卫衣不合适"（在"不合适"前截断）。
_ITEM_PREFIX = re.compile(r"(?:买的|买的是|退的|要退|购入|下单的|商品是)\s*([\u4e00-\u9fffA-Za-z0-9]{2,16})")
_ITEM_STOPS = (
    "不合适", "不能用", "有问题", "坏了", "破损", "想退", "要退", "退货", "退款", "换货",
    "不想要", "不喜欢", "太小", "太大",
)

#: 诉求类关键词 → 归一化标签
INTENT_KEYWORDS: List[Tuple[str, str]] = [
    ("退货", "退货"),
    ("能退吗", "退货"),
    ("能退不", "退货"),
    ("可以退吗", "退货"),
    ("能不能退", "退货"),
    ("能退货", "退货"),
    ("退换", "退换货"),
    ("换货", "换货"),
    ("退款", "退款"),
    ("投诉", "投诉"),
    ("催发货", "催发货"),
    ("催", "催发货"),
    ("没发货", "催发货"),
    ("物流", "查询物流"),
    ("快递", "查询物流"),
    ("到哪", "查询物流"),
    ("查订单", "查询订单"),
    ("看看这个订单", "查询订单"),
    ("看下这个订单", "查询订单"),
    ("订单详情", "查询订单"),
    ("订单状态", "查询订单"),
    ("发票", "开具发票"),
    ("优惠", "优惠咨询"),
    ("运费", "运费咨询"),
    ("保修", "保修咨询"),
]

#: 意图识别里的"正则型"关键词。
#: 用正则是为了容忍中文的自由语序——"改地址 / 改收货地址 / 把地址改成…"
#: 都在表达同一诉求，纯子串匹配会漏掉中间插词的写法（实测漏过"改收货地址"）。
INTENT_PATTERNS: List[Tuple[str, "re.Pattern"]] = [
    ("修改地址", re.compile(r"(改|修改|换).{0,4}(收货)?地址|地址.{0,4}(改|修改|换|填错)")),
    ("查询物流", re.compile(r"(物流|快递|包裹).{0,4}(到哪|在哪|状态|进度|怎么|查)|催.{0,3}发货")),
    ("查询订单", re.compile(r"(查|看).{0,3}(订单|单子)|订单.{0,3}(详情|状态|情况)")),
    ("退货", re.compile(r"(要|想|申请|办理|提交).{0,3}退|能不能退|可以退吗|能退吗")),
    ("退款", re.compile(r"(退款|退钱).{0,4}(进度|到账|状态|到哪)|什么时候.{0,3}退款")),
    ("开具发票", re.compile(r"(开|要|补).{0,3}发票|发票.{0,3}(怎么|开|抬头)")),
    ("投诉", re.compile(r"(投诉|举报|曝光)")),
]

#: 常见城市（用于"我在杭州"这类抽取；不求全，求准）
_KNOWN_CITIES = [
    "北京", "上海", "广州", "深圳", "杭州", "南京", "成都", "重庆", "武汉", "西安",
    "苏州", "天津", "长沙", "郑州", "青岛", "宁波", "厦门", "福州", "合肥", "济南",
    "大连", "沈阳", "昆明", "无锡", "佛山", "东莞", "石家庄", "太原", "南昌", "贵阳",
]


def extract_by_rules(text: str) -> Dict[str, Dict[str, Any]]:
    """正则抽取：高置信、可离线、不会幻觉。"""
    facts: Dict[str, Dict[str, Any]] = {}
    if not text:
        return facts

    for key, pattern in _PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        value = match.group(1).strip()
        if not value:
            continue
        # 订单号统一大写，避免 so123 与 SO123 被当成两个值
        if key == "订单号":
            value = value.upper()
        facts[key] = {"value": value, "source": "regex", "evidence": match.group(0).strip()}

    # 诉求：多关键词命中时全部保留（用户可能一次说两件事）
    intents: List[str] = []
    for keyword, label in INTENT_KEYWORDS:
        if keyword in text and label not in intents:
            intents.append(label)
    for label, pattern in INTENT_PATTERNS:
        if label not in intents and pattern.search(text):
            intents.append(label)
    if intents:
        facts["诉求"] = {
            "value": "、".join(intents),
            "source": "regex",
            "evidence": "关键词命中：" + "、".join(intents),
        }

    # 城市：只认"我在/我在的城市是"这种明确表达，避免把目的地城市误当用户所在地
    city_match = re.search(
        r"(?:我|我们)(?:现在|目前|最近)?(?:在|住在|搬到|定居在|这边是|收货在)\s*([\u4e00-\u9fff]{2,4})",
        text,
    )
    if city_match:
        city = city_match.group(1)
        for known in _KNOWN_CITIES:
            if known in city:
                facts["城市"] = {
                    "value": known,
                    "source": "regex",
                    "evidence": city_match.group(0).strip(),
                }
                break

    # 涉及金额
    amount_match = re.search(r"(\d{1,6}(?:\.\d{1,2})?)\s*(?:元|块钱|块)", text)
    if amount_match:
        facts["涉及金额"] = {
            "value": f"{amount_match.group(1)} 元",
            "source": "regex",
            "evidence": amount_match.group(0).strip(),
        }

    # 退货原因（软槽位，但直接影响运费判定）
    for reason in REASON_KEYWORDS:
        if reason in text:
            facts["退货原因"] = {"value": reason, "source": "regex", "evidence": reason}
            break

    # 商品名称（只在明确表达时抽取）
    item_match = _ITEM_PREFIX.search(text)
    if item_match:
        item = item_match.group(1).strip()
        # 回退到最近的终止符："蓝色卫衣不合适" → "蓝色卫衣"
        cut = len(item)
        for stop in _ITEM_STOPS:
            index = item.find(stop)
            if index > 0:
                cut = min(cut, index)
        item = item[:cut].strip("的 ")
        # 排除把诉求词当成商品名的情况（"我要退货" 里的"退货"）
        if len(item) >= 2 and item not in {"退货", "退款", "换货", "退换", "发票", "地址", "订单里"}:
            facts["商品名称"] = {
                "value": item,
                "source": "regex",
                "evidence": item_match.group(0).strip(),
            }

    return facts


LLM_SYSTEM_PROMPT = """你是一个信息抽取器。从用户的客服对话中抽取**客观事实**，只输出 JSON。

要求：
1. 只抽取用户**明确说过**的信息，绝不推测、绝不补全；
2. 没有提到的字段就不要出现在 JSON 里（不要填空值、不要编造）；
3. 键名只能从下列集合里选：
   称呼 / 城市 / 联系方式 / 会员等级 / 沟通偏好 / 常用收货地址 / 订单号 / 诉求 / 退货原因 / 涉及金额 / 期望处理方式
4. 「诉求」用简短名词短语概括，例如"退货""查询物流""投诉"；
5. 只输出 JSON，不要 Markdown 代码块，不要解释。
"""


def extract_with_llm(
    text: str,
    llm: LLMClient,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """LLM 语义抽取：补正则覆盖不到的表达。失败时返回空字典（不阻断主流程）。"""
    if not llm.online or not (text or "").strip():
        return {}

    messages: List[Dict[str, str]] = [{"role": "system", "content": LLM_SYSTEM_PROMPT}]
    for item in (history or [])[-6:]:
        role = item.get("role")
        content = (item.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append(
        {
            "role": "user",
            "content": f"请从下面这条用户消息中抽取事实，只输出 JSON：\n{text}",
        }
    )

    try:
        result = llm.chat(messages, temperature=0.0, json_mode=True)
    except LLMError:
        return {}

    payload = _loads_lenient(result.text)
    if not isinstance(payload, dict):
        return {}

    facts: Dict[str, Dict[str, Any]] = {}
    allowed = set(FACT_ORDER)
    for key, value in payload.items():
        key = str(key).strip()
        if key not in allowed:
            continue
        if isinstance(value, (list, tuple)):
            value = "、".join(str(v) for v in value if str(v).strip())
        value = str(value).strip()
        if not value or value in {"未知", "无", "null", "None", "-"}:
            continue
        facts[key] = {"value": value[:120], "source": "llm", "evidence": text[:80]}
    return facts


def _loads_lenient(text: str) -> Any:
    """宽松解析模型返回的 JSON（容忍 Markdown 代码块与前后废话）。"""
    raw = (text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = raw.strip("`")
        if "\n" in raw:
            raw = raw.split("\n", 1)[1]
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None


def merge_extractions(
    rules: Dict[str, Dict[str, Any]],
    semantic: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """合并两路抽取结果：**正则优先**。

    正则是确定性抽取（订单号抽错的可能性远低于模型），因此同一键冲突时以正则为准。
    """
    merged: Dict[str, Dict[str, Any]] = dict(semantic)
    merged.update(rules)
    return merged


def select_profile_updates(facts: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """从会话事实里挑出可以沉淀为**跨会话画像**的部分。"""
    return {
        key: str(payload.get("value"))
        for key, payload in facts.items()
        if key in PROFILE_ALLOWED_KEYS and payload.get("value")
    }


def render_facts(facts: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """把版本化事实渲染成适合放进提示词的扁平键值。

    * 当前值直接展示；
    * 发生过改口的，额外附注"（此前为 X）"——让模型知道用户改过口，
      而不是把两个值都当成有效值。
    """
    rendered: Dict[str, str] = {}
    keys = [k for k in FACT_ORDER if k in facts] + [
        k for k in facts if k not in FACT_ORDER
    ]
    for key in keys:
        payload = facts.get(key) or {}
        value = payload.get("value")
        if not value:
            continue
        previous = payload.get("previous")
        if previous and previous != value:
            rendered[key] = f"{value}（此前为 {previous}）"
        else:
            rendered[key] = str(value)
    return rendered
