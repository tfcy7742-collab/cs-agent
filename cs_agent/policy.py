"""平台政策知识库与检索。

⚠️ 这里的政策条文是**为演示编写的示例文本**，不代表任何真实电商平台的实际规则。

检索实现刻意用"中文二元组 + IDF"而不是向量检索：
* 政策条文只有十几条，规模和"两个字的差异就决定答案不同"的场景，
  关键词检索的**可解释性**比语义召回更重要（能说清"命中了哪个关键词"）；
* 零依赖、可离线、结果确定，便于测试；
* 局限也很明确——同义改写召回不了（"运费谁出" vs "运费承担"），
  这一点在 README 里如实写明，不装作是语义检索。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


@dataclass(frozen=True)
class PolicyDoc:
    """一条政策。"""

    doc_id: str
    title: str
    text: str
    keywords: Tuple[str, ...] = ()
    category: str = "售后"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "text": self.text,
            "category": self.category,
            "source": f"平台售后政策 · {self.title}",
        }


#: 演示政策库
POLICIES: List[PolicyDoc] = [
    PolicyDoc(
        doc_id="POL-001",
        title="七天无理由退货",
        category="退货",
        keywords=("七天无理由", "无理由退货", "退货时限", "多久内可退"),
        text=(
            "自签收之日起 7 个自然日内，商品完好、不影响二次销售的，可申请七天无理由退货。"
            "「商品完好」指商品本身、配件、吊牌、包装、赠品齐全，且无使用痕迹。"
            "超过 7 天后不再支持无理由退货，但因质量问题产生的退货不受此限。"
        ),
    ),
    PolicyDoc(
        doc_id="POL-002",
        title="退货运费承担规则",
        category="运费",
        keywords=("运费", "退货运费", "谁承担", "运费谁出", "运费险"),
        text=(
            "退货运费按原因区分：因质量问题、错发、漏发、破损等商家责任产生的退货，"
            "运费由商家承担；因七天无理由、个人原因（不喜欢、买错、尺码不合）产生的退货，"
            "运费由买家承担。订单含运费险的，可按运费险条款获得赔付，具体金额以保险理赔结果为准。"
        ),
    ),
    PolicyDoc(
        doc_id="POL-003",
        title="退款到账时效",
        category="退款",
        keywords=("退款", "到账", "多久到账", "退款时效", "几天到账"),
        text=(
            "退货商品经仓库签收并质检通过后，退款将在 1-3 个工作日内发起；"
            "发起后原路退回，到账时间取决于支付渠道：余额一般即时到账，"
            "银行卡/信用卡通常 1-7 个工作日，具体以银行入账时间为准。"
            "若超时未到账，可提供支付流水申请人工核查。"
        ),
    ),
    PolicyDoc(
        doc_id="POL-004",
        title="不可退换的商品范围",
        category="退货",
        keywords=("不能退", "不可退", "定制", "拆封", "特殊商品"),
        text=(
            "以下商品不支持七天无理由退货：定制类商品（刻字、改尺寸、专属图案）、"
            "已拆封的音像制品与软件、鲜活易腐、已激活的数码产品、"
            "贴身衣物与个人护理用品（拆封后）。上述商品若存在质量问题，仍可申请退换。"
        ),
    ),
    PolicyDoc(
        doc_id="POL-005",
        title="换货流程",
        category="换货",
        keywords=("换货", "换一件", "换尺码", "换颜色"),
        text=(
            "支持换货的情形：尺码/颜色不合适、商品存在质量问题。"
            "换货需在签收后 7 个自然日内提交申请，同款同价可直接换，"
            "价差需补足或退回。换货商品寄回后，新商品一般在仓库签收后 1-3 个工作日发出。"
        ),
    ),
    PolicyDoc(
        doc_id="POL-006",
        title="物流时效与查询",
        category="物流",
        keywords=("物流", "快递", "多久到", "发货", "催发货", "到货时间"),
        text=(
            "现货商品一般在下单后 48 小时内发出；预售商品以商品页标注的发货时间为准。"
            "发货后可通过订单详情页查看实时物流轨迹。"
            "物流信息长时间未更新（超过 48 小时）或出现异常，可申请人工催件核查。"
        ),
    ),
    PolicyDoc(
        doc_id="POL-007",
        title="发票开具规则",
        category="发票",
        keywords=("发票", "开票", "抬头", "税号", "电子发票"),
        text=(
            "订单完成后可在订单详情页申请发票，支持电子发票与纸质发票。"
            "个人抬头填写姓名即可；单位抬头需提供单位名称与纳税人识别号。"
            "电子发票一般在申请后 1-3 个工作日发送至预留邮箱；"
            "发票内容默认按商品明细开具，如需调整请在申请时备注。"
        ),
    ),
    PolicyDoc(
        doc_id="POL-008",
        title="优惠券使用与退回",
        category="优惠",
        keywords=("优惠券", "券", "用不了", "过期", "折扣"),
        text=(
            "优惠券需同时满足使用门槛、适用商品范围与有效期才可用，"
            "具体以结算页显示为准。订单发生退货时，已使用的优惠券在有效期内会原路退回，"
            "已过期的优惠券不再补发。满减活动按退货后的实际支付金额重新计算，"
            "可能出现部分优惠被收回的情况。"
        ),
    ),
    PolicyDoc(
        doc_id="POL-009",
        title="修改收货地址",
        category="订单",
        keywords=("改地址", "修改地址", "换地址", "地址填错"),
        text=(
            "未发货订单可直接在订单详情页修改收货地址，修改后立即生效。"
            "已发货订单无法直接修改，可联系承运商尝试拦截改派，"
            "成功率取决于包裹当前所处环节，不保证成功；若拦截失败需拒收后重新下单。"
        ),
    ),
    PolicyDoc(
        doc_id="POL-010",
        title="价保与降价补差",
        category="价格",
        keywords=("价保", "降价", "补差价", "买贵了"),
        text=(
            "自下单之日起 15 个自然日内，若同一商品在同一平台出现直接降价，"
            "可申请价保补差。价保不适用于优惠券、满减、限时秒杀等促销形式导致的价格变化，"
            "同一订单仅可申请一次价保。"
        ),
    ),
    PolicyDoc(
        doc_id="POL-011",
        title="上门取件服务",
        category="退货",
        keywords=("上门取件", "取件", "寄回", "自己寄"),
        text=(
            "大部分城市支持退货上门取件，可在提交退货申请时选择时间段。"
            "是否支持、可选时段与是否收费以提交申请时页面实际显示为准。"
            "若所在地不支持上门取件，需自行寄回，寄回后请保留运单号以便追踪。"
        ),
    ),
    PolicyDoc(
        doc_id="POL-012",
        title="保修与售后维修",
        category="保修",
        keywords=("保修", "维修", "坏了", "故障", "三包"),
        text=(
            "数码类商品自签收之日起提供 12 个月厂商保修，"
            "非人为损坏可申请免费维修；人为损坏、进液、私自拆修不在保修范围内。"
            "保修需提供订单号与故障描述，寄修往返运费在保修范围内由平台承担。"
        ),
    ),
]

#: 停用词/语气词（对区分政策毫无帮助，却会拉高所有文档的相似度）
_STOPWORDS: Set[str] = {
    "的", "了", "吗", "呢", "吧", "啊", "是", "我", "你", "他", "她", "它",
    "这", "那", "个", "们", "有", "在", "和", "与", "及", "或", "请", "想",
    "要", "会", "能", "可以", "怎么", "什么", "如何", "一下", "请问", "谢谢",
    "你 们", "帮我", "麻烦", "现在", "然后", "还有", "就是", "应该",
}


def _normalise(text: str) -> str:
    return re.sub(r"[\s，。！？、；：,.!?;:（）()\[\]「」【】\"'`~]+", "", (text or "").lower())


def _bigrams(text: str) -> List[str]:
    """中文二元组：不依赖分词，2 字以上的词大多能被覆盖。"""
    cleaned = _normalise(text)
    if len(cleaned) < 2:
        return [cleaned] if cleaned else []
    return [cleaned[i : i + 2] for i in range(len(cleaned) - 1)]


@dataclass
class PolicyHit:
    """一条命中结果（带可核对的依据）。"""

    doc: PolicyDoc
    score: float
    matched_terms: List[str] = field(default_factory=list)
    highlight: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc.doc_id,
            "title": self.doc.title,
            "category": self.doc.category,
            "text": self.doc.text,
            "source": f"平台售后政策 · {self.doc.title}",
            "score": round(self.score, 4),
            "matched_terms": self.matched_terms,
            "highlight": self.highlight,
        }


class PolicyIndex:
    """政策检索索引：关键词命中 + 二元组 IDF 打分。"""

    def __init__(self, docs: Optional[Iterable[PolicyDoc]] = None) -> None:
        self.docs: List[PolicyDoc] = list(docs or POLICIES)
        self._doc_terms: List[Set[str]] = []
        self._doc_bigrams: List[Set[str]] = []
        self._keyword_map: Dict[str, List[int]] = {}

        for index, doc in enumerate(self.docs):
            self._doc_terms.append(set(doc.keywords) | {doc.title} | {doc.category})
            self._doc_bigrams.append(set(_bigrams(f"{doc.title}{doc.text}")) | set(_bigrams("".join(doc.keywords))))
            for keyword in doc.keywords:
                self._keyword_map.setdefault(keyword, []).append(index)

        # 二元组 IDF：出现得越普遍，区分度越低
        total = max(len(self.docs), 1)
        df: Dict[str, int] = {}
        for grams in self._doc_bigrams:
            for gram in grams:
                df[gram] = df.get(gram, 0) + 1
        self._idf: Dict[str, float] = {
            gram: math.log((total + 1) / (count + 0.5)) for gram, count in df.items()
        }

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 3, min_score: float = 0.0) -> List[PolicyHit]:
        """检索政策。

        Args:
            query: 用户问题。
            top_k: 返回条数。
            min_score: 低于该分数不返回——**宁可返回空，也不给不相关的政策**，
                因为把不相关政策当成答案比说"没查到"更糟。

        Returns:
            按分数降序的命中列表。
        """
        cleaned = _normalise(query)
        if not cleaned:
            return []

        query_terms = set(_bigrams(query))
        hits: List[PolicyHit] = []
        for index, doc in enumerate(self.docs):
            score = 0.0
            matched: List[str] = []

            # 1) 关键词命中（最强信号）
            for keyword in doc.keywords:
                if _normalise(keyword) and _normalise(keyword) in cleaned:
                    score += 3.0
                    matched.append(keyword)
            if _normalise(doc.title) and _normalise(doc.title) in cleaned:
                score += 4.0
                matched.append(doc.title)

            # 2) 二元组加权（IDF 越高越有区分度）
            overlap = query_terms & self._doc_bigrams[index]
            gram_score = sum(self._idf.get(gram, 0.0) for gram in overlap)
            score += gram_score * 0.6

            if score <= 0:
                continue
            hits.append(
                PolicyHit(
                    doc=doc,
                    score=score,
                    matched_terms=sorted(set(matched), key=len, reverse=True)[:5],
                    highlight=self._highlight(doc, matched, overlap),
                )
            )

        hits.sort(key=lambda hit: hit.score, reverse=True)
        return [hit for hit in hits if hit.score > min_score][:top_k]

    @staticmethod
    def _highlight(doc: PolicyDoc, matched: List[str], overlap: Set[str]) -> str:
        """摘一句最相关的原文，便于用户/审计核对依据。"""
        if matched:
            return f"命中关键词：{'、'.join(matched[:3])}"
        if overlap:
            sample = "、".join(sorted(overlap)[:4])
            return f"文本片段匹配：{sample}"
        return ""

    # ------------------------------------------------------------------
    def by_id(self, doc_id: str) -> Optional[PolicyDoc]:
        for doc in self.docs:
            if doc.doc_id == doc_id:
                return doc
        return None

    def count(self) -> int:
        return len(self.docs)


_index: Optional[PolicyIndex] = None


def get_index() -> PolicyIndex:
    global _index
    if _index is None:
        _index = PolicyIndex()
    return _index
