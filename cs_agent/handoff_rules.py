"""转人工兜底：可量化的触发规则 + 结构化交接单。

为什么不能"让模型自己决定要不要转人工"：

* **不稳定**：同一个诉求，模型可能这次说"我帮您转"、下次继续自己答；
* **不可测**：无法写测试，也无法解释"为什么这通会话转了人工"；
* **代价高**：该转不转，用户重复投诉；不该转乱转，人工成本被浪费。

所以判定放在**规则层**，每条规则都有名字、权重与可读理由，进轨迹可追溯。
模型只负责"怎么把转接这件事说清楚"，不负责"要不要转"。

五条规则：

| 规则 | 触发条件 | 权重 |
| --- | --- | --- |
| ``explicit_request`` | 用户明确要求转人工 | 立即转 |
| ``out_of_scope`` | 要求超出客服权限（赔偿/加急/免运费…） | 立即转（不能自己答应） |
| ``negative_sentiment`` | 本会话负面情绪累计 | 阈值 |
| ``repeat_unresolved`` | 同一诉求反复提及且没解决 | 阈值 |
| ``answer_loop`` | 连续多轮答非所问（用户重复＋没有工具佐证） | 阈值 |
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# 情绪词典（规则近似，不是情感模型——这一点在 README 里写明）
# ---------------------------------------------------------------------------
NEGATIVE_WORDS: Dict[str, int] = {
    # 强度 3：明确的强烈不满
    "投诉": 3, "太差了": 3, "垃圾": 3, "骗子": 3, "欺x": 3, "曝光": 3, "315": 3,
    "消费者协会": 3, "起诉": 3, "报警": 3, "再也不买": 3, "退一赔三": 3,
    # 强度 2：明显不满
    "生气": 2, "气死": 2, "愤怒": 2, "过分": 2, "离谱": 2, "什么态度": 2,
    "第三次": 2, "很多次": 2, "好几次": 2, "一直没": 2, "还没解决": 2, "没用": 2,
    "不满意": 2, "差评": 2, "无语": 2, "服了": 2, "到底": 2, "怎么回事": 2,
    # 强度 1：轻度烦躁
    "急": 1, "等很久": 1, "太慢": 1, "慢": 1, "有点慢": 1, "这么慢": 1, "催": 1,
    "麻烦": 1, "又": 1, "还没": 1, "怎么回事呢": 1, "烦": 1, "气": 1,
    "几天": 1, "几天了": 1, "好几天": 1, "这么久": 1,
}

#: 超出客服权限的诉求：不能自己答应，必须转人工核实
OUT_OF_SCOPE_PATTERNS: List[tuple] = [
    (re.compile(r"(赔偿|赔钱|赔付|补偿|赔我|三倍|十倍|退一赔三)"), "涉及赔偿，超出客服权限"),
    (re.compile(r"(免运费|免邮|包邮到|运费免了|给我免)"), "涉及运费减免，超出客服权限"),
    (re.compile(r"(加急|优先处理|插队|马上发|立刻发|催一下仓库)"), "涉及加急，需人工核实"),
    (re.compile(r"(额外|特殊).{0,4}(折扣|优惠|券|补偿)"), "涉及额外优惠，超出客服权限"),
    (re.compile(r"(举报|起诉|律师|12315|工商|消协|曝光)"), "涉及投诉升级，需人工介入"),
    (re.compile(r"((金额|价格|价).{0,4}(改|调|少收)|(改|调).{0,4}(金额|价格))"), "涉及金额调整，超出客服权限"),
    # 伪造权限：用户自称是管理员/已获授权，要求跳过流程。**绝不能采信用户自述的权限**，
    # 一律转人工核实。（实测：没有这条规则时会去追问订单号，等于准备照办。）
    (
        re.compile(
            r"(我是|作为).{0,6}(管理员|客服|内部人员|官方|运营|老板)"
            r"|已(经)?(授权|批准|审批)"
            r"|(不用|无需|不必).{0,4}(确认|核实|审核)"
            r"|(跳过|绕过).{0,4}(流程|确认|审核)"
        ),
        "声称已获授权或要求跳过核实流程，必须转人工确认",
    ),
]

#: 明确要求转人工
EXPLICIT_PATTERNS: List[re.Pattern] = [
    re.compile(r"(转人工|找人工|人工客服|真人|转接客服|叫客服|要客服)"),
    re.compile(r"(找|要|叫|请|让).{0,4}(经理|主管|负责人|领导|店长)"),
    re.compile(r"(不要机器人|别用机器人|机器人别回|只能人工)"),
]

#: 表示"事情还没解决"的说法（配合重复意图用于 answer_loop）
UNRESOLVED_MARKERS = (
    "还没解决", "没解决", "没处理", "没有处理", "一直没", "还是没", "仍然没",
    "第.{0,3}次", "又没", "还没好", "还没结果",
)


@dataclass
class Sentiment:
    """情绪判定结果。"""

    score: int = 0
    words: List[str] = field(default_factory=list)
    escalated: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {"score": self.score, "words": self.words[:6]}


def analyse_sentiment(text: str, previous_score: int = 0) -> Sentiment:
    """情绪打分（词典近似）。

    ``previous_score`` 让情绪可以**跨轮累计**——单轮"有点烦"不算什么，
    连说五轮才是真的该转人工了。

    匹配用**字符集合**而不是子串：中文表达变化多，"还没发货 / 还是没发货 /
    一直没动静"都在表达同一件事，子串匹配漏掉一个变体就白搭。
    """
    text = text or ""
    score = 0
    hits: List[str] = []
    for word, weight in NEGATIVE_WORDS.items():
        if word in text and word not in hits:
            hits.append(word)
            score += weight

    # 强化信号：反复/持续未解决的表达（跨轮累计 + 一次加 2 分）
    if ("还" in text and "没" in text) or ("一直" in text and "没" in text):
        hits.append("仍未解决")
        score += 2

    return Sentiment(score=previous_score + score, words=hits, escalated=score >= 3)


@dataclass
class Trigger:
    """一条触发规则命中的记录（进轨迹、写进交接单）。"""

    rule: str
    reason: str
    weight: int = 0
    immediate: bool = False
    evidence: str = ""

    def as_dict(self) -> Dict[str, Any]:
        data = {"rule": self.rule, "reason": self.reason, "weight": self.weight}
        if self.evidence:
            data["evidence"] = self.evidence
        return data


@dataclass
class EscalationVerdict:
    """转人工判定结果。"""

    decision: str = "continue"  # continue | defer | handoff
    score: int = 0
    threshold: int = 5
    triggers: List[Trigger] = field(default_factory=list)
    reason: str = ""
    sentiment: Sentiment = field(default_factory=Sentiment)

    @property
    def should_handoff(self) -> bool:
        return self.decision == "handoff"

    @property
    def should_defer(self) -> bool:
        return self.decision == "defer"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "score": self.score,
            "threshold": self.threshold,
            "reason": self.reason,
            "triggers": [item.as_dict() for item in self.triggers],
            "sentiment": self.sentiment.as_dict(),
        }


@dataclass
class EscalationThresholds:
    """阈值集中在这里，便于调参与测试。"""

    #: 情绪累计达到多少分转人工
    sentiment_score: int = 5
    #: 同一诉求连续提及几轮仍未解决
    repeat_turns: int = 3
    #: 连续几轮"答非所问"（有重复意图、且结果里没有真实工具佐证）
    loop_turns: int = 2
    #: 首次触发但"刚问过问题"时是否再给一轮机会
    allow_defer: bool = True


@dataclass
class TurnObservation:
    """判定所需的本轮事实（由对话层提供，判定逻辑不依赖具体实现）。"""

    turn: int = 0
    intents: List[str] = field(default_factory=list)
    tool_ok: bool = False
    asked_question: bool = False
    #: 历史轮次里出现过的意图（按轮次顺序）
    intent_history: List[List[str]] = field(default_factory=list)
    #: 历史轮次里是否有过成功的工具调用
    tool_ok_history: List[bool] = field(default_factory=list)


class EscalationEngine:
    """按规则判定"该不该转人工"。"""

    def __init__(self, thresholds: Optional[EscalationThresholds] = None) -> None:
        self.thresholds = thresholds or EscalationThresholds()

    # ------------------------------------------------------------------
    def evaluate(
        self,
        user_text: str,
        observation: TurnObservation,
        cumulative_sentiment: int = 0,
    ) -> EscalationVerdict:
        sentiment = analyse_sentiment(user_text, cumulative_sentiment)
        triggers: List[Trigger] = []
        immediate: Optional[Trigger] = None

        # 1) 明确要求转人工
        for pattern in EXPLICIT_PATTERNS:
            match = pattern.search(user_text or "")
            if match:
                immediate = Trigger(
                    rule="explicit_request",
                    reason="用户明确要求转人工",
                    weight=100,
                    immediate=True,
                    evidence=match.group(0),
                )
                break

        # 2) 超出权限（不能自己答应，必须转人工核实）
        if immediate is None:
            for pattern, reason in OUT_OF_SCOPE_PATTERNS:
                match = pattern.search(user_text or "")
                if match:
                    immediate = Trigger(
                        rule="out_of_scope",
                        reason=reason,
                        weight=100,
                        immediate=True,
                        evidence=match.group(0),
                    )
                    break

        # 3) 情绪累计
        if sentiment.score >= self.thresholds.sentiment_score:
            triggers.append(
                Trigger(
                    rule="negative_sentiment",
                    reason=f"负面情绪累计 {sentiment.score} 分（阈值 {self.thresholds.sentiment_score}）",
                    weight=sentiment.score,
                    evidence="、".join(sentiment.words[:4]),
                )
            )

        # 4) 同一诉求反复提及
        repeated = self._repeated_intent(observation)
        if repeated:
            triggers.append(
                Trigger(
                    rule="repeat_unresolved",
                    reason=(
                        f"「{repeated['intent']}」在最近 {repeated['turns']} 轮里被反复提及"
                        "且没有拿到处理结果"
                    ),
                    weight=repeated["turns"],
                    evidence=repeated["intent"],
                )
            )

        # 5) 答非所问循环
        if self._in_answer_loop(observation):
            triggers.append(
                Trigger(
                    rule="answer_loop",
                    reason=(
                        f"连续 {self.thresholds.loop_turns} 轮重复同一诉求"
                        "且没有查到有效信息，疑似无法解决"
                    ),
                    weight=3,
                    evidence=observation.intents[0] if observation.intents else "",
                )
            )

        score = sum(item.weight for item in triggers)

        if immediate is not None:
            return EscalationVerdict(
                decision="handoff",
                score=100,
                threshold=self.thresholds.sentiment_score,
                triggers=[immediate, *triggers],
                reason=immediate.reason,
                sentiment=sentiment,
            )

        if score >= self.thresholds.sentiment_score:
            # 刚问过问题时先不打断：用户正准备补充信息，这时转人工很突兀
            if self.thresholds.allow_defer and observation.asked_question:
                return EscalationVerdict(
                    decision="defer",
                    score=score,
                    threshold=self.thresholds.sentiment_score,
                    triggers=triggers,
                    reason="本轮刚向用户提问，暂缓转人工，等用户回应后再判断",
                    sentiment=sentiment,
                )
            return EscalationVerdict(
                decision="handoff",
                score=score,
                threshold=self.thresholds.sentiment_score,
                triggers=triggers,
                reason="；".join(item.reason for item in triggers),
                sentiment=sentiment,
            )

        return EscalationVerdict(
            decision="continue",
            score=score,
            threshold=self.thresholds.sentiment_score,
            triggers=triggers,
            reason="",
            sentiment=sentiment,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _repeated_intent(observation: TurnObservation) -> Optional[Dict[str, Any]]:
        """同一诉求在最近 N 轮里出现多次，且期间没有成功的工具调用。"""
        if not observation.intents:
            return None
        history = list(observation.intent_history) + [list(observation.intents)]
        tool_history = list(observation.tool_ok_history) + [observation.tool_ok]
        if len(history) < 2:
            return None

        target = observation.intents[0]
        window_turns = min(len(history), 5)
        recent = history[-window_turns:]
        recent_tools = tool_history[-window_turns:]

        count = sum(1 for intents in recent if target in intents)
        if count >= 2 and not any(recent_tools):
            return {"intent": target, "turns": count}
        return None

    def _in_answer_loop(self, observation: TurnObservation) -> bool:
        """连续多轮重复同一诉求且始终没有工具佐证 → 判定为"答非所问循环"。"""
        if not observation.intents:
            return False
        history = list(observation.intent_history) + [list(observation.intents)]
        tool_history = list(observation.tool_ok_history) + [observation.tool_ok]
        need = max(self.thresholds.loop_turns, 2)
        if len(history) < need:
            return False

        target = observation.intents[0]
        tail = history[-need:]
        tail_tools = tool_history[-need:]
        return all(target in intents for intents in tail) and not any(tail_tools)


__all__ = [
    "EscalationEngine",
    "EscalationThresholds",
    "EscalationVerdict",
    "Sentiment",
    "Trigger",
    "TurnObservation",
    "analyse_sentiment",
]
