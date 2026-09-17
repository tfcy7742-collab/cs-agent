"""P5：转人工触发规则测试。

规则层的价值在于**可解释、可测试**。这里逐条验证：
明确要求、超权限、情绪累计、重复未解决、答非所问循环，
以及"刚问过问题先别打断"的推迟策略。
"""

from __future__ import annotations

import pytest

from cs_agent.handoff_rules import (
    EscalationEngine,
    EscalationThresholds,
    TurnObservation,
    analyse_sentiment,
)


def obs(
    intents=None,
    tool_ok=False,
    asked=False,
    history=None,
    tool_history=None,
    turn=1,
) -> TurnObservation:
    return TurnObservation(
        turn=turn,
        intents=list(intents or []),
        tool_ok=tool_ok,
        asked_question=asked,
        intent_history=list(history or []),
        tool_ok_history=list(tool_history or []),
    )


# ---------------------------------------------------------------------------
# 情绪词典
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected_min",
    [
        ("我要投诉你们", 3),
        ("这也太差了吧", 3),
        ("你们什么意思，我要投诉", 3),
        ("怎么还没发货啊", 1),
        ("麻烦帮我看看", 1),
        ("你好，我想咨询一下", 0),
    ],
)
def test_sentiment_scoring(text: str, expected_min: int) -> None:
    assert analyse_sentiment(text).score >= expected_min


def test_sentiment_accumulates_across_turns() -> None:
    """单轮"有点烦"不算什么，连说几轮才是真该转人工了。"""
    first = analyse_sentiment("怎么这么慢")
    second = analyse_sentiment("还是没发货", previous_score=first.score)
    assert second.score > first.score


def test_sentiment_lists_matched_words_for_audit() -> None:
    sentiment = analyse_sentiment("太差了，我要投诉")
    assert sentiment.words
    assert sentiment.score >= 6


# ---------------------------------------------------------------------------
# 立即触发：明确要求 / 超权限
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    ["我要转人工", "转人工客服", "叫你们经理来", "不要机器人，找真人", "帮我转接客服"],
)
def test_explicit_human_request_triggers_handoff(text: str) -> None:
    verdict = EscalationEngine().evaluate(text, obs(intents=["投诉"]))
    assert verdict.should_handoff is True
    assert any(item.rule == "explicit_request" for item in verdict.triggers)


@pytest.mark.parametrize(
    "text,rule",
    [
        ("你们得赔我 100 块", "out_of_scope"),
        ("给我免运费", "out_of_scope"),
        ("能不能加急发货", "out_of_scope"),
        ("再给我一张额外优惠券", "out_of_scope"),
        ("我要去 12315 投诉", "out_of_scope"),
        ("帮我把订单金额改成 100", "out_of_scope"),
    ],
)
def test_out_of_scope_requests_trigger_handoff(text: str, rule: str) -> None:
    """超出权限的诉求不能自己答应——必须转人工核实。"""
    verdict = EscalationEngine().evaluate(text, obs(intents=["补偿"]))
    assert verdict.should_handoff is True
    assert any(item.rule == rule for item in verdict.triggers)
    assert verdict.reason


def test_out_of_scope_is_immediate_even_with_question() -> None:
    """即使本轮刚问过用户，超权限诉求也必须立刻转（不能拖）。"""
    verdict = EscalationEngine().evaluate("你们赔我钱", obs(asked=True))
    assert verdict.should_handoff is True


# ---------------------------------------------------------------------------
# 阈值触发：情绪累计 / 重复未解决 / 答非所问
# ---------------------------------------------------------------------------
def test_low_sentiment_does_not_handoff() -> None:
    verdict = EscalationEngine().evaluate("麻烦帮我看下", obs(intents=["查询物流"]))
    assert verdict.decision == "continue"
    assert verdict.triggers == []


def test_accumulated_sentiment_triggers_handoff() -> None:
    engine = EscalationEngine()
    verdict = engine.evaluate(
        "还是没发货", obs(intents=["催发货"], history=[["催发货"]], tool_history=[True]),
        cumulative_sentiment=5,
    )
    assert verdict.should_handoff is True
    assert any(item.rule == "negative_sentiment" for item in verdict.triggers)
    assert verdict.score >= verdict.threshold


def test_repeat_unresolved_triggers_handoff() -> None:
    """同一诉求反复提、且始终没有拿到有效结果 → 该转人工了。"""
    verdict = EscalationEngine().evaluate(
        "物流到底怎么样了",
        obs(
            intents=["查询物流"],
            tool_ok=False,
            history=[["查询物流"], ["查询物流"]],
            tool_history=[False, False],
        ),
    )
    assert any(item.rule in {"repeat_unresolved", "answer_loop"} for item in verdict.triggers)


def test_repeat_with_successful_tool_does_not_escalate() -> None:
    """查到了结果就不算"反复未解决"——不能因为有重复词就转人工。"""
    verdict = EscalationEngine().evaluate(
        "还有别的快递吗",
        obs(
            intents=["查询物流"],
            tool_ok=True,
            history=[["查询物流"], ["查询物流"]],
            tool_history=[True, True],
        ),
    )
    assert not any(item.rule == "repeat_unresolved" for item in verdict.triggers)


def test_answer_loop_detected() -> None:
    verdict = EscalationEngine().evaluate(
        "我要退货",
        obs(
            intents=["退货"],
            tool_ok=False,
            history=[["退货"], ["退货"]],
            tool_history=[False, False],
        ),
    )
    assert any(item.rule == "answer_loop" for item in verdict.triggers)


# ---------------------------------------------------------------------------
# 推迟策略
# ---------------------------------------------------------------------------
def test_defer_when_question_just_asked() -> None:
    """刚向用户提问（正在等补充信息）时不打断，先给一轮机会。"""
    verdict = EscalationEngine().evaluate(
        "我要退货",
        obs(
            intents=["退货"],
            asked=True,
            history=[["退货"], ["退货"]],
            tool_history=[False, False],
        ),
    )
    assert verdict.decision == "defer"
    assert verdict.should_handoff is False
    assert "暂缓" in verdict.reason


def test_defer_can_be_disabled() -> None:
    engine = EscalationEngine(EscalationThresholds(allow_defer=False))
    verdict = engine.evaluate(
        "我要退货",
        obs(intents=["退货"], asked=True, history=[["退货"], ["退货"]], tool_history=[False, False]),
    )
    assert verdict.should_handoff is True


# ---------------------------------------------------------------------------
# 判定结果的可解释性
# ---------------------------------------------------------------------------
def test_verdict_is_serialisable_for_trace() -> None:
    verdict = EscalationEngine().evaluate("我要投诉，转人工", obs(intents=["投诉"]))
    payload = verdict.as_dict()
    for key in ("decision", "score", "threshold", "reason", "triggers", "sentiment"):
        assert key in payload
    assert payload["triggers"]
    assert payload["triggers"][0]["rule"]


def test_thresholds_are_configurable() -> None:
    engine = EscalationEngine(EscalationThresholds(sentiment_score=1))
    verdict = engine.evaluate("有点慢", obs(intents=["催发货"]))
    assert verdict.should_handoff is True


def test_no_intents_no_history_never_escalates_by_threshold() -> None:
    verdict = EscalationEngine().evaluate("你好", obs())
    assert verdict.decision == "continue"
    assert verdict.score == 0
