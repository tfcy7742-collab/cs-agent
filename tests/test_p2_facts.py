"""P2：事实抽取测试。

重点：
* 正则抽取**不幻觉**（没有的信息不编）；
* LLM 抽取失败/离线时不影响主流程；
* 合并时**正则优先**；
* 只有白名单字段能进跨会话画像。
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from cs_agent.facts import (
    PROFILE_ALLOWED_KEYS,
    SESSION_ONLY_KEYS,
    extract_by_rules,
    extract_with_llm,
    merge_extractions,
    render_facts,
    select_profile_updates,
)


# ---------------------------------------------------------------------------
# 正则抽取
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,key,expected",
    [
        ("我的订单号是 SO20260101", "订单号", "SO20260101"),
        ("订单编号：20260101998877，麻烦查下", "订单号", "20260101998877"),
        ("单号 so20260101 到现在还没发货", "订单号", "SO20260101"),
        ("我的手机是 13812345678", "联系方式", "13812345678"),
        ("我叫张先生", "称呼", "张先生"),
        ("我是李女士，想退货", "称呼", "李女士"),
        ("我在杭州，买的衣服不合适", "城市", "杭州"),
        ("我这边是北京，帮我改地址", "城市", "北京"),
        ("我是金卡会员", "会员等级", "金卡"),
        ("希望简洁一点回复", "沟通偏好", "简洁"),
    ],
)
def test_regex_extraction_hits(text: str, key: str, expected: str) -> None:
    facts = extract_by_rules(text)
    assert key in facts, f"未抽出 {key}：{facts}"
    assert expected in facts[key]["value"]
    assert facts[key]["source"] == "regex"
    assert facts[key]["evidence"], "必须留下抽取依据，便于事后核对"


def test_regex_order_id_normalised_to_upper() -> None:
    """订单号统一大写，避免 so123 与 SO123 被当成两个值反复覆盖。"""
    assert extract_by_rules("单号 so123456")["订单号"]["value"] == "SO123456"
    assert extract_by_rules("单号 SO123456")["订单号"]["value"] == "SO123456"


def test_regex_does_not_hallucinate() -> None:
    """没说过的信息绝不能出现——这是"不编造"在最底层的一道闸。"""
    facts = extract_by_rules("你好，我想咨询一下")
    assert facts == {}


def test_regex_ignores_plain_numbers_as_order_id() -> None:
    """随口提到的数字不该被当成订单号（宁可漏抽，也不能抽错）。"""
    facts = extract_by_rules("我买了 3 件，花了 199 元，5 天了还没到")
    assert "订单号" not in facts
    assert facts["涉及金额"]["value"] == "199 元"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("我要退货", "退货"),
        ("这个要退款", "退款"),
        ("我的快递到哪了", "查询物流"),
        ("你们再不处理我就投诉了", "投诉"),
        ("麻烦帮我改地址", "修改地址"),
        ("怎么还没发货", "催发货"),
    ],
)
def test_intent_detection(text: str, expected: str) -> None:
    facts = extract_by_rules(text)
    assert "诉求" in facts
    assert expected in facts["诉求"]["value"]


def test_intent_can_hold_multiple() -> None:
    facts = extract_by_rules("我要退货，另外帮我查下物流")
    assert "退货" in facts["诉求"]["value"]
    assert "查询物流" in facts["诉求"]["value"]


def test_regex_requires_explicit_city_expression() -> None:
    """"我想买杭州的茶叶"不该被当成用户所在地。"""
    facts = extract_by_rules("我想买杭州的茶叶")
    assert "城市" not in facts


@pytest.mark.parametrize(
    "text,expected",
    [
        ("质量有问题", "质量有问题"),
        ("收到就破损了", "破损"),
        ("尺码不合适想换一件", "尺码不合适"),
        ("我要七天无理由退货", "七天无理由"),
        ("就是不喜欢了", "不喜欢"),
    ],
)
def test_reason_extraction(text: str, expected: str) -> None:
    """退货原因直接影响"运费谁承担"，是软槽位里最值得抽的一个。"""
    facts = extract_by_rules(text)
    assert "退货原因" in facts, f"未抽出退货原因：{facts}"
    assert facts["退货原因"]["value"] == expected


def test_reason_not_extracted_when_unrelated() -> None:
    assert "退货原因" not in extract_by_rules("你好，我想查下物流")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("我买的蓝色卫衣不合适", "蓝色卫衣"),
        ("要退订单里的运动鞋", "订单里的运动鞋"),
    ],
)
def test_item_extraction(text: str, expected: str) -> None:
    facts = extract_by_rules(text)
    assert facts["商品名称"]["value"] == expected


def test_item_extraction_ignores_intent_words() -> None:
    """"我要退货"不能被当成商品名。"""
    assert "商品名称" not in extract_by_rules("我要退货")


# ---------------------------------------------------------------------------
# LLM 抽取
# ---------------------------------------------------------------------------
class _FakeLLM:
    def __init__(self, text: str = "", online: bool = True) -> None:
        self._text = text
        self.online = online

    def chat(self, messages, **kwargs):  # noqa: ANN001
        class _Result:
            def __init__(self, text: str) -> None:
                self.text = text

        return _Result(self._text)


def test_llm_extraction_parses_json() -> None:
    llm = _FakeLLM('{"称呼":"王先生","诉求":"查询发票"}')
    facts = extract_with_llm("帮我开个发票", llm)
    assert facts["称呼"]["value"] == "王先生"
    assert facts["诉求"]["value"] == "查询发票"
    assert facts["称呼"]["source"] == "llm"


def test_llm_extraction_tolerates_markdown_fence() -> None:
    llm = _FakeLLM('```json\n{"诉求":"退货"}\n```')
    assert extract_with_llm("我要退", llm)["诉求"]["value"] == "退货"


def test_llm_extraction_drops_unknown_and_empty_keys() -> None:
    llm = _FakeLLM('{"诉求":"退货","乱编字段":"x","称呼":"无","城市":""}')
    facts = extract_with_llm("我要退货", llm)
    assert set(facts) == {"诉求"}


def test_llm_extraction_returns_empty_when_offline() -> None:
    llm = _FakeLLM('{"诉求":"退货"}', online=False)
    assert extract_with_llm("我要退货", llm) == {}


def test_llm_extraction_handles_broken_json() -> None:
    assert extract_with_llm("我要退货", _FakeLLM("完全不是 JSON")) == {}


def test_llm_extraction_handles_api_error() -> None:
    from cs_agent.llm import LLMError

    class _Broken:
        online = True

        def chat(self, messages, **kwargs):  # noqa: ANN001
            raise LLMError("接口故障")

    assert extract_with_llm("我要退货", _Broken()) == {}


def test_merge_prefers_regex() -> None:
    """两路都抽到订单号时以正则为准（确定性抽取优先）。"""
    rules = {"订单号": {"value": "SO111111", "source": "regex"}}
    semantic = {"订单号": {"value": "SO999999", "source": "llm"}, "诉求": {"value": "退货"}}
    merged = merge_extractions(rules, semantic)
    assert merged["订单号"]["value"] == "SO111111"
    assert merged["诉求"]["value"] == "退货"


# ---------------------------------------------------------------------------
# 画像白名单
# ---------------------------------------------------------------------------
def test_only_whitelisted_keys_go_to_profile() -> None:
    facts = {
        "称呼": {"value": "张先生"},
        "城市": {"value": "北京"},
        "订单号": {"value": "SO1"},
        "诉求": {"value": "退货"},
    }
    updates = select_profile_updates(facts)
    assert set(updates) == {"称呼", "城市"}
    assert "订单号" not in updates
    assert "诉求" not in updates


def test_session_only_and_profile_keys_do_not_overlap() -> None:
    """两组白名单不能有交集，否则规则会自相矛盾。"""
    assert not (SESSION_ONLY_KEYS & PROFILE_ALLOWED_KEYS)
    assert "订单号" in SESSION_ONLY_KEYS


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------
def test_render_facts_marks_changed_values() -> None:
    """"改过口"必须让模型看见，否则它会拿旧值当有效值。"""
    facts = {
        "城市": {"value": "上海", "previous": "北京"},
        "称呼": {"value": "张先生"},
    }
    rendered = render_facts(facts)
    assert rendered["城市"] == "上海（此前为 北京）"
    assert rendered["称呼"] == "张先生"


def test_render_facts_orders_identity_first() -> None:
    facts = {"订单号": {"value": "SO1"}, "称呼": {"value": "张先生"}}
    assert list(render_facts(facts)) == ["称呼", "订单号"]
