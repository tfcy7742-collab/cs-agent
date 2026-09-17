"""P4：政策检索测试。

盯两件事：
1. **能召回**：常见问法要命中对应条款；
2. **宁缺勿滥**：不相关时返回空，而不是拿一条不相干的政策凑数——
   把不相关政策当答案是比"没查到"更严重的错误。
"""

from __future__ import annotations

import pytest

from cs_agent.policy import POLICIES, PolicyIndex, get_index


# ---------------------------------------------------------------------------
# 召回
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "query,expected_doc",
    [
        ("退货运费谁承担", "POL-002"),
        ("退货运费一般谁出", "POL-002"),
        ("退款多久能到账", "POL-003"),
        ("七天无理由退货的时限是多久", "POL-001"),
        ("定制商品能退吗", "POL-004"),
        ("怎么换尺码", "POL-005"),
        ("为什么我的快递一直没更新", "POL-006"),
        ("发票抬头怎么填", "POL-007"),
        ("优惠券退款会退回吗", "POL-008"),
        ("地址填错了能改吗", "POL-009"),
        ("买完就降价了能补差价吗", "POL-010"),
        ("支持上门取件吗", "POL-011"),
        ("耳机坏了保修怎么走", "POL-012"),
    ],
)
def test_policy_recall(query: str, expected_doc: str) -> None:
    hits = get_index().search(query, top_k=3)
    assert hits, f"「{query}」没有任何命中"
    assert expected_doc in {hit.doc.doc_id for hit in hits}, (
        f"「{query}」未命中 {expected_doc}，实际：{[h.doc.doc_id for h in hits]}"
    )


def test_top_hit_is_relevant_for_common_questions() -> None:
    """最常见的两个问题，期望第一条就是对的。"""
    assert get_index().search("退货运费谁承担")[0].doc.doc_id == "POL-002"
    assert get_index().search("退款多久到账")[0].doc.doc_id == "POL-003"


# ---------------------------------------------------------------------------
# 宁缺勿滥
# ---------------------------------------------------------------------------
def test_irrelevant_query_returns_empty() -> None:
    """完全无关的问题必须返回空——不能拿一条不相关的政策当答案。"""
    assert get_index().search("今天天气怎么样") == []
    assert get_index().search("帮我写一首诗") == []
    assert get_index().search("") == []


def test_min_score_filters_weak_matches() -> None:
    index = get_index()
    loose = index.search("售后", top_k=5)
    strict = index.search("售后", top_k=5, min_score=8.0)
    assert len(strict) <= len(loose)


def test_hits_carry_source_and_evidence() -> None:
    """命中的每一条都要能核对依据：来源、原文、命中词。"""
    hit = get_index().search("退货运费谁承担")[0]
    payload = hit.as_dict()
    assert payload["source"].startswith("平台售后政策")
    assert payload["text"]
    assert payload["matched_terms"] or payload["highlight"]
    assert payload["score"] > 0


def test_top_k_is_clamped() -> None:
    index = get_index()
    assert len(index.search("退货", top_k=100)) <= index.count()
    assert len(index.search("退货", top_k=1)) <= 1


# ---------------------------------------------------------------------------
# 索引本身
# ---------------------------------------------------------------------------
def test_index_covers_all_policies() -> None:
    index = get_index()
    assert index.count() == len(POLICIES)
    for doc in POLICIES:
        assert index.by_id(doc.doc_id) is doc


def test_every_policy_has_keywords_and_source_ready() -> None:
    """没有关键词的政策基本召不回来，等于白写。"""
    for doc in POLICIES:
        assert doc.keywords, f"{doc.doc_id} 没有关键词"
        assert doc.text and doc.title


def test_policy_ids_are_unique() -> None:
    ids = [doc.doc_id for doc in POLICIES]
    assert len(ids) == len(set(ids))


def test_custom_index_is_isolated() -> None:
    """自建索引不影响全局索引（测试可隔离）。"""
    from cs_agent.policy import PolicyDoc

    index = PolicyIndex([PolicyDoc("X1", "测试条款", "仅供测试的条款内容", ("测试甲",))])
    assert index.count() == 1
    assert index.search("测试甲")[0].doc.doc_id == "X1"
    assert get_index().count() == len(POLICIES)
