"""P2：版本化事实与跨会话画像测试。"""

from __future__ import annotations

from cs_agent.memory import MemoryManager
from cs_agent.storage import SessionStore


def test_merge_facts_records_first_value(store: SessionStore, service) -> None:
    sid = store.create_session(user_id="u1")["id"]
    changes = store.merge_facts(sid, {"称呼": {"value": "张先生", "source": "regex"}}, turn=1)
    assert changes == [], "首次抽取不算'改口'"
    facts = store.get_facts(sid)
    assert facts["称呼"]["value"] == "张先生"
    assert "previous" not in facts["称呼"]


def test_merge_facts_keeps_previous_on_change(store: SessionStore) -> None:
    """改口后：新值生效、旧值保留——既不串味，也不丢历史。"""
    sid = store.create_session()["id"]
    store.merge_facts(sid, {"城市": {"value": "北京"}}, turn=1)
    changes = store.merge_facts(sid, {"城市": {"value": "上海"}}, turn=3)

    assert len(changes) == 1
    assert changes[0]["key"] == "城市"
    assert changes[0]["old"] == "北京"
    assert changes[0]["new"] == "上海"

    facts = store.get_facts(sid)
    assert facts["城市"]["value"] == "上海"
    assert facts["城市"]["previous"] == "北京"


def test_same_value_twice_is_not_a_change(store: SessionStore) -> None:
    """重复说同一件事不能产生"改口"噪音（否则轨迹里全是假变更）。"""
    sid = store.create_session()["id"]
    store.merge_facts(sid, {"城市": {"value": "北京"}}, turn=1)
    changes = store.merge_facts(sid, {"城市": {"value": "北京"}}, turn=2)
    assert changes == []
    assert "previous" not in store.get_facts(sid)["城市"]


def test_previous_survives_when_value_set_back(store: SessionStore) -> None:
    """改回旧值也要留痕：北京→上海→北京，previous 应记为上海。"""
    sid = store.create_session()["id"]
    store.merge_facts(sid, {"城市": {"value": "北京"}}, turn=1)
    store.merge_facts(sid, {"城市": {"value": "上海"}}, turn=2)
    store.merge_facts(sid, {"城市": {"value": "北京"}}, turn=4)
    facts = store.get_facts(sid)["城市"]
    assert facts["value"] == "北京"
    assert facts["previous"] == "上海"


def test_merge_facts_ignores_empty_values(store: SessionStore) -> None:
    sid = store.create_session()["id"]
    store.merge_facts(sid, {"城市": {"value": ""}, "称呼": {"value": None}}, turn=1)
    assert store.get_facts(sid) == {}


def test_facts_are_per_session(store: SessionStore) -> None:
    """订单号这类会话事实绝不能串到别的会话。"""
    a = store.create_session(user_id="u1")["id"]
    b = store.create_session(user_id="u1")["id"]
    store.merge_facts(a, {"订单号": {"value": "SO-A"}}, turn=1)
    store.merge_facts(b, {"订单号": {"value": "SO-B"}}, turn=1)
    assert store.get_facts(a)["订单号"]["value"] == "SO-A"
    assert store.get_facts(b)["订单号"]["value"] == "SO-B"


# ---------------------------------------------------------------------------
# 每轮抽取（含 LLM 关掉的情况）
# ---------------------------------------------------------------------------
def test_ingest_user_message_extracts_and_persists(service, store: SessionStore) -> None:
    sid = service.open_session(user_id="u1")["id"]
    flags = service.memory.ingest_user_message(sid, "我叫张先生，订单号是 SO20260101，要退货", turn=1)

    assert "称呼" in flags.fact_keys
    assert "订单号" in flags.fact_keys
    assert "诉求" in flags.fact_keys
    assert flags.extract_sources.get("regex", 0) >= 3
    # 离线模式下不应有 llm 来源
    assert "llm" not in flags.extract_sources

    facts = store.get_facts(sid)
    assert facts["订单号"]["value"] == "SO20260101"
    assert facts["诉求"]["value"].startswith("退货")


def test_ingest_reports_changes_with_readable_message(service, store: SessionStore) -> None:
    sid = service.open_session()["id"]
    service.memory.ingest_user_message(sid, "我在北京", turn=1)
    flags = service.memory.ingest_user_message(sid, "我现在在上海了", turn=3)

    assert flags.fact_changes, "应记录一次改口"
    change = flags.fact_changes[0]
    assert change["key"] == "城市"
    assert "北京" in change["message"] and "上海" in change["message"]


def test_ingest_skips_llm_when_offline(service) -> None:
    sid = service.open_session()["id"]
    flags = service.memory.ingest_user_message(sid, "我要退款", turn=1)
    assert flags.extract_sources == {"regex": 1}


# ---------------------------------------------------------------------------
# 画像沉淀
# ---------------------------------------------------------------------------
def test_profile_gets_stable_attributes_only(service, store: SessionStore) -> None:
    sid = service.open_session(user_id="u9")["id"]
    service.memory.ingest_user_message(sid, "我叫李女士，我在杭州，订单号 SO20260101", turn=1)
    profile = service.memory.sync_profile(sid)

    assert profile["attributes"]["称呼"] == "李女士"
    assert profile["attributes"]["城市"] == "杭州"
    # 会话级信息绝不进画像
    assert "订单号" not in profile["attributes"]
    assert "SO20260101" not in str(profile["attributes"])


def test_profile_persists_across_sessions(service, store: SessionStore) -> None:
    """跨会话记忆的核心断言：新会话能读到上次沉淀的资料。"""
    first = service.open_session(user_id="u9")["id"]
    service.memory.ingest_user_message(first, "我叫王先生，我在北京", turn=1)
    service.memory.sync_profile(first)

    second = service.open_session(user_id="u9")["id"]
    context = service.memory.context(second)
    assert context["profile"]["称呼"] == "王先生"
    assert context["profile"]["城市"] == "北京"

    # 另一个用户不该看到这些资料
    other = service.open_session(user_id="u-other")["id"]
    assert service.memory.context(other)["profile"] == {}


def test_profile_drops_session_level_keys_from_legacy_data(service, store: SessionStore) -> None:
    """防御：即使画像里被写进过会话级字段，同步时也要清掉。"""
    store.save_profile("u-legacy", {"attributes": {"订单号": "SO-OLD", "称呼": "老张"}})
    sid = service.open_session(user_id="u-legacy")["id"]
    profile = service.memory.sync_profile(sid)
    assert "订单号" not in profile["attributes"]
    assert profile["attributes"]["称呼"] == "老张"


def test_profile_history_is_bounded(service, store: SessionStore) -> None:
    from cs_agent.memory import PROFILE_HISTORY_LIMIT

    for index in range(PROFILE_HISTORY_LIMIT + 3):
        sid = service.open_session(user_id="u-hist")["id"]
        service.memory.ingest_user_message(sid, f"第 {index} 次咨询，我要退货", turn=1)
        service.memory.compress(sid, turn=1)  # 无历史时不会压缩，但会走到画像同步
        service.memory.sync_profile(sid)

    profile = store.get_profile("u-hist")
    assert len(profile["history"]) <= PROFILE_HISTORY_LIMIT


def test_profile_attributes_render_into_prompt(service) -> None:
    sid = service.open_session(user_id="u-render")["id"]
    service.memory.ingest_user_message(sid, "我叫赵先生", turn=1)
    service.memory.sync_profile(sid)

    next_session = service.open_session(user_id="u-render")["id"]
    messages = service.build_messages(next_session)
    assert "赵先生" in messages[0]["content"]
    assert "长期资料" in messages[0]["content"]
