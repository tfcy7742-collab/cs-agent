"""P2：滚动摘要与长会话测试。

这是"长会话"的主战场。要证明三件事：

1. **会压缩**：超阈值后确实把最老的原文折进摘要，且原文仍完整留在库里；
2. **有上限**：聊到几十轮后，真正送进模型的工作记忆**不随时长线性增长**；
3. **不丢关键信息**：被压缩掉的那一轮里的订单号，在后续轮次仍能被模型看到
   （因为结构化事实是独立通道，不依赖摘要）。
"""

from __future__ import annotations

from cs_agent.memory import MemoryManager


def _chat(service, sid: str, text: str) -> dict:
    return service.chat_once(sid, text)


# ---------------------------------------------------------------------------
# 压缩触发
# ---------------------------------------------------------------------------
def test_no_compression_when_history_is_short(service, store) -> None:
    sid = service.open_session()["id"]
    for index in range(2):
        _chat(service, sid, f"第 {index + 1} 轮")

    assert service.memory.needs_compression(sid) is False
    assert store.latest_summary(sid) is None


def test_compression_triggers_after_threshold(service, store, settings) -> None:
    sid = service.open_session()["id"]
    # 阈值：未压缩条数 > working_recent_turns * 2（配置里是 4）
    for index in range(6):
        _chat(service, sid, f"第 {index + 1} 轮问题")

    assert store.latest_summary(sid) is not None
    summary = store.latest_summary(sid)["content"]
    assert "第 1 轮问题" in summary
    # 最老的原文被标记覆盖，但**仍然存在**
    messages = store.list_messages(sid, roles=["user", "assistant"])
    assert len(messages) == 12, "原文必须完整保留，只是不再进工作记忆"
    assert any(m["compressed"] == 1 for m in messages)


def test_working_memory_keeps_recent_turns(service, store, settings) -> None:
    sid = service.open_session()["id"]
    for index in range(8):
        _chat(service, sid, f"问题{index + 1}")

    working = service.memory.working_messages(sid)
    assert len(working) <= settings.working_recent_turns * 2
    # 最近一轮必须在窗口里
    assert "问题8" in working[-1]["content"]


def test_compression_keeps_facts_out_of_summary_path(service, store) -> None:
    """关键断言：订单号在第 1 轮给出，聊到第 10 轮仍要在系统提示里可见。

    这是"结构化事实独立于摘要"的价值——摘要会有损，事实不会。
    """
    sid = service.open_session()["id"]
    _chat(service, sid, "我的订单号是 SO20260101")
    for index in range(10):
        _chat(service, sid, f"继续问第 {index + 1} 个问题")

    # 第 1 轮的原文确实已被压缩
    first = store.list_messages(sid, roles=["user"])[0]
    assert first["compressed"] == 1

    # 但事实还在，并且被渲染进本轮提示
    assert store.get_facts(sid)["订单号"]["value"] == "SO20260101"
    messages = service.build_messages(sid)
    system = messages[0]["content"]
    assert "SO20260101" in system
    assert "摘要" in system


def test_compression_ratio_is_measured_not_claimed(service, store) -> None:
    """压缩比要如实测量并暴露，而不是宣称"无损压缩"。"""
    sid = service.open_session()["id"]
    for index in range(8):
        _chat(service, sid, f"第 {index + 1} 轮咨询，我的订单有问题" * 3)

    report = service.memory.memory_report(sid)
    assert report["has_summary"] is True
    assert report["summary_chars"] > 0
    assert 0 < report["compression_ratio"] <= 1.0
    assert report["messages_compressed"] > 0


def test_summary_is_extractive_when_offline(service, store) -> None:
    """离线时摘要走抽取式，并**明确标注**，不假装是语义压缩。"""
    sid = service.open_session()["id"]
    for index in range(6):
        _chat(service, sid, f"第 {index + 1} 轮")

    summary = store.latest_summary(sid)["content"]
    assert "抽取式" in summary
    assert "第 1 轮" in summary
    meta = store.latest_summary(sid)["meta"]
    assert meta["stats"]["source"] == "extractive"


def test_summary_is_updated_not_appended(service, store) -> None:
    """摘要就地更新：同一会话只保留一条，避免摘要层层叠加变成新负担。"""
    sid = service.open_session()["id"]
    for index in range(6):
        _chat(service, sid, f"第 {index + 1} 轮")
    first_count = len(store.list_messages(sid, roles=["summary"]))

    for index in range(6, 14):
        _chat(service, sid, f"第 {index + 1} 轮")

    summaries = store.list_messages(sid, roles=["summary"])
    assert first_count == 1
    assert len(summaries) == 1
    assert summaries[0]["meta"]["covers_until_message_id"] > 0


def test_compression_covers_old_and_keeps_window(service, store, settings) -> None:
    sid = service.open_session()["id"]
    for index in range(10):
        _chat(service, sid, f"第 {index + 1} 轮")

    summary = store.latest_summary(sid)
    covers = summary["meta"]["covers_until_message_id"]
    working = service.memory.working_messages(sid)
    assert all(m["id"] > covers for m in working), "工作记忆里不该有已覆盖的原文"
    assert len(working) <= settings.working_recent_turns * 2


# ---------------------------------------------------------------------------
# 长会话：这是本阶段的验收核心
# ---------------------------------------------------------------------------
def test_fifty_turns_working_memory_stays_bounded(service, store, settings) -> None:
    """聊 50 轮后，真正进模型的工作记忆必须**不随时长线性增长**。"""
    sid = service.open_session()["id"]
    samples = []
    for index in range(50):
        _chat(service, sid, f"第 {index + 1} 轮：请帮我看看订单状态和物流进度，谢谢。")
        if index + 1 in (10, 30, 50):
            samples.append(service.session_stats(sid)["working_chars"])

    # 工作记忆规模应基本稳定（允许小幅波动）
    assert max(samples) <= min(samples) * 1.6, f"工作记忆增长过快：{samples}"

    report = service.memory.memory_report(sid)
    assert report["turns"] == 50
    assert report["messages_total"] == 100
    assert report["working_messages"] <= settings.working_recent_turns * 2
    assert report["total_chars"] > report["uncompressed_chars"]


def test_long_conversation_keeps_early_fact(service, store) -> None:
    """50 轮之后，第 1 轮给出的称呼仍要出现在提示里。"""
    sid = service.open_session(user_id="u-long")["id"]
    _chat(service, sid, "我叫孙女士，我的手机号是 13900001111")
    for index in range(49):
        _chat(service, sid, f"第 {index + 2} 轮普通咨询")

    messages = service.build_messages(sid)
    system = messages[0]["content"]
    assert "孙女士" in system or "13900001111" in system
    report = service.memory.memory_report(sid)
    assert report["fact_count"] >= 1


def test_memory_report_shape(service) -> None:
    sid = service.open_session()["id"]
    _chat(service, sid, "你好")
    report = service.memory.memory_report(sid)
    for key in (
        "turns",
        "messages_total",
        "messages_compressed",
        "messages_uncompressed",
        "working_messages",
        "working_chars",
        "has_summary",
        "compression_ratio",
        "facts",
        "profile_attributes",
    ):
        assert key in report


def test_manual_compress_endpoint_flow(service, store) -> None:
    """手动压缩接口（调试用）：没有可压缩内容时应该是安全的空操作。"""
    sid = service.open_session()["id"]
    _chat(service, sid, "只有一轮")
    flags = service.memory.compress(sid)
    assert flags.summary_updated is False
    assert store.latest_summary(sid) is None


def test_compress_twice_is_idempotent(service, store) -> None:
    sid = service.open_session()["id"]
    for index in range(8):
        _chat(service, sid, f"第 {index + 1} 轮")

    first = service.memory.compress(sid)
    second = service.memory.compress(sid)
    assert first.summary_updated is True
    # 第二轮没有新的可压缩内容（窗口之外已经压完）
    assert second.summary_updated is False
    assert len(store.list_messages(sid, roles=["summary"])) == 1
