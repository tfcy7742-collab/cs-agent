"""P4：技能编排集成测试（对话流程 + 工具）。

覆盖四件事：
1. 有订单号时查询类工具真的被调用，结果进提示；
2. 政策问题走检索，并把来源带进提示；
3. **工具查不到时，如实告知的指令必须进提示**（不编造的最后一道闸）；
4. 写操作的 确认 → 执行 完整闭环，未确认前数据不变。
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from cs_agent.conversation import ConversationService, _parse_frame
from cs_agent.state import StateSnapshot
from cs_agent.tools.mock_data import get_repository, reset_repository


@pytest.fixture(autouse=True)
def _clean_data():
    reset_repository()
    yield
    reset_repository()


def run_turn(service: ConversationService, sid: str, text: str) -> Dict[str, Any]:
    """跑一轮，返回本轮的计划、工具执行情况与**真正发给模型的系统提示**。"""
    payload_start: Dict[str, Any] = {}
    done: Dict[str, Any] = {}
    for frame in service.stream_turn(sid, text):
        event, payload = _parse_frame(frame)
        if event == "start":
            payload_start = payload
        elif event == "done":
            done = payload
    return {
        "plan": payload_start.get("plan") or {},
        "tools": payload_start.get("tools") or {},
        "prompt": payload_start.get("system_prompt") or "",
        "reply": done.get("content", ""),
    }


def system_prompt(result: Dict[str, Any]) -> str:
    """本轮送给模型的系统提示（含工具结果）。"""
    return result["prompt"]


# ---------------------------------------------------------------------------
# 查询类工具
# ---------------------------------------------------------------------------
def test_logistics_query_calls_tool_and_injects_data(service, store) -> None:
    sid = service.open_session()["id"]
    result = run_turn(service, sid, "订单号 SO20260101，帮我查下物流")

    runs = result["tools"].get("runs") or []
    assert any(run["tool"] == "track_logistics" and run["ok"] for run in runs), runs

    assert "工具执行结果" in system_prompt(result)
    assert "顺丰速运" in system_prompt(result), "工具查到的事实必须进提示"
    assert "禁止编造" in system_prompt(result)
    assert "本地模拟" in system_prompt(result), "数据来源必须如实标注"


def test_order_query_returns_status(service, store) -> None:
    sid = service.open_session()["id"]
    result = run_turn(service, sid, "订单号 SO20260102，帮我看看这个订单")
    runs = result["tools"].get("runs") or []
    assert any(run["tool"] == "query_order" for run in runs), runs
    assert "已签收" in system_prompt(result)


def test_refund_query_without_record_tells_the_truth(service, store) -> None:
    """没有退款记录时，提示里必须出现"如实告知"的指令。"""
    sid = service.open_session()["id"]
    result = run_turn(service, sid, "订单号 SO20260101，退款到哪一步了")

    runs = result["tools"].get("runs") or []
    assert any(run["tool"] == "query_refund" for run in runs), runs
    assert "没有退款记录" in system_prompt(result)
    assert "必须如实告知" in system_prompt(result)


def test_unknown_order_is_reported_not_fabricated(service, store) -> None:
    """查不到订单：提示里必须明确"查询失败 + 原因"，并要求如实告知。"""
    sid = service.open_session()["id"]
    result = run_turn(service, sid, "订单号 SO99999999，帮我查下物流")

    runs = result["tools"].get("runs") or []
    assert runs and runs[0]["ok"] is False
    assert runs[0]["error_code"] == "not_found"

    assert "查询失败" in system_prompt(result)
    assert "不要猜测或编造" in system_prompt(result)


def test_tool_not_called_without_order_id(service, store) -> None:
    """缺订单号时不空调用工具，而是先追问（P3 的追问逻辑接管）。"""
    sid = service.open_session()["id"]
    result = run_turn(service, sid, "帮我查下物流")

    assert result["plan"]["action"] == "ask"
    assert not result["tools"].get("runs"), "缺参数时不应调用工具"
    assert "订单号" in result["prompt"], "追问话术必须进提示"


# ---------------------------------------------------------------------------
# 政策检索
# ---------------------------------------------------------------------------
def test_policy_question_uses_search_tool(service, store) -> None:
    sid = service.open_session()["id"]
    result = run_turn(service, sid, "退货运费一般谁承担？")

    runs = result["tools"].get("runs") or []
    assert any(run["tool"] == "search_policy" and run["ok"] for run in runs), runs

    assert "平台售后政策" in system_prompt(result), "政策来源必须进提示"
    assert "POL-002" in system_prompt(result) or "退货运费承担规则" in system_prompt(result)
    assert "依据来源" in system_prompt(result)


def test_policy_no_match_is_honest(service, store) -> None:
    """问了个政策库里没有的问题：必须告知"没查到"，而不是编一条政策。"""
    sid = service.open_session()["id"]
    result = run_turn(service, sid, "请问会员积分怎么兑换？")

    runs = result["tools"].get("runs") or []
    assert runs and runs[0]["ok"] is False
    assert runs[0]["error_code"] == "no_policy_match"
    assert "没有找到" in system_prompt(result)


def test_return_eligibility_includes_citations(service, store) -> None:
    sid = service.open_session()["id"]
    result = run_turn(service, sid, "订单号 SO20260103，这个能退吗")

    runs = result["tools"].get("runs") or []
    assert any(run["tool"] == "check_returnable" for run in runs), runs
    assert "依据条款" in system_prompt(result)
    assert "POL-004" in system_prompt(result), "定制商品应引用不可退换条款"


# ---------------------------------------------------------------------------
# 写操作：确认闭环
# ---------------------------------------------------------------------------
def test_return_submission_requires_confirmation_then_executes(service, store) -> None:
    """完整闭环：申请 → 挂起等确认 → 用户确认 → 真正执行。"""
    sid = service.open_session()["id"]

    # 1) 明确申请退货 → 应挂起并请确认，数据不变
    first = run_turn(service, sid, "订单号 SO20260102，尺码不合适，我要退货")
    runs = first["tools"].get("runs") or []
    assert any(run["needs_confirmation"] for run in runs), runs
    assert first["tools"].get("confirmation_prompt")
    assert "不可撤销" in system_prompt(first)
    assert get_repository().get("SO20260102").status == "delivered", "未确认前不能改数据"

    # 2) 挂起状态被持久化（下一轮要能续办）
    assert (store.get_state(sid).get("pending_write") or {}).get("tool") == "submit_return_request"

    # 3) 用户确认 → 真正执行
    second = run_turn(service, sid, "确认提交")
    runs2 = second["tools"].get("runs") or []
    assert any(run["tool"] == "submit_return_request" and run["ok"] for run in runs2), runs2
    assert get_repository().get("SO20260102").status == "returning"
    assert not store.get_state(sid).get("pending_write"), "执行后应清掉挂起状态"


def test_return_submission_can_be_cancelled(service, store) -> None:
    """用户反悔：取消后数据必须保持不变。"""
    sid = service.open_session()["id"]
    run_turn(service, sid, "订单号 SO20260102，尺码不合适，我要退货")
    result = run_turn(service, sid, "算了不用了")

    assert "已取消这次操作" in system_prompt(result)
    assert get_repository().get("SO20260102").status == "delivered"
    assert not store.get_state(sid).get("pending_write")


def test_ambiguous_reply_does_not_execute_write(service, store) -> None:
    """用户答非所问时不能擅自执行写操作，只能再问一次。"""
    sid = service.open_session()["id"]
    run_turn(service, sid, "订单号 SO20260102，尺码不合适，我要退货")
    result = run_turn(service, sid, "这个运费谁出啊")

    assert result["tools"].get("confirmation_prompt"), "应再次请求确认"
    assert get_repository().get("SO20260102").status == "delivered"


def test_write_on_undelivered_order_is_rejected(service, store) -> None:
    """未签收的订单不能提交退货——工具会拒绝，且理由要传进提示。"""
    sid = service.open_session()["id"]
    run_turn(service, sid, "订单号 SO20260101，我不想要了，我要退货")
    result = run_turn(service, sid, "确认")

    runs = result["tools"].get("runs") or []
    assert runs and runs[0]["ok"] is False
    assert runs[0]["error_code"] == "not_delivered"
    assert "尚未签收" in system_prompt(result)
    assert get_repository().get("SO20260101").status == "shipped"


# ---------------------------------------------------------------------------
# 工具清单
# ---------------------------------------------------------------------------
def test_tool_specs_available_for_prompt_or_frontend(service) -> None:
    specs = service.skills.registry.specs()
    assert len(specs) == 6
    assert all("description" in spec for spec in specs)
    assert any(spec["requires_confirmation"] for spec in specs)


def test_trace_records_tool_runs(service, store) -> None:
    sid = service.open_session()["id"]
    run_turn(service, sid, "订单号 SO20260101，帮我查下物流")
    trace = service.latest_trace(sid)
    assert trace is not None
    assert "tools" in trace
    assert trace["tools"]["runs"][0]["tool"] == "track_logistics"
