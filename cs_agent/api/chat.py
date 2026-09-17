"""会话与消息接口。

约定：
* 所有业务错误都返回**结构化中文提示**（前端直接展示，不用再翻译）；
* 流式接口用 SSE，事件名见 ``conversation.py`` 顶部注释；
* 非流式接口与流式接口走**同一条执行路径**（``chat_once`` 消费同一生成器），
  避免"流式和非流式行为不一致"这类经典问题。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..app import get_store
from ..config import get_settings
from ..conversation import ConversationService
from ..state import StateSnapshot
from ..tasks import missing_required, task_for_intent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])

_service: Optional[ConversationService] = None


def get_service() -> ConversationService:
    """会话服务单例。

    用惰性单例而不是模块级实例，是为了让测试能在改配置（例如切离线模式）后
    通过 ``reset_service()`` 重建。
    """
    global _service
    if _service is None:
        _service = ConversationService(get_store(), get_settings())
    return _service


def reset_service() -> None:
    global _service
    _service = None


class CreateSessionRequest(BaseModel):
    user_id: str = Field("anonymous", description="用户标识（用于跨会话记忆）")
    title: str = Field("", description="会话标题，留空则用首条消息生成")


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户消息", min_length=1)
    session_id: Optional[str] = Field(
        None, description="会话 ID；不传则自动新建一个会话"
    )
    user_id: str = Field("anonymous", description="用户标识（新建会话时使用）")
    stream: bool = Field(True, description="是否流式返回")


class ChatResponse(BaseModel):
    session_id: str
    turn: int
    content: str
    latency_ms: int = 0
    usage: Dict[str, int] = Field(default_factory=dict)
    offline: bool = False
    error: str = ""


class HandoffReplyRequest(BaseModel):
    text: str = Field(..., description="坐席回复内容", min_length=1)
    agent: str = Field("人工客服", description="坐席名称")


class HandoffNoteRequest(BaseModel):
    note: str = Field("", description="办结备注")
    agent: str = Field("人工客服", description="坐席名称")


# ---------------------------------------------------------------------------
# 评测与观测（P6）
# ---------------------------------------------------------------------------
@router.get("/eval/scenarios", response_model=Dict[str, Any])
async def list_eval_scenarios(rounds: int = Query(20, ge=5, le=100)) -> Dict[str, Any]:
    """列出评测场景（便于查看覆盖了哪些方向）。"""
    from ..eval_scenarios import all_scenarios

    scenarios = all_scenarios(rounds=rounds)
    return {
        "total": len(scenarios),
        "scenarios": [
            {
                "key": item.key,
                "name": item.name,
                "tags": item.tags,
                "turns": len(item.turns),
                "expect_handoff": item.expect_handoff,
            }
            for item in scenarios
        ],
    }


@router.post("/eval/run", response_model=Dict[str, Any])
async def run_eval(
    rounds: int = Query(20, ge=5, le=60),
    only: Optional[str] = Query(None, description="只跑带这些标签的场景（逗号分隔）"),
) -> Dict[str, Any]:
    """跑一次评测并返回指标。

    注意：**接口永远以离线模式运行**（不调用真实模型），
    因此报告里 `mode=offline`，只反映工具路由/状态/规则。
    回答质量请用 `scripts/run_eval.py --online` 或在线验收脚本。
    """
    from ..eval_scenarios import all_scenarios
    from ..evaluation import Evaluator

    service = get_service()
    scenarios = all_scenarios(rounds=rounds)
    if only:
        wanted = {item.strip() for item in only.split(",") if item.strip()}
        scenarios = [item for item in scenarios if wanted & set(item.tags)]

    evaluator = Evaluator(service, get_store(), online=False)
    report = evaluator.run(scenarios)
    return {"report": report, "readable": Evaluator.format_report(report)}


@router.get("/sessions/{session_id}/traces", response_model=Dict[str, Any])
async def get_session_traces(session_id: str) -> Dict[str, Any]:
    """逐轮轨迹：每轮的决策、工具、记忆、转人工与用量。

    长会话出问题时靠它定位：哪一轮开始变慢、哪一轮开始重复追问、
    哪一轮压缩了上下文、token 怎么长的。
    """
    store = get_store()
    if store.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail=f"会话不存在：{session_id}")
    traces: List[Dict[str, Any]] = []
    prompt_tokens = 0
    for message in store.list_messages(session_id, roles=["assistant"]):
        trace = (message.get("meta") or {}).get("trace")
        if not trace:
            continue
        prompt_tokens += int(trace.get("prompt_tokens") or 0)
        traces.append(
            {
                "turn": trace.get("turn"),
                "prompt_tokens": trace.get("prompt_tokens"),
                "completion_tokens": trace.get("completion_tokens"),
                "prompt_chars": trace.get("prompt_chars"),
                "history_messages": trace.get("history_messages"),
                "compressed": trace.get("compressed"),
                "latency_ms": trace.get("latency_ms"),
                "first_token_ms": trace.get("first_token_ms"),
                "model": trace.get("model"),
                "plan": (trace.get("plan") or {}).get("action"),
                "tools": [run.get("tool") for run in (trace.get("tools") or {}).get("runs", [])],
                "escalation": (trace.get("escalation") or {}).get("decision"),
                "handoff_id": (trace.get("handoff") or {}).get("handoff_id", ""),
                "memory": {
                    "summary_updated": (trace.get("memory") or {}).get("summary_updated"),
                    "compression_ratio": (trace.get("memory") or {}).get("compression_ratio"),
                },
            }
        )
    return {
        "session_id": session_id,
        "turns": len(traces),
        "prompt_tokens_total": prompt_tokens,
        "traces": traces,
    }


# ---------------------------------------------------------------------------
# 人工工作台（P5）
# ---------------------------------------------------------------------------
@router.get("/handoffs", response_model=Dict[str, Any])
async def list_handoffs(
    status: Optional[str] = Query(
        None, description="按状态过滤：waiting / accepted / resolved"
    ),
    limit: int = Query(50, ge=1, le=200),
) -> Dict[str, Any]:
    """坐席工作台：工单列表（等待中的排最前）。"""
    items = get_service().handoffs.workspace(status=status, limit=limit)
    return {
        "handoffs": items,
        "total": len(items),
        "waiting": sum(1 for item in items if item["status"] == "waiting"),
    }


@router.get("/handoffs/{handoff_id}", response_model=Dict[str, Any])
async def get_handoff(handoff_id: str) -> Dict[str, Any]:
    """工单详情：完整交接单 + 会话消息 + 结构化状态。"""
    detail = get_service().handoffs.detail(handoff_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"工单不存在：{handoff_id}")
    return detail


@router.post("/handoffs/{handoff_id}/accept", response_model=Dict[str, Any])
async def accept_handoff(handoff_id: str, agent: str = Query("坐席")) -> Dict[str, Any]:
    """坐席认领工单。"""
    handoff = get_service().handoffs.accept(handoff_id, agent=agent)
    if handoff is None:
        raise HTTPException(status_code=404, detail=f"工单不存在：{handoff_id}")
    return {"handoff": handoff}


@router.post("/handoffs/{handoff_id}/reply", response_model=Dict[str, Any])
async def reply_handoff(handoff_id: str, payload: HandoffReplyRequest) -> Dict[str, Any]:
    """坐席回复：写入会话并让会话回到活跃态，用户可接着对话。

    回复后会话**不重建状态**——已核实的槽位与事实原样保留，
    所以人工接手后用户不会被再问一遍订单号。
    """
    handoff = get_service().handoffs.reply(handoff_id, payload.text, agent=payload.agent)
    if handoff is None:
        raise HTTPException(status_code=404, detail=f"工单不存在：{handoff_id}")
    session_id = handoff["session_id"]
    return {
        "handoff": handoff,
        "session": get_store().get_session(session_id),
        "state": StateSnapshot.load(get_store(), session_id).summary(),
    }


@router.post("/handoffs/{handoff_id}/resolve", response_model=Dict[str, Any])
async def resolve_handoff(handoff_id: str, payload: HandoffNoteRequest) -> Dict[str, Any]:
    """办结工单。"""
    handoff = get_service().handoffs.resolve(handoff_id, note=payload.note)
    if handoff is None:
        raise HTTPException(status_code=404, detail=f"工单不存在：{handoff_id}")
    return {"handoff": handoff}


# ---------------------------------------------------------------------------
# 会话
# ---------------------------------------------------------------------------
@router.post("/sessions", response_model=Dict[str, Any])
async def create_session(payload: CreateSessionRequest) -> Dict[str, Any]:
    """新建会话。"""
    session = get_service().open_session(user_id=payload.user_id, title=payload.title)
    return {"session": session}


@router.get("/sessions", response_model=Dict[str, Any])
async def list_sessions(
    user_id: Optional[str] = Query(None, description="按用户过滤"),
    limit: int = Query(20, ge=1, le=200),
) -> Dict[str, Any]:
    """会话列表（按最近更新倒序）。"""
    sessions = get_store().list_sessions(user_id=user_id, limit=limit)
    return {"sessions": sessions, "total": len(sessions)}


@router.get("/sessions/{session_id}", response_model=Dict[str, Any])
async def get_session(session_id: str) -> Dict[str, Any]:
    """会话详情 + 消息历史 + 统计。"""
    store = get_store()
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"会话不存在：{session_id}")
    return {
        "session": session,
        "messages": store.list_messages(session_id),
        "stats": get_service().session_stats(session_id),
    }


@router.get("/sessions/{session_id}/memory", response_model=Dict[str, Any])
async def get_session_memory(session_id: str) -> Dict[str, Any]:
    """会话的记忆体检：分层记忆各层规模、压缩比、事实与画像。

    长会话的"记忆有没有真的生效"靠这个接口看，而不是靠感觉。
    """
    store = get_store()
    if store.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail=f"会话不存在：{session_id}")
    summary = store.latest_summary(session_id)
    return {
        "report": get_service().memory.memory_report(session_id),
        "summary": (summary or {}).get("content") or "",
        "facts": store.get_facts(session_id),
    }


@router.get("/sessions/{session_id}/state", response_model=Dict[str, Any])
async def get_session_state(session_id: str) -> Dict[str, Any]:
    """会话的结构化状态：正在办什么、还缺哪些信息、还有哪些待办。"""
    store = get_store()
    if store.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail=f"会话不存在：{session_id}")
    snapshot = StateSnapshot.load(store, session_id)
    detail = snapshot.summary()
    detail["pending_tasks"] = [
        {"kind": t.kind, "name": t.name, "missing": _missing_slots(t)}
        for t in snapshot.pending_tasks
    ]
    active = snapshot.active_task
    detail["active_missing"] = _missing_slots(active) if active else []
    return detail


def _missing_slots(task: Any) -> List[str]:
    if task is None:
        return []
    task_def = task_for_intent(task.kind)
    if task_def is None:
        return []
    return missing_required(task_def, task.filled)


@router.post("/sessions/{session_id}/compact", response_model=Dict[str, Any])
async def compact_session(session_id: str) -> Dict[str, Any]:
    """手动触发一次滚动压缩（调试与演示用；正常流程会自动触发）。"""
    store = get_store()
    if store.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail=f"会话不存在：{session_id}")
    flags = get_service().memory.compress(session_id)
    return {"memory": flags.as_dict(), "report": get_service().memory.memory_report(session_id)}


@router.delete("/sessions/{session_id}", response_model=Dict[str, Any])
async def delete_session(session_id: str) -> Dict[str, Any]:
    if not get_store().delete_session(session_id):
        raise HTTPException(status_code=404, detail=f"会话不存在：{session_id}")
    return {"deleted": True, "session_id": session_id}


# ---------------------------------------------------------------------------
# 对话
# ---------------------------------------------------------------------------
def _resolve_session(session_id: Optional[str], user_id: str) -> str:
    """拿到有效会话 ID；不存在就新建（前端首次进入时可只发消息不带 ID）。"""
    service = get_service()
    if session_id:
        if get_store().get_session(session_id) is None:
            raise HTTPException(status_code=404, detail=f"会话不存在：{session_id}")
        return session_id
    return str(service.open_session(user_id=user_id)["id"])


def _sse_response(service: ConversationService, session_id: str, message: str) -> StreamingResponse:
    return StreamingResponse(
        service.stream_turn(session_id, message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 反向代理下禁用缓冲，否则流式会攒成一坨
            "X-Session-Id": session_id,
        },
    )


def _read_sse(response: Any) -> List[Dict[str, Any]]:
    """把 SSE 响应解析成事件列表。

    ``StreamingResponse`` 在 TestClient 下是"整段文本"，在真实网络下是增量流，
    两种形态都要能解析，所以按文本行解析（而不是依赖 ``iter_lines`` 的行号语义）。
    """
    import json as _json

    raw = getattr(response, "text", None)
    if raw is None:  # pragma: no cover - 防御性分支
        content = getattr(response, "content", b"")
        raw = content.decode("utf-8") if isinstance(content, bytes) else str(content)

    events: List[Dict[str, Any]] = []
    event = ""
    for line in str(raw).splitlines():
        if line.startswith("event: "):
            event = line[7:].strip()
        elif line.startswith("data: "):
            try:
                events.append({"event": event, "data": _json.loads(line[6:])})
            except ValueError:  # pragma: no cover - 防御性分支
                continue
    return events


@router.post("/chat")
async def chat(payload: ChatRequest) -> Any:
    """发一条消息。

    ``stream=true``（默认）返回 SSE 流；``stream=false`` 返回一次性 JSON，
    便于脚本与测试调用。
    """
    session_id = _resolve_session(payload.session_id, payload.user_id)
    service = get_service()

    if payload.stream:
        return _sse_response(service, session_id, payload.message)

    result = service.chat_once(session_id, payload.message)
    # 用 is not None 判断：成功时 done 事件的 error 字段是空字符串
    if result.get("error") is not None and "turn" not in result:
        return JSONResponse_with_session(
            {"detail": result.get("error") or "本轮对话未能完成"},
            session_id,
            status_code=502,
        )
    return ChatResponse(
        session_id=session_id,
        turn=int(result.get("turn") or 0),
        content=result.get("content") or "",
        latency_ms=int(result.get("latency_ms") or 0),
        usage=result.get("usage") or {},
        offline=bool(result.get("offline")),
        error=result.get("error") or "",
    )


@router.get("/chat/stream")
async def chat_stream(
    message: str = Query(..., min_length=1, description="用户消息"),
    session_id: Optional[str] = Query(None),
    user_id: str = Query("anonymous"),
) -> StreamingResponse:
    """SSE 版对话（EventSource 只能发 GET，所以补一个 GET 入口）。"""
    resolved = _resolve_session(session_id, user_id)
    return _sse_response(get_service(), resolved, message)


def JSONResponse_with_session(content: Dict[str, Any], session_id: str, status_code: int):
    """带会话 ID 的错误响应（方便调用方在失败后仍能继续这个会话）。"""
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=status_code, content=content, headers={"X-Session-Id": session_id}
    )
