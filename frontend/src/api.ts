/**
 * 后端调用封装。
 *
 * 流式对话用 **fetch + ReadableStream** 而不是 EventSource：
 * EventSource 只能发 GET 且无法带请求体，而这里需要 POST 一段 JSON
 * （带 session_id / user_id）。自己解析 SSE 帧也更可控。
 */

import type {
  EvalReport,
  HandoffDetail,
  HandoffItem,
  Health,
  MemoryReport,
  Session,
  SessionDetail,
  StartPayload,
  DonePayload,
  Traces
} from './types'

/**
 * 接口基址。
 *
 * 因为界面挂在后端 `/ui` 下，页面地址不是站点根目录——
 * 用相对路径 `api/xxx` 会被解析成 `/ui/api/xxx` 而 404。
 * 这里统一用**站点绝对路径**，开发（Vite 代理）与生产（同源）都成立。
 */
const API_BASE = '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...init
  })
  if (!response.ok) {
    let detail = `请求失败（HTTP ${response.status}）`
    try {
      const body = await response.json()
      if (body?.detail) detail = String(body.detail)
    } catch {
      /* 保持默认提示 */
    }
    throw new Error(detail)
  }
  return (await response.json()) as T
}

export const api = {
  health: () => request<Health>(`${API_BASE}/health`),

  listSessions: (userId?: string) =>
    request<{ sessions: Session[] }>(
      `${API_BASE}/sessions${userId ? `?user_id=${encodeURIComponent(userId)}` : ''}`
    ),

  createSession: (userId: string, title = '') =>
    request<{ session: Session }>(`${API_BASE}/sessions`, {
      method: 'POST',
      body: JSON.stringify({ user_id: userId, title })
    }),

  sessionDetail: (id: string) => request<SessionDetail>(`${API_BASE}/sessions/${id}`),

  memory: (id: string) => request<MemoryReport>(`${API_BASE}/sessions/${id}/memory`),

  traces: (id: string) => request<Traces>(`${API_BASE}/sessions/${id}/traces`),

  handoffs: (status?: string) =>
    request<{ handoffs: HandoffItem[]; total: number; waiting: number }>(
      `${API_BASE}/handoffs${status ? `?status=${status}` : ''}`
    ),

  handoffDetail: (id: string) => request<HandoffDetail>(`${API_BASE}/handoffs/${id}`),

  acceptHandoff: (id: string, agent = '坐席') =>
    request<unknown>(`${API_BASE}/handoffs/${id}/accept?agent=${encodeURIComponent(agent)}`, {
      method: 'POST'
    }),

  replyHandoff: (id: string, text: string, agent = '人工客服') =>
    request<unknown>(`${API_BASE}/handoffs/${id}/reply`, {
      method: 'POST',
      body: JSON.stringify({ text, agent })
    }),

  resolveHandoff: (id: string, note: string) =>
    request<unknown>(`${API_BASE}/handoffs/${id}/resolve`, {
      method: 'POST',
      body: JSON.stringify({ note, agent: '人工客服' })
    }),

  evalScenarios: () =>
    request<{ total: number; scenarios: { key: string; name: string; tags: string[]; turns: number }[] }>(
      `${API_BASE}/eval/scenarios`
    ),

  runEval: (rounds = 12, only?: string) =>
    request<EvalReport>(
      `${API_BASE}/eval/run?rounds=${rounds}${only ? `&only=${encodeURIComponent(only)}` : ''}`,
      { method: 'POST' }
    )
}

export interface StreamHandlers {
  onStart?: (payload: StartPayload) => void
  onDelta?: (text: string) => void
  onDone?: (payload: DonePayload) => void
  onError?: (message: string, retryable: boolean) => void
}

/**
 * 发一条消息并消费 SSE 流。
 *
 * SSE 帧以空行结束，`event:` 与 `data:` 各占一行——必须按帧累积，
 * 否则载荷里的换行会被误当成帧边界（后端 README 里记了这个坑）。
 */
export async function streamChat(
  sessionId: string | null,
  message: string,
  userId: string,
  handlers: StreamHandlers,
  signal?: AbortSignal
): Promise<void> {
  const response = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      session_id: sessionId,
      user_id: userId,
      stream: true
    }),
    signal
  })

  if (!response.ok) {
    let detail = `请求失败（HTTP ${response.status}）`
    try {
      const body = await response.json()
      if (body?.detail) detail = String(body.detail)
    } catch {
      /* 保持默认提示 */
    }
    handlers.onError?.(detail, false)
    return
  }
  if (!response.body) {
    handlers.onError?.('响应没有可读流', false)
    return
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  const flushFrame = (frame: string) => {
    let event = ''
    const dataLines: string[] = []
    for (const line of frame.split('\n')) {
      if (line.startsWith('event: ')) event = line.slice(7).trim()
      else if (line.startsWith('data: ')) dataLines.push(line.slice(6))
    }
    if (!dataLines.length) return
    let payload: Record<string, unknown>
    try {
      payload = JSON.parse(dataLines.join('\n'))
    } catch {
      return
    }
    if (event === 'start') handlers.onStart?.(payload as unknown as StartPayload)
    else if (event === 'delta') handlers.onDelta?.(String(payload.text ?? ''))
    else if (event === 'done') handlers.onDone?.(payload as unknown as DonePayload)
    else if (event === 'error')
      handlers.onError?.(String(payload.message ?? '未知错误'), Boolean(payload.retryable))
  }

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    // 帧之间以空行分隔
    let index = buffer.indexOf('\n\n')
    while (index !== -1) {
      const frame = buffer.slice(0, index)
      buffer = buffer.slice(index + 2)
      flushFrame(frame)
      index = buffer.indexOf('\n\n')
    }
  }
  if (buffer.trim()) flushFrame(buffer)
}
