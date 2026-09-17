import { useCallback, useEffect, useRef, useState } from 'react'
import { api, streamChat } from './api'
import type { DonePayload, Health, Session, StartPayload, Message } from './types'
import Inspector from './components/Inspector'
import SessionList from './components/SessionList'
import Workspace from './components/Workspace'
import EvalPanel from './components/EvalPanel'

type View = 'chat' | 'workspace' | 'eval'

const USER_ID = 'web-user'

export default function App() {
  const [view, setView] = useState<View>('chat')
  const [health, setHealth] = useState<Health | null>(null)
  const [sessions, setSessions] = useState<Session[]>([])
  const [current, setCurrent] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [draft, setDraft] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [streamText, setStreamText] = useState('')
  const [startInfo, setStartInfo] = useState<StartPayload | null>(null)
  const [lastDone, setLastDone] = useState<DonePayload | null>(null)
  const [error, setError] = useState('')
  const [refreshKey, setRefreshKey] = useState(0)
  const messagesRef = useRef<HTMLDivElement>(null)

  const loadHealth = useCallback(async () => {
    try {
      setHealth(await api.health())
    } catch (exc) {
      setError((exc as Error).message)
    }
  }, [])

  const loadSessions = useCallback(async () => {
    try {
      const body = await api.listSessions(USER_ID)
      setSessions(body.sessions)
    } catch (exc) {
      setError((exc as Error).message)
    }
  }, [])

  const openSession = useCallback(async (id: string) => {
    setCurrent(id)
    setStartInfo(null)
    setLastDone(null)
    try {
      const detail = await api.sessionDetail(id)
      setMessages(detail.messages.filter((m) => m.role !== 'summary'))
    } catch (exc) {
      setError((exc as Error).message)
    }
  }, [])

  useEffect(() => {
    void loadHealth()
    void loadSessions()
  }, [loadHealth, loadSessions])

  useEffect(() => {
    const node = messagesRef.current
    if (node) node.scrollTop = node.scrollHeight
  }, [messages, streamText])

  const newSession = async () => {
    try {
      const body = await api.createSession(USER_ID)
      setSessions((prev) => [body.session, ...prev])
      await openSession(body.session.id)
      setView('chat')
    } catch (exc) {
      setError((exc as Error).message)
    }
  }

  const send = async () => {
    const text = draft.trim()
    if (!text || streaming) return
    setError('')
    setDraft('')
    setStreaming(true)
    setStreamText('')
    setStartInfo(null)
    setLastDone(null)

    // 乐观插入用户消息，并留一条空的助手消息（流式往里填字）
    const now = new Date().toISOString()
    setMessages((prev) => [
      ...prev,
      {
        id: Date.now(),
        session_id: current ?? '',
        role: 'user',
        content: text,
        turn: 0,
        token_estimate: 0,
        meta: {},
        created_at: now
      } as Message
    ])

    try {
      await streamChat(current, text, USER_ID, {
        onStart: (payload) => setStartInfo(payload),
        onDelta: (piece) => setStreamText((prev) => prev + piece),
        onDone: (payload) => {
          setLastDone(payload)
          setMessages((prev) => [
            ...prev,
            {
              id: Date.now() + 1,
              session_id: current ?? '',
              role: 'assistant',
              content: payload.content,
              turn: payload.turn,
              token_estimate: payload.usage?.completion_tokens ?? 0,
              meta: {},
              created_at: new Date().toISOString()
            } as Message
          ])
          setStreamText('')
          void loadSessions()
          setRefreshKey((prev) => prev + 1)
        },
        onError: (message) => setError(message)
      })
    } catch (exc) {
      setError((exc as Error).message)
    } finally {
      setStreaming(false)
      // 拉到真实 session id（首次对话是后端自动建的会话）
      if (!current) {
        void loadSessions()
        try {
          const body = await api.listSessions(USER_ID)
          setSessions(body.sessions)
          const latest = body.sessions[0]
          if (latest) setCurrent(latest.id)
        } catch {
          /* 忽略：下一轮会重新拉 */
        }
      }
    }
  }

  const onKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      void send()
    }
  }

  const online = health?.llm.online

  return (
    <div className="app">
      <div className="topbar">
        <h1>长会话客服 Agent</h1>
        {health && (
          <span className={`badge ${online ? 'ok' : 'warn'}`}>
            {online ? `在线 · ${health.llm.model}` : '离线模板模式'}
          </span>
        )}
        <span className="badge">数据：本地模拟</span>
        <div className="spacer" />
        <div className="nav">
          <button className={view === 'chat' ? 'active' : ''} onClick={() => setView('chat')}>
            对话
          </button>
          <button className={view === 'workspace' ? 'active' : ''} onClick={() => setView('workspace')}>
            人工工作台
          </button>
          <button className={view === 'eval' ? 'active' : ''} onClick={() => setView('eval')}>
            评测
          </button>
        </div>
      </div>

      {view === 'chat' && (
        <div className="layout">
          <SessionList
            sessions={sessions}
            current={current}
            onSelect={openSession}
            onCreate={newSession}
            onRefresh={loadSessions}
          />

          <div className="chat">
            <div className="messages" ref={messagesRef}>
              {messages.length === 0 && !streaming && (
                <div className="empty">
                  还没有对话。可以直接问，例如：
                  <div className="small" style={{ marginTop: 8 }}>
                    「订单号 SO20260101，帮我看看快递到哪了」
                    <br />
                    「退货运费一般谁承担？」
                    <br />
                    「订单号 SO20260102，尺码不合适，我要退货」
                  </div>
                </div>
              )}

              {messages.map((item) => (
                <MessageRow key={item.id} message={item} />
              ))}

              {streaming && (
                <div className="msg assistant">
                  <div className="who">客</div>
                  <div className="bubble">
                    <div className="text">{streamText || '思考中…'}</div>
                    <div className="sub">正在生成…</div>
                  </div>
                </div>
              )}

              {refreshKey > 0 && startInfo?.handoff?.handoff_id && !streaming && (
                <div className="card">
                  <div className="title">
                    <span className="badge warn">已转人工</span> 工单 {startInfo.handoff.handoff_id}
                  </div>
                  <div className="small muted">转接原因：{startInfo.handoff.reason}</div>
                  <div className="small" style={{ marginTop: 6 }}>
                    到「人工工作台」可以查看交接单、以坐席身份回复，然后继续这个会话。
                  </div>
                </div>
              )}
            </div>

            <div className="composer">
              <div className="row">
                <textarea
                  rows={2}
                  value={draft}
                  placeholder="输入消息，Enter 发送，Shift+Enter 换行"
                  onChange={(event) => setDraft(event.target.value)}
                  onKeyDown={onKeyDown}
                  disabled={streaming}
                />
                <button className="primary" onClick={send} disabled={streaming || !draft.trim()}>
                  {streaming ? '生成中' : '发送'}
                </button>
              </div>
              <div className="hint">
                {error && <span className="error-text">⚠ {error}</span>}
                {!error && lastDone && (
                  <>
                    本轮 {lastDone.latency_ms} ms（首字 {lastDone.first_token_ms} ms）·
                    tokens {lastDone.usage?.prompt_tokens ?? 0}/{lastDone.usage?.completion_tokens ?? 0}
                    {lastDone.error ? ` · ⚠ ${lastDone.error}` : ''}
                  </>
                )}
              </div>
            </div>
          </div>

          <Inspector
            sessionId={current}
            start={startInfo}
            onChanged={() => setRefreshKey((prev) => prev + 1)}
          />
        </div>
      )}

      {view === 'workspace' && <Workspace onChanged={() => setRefreshKey((prev) => prev + 1)} />}
      {view === 'eval' && <EvalPanel />}
    </div>
  )
}

function MessageRow({ message }: { message: Message }) {
  const label =
    message.role === 'user' ? '我' : message.role === 'human_agent' ? '人工' : '客'
  const className = message.role === 'user' ? 'user' : message.role === 'human_agent' ? 'human_agent' : 'assistant'
  return (
    <div className={`msg ${className}`}>
      <div className="who">{label}</div>
      <div className="bubble">
        <div className="text">{message.content}</div>
        <div className="sub">
          {message.role === 'human_agent' ? '人工坐席 · ' : ''}
          {message.created_at}
        </div>
      </div>
    </div>
  )
}
