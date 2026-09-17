import { useCallback, useEffect, useState } from 'react'
import { api } from '../api'
import type { HandoffDetail, HandoffItem } from '../types'

interface Props {
  onChanged: () => void
}

const STATUS_TEXT: Record<string, string> = {
  waiting: '等待接手',
  accepted: '处理中',
  resolved: '已办结'
}

/**
 * 人工工作台：把"转人工"这条链路做完整。
 *
 * 左侧待办队列，右侧**交接单**——坐席不用从零问一遍，
 * 回复后会话自动恢复活跃，用户可以接着对话。
 */
export default function Workspace({ onChanged }: Props) {
  const [items, setItems] = useState<HandoffItem[]>([])
  const [waiting, setWaiting] = useState(0)
  const [selected, setSelected] = useState<string | null>(null)
  const [detail, setDetail] = useState<HandoffDetail | null>(null)
  const [reply, setReply] = useState('')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')

  const load = useCallback(async () => {
    setError('')
    try {
      const body = await api.handoffs()
      setItems(body.handoffs)
      setWaiting(body.waiting)
      if (!selected && body.handoffs.length) setSelected(body.handoffs[0].handoff_id)
    } catch (exc) {
      setError((exc as Error).message)
    }
  }, [selected])

  const loadDetail = useCallback(async (id: string) => {
    try {
      setDetail(await api.handoffDetail(id))
    } catch (exc) {
      setError((exc as Error).message)
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  useEffect(() => {
    if (selected) void loadDetail(selected)
  }, [selected, loadDetail])

  const act = async (action: 'accept' | 'reply' | 'resolve') => {
    if (!selected) return
    setError('')
    setNotice('')
    try {
      if (action === 'accept') {
        await api.acceptHandoff(selected)
        setNotice('已认领工单')
      } else if (action === 'reply') {
        if (!reply.trim()) return
        await api.replyHandoff(selected, reply.trim())
        setReply('')
        setNotice('回复已发出，用户会话已恢复，可以继续对话')
      } else {
        await api.resolveHandoff(selected, '已处理完成')
        setNotice('工单已办结')
      }
      await load()
      await loadDetail(selected)
      onChanged()
    } catch (exc) {
      setError((exc as Error).message)
    }
  }

  return (
    <div className="layout">
      <div className="sidebar" style={{ width: 320 }}>
        <div className="head">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <strong className="small">工单队列</strong>
            <span className={`badge ${waiting ? 'warn' : 'ok'}`}>等待 {waiting}</span>
          </div>
        </div>
        <div className="session-list">
          {items.length === 0 && (
            <div className="empty small">
              还没有工单。
              <div style={{ marginTop: 6 }}>在对话里说「我要转人工」就会出现。</div>
            </div>
          )}
          {items.map((item) => (
            <div
              key={item.handoff_id}
              className={`session-item ${item.handoff_id === selected ? 'active' : ''}`}
              onClick={() => setSelected(item.handoff_id)}
            >
              <div className="title">
                <span className={`badge ${item.status === 'waiting' ? 'warn' : item.status === 'resolved' ? 'ok' : ''}`}>
                  {STATUS_TEXT[item.status] ?? item.status}
                </span>{' '}
                {item.demand || item.session_title}
              </div>
              <div className="meta mono">{item.handoff_id}</div>
              <div className="meta">
                轮次 {item.turns} · 情绪 {item.sentiment}
                {item.blocker ? ` · ${item.blocker.slice(0, 22)}` : ''}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="chat" style={{ padding: 0, overflowY: 'auto' }}>
        {!detail && <div className="empty">选择左侧工单查看交接单</div>}
        {detail && (
          <div style={{ padding: 18 }}>
            {error && <div className="error-text">⚠ {error}</div>}
            {notice && <div className="card"><span className="badge ok">完成</span> {notice}</div>}

            <div className="card">
              <div className="title">
                交接单 <span className="mono">{detail.handoff.id}</span>
              </div>
              <div className="small muted">
                状态 {STATUS_TEXT[detail.handoff.status] ?? detail.handoff.status} · 转接原因 {detail.handoff.reason}
              </div>
              <div style={{ marginTop: 8 }}>
                {detail.handoff.triggers?.map((item) => (
                  <span className="chip warn" key={item.rule} title={item.reason}>
                    {item.rule}
                  </span>
                ))}
              </div>
            </div>

            <h3 className="small muted">交接单内容（坐席视角）</h3>
            <div className="pre">{detail.packet_text}</div>

            <h3 className="small muted">会话上下文（{detail.messages.length} 条）</h3>
            <div className="pre">
              {detail.messages
                .filter((item) => item.role !== 'summary')
                .map((item) => {
                  const who =
                    item.role === 'user' ? '用户' : item.role === 'human_agent' ? '人工' : '客服'
                  return `[${who}] ${item.content}`
                })
                .join('\n\n')}
            </div>

            <h3 className="small muted">结构化状态</h3>
            <div className="card">
              <div className="kv">
                <span className="k">槽位</span>
                <span className="v">
                  {Object.entries(detail.state.slots ?? {})
                    .map(([k, v]) => `${k}=${v}`)
                    .join('；') || '无'}
                </span>
              </div>
              <div className="kv">
                <span className="k">待办</span>
                <span className="v">
                  {(detail.state.tasks ?? []).map((task) => (task.name || task.kind)).join('、') || '无'}
                </span>
              </div>
            </div>

            <div className="card">
              <textarea
                rows={3}
                placeholder="以人工客服身份回复用户（会写入会话，用户可继续对话）"
                value={reply}
                onChange={(event) => setReply(event.target.value)}
              />
              <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
                <button onClick={() => act('accept')} disabled={detail.handoff.status !== 'waiting'}>
                  认领
                </button>
                <button className="primary" onClick={() => act('reply')} disabled={!reply.trim()}>
                  回复
                </button>
                <button onClick={() => act('resolve')} disabled={detail.handoff.status === 'resolved'}>
                  办结
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
