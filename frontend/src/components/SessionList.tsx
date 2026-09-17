import type { Session } from '../types'

interface Props {
  sessions: Session[]
  current: string | null
  onSelect: (id: string) => void
  onCreate: () => void
  onRefresh: () => void
}

const STATUS_TEXT: Record<string, string> = {
  active: '进行中',
  waiting_human: '等待人工',
  closed: '已结束'
}

export default function SessionList({ sessions, current, onSelect, onCreate, onRefresh }: Props) {
  return (
    <div className="sidebar">
      <div className="head">
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="primary" style={{ flex: 1 }} onClick={onCreate}>
            + 新会话
          </button>
          <button className="ghost" onClick={onRefresh} title="刷新列表">
            ⟳
          </button>
        </div>
      </div>
      <div className="session-list">
        {sessions.length === 0 && <div className="empty small">还没有会话</div>}
        {sessions.map((item) => (
          <div
            key={item.id}
            className={`session-item ${item.id === current ? 'active' : ''}`}
            onClick={() => onSelect(item.id)}
          >
            <div className="title">{item.title || '（未命名会话）'}</div>
            <div className="meta">
              {item.turn_count} 轮 ·{' '}
              <span className={item.status === 'waiting_human' ? 'error-text' : ''}>
                {STATUS_TEXT[item.status] ?? item.status}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
