import { useCallback, useEffect, useState } from 'react'
import { api } from '../api'
import type { MemoryReport, StartPayload, StateSnapshot, Traces } from '../types'

interface Props {
  sessionId: string | null
  start: StartPayload | null
  onChanged: () => void
}

type Tab = 'decision' | 'state' | 'memory' | 'traces'

/**
 * 右侧检视面板：把 Agent 的"内部状态"直接摊开给使用者看。
 *
 * 这是长会话项目最需要的东西——用户看到的是一句话回复，
 * 但真正决定行为的是槽位、待办、压缩与转人工判定。
 */
export default function Inspector({ sessionId, start, onChanged }: Props) {
  const [tab, setTab] = useState<Tab>('decision')
  const [state, setState] = useState<StateSnapshot | null>(null)
  const [memory, setMemory] = useState<MemoryReport | null>(null)
  const [traces, setTraces] = useState<Traces | null>(null)
  const [error, setError] = useState('')

  const load = useCallback(async () => {
    if (!sessionId) return
    setError('')
    try {
      // 结构化状态在后端是独立接口（含待办栈与仍缺的槽位）
      const [stateBody, memoryBody, traceBody] = await Promise.all([
        fetchState(sessionId),
        api.memory(sessionId),
        api.traces(sessionId)
      ])
      setState(stateBody)
      setMemory(memoryBody)
      setTraces(traceBody)
    } catch (exc) {
      setError((exc as Error).message)
    }
  }, [sessionId])

  useEffect(() => {
    void load()
  }, [load, start, onChanged])

  if (!sessionId) {
    return (
      <div className="inspector">
        <div className="empty small">选择一个会话后，这里会显示它的状态、记忆与轨迹。</div>
      </div>
    )
  }

  const plan = start?.plan
  const tools = start?.tools?.runs ?? []
  const escalation = start?.escalation
  const memoryFlags = start?.memory

  return (
    <div className="inspector">
      <div className="tabs">
        <button className={`tiny ${tab === 'decision' ? 'primary' : ''}`} onClick={() => setTab('decision')}>
          本轮决策
        </button>
        <button className={`tiny ${tab === 'state' ? 'primary' : ''}`} onClick={() => setTab('state')}>
          状态
        </button>
        <button className={`tiny ${tab === 'memory' ? 'primary' : ''}`} onClick={() => setTab('memory')}>
          记忆
        </button>
        <button className={`tiny ${tab === 'traces' ? 'primary' : ''}`} onClick={() => setTab('traces')}>
          轨迹
        </button>
      </div>

      {error && <div className="error-text">⚠ {error}</div>}

      {tab === 'decision' && (
        <>
          <h3>本轮计划</h3>
          {!plan && <div className="small muted">还没有发起对话</div>}
          {plan && (
            <div className="card">
              <div className="kv">
                <span className="k">动作</span>
                <span className="v">{actionText(plan.action)}</span>
              </div>
              {plan.intents?.length ? (
                <div className="kv">
                  <span className="k">识别意图</span>
                  <span className="v">{plan.intents.join('、')}</span>
                </div>
              ) : null}
              {plan.ask && (
                <div className="kv">
                  <span className="k">追问</span>
                  <span className="v">{plan.ask.target}</span>
                </div>
              )}
              {plan.clarification && (
                <div className="small" style={{ marginTop: 6 }}>
                  <span className="badge warn">需要澄清指代</span>
                </div>
              )}
              {plan.new_tasks?.length ? (
                <div style={{ marginTop: 6 }}>
                  <span className="small muted">新登记待办：</span>
                  {plan.new_tasks.map((item) => (
                    <span className="chip" key={item}>
                      {item}
                    </span>
                  ))}
                </div>
              ) : null}
            </div>
          )}

          <h3>工具调用</h3>
          {tools.length === 0 && <div className="small muted">本轮没有调用工具</div>}
          {tools.map((run, index) => (
            <div className="card" key={index}>
              <div className="kv">
                <span className="k mono">{run.tool}</span>
                <span className={`chip ${run.ok ? 'ok' : 'danger'}`}>
                  {run.ok ? '成功' : run.error_code || '失败'} · {run.latency_ms}ms
                </span>
              </div>
              {run.error && <div className="error-text">{run.error}</div>}
              {run.needs_confirmation && <div className="chip warn">等待用户确认</div>}
            </div>
          ))}
          {start?.tools?.confirmation_prompt && (
            <div className="card">
              <div className="title">待确认操作</div>
              <div className="small">{start.tools.confirmation_prompt}</div>
            </div>
          )}
          {start?.tools?.honest_failures?.length ? (
            <div className="card">
              <div className="title">如实告知用户</div>
              {start.tools.honest_failures.map((item, index) => (
                <div className="small" key={index}>
                  · {item}
                </div>
              ))}
            </div>
          ) : null}

          <h3>转人工判定</h3>
          {escalation ? (
            <div className="card">
              <div className="kv">
                <span className="k">结论</span>
                <span className={`chip ${escalation.decision === 'handoff' ? 'danger' : escalation.decision === 'defer' ? 'warn' : 'ok'}`}>
                  {escalation.decision === 'handoff' ? '转人工' : escalation.decision === 'defer' ? '暂缓' : '继续服务'}
                </span>
              </div>
              <div className="kv">
                <span className="k">情绪分</span>
                <span className="v">
                  {escalation.sentiment?.score ?? 0} / 阈值 {escalation.threshold}
                </span>
              </div>
              {escalation.reason && <div className="small muted">{escalation.reason}</div>}
              <div style={{ marginTop: 6 }}>
                {escalation.triggers?.map((item) => (
                  <span className="chip warn" key={item.rule} title={item.reason}>
                    {item.rule}
                  </span>
                ))}
              </div>
              {start?.handoff?.handoff_id && (
                <div className="small" style={{ marginTop: 6 }}>
                  工单：<span className="mono">{start.handoff.handoff_id}</span>
                </div>
              )}
            </div>
          ) : (
            <div className="small muted">暂无判定</div>
          )}
        </>
      )}

      {tab === 'state' && <StateView state={state} />}

      {tab === 'memory' && (
        <>
          <h3>本轮记忆操作</h3>
          {memoryFlags ? (
            <div className="card">
              <div className="kv">
                <span className="k">抽取到的事实</span>
                <span className="v">{memoryFlags.fact_keys?.join('、') || '无'}</span>
              </div>
              <div className="kv">
                <span className="k">摘要更新</span>
                <span className="v">{memoryFlags.summary_updated ? memoryFlags.summary_source : '否'}</span>
              </div>
              {memoryFlags.compressed_messages ? (
                <div className="kv">
                  <span className="k">本轮压缩</span>
                  <span className="v">
                    {memoryFlags.compressed_messages} 条（比 {memoryFlags.compression_ratio}）
                  </span>
                </div>
              ) : null}
              {memoryFlags.fact_changes?.map((change, index) => (
                <div className="small" key={index}>
                  改口：{change.message}
                </div>
              ))}
            </div>
          ) : (
            <div className="small muted">暂无</div>
          )}

          <h3>会话记忆规模</h3>
          {memory ? (
            <>
              <div className="grid">
                <div className="stat">
                  <div className="n">{num(memory.report.total_chars)}</div>
                  <div className="l">历史总字符</div>
                </div>
                <div className="stat">
                  <div className="n">{num(memory.report.working_chars)}</div>
                  <div className="l">本轮工作记忆</div>
                </div>
                <div className="stat">
                  <div className="n">{num(memory.report.messages_compressed)}</div>
                  <div className="l">已压缩条数</div>
                </div>
                <div className="stat">
                  <div className="n">{String(memory.report.compression_ratio)}</div>
                  <div className="l">压缩比（越小越省）</div>
                </div>
              </div>
              {memory.summary && (
                <>
                  <h3>滚动摘要</h3>
                  <div className="pre">{memory.summary}</div>
                </>
              )}
              <h3>版本化事实</h3>
              {Object.keys(memory.facts).length === 0 && <div className="small muted">还没有抽取到事实</div>}
              {Object.entries(memory.facts).map(([key, value]) => (
                <div className="kv" key={key}>
                  <span className="k">{key}</span>
                  <span className="v">
                    {value.value}
                    {value.previous ? `（此前为 ${value.previous}）` : ''}
                  </span>
                </div>
              ))}
            </>
          ) : (
            <div className="small muted">暂无</div>
          )}
        </>
      )}

      {tab === 'traces' && (
        <>
          <h3>逐轮轨迹{traces ? `（共 ${traces.turns} 轮，prompt 累计 ${num(traces.prompt_tokens_total)} tokens）` : ''}</h3>
          {!traces?.traces.length && <div className="small muted">暂无轨迹</div>}
          {traces?.traces.length ? (
            <table>
              <thead>
                <tr>
                  <th>轮</th>
                  <th>动作</th>
                  <th>工具</th>
                  <th>tokens</th>
                  <th>耗时</th>
                </tr>
              </thead>
              <tbody>
                {traces.traces.map((item) => (
                  <tr key={item.turn}>
                    <td>{item.turn}</td>
                    <td>{actionText(item.plan)}</td>
                    <td className="mono">{item.tools.join(',') || '-'}</td>
                    <td>
                      {item.prompt_tokens}/{item.completion_tokens}
                    </td>
                    <td>{item.latency_ms}ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
        </>
      )}
    </div>
  )
}

async function fetchState(sessionId: string): Promise<StateSnapshot> {
  const response = await fetch(`/api/sessions/${sessionId}/state`)
  if (!response.ok) return {}
  return (await response.json()) as StateSnapshot
}

function StateView({ state }: { state: StateSnapshot | null }) {
  if (!state) return <div className="small muted">加载中…</div>
  const slots = state.slots ?? {}
  const tasks = state.tasks ?? []
  return (
    <>
      <h3>槽位（已知信息）</h3>
      <div className="card">
        {Object.keys(slots).length === 0 && <div className="small muted">还没有已知信息</div>}
        {Object.entries(slots).map(([key, value]) => (
          <div className="kv" key={key}>
            <span className="k">{key}</span>
            <span className="v">{value}</span>
          </div>
        ))}
      </div>

      <h3>待办栈</h3>
      {tasks.length === 0 && <div className="small muted">没有未完成的待办</div>}
      {tasks.map((task, index) => (
        <div className="card" key={`${task.kind}-${index}`}>
          <div className="kv">
            <span className="k">
              {task.name || task.kind}
              {index === 0 ? <span className="chip ok">当前</span> : <span className="chip">排队</span>}
            </span>
            <span className="v">{task.status}</span>
          </div>
          {Object.keys(task.filled ?? {}).length > 0 && (
            <div className="small muted">
              已填：{Object.entries(task.filled).map(([k, v]) => `${k}=${v}`).join('；')}
            </div>
          )}
          {(state.pending_tasks ?? []).find((item) => item.kind === task.kind)?.missing?.length ? (
            <div className="small error-text">
              仍缺：
              {(state.pending_tasks ?? []).find((item) => item.kind === task.kind)?.missing.join('、')}
            </div>
          ) : null}
        </div>
      ))}

      {state.asked_slots?.length ? (
        <>
          <h3>已追问过的槽位</h3>
          <div>
            {state.asked_slots.map((slot) => (
              <span className="chip" key={slot}>
                {slot}
              </span>
            ))}
          </div>
        </>
      ) : null}

      {state.active_missing?.length ? (
        <div className="card" style={{ marginTop: 10 }}>
          <div className="title">当前任务还缺</div>
          <div className="small error-text">{state.active_missing.join('、')}</div>
        </div>
      ) : null}
    </>
  )
}

function actionText(action?: string) {
  if (action === 'ask') return '先追问'
  if (action === 'answer') return '直接回答'
  if (action === 'tool') return '调用工具'
  return action || '-'
}

function num(value: unknown) {
  return typeof value === 'number' ? String(value) : String(value ?? '-')
}
