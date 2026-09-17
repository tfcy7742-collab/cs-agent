import { useEffect, useState } from 'react'
import { api } from '../api'
import type { EvalReport } from '../types'

/**
 * 评测面板：一键跑离线评测集并展示指标。
 *
 * 面板上明确标注**离线评测评什么、不评什么**——
 * 否则很容易把"18/18 通过"误读成"回答质量满分"。
 */
export default function EvalPanel() {
  const [scenarios, setScenarios] = useState<{ key: string; name: string; tags: string[]; turns: number }[]>([])
  const [report, setReport] = useState<EvalReport | null>(null)
  const [running, setRunning] = useState(false)
  const [rounds, setRounds] = useState(12)
  const [only, setOnly] = useState('')
  const [error, setError] = useState('')

  useEffect(() => {
    api
      .evalScenarios()
      .then((body) => setScenarios(body.scenarios))
      .catch((exc: Error) => setError(exc.message))
  }, [])

  const run = async () => {
    setRunning(true)
    setError('')
    try {
      setReport(await api.runEval(rounds, only || undefined))
    } catch (exc) {
      setError((exc as Error).message)
    } finally {
      setRunning(false)
    }
  }

  const metrics = report?.report.metrics
  const tagOptions = Array.from(new Set(scenarios.flatMap((item) => item.tags)))

  return (
    <div style={{ padding: 18, overflowY: 'auto', flex: 1 }}>
      {error && <div className="error-text">⚠ {error}</div>}

      <div className="card">
        <div className="title">评测集</div>
        <div className="small muted">
          共 {scenarios.length} 个场景。接口里的评测**固定为离线模式**（不调用真实模型），
          因此报告只反映工具路由 / 会话状态 / 转人工规则 / 注入场景的副作用，
          <strong>不反映回答质量</strong>。回答质量请用
          <span className="mono"> scripts/run_eval.py --online</span>。
        </div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end', marginTop: 12, flexWrap: 'wrap' }}>
          <label className="small">
            长会话轮数
            <input
              type="number"
              min={5}
              max={60}
              value={rounds}
              onChange={(event) => setRounds(Number(event.target.value))}
              style={{ width: 90 }}
            />
          </label>
          <label className="small">
            只跑标签
            <select value={only} onChange={(event) => setOnly(event.target.value)} style={{ width: 150 }}>
              <option value="">全部</option>
              {tagOptions.map((tag) => (
                <option value={tag} key={tag}>
                  {tag}
                </option>
              ))}
            </select>
          </label>
          <button className="primary" onClick={run} disabled={running}>
            {running ? '评测中…' : '开始评测'}
          </button>
        </div>
      </div>

      {metrics && (
        <>
          <h3 className="small muted">指标（模式：{report?.report.mode}）</h3>
          <div className="grid">
            <Stat label="场景通过率" value={`${Math.round(metrics.pass_rate * 100)}%`} sub={`${metrics.passed}/${metrics.scenarios}`} />
            <Stat
              label="任务完成率"
              value={`${Math.round(metrics.task_success_rate * 100)}%`}
              sub={`业务目标场景 ${metrics.task_scope}`}
            />
            <Stat label="转人工准确率" value={`${Math.round(metrics.handoff_accuracy * 100)}%`} sub="规则层应恒为 100%" />
            <Stat label="重复追问率" value={`${Math.round(metrics.repeat_ask_rate * 100)}%`} sub="越低越好" />
            <Stat
              label="对抗通过率"
              value={`${Math.round(metrics.safety_pass_rate * 100)}%`}
              sub={`注入场景 ${metrics.safety_scenarios}`}
            />
            <Stat label="失败断言" value={String(metrics.total_failures)} sub="应为 0" />
          </div>

          <h3 className="small muted">场景结果</h3>
          <table>
            <thead>
              <tr>
                <th>结果</th>
                <th>场景</th>
                <th>失败断言</th>
              </tr>
            </thead>
            <tbody>
              {report?.report.scenarios.map((item) => (
                <tr key={item.key}>
                  <td>
                    <span className={`chip ${item.ok ? 'ok' : 'danger'}`}>{item.ok ? '通过' : '失败'}</span>
                  </td>
                  <td>{item.name}</td>
                  <td className="small">
                    {item.failures.length ? item.failures.map((text) => <div key={text}>· {text}</div>) : '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      {!report && (
        <>
          <h3 className="small muted">场景清单</h3>
          <table>
            <thead>
              <tr>
                <th>场景</th>
                <th>标签</th>
                <th>轮次</th>
              </tr>
            </thead>
            <tbody>
              {scenarios.map((item) => (
                <tr key={item.key}>
                  <td>{item.name}</td>
                  <td>{item.tags.join('、')}</td>
                  <td>{item.turns}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  )
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="stat">
      <div className="n">{value}</div>
      <div className="l">{label}</div>
      {sub && <div className="l" style={{ opacity: 0.7 }}>{sub}</div>}
    </div>
  )
}
