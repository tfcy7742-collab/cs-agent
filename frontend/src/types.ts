/**
 * 后端接口类型定义（与 cs_agent 的响应结构一一对应）。
 *
 * 只声明前端真正用到的字段，避免"类型文件比代码还长"。
 */

export interface Health {
  status: string
  version: string
  llm: { mode: string; online: boolean; model: string; api_key: string }
  storage: { db_path: string; sessions: number }
  conversation: { working_recent_turns: number; history_compress_chars: number }
}

export interface Session {
  id: string
  user_id: string
  title: string
  status: string
  turn_count: number
  created_at: string
  updated_at: string
}

export interface Message {
  id: number
  session_id: string
  role: 'user' | 'assistant' | 'summary' | 'human_agent'
  content: string
  turn: number
  token_estimate: number
  meta: Record<string, unknown>
  created_at: string
}

export interface PlanInfo {
  action?: string
  intents?: string[]
  task_kind?: string
  new_tasks?: string[]
  pending_tasks?: string[]
  completed_tasks?: string[]
  ask?: { kind: string; target: string }
  clarification?: string
  consulted?: string[]
}

export interface ToolRun {
  tool: string
  params: Record<string, unknown>
  ok: boolean
  latency_ms: number
  error?: string
  error_code?: string
  needs_confirmation?: boolean
}

export interface MemoryFlags {
  fact_keys?: string[]
  fact_changes?: { key: string; old?: string; new?: string; message?: string }[]
  summary_updated?: boolean
  summary_source?: string
  compressed_messages?: number
  compression_ratio?: number
  extract_sources?: Record<string, number>
}

export interface Escalation {
  decision?: string
  score?: number
  threshold?: number
  reason?: string
  triggers?: { rule: string; reason: string; weight: number }[]
  sentiment?: { score: number; words: string[] }
}

export interface StateSnapshot {
  slots?: Record<string, string>
  tasks?: { kind: string; name: string; status: string; filled: Record<string, string> }[]
  active_task?: string
  pending_count?: number
  asked_slots?: string[]
  active_missing?: string[]
  pending_tasks?: { kind: string; name: string; missing: string[] }[]
}

/** start 事件的载荷（本轮的工作记忆与决策） */
export interface StartPayload {
  turn: number
  history_messages: number
  prompt_chars: number
  prompt_token_estimate: number
  compressed: boolean
  offline: boolean
  model: string
  memory?: MemoryFlags
  plan?: PlanInfo
  state?: StateSnapshot
  tools?: { runs?: ToolRun[]; confirmation_prompt?: string; honest_failures?: string[] }
  escalation?: Escalation
  handoff?: {
    handoff_id?: string
    reason?: string
    already_waiting?: boolean
    notice?: string
  }
}

export interface DonePayload {
  turn: number
  content: string
  latency_ms: number
  first_token_ms: number
  usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number }
  offline: boolean
  model: string
  error: string
  trace?: Record<string, unknown>
}

export interface SessionDetail {
  session: Session
  messages: Message[]
  stats: {
    messages: number
    turns: number
    total_chars: number
    token_estimate: number
    has_summary: boolean
    summary_chars: number
    working_messages: number
    working_chars: number
  }
}

export interface MemoryReport {
  report: Record<string, number | string | boolean | Record<string, string>>
  summary: string
  facts: Record<string, { value: string; previous?: string; source?: string }>
}

export interface TraceItem {
  turn: number
  prompt_tokens: number
  completion_tokens: number
  prompt_chars: number
  history_messages: number
  compressed: boolean
  latency_ms: number
  first_token_ms: number
  model: string
  plan: string
  tools: string[]
  escalation: string
  handoff_id: string
  memory: { summary_updated: boolean; compression_ratio: number }
}

export interface Traces {
  session_id: string
  turns: number
  prompt_tokens_total: number
  traces: TraceItem[]
}

export interface HandoffItem {
  handoff_id: string
  session_id: string
  user_id: string
  status: string
  reason: string
  created_at: string
  turns: number
  demand: string
  blocker: string
  sentiment: number
  session_title: string
}

export interface HandoffDetail {
  handoff: {
    id: string
    session_id: string
    status: string
    reason: string
    triggers: { rule: string; reason: string }[]
    packet: Record<string, unknown>
  }
  packet_text: string
  messages: Message[]
  state: StateSnapshot
}

export interface EvalMetrics {
  scenarios: number
  passed: number
  pass_rate: number
  task_success_rate: number
  task_scope: number
  first_touch_rate: number
  handoff_accuracy: number
  repeat_ask_rate: number
  safety_scenarios: number
  safety_pass_rate: number
  total_failures: number
}

export interface EvalReport {
  report: { mode: string; metrics: EvalMetrics; scenarios: { key: string; name: string; ok: boolean; failures: string[] }[] }
  readable: string
}
