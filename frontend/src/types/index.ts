// 类型定义 - 智能就医助手

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp?: string
  toolCalls?: ToolCallInfo[]
  sources?: SourceInfo[]
}

export interface ToolCallInfo {
  tool_name: string
  tool_description: string
  arguments: Record<string, any>
  result_preview?: string
  status?: 'calling' | 'success' | 'failed' | 'rejected'
  duration?: number
  iteration?: number
  risk_level?: 'low' | 'medium' | 'high'
  risk_label?: string
  retry_count?: number
  used_fallback?: boolean
  error_type?: string
  error_message?: string
}

export interface SourceInfo {
  name: string
  url?: string
  snippet?: string
}

export interface StreamEvent {
  type: 'thought' | 'tool_call' | 'tool_result' | 'answer' | 'answer_delta' | 'answer_reset' | 'done' | 'approval_required'
  content?: string
  tool_name?: string
  tool_description?: string
  arguments?: Record<string, any>
  status?: string
  result_preview?: string
  duration?: number
  iteration?: number
  done?: boolean
  conversation_id?: string
  iteration_count?: number
  tool_calls_count?: number
  tool_calls_history?: ToolCallInfo[]
  risk_level?: 'low' | 'medium' | 'high'
  risk_label?: string
  retry_count?: number
  used_fallback?: boolean
  // approval_required 事件专用
  title?: string
  message?: string
  warning?: string
  confirm_text?: string
  cancel_text?: string
  tool_args?: Record<string, any>
  conversation_id?: string
}

export interface ConversationInfo {
  conversation_id: string
  title: string
  created_at: string
  last_message: string
  message_count: number
}

export interface ChatRequest {
  message: string
  conversation_id?: string
  history?: ChatMessage[]
  city?: string
}

export interface ChatResponse {
  success: boolean
  conversation_id: string
  answer: string
  tool_calls_history: ToolCallInfo[]
  iteration_count: number
}
