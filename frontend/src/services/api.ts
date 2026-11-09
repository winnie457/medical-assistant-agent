import axios from 'axios'
import type { ChatRequest, ChatResponse, ConversationInfo, StreamEvent } from '@/types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器
apiClient.interceptors.request.use(
  (config) => {
    console.log('发送请求:', config.method?.toUpperCase(), config.url)
    return config
  },
  (error) => {
    console.error('请求错误:', error)
    return Promise.reject(error)
  }
)

// 响应拦截器
apiClient.interceptors.response.use(
  (response) => {
    console.log('收到响应:', response.status, response.config.url)
    return response
  },
  (error) => {
    console.error('响应错误:', error.response?.status, error.message)
    return Promise.reject(error)
  }
)

/**
 * 发送消息（非流式）
 */
export async function sendMessage(request: ChatRequest): Promise<ChatResponse> {
  try {
    const response = await apiClient.post<ChatResponse>('/api/chat', request)
    return response.data
  } catch (error: any) {
    console.error('发送消息失败:', error)
    throw new Error(error.response?.data?.detail || error.message || '发送消息失败')
  }
}

/**
 * 发送消息（流式）
 */
export async function sendMessageStream(
  request: ChatRequest,
  onEvent: (event: StreamEvent) => void
): Promise<void> {
  return new Promise((resolve, reject) => {
    // 注意：EventSource 默认只支持 GET，这里改用 fetch + ReadableStream
    fetch(`${API_BASE_URL}/api/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(request)
    })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`)
        }

        const reader = response.body?.getReader()
        const decoder = new TextDecoder()
        let buffer = ''

        if (!reader) {
          throw new Error('无法读取响应流')
        }

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() || ''

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const eventData: StreamEvent = JSON.parse(line.slice(6))
                onEvent(eventData)

                if (eventData.type === 'done') {
                  resolve()
                }
              } catch (e) {
                console.error('解析SSE事件失败:', e)
              }
            }
          }
        }
        resolve()
      })
      .catch((error) => {
        reject(error)
      })
  })
}

/**
 * 获取会话列表
 */
export async function getConversations(): Promise<ConversationInfo[]> {
  try {
    const response = await apiClient.get('/api/chat/conversations')
    return response.data.data || []
  } catch (error: any) {
    console.error('获取会话列表失败:', error)
    throw new Error(error.message || '获取会话列表失败')
  }
}

/**
 * 获取会话详情
 */
export async function getConversation(conversationId: string) {
  try {
    const response = await apiClient.get(`/api/chat/conversations/${conversationId}`)
    return response.data.data
  } catch (error: any) {
    console.error('获取会话详情失败:', error)
    throw new Error(error.message || '获取会话详情失败')
  }
}

/**
 * 删除会话
 */
export async function deleteConversation(conversationId: string): Promise<void> {
  try {
    await apiClient.delete(`/api/chat/conversations/${conversationId}`)
  } catch (error: any) {
    console.error('删除会话失败:', error)
    throw new Error(error.message || '删除会话失败')
  }
}

/**
 * 确认执行高风险工具（HITL）
 */
export async function approveTool(conversationId: string): Promise<any> {
  try {
    const response = await apiClient.post('/api/chat/approve', {
      conversation_id: conversationId
    })
    return response.data
  } catch (error: any) {
    console.error('确认失败:', error)
    throw new Error(error.response?.data?.detail || error.message || '确认失败')
  }
}

/**
 * 拒绝执行高风险工具（HITL）
 */
export async function rejectTool(conversationId: string, reason: string = '用户取消'): Promise<any> {
  try {
    const response = await apiClient.post('/api/chat/reject', {
      conversation_id: conversationId,
      reason: reason
    })
    return response.data
  } catch (error: any) {
    console.error('拒绝失败:', error)
    throw new Error(error.response?.data?.detail || error.message || '拒绝失败')
  }
}

/**
 * 获取用户健康画像
 */
export async function getUserProfile(userId: string = 'default'): Promise<any> {
  try {
    const response = await apiClient.get(`/api/chat/profile?user_id=${userId}`)
    return response.data.data
  } catch (error: any) {
    console.error('获取用户画像失败:', error)
    throw new Error(error.message || '获取用户画像失败')
  }
}

/**
 * 删除一条记忆
 */
export async function deleteMemory(memoryId: string, userId: string = 'default'): Promise<any> {
  try {
    const response = await apiClient.delete(`/api/chat/profile/memories/${memoryId}?user_id=${userId}`)
    return response.data
  } catch (error: any) {
    console.error('删除记忆失败:', error)
    throw new Error(error.message || '删除失败')
  }
}

/**
 * 清空所有记忆
 */
export async function clearAllMemories(userId: string = 'default'): Promise<any> {
  try {
    const response = await apiClient.delete(`/api/chat/profile/memories?user_id=${userId}`)
    return response.data
  } catch (error: any) {
    console.error('清空记忆失败:', error)
    throw new Error(error.message || '清空失败')
  }
}

/**
 * 健康检查
 */
export async function healthCheck(): Promise<any> {
  try {
    const response = await apiClient.get('/health')
    return response.data
  } catch (error: any) {
    console.error('健康检查失败:', error)
    throw new Error(error.message || '健康检查失败')
  }
}

export default apiClient
