<template>
  <div class="chat-container">
    <!-- 左侧边栏：会话列表 -->
    <div class="sidebar">
      <div class="sidebar-header">
        <div class="logo">
          <span class="logo-icon">🏥</span>
          <span class="logo-text">医小助</span>
        </div>
        <a-button type="primary" block @click="startNewChat" class="new-chat-btn">
          <span class="btn-icon">+</span>
          <span>新建咨询</span>
        </a-button>
      </div>

      <div class="city-selector">
        <a-select v-model:value="currentCity" size="small" class="city-select">
          <a-select-option value="北京">📍 北京</a-select-option>
          <a-select-option value="上海">📍 上海</a-select-option>
          <a-select-option value="广州">📍 广州</a-select-option>
          <a-select-option value="深圳">📍 深圳</a-select-option>
          <a-select-option value="杭州">📍 杭州</a-select-option>
          <a-select-option value="成都">📍 成都</a-select-option>
        </a-select>
      </div>

      <div class="conversation-list">
        <div class="list-title">历史咨询</div>
        <div v-if="conversations.length === 0" class="empty-hint">
          暂无历史咨询
        </div>
        <div
          v-for="conv in conversations"
          :key="conv.conversation_id"
          class="conversation-item"
          :class="{ active: conv.conversation_id === currentConversationId }"
          @click="loadConversation(conv.conversation_id)"
        >
          <div class="conv-icon">💬</div>
          <div class="conv-info">
            <div class="conv-title">{{ conv.title }}</div>
            <div class="conv-preview">{{ conv.last_message }}</div>
          </div>
          <a-button
            type="text"
            size="small"
            danger
            class="delete-btn"
            @click.stop="handleDeleteConversation(conv.conversation_id)"
          >
            🗑️
          </a-button>
        </div>
      </div>

      <!-- 用户健康画像 -->
      <div class="health-profile">
        <div class="profile-header" @click="toggleProfilePanel">
          <span class="profile-icon">🧠</span>
          <span class="profile-title">健康画像</span>
          <span class="profile-count">{{ profileMemoryCount }} 条</span>
          <span class="profile-toggle">{{ profileExpanded ? '▲' : '▼' }}</span>
        </div>
        <div v-if="profileExpanded" class="profile-content">
          <div v-if="profileLoading" class="profile-loading">加载中...</div>
          <div v-else-if="profileMemories.length === 0" class="profile-empty">
            暂无健康记忆<br/>
            <span class="profile-hint">对话中会自动提取您的健康信息</span>
          </div>
          <div v-else class="profile-memory-list">
            <div
              v-for="mem in profileMemories"
              :key="mem.id"
              class="profile-memory-item"
            >
              <div class="memory-cat">{{ getCategoryLabel(mem.category) }}</div>
              <div class="memory-value">{{ mem.value }}</div>
              <div class="memory-meta">
                <span class="confidence-tag" :class="'conf-' + mem.confidence">
                  {{ getConfidenceLabel(mem.confidence) }}
                </span>
                <a-button
                  type="text"
                  size="small"
                  danger
                  class="memory-delete"
                  @click.stop="handleDeleteMemory(mem.id)"
                >
                  删除
                </a-button>
              </div>
            </div>
          </div>
          <div v-if="profileMemories.length > 0" class="profile-actions">
            <a-button size="small" danger type="text" @click="handleClearMemories">
              清空所有记忆
            </a-button>
          </div>
        </div>
      </div>

      <div class="sidebar-footer">
        <div class="footer-info">
          <span>🏥 智能就医助手</span>
          <span class="version">v2.0.0</span>
        </div>
        <div class="disclaimer">
          ⚠️ 仅供参考，不能替代医生诊断
        </div>
      </div>
    </div>

    <!-- 右侧聊天区 -->
    <div class="chat-main">
      <!-- 聊天头部 -->
      <div class="chat-header">
        <div class="chat-title">
          <span class="status-dot online"></span>
          <span>医小助 - 您的智能就医顾问</span>
        </div>
        <div class="chat-subtitle">
          基于 ReAct Agent · 7+ 医疗工具 · 专业贴心
        </div>
      </div>

      <!-- 消息区域 -->
      <div class="messages-container" ref="messagesContainer">
        <!-- 欢迎消息 -->
        <div v-if="messages.length === 0" class="welcome-section">
          <div class="welcome-icon">🏥</div>
          <h2>您好，我是医小助</h2>
          <p class="welcome-desc">
            我可以帮您分析症状、推荐科室、查找医院、规划就医路线、了解用药知识。
          </p>
          <div class="quick-actions">
            <div class="quick-action" @click="quickAsk('我最近头疼，应该挂什么科？')">
              <span class="qa-icon">🩺</span>
              <span>我头疼该挂什么科？</span>
            </div>
            <div class="quick-action" @click="quickAsk('北京有哪些三甲医院？')">
              <span class="qa-icon">🏥</span>
              <span>附近有哪些三甲医院？</span>
            </div>
            <div class="quick-action" @click="quickAsk('感冒了吃什么药好？')">
              <span class="qa-icon">💊</span>
              <span>感冒了吃什么药？</span>
            </div>
            <div class="quick-action" @click="quickAsk('高血压平时要注意什么？')">
              <span class="qa-icon">📚</span>
              <span>高血压要注意什么？</span>
            </div>
          </div>
        </div>

        <!-- 消息列表 -->
        <div
          v-for="(msg, index) in messages"
          :key="index"
          class="message-item"
          :class="msg.role"
        >
          <div v-if="msg.role === 'user'" class="message-wrapper user-wrapper">
            <div class="message-content user-content">
              {{ msg.content }}
            </div>
            <div class="avatar user-avatar">👤</div>
          </div>

          <div v-else class="message-wrapper assistant-wrapper">
            <div class="avatar assistant-avatar">🏥</div>
            <div class="message-content assistant-content">
              <div v-if="msg.thinking" class="thinking-indicator">
                <span class="thinking-text">正在思考...</span>
              </div>
              <div v-else-if="msg.content" class="message-text" v-html="formatMessage(msg.content)"></div>

              <!-- 工具调用展示 -->
              <div v-if="msg.toolCalls && msg.toolCalls.length > 0" class="tool-calls-section">
                <div class="tool-calls-title">
                  <span>🔧 调用工具</span>
                  <span class="tool-count">{{ msg.toolCalls.length }} 个</span>
                </div>
                <div
                  v-for="(tool, tIndex) in msg.toolCalls"
                  :key="tIndex"
                  class="tool-call-item"
                  :class="tool.status"
                >
                  <div class="tool-header" @click="toggleToolDetail(index, tIndex)">
                    <div class="tool-info">
                      <span class="tool-status-icon">
                        <span v-if="tool.status === 'calling'" class="loading-spin">⏳</span>
                        <span v-else-if="tool.status === 'success'">✅</span>
                        <span v-else-if="tool.status === 'rejected'">🚫</span>
                        <span v-else>❌</span>
                      </span>
                      <span class="tool-name">{{ getToolDisplayName(tool.tool_name) }}</span>
                      <span
                        v-if="tool.risk_level === 'high'"
                        class="risk-tag risk-high"
                      >高风险</span>
                      <span
                        v-else-if="tool.risk_level === 'medium'"
                        class="risk-tag risk-medium"
                      >中风险</span>
                    </div>
                    <div class="tool-meta">
                      <span v-if="tool.used_fallback" class="fallback-tag">已降级</span>
                      <span v-if="tool.retry_count && tool.retry_count > 0" class="retry-tag">
                        重试{{ tool.retry_count }}次
                      </span>
                      <span v-if="tool.duration" class="tool-duration">{{ tool.duration }}s</span>
                      <span class="expand-icon">{{ expandedTools[`${index}-${tIndex}`] ? '▲' : '▼' }}</span>
                    </div>
                  </div>
                  <div v-if="expandedTools[`${index}-${tIndex}`]" class="tool-detail">
                    <div class="tool-args">
                      <div class="detail-label">参数:</div>
                      <pre>{{ JSON.stringify(tool.arguments, null, 2) }}</pre>
                    </div>
                    <div v-if="tool.result_preview" class="tool-result">
                      <div class="detail-label">结果预览:</div>
                      <pre>{{ tool.result_preview }}</pre>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- 正在输入指示器 -->
        <div v-if="isTyping" class="message-item assistant">
          <div class="message-wrapper assistant-wrapper">
            <div class="avatar assistant-avatar">🏥</div>
            <div class="message-content assistant-content">
              <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 输入区域 -->
      <div class="input-area">
        <div class="input-wrapper">
          <a-textarea
            v-model:value="inputMessage"
            placeholder="描述您的症状或问题，我来帮您分析..."
            :rows="1"
            :auto-size="{ minRows: 1, maxRows: 5 }"
            class="chat-input"
            @pressEnter="handleSend"
            :disabled="isLoading"
          />
          <a-button
            type="primary"
            class="send-btn"
            @click="handleSend"
            :loading="isLoading"
            :disabled="!inputMessage.trim()"
          >
            <template v-if="!isLoading">
              <span>发送</span>
            </template>
            <template v-else>
              <span>思考中</span>
            </template>
          </a-button>
        </div>
        <div class="input-hint">
          按 Enter 发送，Shift+Enter 换行 · 信息仅供参考，不能替代医生诊断
        </div>
      </div>
    </div>
  </div>

  <!-- 高风险操作确认弹窗 -->
  <a-modal
    v-model:open="approvalVisible"
    :title="approvalData?.title || '确认操作'"
    :confirm-loading="approvalLoading"
    ok-type="danger"
    :ok-text="approvalData?.confirm_text || '确认'"
    :cancel-text="approvalData?.cancel_text || '取消'"
    @ok="handleApprove"
    @cancel="handleReject"
    :mask-closable="false"
    :esc-to-close="false"
    width="480px"
  >
    <div class="approval-modal">
      <div class="approval-icon">⚠️</div>
      <div class="approval-message">
        {{ approvalData?.message }}
      </div>
      <div v-if="approvalData?.warning" class="approval-warning">
        <div class="warning-title">重要提示</div>
        <div class="warning-content">{{ approvalData.warning }}</div>
      </div>
      <div class="approval-tip">
        💡 如您对用药有任何疑问，建议咨询专业医生或药师。
      </div>
    </div>
  </a-modal>
</template>

<script setup lang="ts">
import { ref, nextTick, onMounted } from 'vue'
import { message, Modal } from 'ant-design-vue'
import { sendMessageStream, getConversations, getConversation, deleteConversation, approveTool, rejectTool, getUserProfile, deleteMemory, clearAllMemories } from '@/services/api'
import type { ChatMessage, StreamEvent } from '@/types'

const messagesContainer = ref<HTMLElement | null>(null)
const inputMessage = ref('')
const isLoading = ref(false)
const isTyping = ref(false)
// 风险确认弹窗状态
const approvalVisible = ref(false)
const approvalData = ref<{
  tool_name: string
  tool_args: Record<string, any>
  title: string
  message: string
  warning: string
  confirm_text: string
  cancel_text: string
  iteration: number
} | null>(null)
const approvalConversationId = ref('')
// 被中断的那次运行标识，审批续跑时回传给服务端
const approvalRunId = ref('')
// 审批后续跑的事件要回填到同一条助手消息
const approvalMsgIndex = ref(-1)
const approvalLoading = ref(false)

// 用户健康画像
const profileExpanded = ref(false)
const profileLoading = ref(false)
const profileMemories = ref<any[]>([])
const profileMemoryCount = ref(0)
const userId = ref('default')
const messages = ref<ChatMessage[]>([])
// 记录哪些消息已经开始接收答案增量（首个增量到达时清空“分析中”占位文案）
const answerStarted = new Set<number>()
const conversations = ref<any[]>([])
const currentConversationId = ref<string | null>(null)
const currentCity = ref('北京')
const expandedTools = ref<Record<string, boolean>>({})

const toolDisplayNames: Record<string, string> = {
  symptom_analyzer: '🩺 症状分析',
  hospital_search: '🏥 医院搜索',
  department_guide: '📋 科室导诊',
  medical_knowledge_search: '📚 医疗知识',
  route_to_hospital: '🗺️ 路线规划',
  medication_consultation: '💊 用药咨询',
  health_news_search: '📰 健康资讯'
}

function getToolDisplayName(name: string): string {
  return toolDisplayNames[name] || name
}

function formatMessage(content: string): string {
  // 简单的 markdown 格式化
  let html = content
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>')
    .replace(/^### (.*?)$/gm, '<h3>$1</h3>')
    .replace(/^## (.*?)$/gm, '<h2>$1</h2>')
    .replace(/- (.*?)$/gm, '• $1')
  return html
}

function toggleToolDetail(msgIndex: number, toolIndex: number) {
  const key = `${msgIndex}-${toolIndex}`
  expandedTools.value[key] = !expandedTools.value[key]
}

function scrollToBottom() {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
}

function quickAsk(question: string) {
  inputMessage.value = question
  handleSend()
}

function startNewChat() {
  messages.value = []
  currentConversationId.value = null
  inputMessage.value = ''
}

async function loadConversation(convId: string) {
  if (isLoading.value) {
    message.warning('请等待当前回答完成后再切换会话')
    return
  }

  currentConversationId.value = convId
  message.loading('加载会话中...', 0.5)

  try {
    const convData = await getConversation(convId)
    // 转换后端消息格式为前端格式
    const msgs: ChatMessage[] = (convData.messages || []).map((msg: any) => ({
      role: msg.role,
      content: msg.content,
      timestamp: msg.timestamp,
      toolCalls: msg.tool_calls || [],
      thinking: false
    }))
    messages.value = msgs
    scrollToBottom()
  } catch (e: any) {
    message.error(e.message || '加载会话失败')
  }
}

async function handleDeleteConversation(convId: string) {
  try {
    await deleteConversation(convId)
    conversations.value = conversations.value.filter(c => c.conversation_id !== convId)
    if (currentConversationId.value === convId) {
      startNewChat()
    }
    message.success('会话已删除')
  } catch (e: any) {
    message.error(e.message || '删除失败')
  }
}

async function handleSend() {
  const text = inputMessage.value.trim()
  if (!text || isLoading.value) return

  isLoading.value = true
  isTyping.value = true

  // 添加用户消息
  messages.value.push({
    role: 'user',
    content: text
  })

  // 添加一个空的助手消息占位
  const assistantMsgIndex = messages.value.length
  answerStarted.delete(assistantMsgIndex)
  messages.value.push({
    role: 'assistant',
    content: '',
    thinking: true,
    toolCalls: []
  })

  inputMessage.value = ''
  scrollToBottom()

  try {
    await sendMessageStream(
      {
        message: text,
        conversation_id: currentConversationId.value || undefined,
        city: currentCity.value
      },
      (event: StreamEvent) => {
        handleStreamEvent(event, assistantMsgIndex)
      }
    )
  } catch (error: any) {
    messages.value[assistantMsgIndex].content = '抱歉，服务暂时不可用，请稍后再试。'
    messages.value[assistantMsgIndex].thinking = false
    message.error(error.message || '发送失败')
  } finally {
    isLoading.value = false
    isTyping.value = false
    scrollToBottom()
  }
}

function handleStreamEvent(event: StreamEvent, msgIndex: number) {
  const msg = messages.value[msgIndex]

  switch (event.type) {
    case 'thought':
      msg.content = event.content || '分析中...'
      break

    case 'tool_call':
      msg.thinking = false
      isTyping.value = true
      msg.toolCalls?.push({
        tool_name: event.tool_name || '',
        tool_description: event.tool_description || '',
        arguments: event.arguments || {},
        status: 'calling',
        iteration: event.iteration,
        risk_level: event.risk_level as 'low' | 'medium' | 'high',
        risk_label: event.risk_label
      })
      break

    case 'approval_required':
      // 高风险操作挂起 → 弹窗；确认/拒绝后由服务端续流继续推送
      approvalData.value = {
        tool_name: event.tool_name || '',
        tool_args: event.tool_args || {},
        title: event.title || '确认操作',
        message: event.message || '',
        warning: event.warning || '',
        confirm_text: event.confirm_text || '确认',
        cancel_text: event.cancel_text || '取消',
        iteration: event.iteration || 0
      }
      // 从事件里取 conversation_id / run_id（新会话时 currentConversationId 还没设置）
      approvalConversationId.value = event.conversation_id || currentConversationId.value || ''
      if (event.conversation_id) {
        currentConversationId.value = event.conversation_id
      }
      approvalRunId.value = event.run_id || ''
      approvalMsgIndex.value = msgIndex
      approvalVisible.value = true
      loadConversations()
      break

    case 'tool_result':
      if (msg.toolCalls && msg.toolCalls.length > 0) {
        const lastTool = msg.toolCalls[msg.toolCalls.length - 1]
        lastTool.status = event.status as 'success' | 'failed' | 'rejected'
        lastTool.result_preview = event.result_preview
        lastTool.duration = event.duration
        lastTool.risk_level = event.risk_level as 'low' | 'medium' | 'high'
        lastTool.retry_count = event.retry_count
        lastTool.used_fallback = event.used_fallback
      }
      break

    case 'answer_delta':
      // token 级增量：首次到达时清空思考占位，之后逐段追加
      msg.thinking = false
      isTyping.value = false
      if (!answerStarted.has(msgIndex)) {
        answerStarted.add(msgIndex)
        msg.content = ''
      }
      msg.content = (msg.content || '') + (event.content || '')
      break

    case 'answer_reset':
      // 模型转入工具调用，丢弃之前推送的正文增量
      answerStarted.delete(msgIndex)
      msg.content = ''
      break

    case 'answer':
      msg.thinking = false
      isTyping.value = false
      // 最终答案覆盖增量结果，保证与服务端落库内容一致
      msg.content = event.content || ''
      break

    case 'done':
      msg.thinking = false
      isTyping.value = false
      if (event.conversation_id) {
        currentConversationId.value = event.conversation_id
      }
      // 刷新会话列表
      loadConversations()
      // 如果健康画像面板展开了，刷新一下（可能有新记忆）
      if (profileExpanded.value) {
        setTimeout(() => loadUserProfile(), 500)
      }
      break
  }

  scrollToBottom()
}

async function loadConversations() {
  try {
    const data = await getConversations()
    conversations.value = data
  } catch (e) {
    console.error('加载会话列表失败:', e)
  }
}

onMounted(() => {
  loadConversations()
})

// ========== 风险分级确认弹窗 ==========

async function handleApprove() {
  if (!approvalRunId.value) return

  const msgIndex = approvalMsgIndex.value
  approvalLoading.value = true
  approvalVisible.value = false
  isLoading.value = true

  try {
    await approveTool(
      approvalRunId.value,
      approvalConversationId.value,
      userId.value,
      (event) => handleStreamEvent(event, msgIndex)
    )
  } catch (e: any) {
    message.error(e.message || '确认失败')
  } finally {
    approvalLoading.value = false
    isLoading.value = false
    isTyping.value = false
  }
}

async function handleReject() {
  if (!approvalRunId.value) return

  const msgIndex = approvalMsgIndex.value
  approvalLoading.value = true
  approvalVisible.value = false
  isLoading.value = true

  try {
    await rejectTool(
      approvalRunId.value,
      approvalConversationId.value,
      userId.value,
      '用户取消查看用药建议',
      (event) => handleStreamEvent(event, msgIndex)
    )
  } catch (e: any) {
    message.error(e.message || '操作失败')
  } finally {
    approvalLoading.value = false
    isLoading.value = false
    isTyping.value = false
  }
}

// ========== 用户健康画像 ==========

function toggleProfilePanel() {
  profileExpanded.value = !profileExpanded.value
  if (profileExpanded.value && profileMemories.value.length === 0) {
    loadUserProfile()
  }
}

async function loadUserProfile() {
  profileLoading.value = true
  try {
    const data = await getUserProfile(userId.value)
    profileMemories.value = data.memories || []
    profileMemoryCount.value = data.memory_count || 0
  } catch (e: any) {
    console.error('加载画像失败:', e)
  } finally {
    profileLoading.value = false
  }
}

function getCategoryLabel(category: string): string {
  const map: Record<string, string> = {
    basic_info: '基本信息',
    allergy: '过敏史',
    chronic_disease: '慢性病',
    surgery_history: '手术史',
    medication: '用药史',
    family_history: '家族病史',
    lifestyle: '生活习惯',
    symptom_history: '症状历史',
    other: '其他',
  }
  return map[category] || category
}

function getConfidenceLabel(confidence: string): string {
  const map: Record<string, string> = {
    high: '高置信',
    medium: '中置信',
    low: '低置信',
    confirmed: '已确认',
  }
  return map[confidence] || confidence
}

async function handleDeleteMemory(memoryId: string) {
  try {
    await deleteMemory(memoryId, userId.value)
    message.success('已删除该记忆')
    loadUserProfile()
  } catch (e: any) {
    message.error(e.message || '删除失败')
  }
}

async function handleClearMemories() {
  Modal.confirm({
    title: '确认清空所有记忆？',
    content: '清空后健康画像将重置，无法恢复。',
    okText: '确认清空',
    okType: 'danger',
    cancelText: '取消',
    onOk: async () => {
      try {
        await clearAllMemories(userId.value)
        message.success('已清空所有记忆')
        loadUserProfile()
      } catch (e: any) {
        message.error(e.message || '清空失败')
      }
    }
  })
}
</script>

<style scoped>
.chat-container {
  display: flex;
  height: 100vh;
  background: #f5f7fa;
}

/* 侧边栏 */
.sidebar {
  width: 280px;
  background: linear-gradient(180deg, #1a2332 0%, #0f1923 100%);
  display: flex;
  flex-direction: column;
  color: white;
}

.sidebar-header {
  padding: 20px 16px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.logo {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
}

.logo-icon {
  font-size: 32px;
}

.logo-text {
  font-size: 20px;
  font-weight: 700;
  background: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.new-chat-btn {
  background: linear-gradient(135deg, #00d4ff 0%, #00aa88 100%) !important;
  border: none !important;
  height: 44px;
  border-radius: 12px;
  font-weight: 600;
}

.new-chat-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(0, 212, 255, 0.4);
}

.btn-icon {
  font-size: 18px;
  margin-right: 4px;
}

.city-selector {
  padding: 12px 16px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.city-select {
  width: 100%;
}

.conversation-list {
  flex: 1;
  overflow-y: auto;
  padding: 12px 8px;
}

.list-title {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.5);
  padding: 8px;
  text-transform: uppercase;
  letter-spacing: 1px;
}

.empty-hint {
  text-align: center;
  color: rgba(255, 255, 255, 0.4);
  padding: 20px;
  font-size: 14px;
}

.conversation-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px;
  border-radius: 10px;
  cursor: pointer;
  transition: all 0.2s ease;
  margin-bottom: 4px;
}

.conversation-item:hover {
  background: rgba(255, 255, 255, 0.08);
}

.conversation-item.active {
  background: rgba(0, 212, 255, 0.15);
  border: 1px solid rgba(0, 212, 255, 0.3);
}

.conv-icon {
  font-size: 18px;
  flex-shrink: 0;
}

.conv-info {
  flex: 1;
  min-width: 0;
}

.conv-title {
  font-size: 14px;
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.conv-preview {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.5);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin-top: 2px;
}

.delete-btn {
  opacity: 0;
  transition: opacity 0.2s;
}

.conversation-item:hover .delete-btn {
  opacity: 1;
}

.sidebar-footer {
  padding: 16px;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
}

.footer-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 13px;
  margin-bottom: 8px;
}

.version {
  color: rgba(255, 255, 255, 0.5);
}

.disclaimer {
  font-size: 11px;
  color: rgba(255, 255, 255, 0.4);
  text-align: center;
}

/* 聊天主区域 */
.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.chat-header {
  padding: 16px 24px;
  background: white;
  border-bottom: 1px solid #e8e8e8;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}

.chat-title {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 16px;
  font-weight: 600;
  color: #333;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.status-dot.online {
  background: #52c41a;
  box-shadow: 0 0 0 3px rgba(82, 196, 26, 0.2);
}

.chat-subtitle {
  font-size: 12px;
  color: #999;
  margin-top: 4px;
  margin-left: 18px;
}

/* 消息区域 */
.messages-container {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
}

/* 欢迎区域 */
.welcome-section {
  text-align: center;
  padding: 60px 20px;
}

.welcome-icon {
  font-size: 64px;
  margin-bottom: 20px;
}

.welcome-section h2 {
  font-size: 28px;
  color: #333;
  margin-bottom: 12px;
}

.welcome-desc {
  font-size: 16px;
  color: #666;
  max-width: 500px;
  margin: 0 auto 32px;
  line-height: 1.6;
}

.quick-actions {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
  max-width: 600px;
  margin: 0 auto;
}

.quick-action {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px 20px;
  background: white;
  border: 2px solid #e8e8e8;
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.2s ease;
  text-align: left;
}

.quick-action:hover {
  border-color: #00d4ff;
  box-shadow: 0 4px 12px rgba(0, 212, 255, 0.15);
  transform: translateY(-2px);
}

.qa-icon {
  font-size: 24px;
  flex-shrink: 0;
}

.quick-action span:last-child {
  font-size: 14px;
  color: #555;
}

/* 消息样式 */
.message-item {
  margin-bottom: 20px;
}

.message-wrapper {
  display: flex;
  gap: 12px;
  max-width: 80%;
}

.user-wrapper {
  flex-direction: row-reverse;
  margin-left: auto;
}

.assistant-wrapper {
  margin-right: auto;
}

.avatar {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  flex-shrink: 0;
}

.user-avatar {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.assistant-avatar {
  background: linear-gradient(135deg, #00d4ff 0%, #00aa88 100%);
}

.message-content {
  padding: 12px 16px;
  border-radius: 16px;
  line-height: 1.6;
  word-wrap: break-word;
}

.user-content {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-bottom-right-radius: 4px;
}

.assistant-content {
  background: white;
  color: #333;
  border: 1px solid #e8e8e8;
  border-bottom-left-radius: 4px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}

.message-text {
  font-size: 15px;
}

.message-text :deep(h2),
.message-text :deep(h3) {
  margin: 16px 0 8px;
  color: #333;
}

.message-text :deep(strong) {
  color: #1890ff;
}

/* 思考中指示器 */
.thinking-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #999;
  font-size: 14px;
}

.typing-indicator {
  display: flex;
  gap: 4px;
  padding: 4px 0;
}

.typing-indicator span {
  width: 8px;
  height: 8px;
  background: #ccc;
  border-radius: 50%;
  animation: typing 1.4s infinite ease-in-out;
}

.typing-indicator span:nth-child(1) {
  animation-delay: 0s;
}

.typing-indicator span:nth-child(2) {
  animation-delay: 0.2s;
}

.typing-indicator span:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes typing {
  0%, 60%, 100% {
    transform: translateY(0);
    opacity: 0.4;
  }
  30% {
    transform: translateY(-6px);
    opacity: 1;
  }
}

/* 工具调用展示 */
.tool-calls-section {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px dashed #e8e8e8;
}

.tool-calls-title {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 13px;
  color: #666;
  margin-bottom: 8px;
}

.tool-count {
  background: #f0f5ff;
  color: #1890ff;
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 11px;
}

.tool-call-item {
  background: #fafafa;
  border: 1px solid #e8e8e8;
  border-radius: 8px;
  margin-bottom: 8px;
  overflow: hidden;
}

.tool-call-item.success {
  border-color: #b7eb8f;
  background: #f6ffed;
}

.tool-call-item.failed {
  border-color: #ffa39e;
  background: #fff2f0;
}

.tool-call-item.rejected {
  border-color: #ffc069;
  background: #fff7e6;
}

/* 风险标签 */
.risk-tag {
  font-size: 11px;
  padding: 1px 6px;
  border-radius: 4px;
  font-weight: 500;
}

.risk-tag.risk-high {
  background: #fff1f0;
  color: #cf1322;
  border: 1px solid #ffa39e;
}

.risk-tag.risk-medium {
  background: #fff7e6;
  color: #d46b08;
  border: 1px solid #ffd591;
}

/* 重试/降级标签 */
.retry-tag {
  font-size: 11px;
  color: #d46b08;
  background: #fff7e6;
  padding: 1px 6px;
  border-radius: 4px;
}

.fallback-tag {
  font-size: 11px;
  color: #1890ff;
  background: #e6f7ff;
  padding: 1px 6px;
  border-radius: 4px;
}

.tool-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
  cursor: pointer;
}

.tool-info {
  display: flex;
  align-items: center;
  gap: 8px;
}

.tool-status-icon {
  font-size: 14px;
}

.loading-spin {
  display: inline-block;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.tool-name {
  font-size: 13px;
  font-weight: 500;
  color: #555;
}

.tool-meta {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: #999;
}

.expand-icon {
  font-size: 10px;
}

.tool-detail {
  padding: 8px 12px;
  border-top: 1px solid #e8e8e8;
  background: white;
}

.detail-label {
  font-size: 11px;
  color: #999;
  margin-bottom: 4px;
}

.tool-detail pre {
  background: #f5f5f5;
  padding: 8px;
  border-radius: 4px;
  font-size: 12px;
  margin: 0 0 8px 0;
  max-height: 150px;
  overflow: auto;
  white-space: pre-wrap;
  word-wrap: break-word;
}

/* 输入区域 */
.input-area {
  padding: 16px 24px;
  background: white;
  border-top: 1px solid #e8e8e8;
}

.input-wrapper {
  display: flex;
  gap: 12px;
  align-items: flex-end;
  max-width: 900px;
  margin: 0 auto;
}

.chat-input :deep(.ant-input) {
  border-radius: 12px;
  border: 2px solid #e8e8e8;
  transition: all 0.2s ease;
  font-size: 15px;
  padding: 12px 16px;
}

.chat-input :deep(.ant-input:hover) {
  border-color: #00d4ff;
}

.chat-input :deep(.ant-input:focus) {
  border-color: #00d4ff;
  box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.1);
}

.send-btn {
  height: 44px !important;
  padding: 0 24px !important;
  border-radius: 12px !important;
  background: linear-gradient(135deg, #00d4ff 0%, #00aa88 100%) !important;
  border: none !important;
  font-weight: 600;
  flex-shrink: 0;
}

.send-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(0, 212, 255, 0.4);
}

.input-hint {
  text-align: center;
  font-size: 12px;
  color: #bbb;
  margin-top: 8px;
}

/* 滚动条 */
.messages-container::-webkit-scrollbar,
.conversation-list::-webkit-scrollbar {
  width: 6px;
}

.messages-container::-webkit-scrollbar-track,
.conversation-list::-webkit-scrollbar-track {
  background: transparent;
}

.messages-container::-webkit-scrollbar-thumb,
.conversation-list::-webkit-scrollbar-thumb {
  background: #ddd;
  border-radius: 3px;
}

.messages-container::-webkit-scrollbar-thumb:hover,
.conversation-list::-webkit-scrollbar-thumb:hover {
  background: #ccc;
}

/* 风险确认弹窗 */
.approval-modal {
  text-align: center;
  padding: 10px 0;
}

.approval-icon {
  font-size: 48px;
  margin-bottom: 16px;
}

.approval-message {
  font-size: 15px;
  line-height: 1.6;
  color: #333;
  margin-bottom: 16px;
}

.approval-warning {
  background: #fff2e8;
  border: 1px solid #ffd591;
  border-radius: 6px;
  padding: 12px 16px;
  margin-bottom: 16px;
  text-align: left;
}

.warning-title {
  font-weight: 600;
  color: #d46b08;
  margin-bottom: 6px;
  font-size: 14px;
}

.warning-content {
  font-size: 13px;
  color: #874d00;
  line-height: 1.5;
}

.approval-tip {
  font-size: 13px;
  color: #999;
  margin-top: 8px;
}

/* 健康画像面板 */
.health-profile {
  margin: 0 12px 12px;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 8px;
  overflow: hidden;
}

.profile-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
  cursor: pointer;
  color: rgba(255, 255, 255, 0.85);
  font-size: 13px;
  transition: background 0.2s;
}

.profile-header:hover {
  background: rgba(255, 255, 255, 0.08);
}

.profile-icon {
  font-size: 16px;
}

.profile-title {
  flex: 1;
  font-weight: 500;
}

.profile-count {
  font-size: 11px;
  color: rgba(255, 255, 255, 0.5);
  background: rgba(255, 255, 255, 0.1);
  padding: 1px 6px;
  border-radius: 10px;
}

.profile-toggle {
  font-size: 11px;
  color: rgba(255, 255, 255, 0.4);
}

.profile-content {
  padding: 0 12px 12px;
  border-top: 1px solid rgba(255, 255, 255, 0.06);
}

.profile-loading,
.profile-empty {
  text-align: center;
  padding: 16px 8px;
  color: rgba(255, 255, 255, 0.4);
  font-size: 12px;
  line-height: 1.6;
}

.profile-hint {
  font-size: 11px;
  color: rgba(255, 255, 255, 0.3);
}

.profile-memory-list {
  max-height: 240px;
  overflow-y: auto;
  padding-top: 8px;
}

.profile-memory-item {
  background: rgba(255, 255, 255, 0.06);
  border-radius: 6px;
  padding: 8px 10px;
  margin-bottom: 6px;
}

.memory-cat {
  font-size: 10px;
  color: #4caf93;
  font-weight: 500;
  margin-bottom: 4px;
}

.memory-value {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.8);
  line-height: 1.4;
  margin-bottom: 6px;
  word-break: break-all;
}

.memory-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.confidence-tag {
  font-size: 10px;
  padding: 1px 5px;
  border-radius: 3px;
}

.confidence-tag.conf-high {
  background: rgba(76, 175, 147, 0.2);
  color: #4caf93;
}

.confidence-tag.conf-medium {
  background: rgba(255, 193, 7, 0.2);
  color: #ffc107;
}

.confidence-tag.conf-low {
  background: rgba(158, 158, 158, 0.2);
  color: #9e9e9e;
}

.memory-delete {
  font-size: 11px !important;
  padding: 0 !important;
  color: rgba(255, 77, 79, 0.6) !important;
}

.memory-delete:hover {
  color: #ff4d4f !important;
}

.profile-actions {
  text-align: center;
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid rgba(255, 255, 255, 0.06);
}
</style>
