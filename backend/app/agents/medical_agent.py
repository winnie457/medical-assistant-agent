"""智能就医助手 ReAct Agent - 基于 LangGraph 实现"""

import json
import time
import uuid
import asyncio
import logging
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    ToolMessage,
    BaseMessage,
    SystemMessage,
    message_chunk_to_message,
)
from langchain_openai import ChatOpenAI
from ..services.llm_service import get_llm
from ..config import get_settings
from ..tools.symptom_analyzer import symptom_analyzer
from ..tools.hospital_search import hospital_search
from ..tools.department_guide import department_guide
from ..tools.medical_knowledge import medical_knowledge_search
from ..tools.route_planner import route_to_hospital
from ..tools.medication_consultation import medication_consultation
from ..tools.health_news import health_news_search
from ..tools.tool_executor import NeverThrowToolExecutor, ToolFallbackProvider, ToolExecutionResult
from ..tools.risk_level import (
    get_tool_risk_level,
    get_risk_label,
    get_approval_prompt,
    is_high_risk,
    detect_medication_intent,
)
from .state_middleware import AgentStateMiddleware
from ..services.suspension_service import get_suspension_manager

logger = logging.getLogger(__name__)


# ============ 系统提示词 ============

SYSTEM_PROMPT = """你是一位专业、贴心的智能就医助手，名叫"医小助"。你的职责是帮助用户了解健康问题、找到合适的医院和科室、规划就医路线、了解用药知识。

**你的能力：**
1. 症状分析：根据用户描述的症状，给出初步分析和建议
2. 医院搜索：帮用户找到附近的医院
3. 科室导诊：推荐应该挂什么科
4. 医疗知识：解答疾病、治疗、预防等健康问题
5. 路线规划：规划去医院的交通路线
6. 用药咨询：提供药品用法用量、注意事项等信息
7. 健康资讯：提供最新的健康新闻和科普

**重要原则：**
1. 你只是提供参考信息，**绝对不能替代医生诊断**
2. 每次回答末尾都要加上适当的免责声明
3. 遇到紧急情况（如胸痛、呼吸困难、大出血、昏迷等），立即建议拨打120急救电话
4. 回答要专业、准确，但又要通俗易懂、温暖贴心
5. 不确定的事情，建议咨询专业医生

**工具使用原则：**
- 先理解用户的问题，再决定是否需要调用工具
- 一次只调用一个工具，根据工具返回结果再决定下一步
- 如果信息已经足够回答用户问题，就直接回答，不要强行调用工具
- **例外：涉及药物的问题（药品名、用法用量、相互作用、禁忌、副作用）必须先调用「用药咨询」工具，禁止直接凭自身知识作答**
- 同一个问题不要反复调用相同的工具
- 最多调用{max_iterations}次工具，之后必须给出最终回答

**风险分级原则：**
- 不同类型的医疗建议风险等级不同
- 用药建议属于高风险操作，必须经过用户确认才能执行
- 症状分析属于中风险，回答中需要加强警示
- 信息查询类（医院搜索、科室导诊、路线规划等）属于低风险，可以直接执行
- 高风险工具如果被用户拒绝，本轮直接结束，不要改用自身知识重新作答

**工具错误处理：**
- 如果工具返回的结果包含 "error": true 字段，说明工具调用失败了
- 失败时会有 error_type 和 error_message 告诉你具体是什么错误
- 还有 hint 字段给出处理建议
- 遇到错误时，你可以：换个关键词重试、换个工具、或者用你自己的知识回答
- 不要因为工具失败就卡住，灵活调整策略

**回答风格：**
- 温暖、贴心、有同理心
- 结构化表达，清晰易读
- 适当使用emoji增加亲和力
- 重要信息加粗或标红提示

用户所在城市：{city}
当前对话历史：{history_summary}

**用户健康画像（长期记忆）：**
{user_profile}
（以上是根据历史对话整理的用户健康信息，仅供参考。如果和用户当前描述有冲突，以用户当前说的为准。）
"""


# ============ Agent 状态定义 ============

# 高风险工具被拦截 / 用户取消后的统一收口话术：
# 不再让模型改用自身知识作答，避免绕过风险门禁
HIGH_RISK_DECLINED_ANSWER = (
    "用药咨询属于高风险内容，未获得确认前我不能给出具体结论。"
    "用药涉及个人体质、既往病史和合并用药，建议携带正在服用的药品咨询医生或药师。"
    "如果需要，我可以帮你查附近医院或对应的就诊科室。"
)

class AgentState(TypedDict):
    """Agent 状态"""
    # 使用 add_messages reducer：节点只需返回新增消息，由框架负责追加，
    # 避免整表覆盖导致 ToolMessage 与 AIMessage.tool_calls 配对断裂
    messages: Annotated[List[BaseMessage], add_messages]
    city: str
    iteration_count: int
    max_iterations: int
    tool_calls_history: List[Dict[str, Any]]
    final_answer: Optional[str]
    user_profile: Optional[str]


# ============ 工具列表 ============

def get_tools():
    """获取所有可用工具"""
    return [
        symptom_analyzer,
        hospital_search,
        department_guide,
        medical_knowledge_search,
        route_to_hospital,
        medication_consultation,
        health_news_search,
    ]


# ============ 核心 Agent 类 ============

class MedicalAgent:
    """智能就医助手 Agent (ReAct 模式)"""

    def __init__(self):
        self.llm = get_llm()
        self.tools = get_tools()
        self.settings = get_settings()
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.graph = self._build_graph()
        # Never-Throw 工具执行器（带降级方案）
        self.tool_executor = NeverThrowToolExecutor(
            tools=self.tools,
            fallback_providers=ToolFallbackProvider.get_all_fallbacks()
        )

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 状态图"""
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("agent", self._agent_node)
        # 与非流式接口共用同一套容错执行器 + 高风险 fail-closed 门禁
        workflow.add_node("tools", self._safe_tools_node)

        # 设置入口
        workflow.set_entry_point("agent")

        # 添加条件边：agent -> tools 或 agent -> END
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )

        # tools -> agent；若本轮因高危工具拦截已收口，则直接结束
        workflow.add_conditional_edges(
            "tools",
            lambda state: "end" if state.get("final_answer") else "continue",
            {
                "continue": "agent",
                "end": END
            }
        )

        return workflow.compile()

    def _agent_node(self, state: AgentState) -> Dict[str, Any]:
        """Agent 节点：思考并决定下一步行动"""
        # 与流式路径保持一致：每轮迭代前先做状态校验与修复
        state, validation_result = AgentStateMiddleware.validate_and_fix(state)
        if not validation_result.is_valid:
            logger.warning(f"[状态校验] 非流式路径修复了异常状态: {validation_result.issues}")

        messages = state["messages"]
        city = state.get("city", "北京")
        iteration = state.get("iteration_count", 0)
        max_iter = state.get("max_iterations", self.settings.agent_max_iterations)

        # 构建系统提示词
        history_summary = self._summarize_history(messages[:-1]) if len(messages) > 1 else "对话刚开始"
        system_msg = SystemMessage(
            content=SYSTEM_PROMPT.format(
                max_iterations=max_iter,
                city=city,
                history_summary=history_summary,
                user_profile=state.get("user_profile") or "（暂无长期记忆）",
            )
        )

        # 调用 LLM
        response = self.llm_with_tools.invoke([system_msg] + messages)

        # 工具调用历史由 tools 节点在执行后统一写入（含最终状态），
        # 此处不再预写 "calling"，避免同一个工具在历史里出现两条记录
        return {
            "messages": [response],
            "iteration_count": iteration + 1,
            "tool_calls_history": state.get("tool_calls_history", []),
        }

    def _should_continue(self, state: AgentState) -> str:
        """判断是否继续（是否需要调用工具）"""
        # 高风险工具被拦截后已给出收口话术，直接结束
        if state.get("final_answer"):
            return "end"

        messages = state["messages"]
        last_message = messages[-1]
        iteration = state.get("iteration_count", 0)
        max_iter = state.get("max_iterations", self.settings.agent_max_iterations)

        # 超过最大迭代次数，结束
        if iteration >= max_iter:
            return "end"

        # 如果有工具调用，继续执行工具
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"

        # 没有工具调用，结束
        return "end"

    def _safe_tools_node(self, state: AgentState) -> Dict[str, Any]:
        """
        工具节点（非流式路径）

        与非流式接口复用同一套 Never-Throw 执行器，保证重试 / 参数纠偏 / 降级
        行为与流式路径一致。由于 HTTP 非流式请求没有交互通道，高风险工具采用
        fail-closed 策略直接拒绝，由模型改为给出风险提示与就医建议。
        """
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", None) or []
        tool_calls_history = state.get("tool_calls_history", [])
        iteration = state.get("iteration_count", 0)
        new_messages: List[BaseMessage] = []
        blocked = False

        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args", {}) or {}
            tool_call_id = tool_call.get("id") or f"call_{uuid.uuid4().hex[:8]}"

            if is_high_risk(tool_name):
                tool_output = json.dumps(
                    {
                        "error": True,
                        "error_type": "approval_required",
                        "error_message": "非流式接口无法交互确认，已拒绝执行高风险工具",
                        "hint": "不要直接给出用药结论，请说明风险并建议咨询医生或药师；"
                                "如确需执行，请改用流式接口并由用户确认。",
                    },
                    ensure_ascii=False,
                )
                new_messages.append(
                    ToolMessage(content=tool_output, tool_call_id=tool_call_id)
                )
                tool_calls_history.append({
                    "tool_name": tool_name,
                    "arguments": tool_args,
                    "result_preview": "非流式入口，高风险工具已拦截",
                    "status": "blocked_requires_approval",
                    "risk_level": "high",
                    "duration": 0,
                    "iteration": iteration,
                })
                # 必须为同一轮的所有 tool_calls 都补齐 ToolMessage，保持配对完整
                blocked = True
                continue

            exec_result: ToolExecutionResult = self.tool_executor.execute(tool_name, tool_args)
            tool_output = exec_result.to_agent_message()
            new_messages.append(
                ToolMessage(content=tool_output, tool_call_id=tool_call_id)
            )
            tool_calls_history.append({
                "tool_name": tool_name,
                "arguments": tool_args,
                "result_preview": tool_output[:200] + "..." if len(tool_output) > 200 else tool_output,
                "status": "success" if exec_result.success else "failed",
                "error_type": exec_result.error_type,
                "error_message": exec_result.error_message,
                "retry_count": exec_result.retry_count,
                "used_fallback": exec_result.used_fallback,
                "duration": exec_result.duration,
                "iteration": iteration,
            })

        update: Dict[str, Any] = {
            "messages": new_messages,
            "tool_calls_history": tool_calls_history,
        }
        if blocked:
            # 高危工具被拦截 → 收口，不再让模型改用自身知识作答
            update["final_answer"] = HIGH_RISK_DECLINED_ANSWER
        return update

    def _summarize_history(self, messages: List[BaseMessage]) -> str:
        """简要总结对话历史"""
        if not messages:
            return "对话刚开始"

        user_msgs = [m for m in messages if isinstance(m, HumanMessage)]
        if len(user_msgs) <= 2:
            return "对话刚开始"

        return f"已进行{len(user_msgs)}轮对话"

    def chat(self, message: str, history: Optional[List[BaseMessage]] = None,
             city: str = "北京", user_id: str = "default") -> Dict[str, Any]:
        """
        非流式对话

        Args:
            message: 用户消息
            history: 对话历史
            city: 所在城市
            user_id: 用户标识（用于加载长期记忆）

        Returns:
            包含回答、工具调用历史等信息的字典
        """
        from ..memory.memory_manager import get_memory_manager

        # 非流式路径同样注入长期记忆，保持与流式一致
        profile = get_memory_manager().get_profile(user_id)

        # 【查询级前置门禁】用药类问题即使模型选择直接作答，也必须先经用户确认；
        # 非流式入口没有交互确认通道 → fail-closed
        if detect_medication_intent(message):
            logger.info("[HITL] 非流式入口识别到用药类问题，已按拒绝处理")
            return {
                "answer": HIGH_RISK_DECLINED_ANSWER,
                "messages": [
                    HumanMessage(content=message),
                    AIMessage(content=HIGH_RISK_DECLINED_ANSWER),
                ],
                "tool_calls_history": [{
                    "tool_name": "medication_consultation",
                    "arguments": {"query": message},
                    "result_preview": "非流式入口，用药类问题已拦截",
                    "status": "blocked_requires_approval",
                    "risk_level": "high",
                    "duration": 0,
                    "iteration": 1,
                }],
                "iteration_count": 0,
            }

        messages = history.copy() if history else []
        messages.append(HumanMessage(content=message))

        initial_state: AgentState = {
            "messages": messages,
            "city": city,
            "iteration_count": 0,
            "max_iterations": self.settings.agent_max_iterations,
            "tool_calls_history": [],
            "final_answer": None,
            "user_profile": profile.format_for_prompt(),
        }

        result = self.graph.invoke(initial_state)
        final_messages = result["messages"]
        final_answer = (
            result.get("final_answer")
            or (final_messages[-1].content if final_messages else "抱歉，我暂时无法回答这个问题。")
        )

        return {
            "answer": final_answer,
            "messages": final_messages,
            "tool_calls_history": result.get("tool_calls_history", []),
            "iteration_count": result.get("iteration_count", 0)
        }

    async def chat_stream(self, message: str, history: Optional[List[BaseMessage]] = None,
                          city: str = "北京", conversation_id: str = None, user_id: str = "default"):
        """
        流式对话（生成器方式）

        Yields:
            不同类型的事件：thought, tool_call, tool_result, answer, done,
            approval_required（高风险工具等待用户确认）
        """
        suspension_manager = get_suspension_manager()
        from ..memory.memory_manager import get_memory_manager
        memory_manager = get_memory_manager()
        
        # 加载用户健康画像（长期记忆）
        profile = memory_manager.get_profile(user_id)
        user_profile_text = profile.format_for_prompt()
        
        messages = history.copy() if history else []
        messages.append(HumanMessage(content=message))

        state: AgentState = {
            "messages": messages,
            "city": city,
            "iteration_count": 0,
            "max_iterations": self.settings.agent_max_iterations,
            "tool_calls_history": [],
            "final_answer": None,
            "user_profile": user_profile_text,
        }

        # 【查询级前置门禁】用药类问题先确认再进入 ReAct 循环。
        # 只拦工具调用是不够的：模型可以选择“直接回答”而绕过确认，因此入口先做确定性识别。
        approved_tool_names: set[str] = set()
        if detect_medication_intent(message):
            approval_prompt = get_approval_prompt("medication_consultation") or {}
            tool_args = {"query": message}
            yield {
                "type": "approval_required",
                "tool_name": "medication_consultation",
                "tool_args": tool_args,
                "risk_level": "high",
                "title": approval_prompt.get("title", "确认操作"),
                "message": approval_prompt.get("message", "用药建议属于高风险信息，是否继续查看？"),
                "warning": approval_prompt.get("warning", ""),
                "confirm_text": approval_prompt.get("confirm_text", "确认"),
                "cancel_text": approval_prompt.get("cancel_text", "取消"),
                "iteration": 1,
            }

            approved = False
            if conversation_id:
                approved = await suspension_manager.suspend(
                    conversation_id=conversation_id,
                    tool_name="medication_consultation",
                    tool_args=tool_args,
                )
            else:
                logger.warning("[HITL] 缺少 conversation_id，用药类问题按拒绝处理")

            if not approved:
                state_ref = suspension_manager.get_suspended_state(conversation_id) if conversation_id else None
                reject_reason = getattr(state_ref, "reject_reason", None) or "用户取消"
                state["tool_calls_history"].append({
                    "tool_name": "medication_consultation",
                    "arguments": tool_args,
                    "result_preview": "用药类问题未获确认，已拦截",
                    "status": "rejected",
                    "risk_level": "high",
                    "duration": 0,
                    "iteration": 1,
                })
                yield {
                    "type": "tool_result",
                    "tool_name": "medication_consultation",
                    "status": "rejected",
                    "error_type": "user_rejected",
                    "error_message": reject_reason,
                    "iteration": 1,
                }
                yield {"type": "answer", "content": HIGH_RISK_DECLINED_ANSWER, "done": True}
                yield {
                    "type": "done",
                    "iteration_count": 0,
                    "tool_calls_count": len(state["tool_calls_history"]),
                    "tool_calls_history": state["tool_calls_history"],
                }
                return

            # 本轮已获得用户确认，后续同名工具不再重复询问
            approved_tool_names.add("medication_consultation")

        # 手动执行 ReAct 循环
        for i in range(self.settings.agent_max_iterations):
            # 【状态校验中间件】每次迭代前校验并修复状态
            state, validation_result = AgentStateMiddleware.validate_and_fix(state)
            if not validation_result.is_valid and validation_result.fixes:
                logger.info(
                    f"[状态校验] 第{i+1}轮: "
                    f"发现{len(validation_result.issues)}个问题，"
                    f"修复{len(validation_result.fixes)}个"
                )

            # Agent 思考
            yield {
                "type": "thought",
                "content": f"第{i+1}步：分析问题中...",
                "iteration": i + 1
            }

            # 构建系统提示词
            history_summary = self._summarize_history(state["messages"][:-1])
            system_msg = SystemMessage(
                content=SYSTEM_PROMPT.format(
                    max_iterations=self.settings.agent_max_iterations,
                    city=city,
                    history_summary=history_summary,
                    user_profile=user_profile_text,
                )
            )

            # 调用 LLM（token 级流式：正文增量推送，tool_call 增量累积）
            emitted_delta = False
            reset_sent = False
            full_chunk: Optional[AIMessageChunk] = None
            async for chunk in self.llm_with_tools.astream([system_msg] + state["messages"]):
                # AIMessageChunk 支持 + 运算：content 拼接、tool_call_chunks 按 index 合并
                full_chunk = chunk if full_chunk is None else full_chunk + chunk

                if getattr(chunk, "tool_call_chunks", None):
                    # 一旦转入工具调用，之前误推的正文需要通知前端清空
                    if emitted_delta and not reset_sent:
                        reset_sent = True
                        yield {"type": "answer_reset", "content": ""}
                elif chunk.content:
                    text = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                    emitted_delta = True
                    yield {"type": "answer_delta", "content": text}

            # 合并完成后还原为标准 AIMessage，保证 .tool_calls / .content 可用
            response = (
                message_chunk_to_message(full_chunk)
                if full_chunk is not None
                else AIMessage(content="")
            )
            state["messages"].append(response)
            state["iteration_count"] = i + 1

            # 检查是否有工具调用
            tool_calls = response.tool_calls if hasattr(response, 'tool_calls') and response.tool_calls else []

            if tool_calls:
                rejected_exit = False
                for tool_call in tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})

                    # 发送工具调用事件
                    yield {
                        "type": "tool_call",
                        "tool_name": tool_name,
                        "tool_description": self._get_tool_description(tool_name),
                        "arguments": tool_args,
                        "risk_level": get_tool_risk_level(tool_name).value,
                        "risk_label": get_risk_label(tool_name),
                        "iteration": i + 1
                    }

                    # 【风险分级门禁】高风险工具需要用户确认；本轮已确认过的工具不再重复询问
                    if is_high_risk(tool_name) and tool_name not in approved_tool_names:
                        approval_prompt = get_approval_prompt(tool_name)
                        
                        # 发送"需要确认"事件
                        yield {
                            "type": "approval_required",
                            "tool_name": tool_name,
                            "tool_args": tool_args,
                            "risk_level": "high",
                            "title": approval_prompt["title"] if approval_prompt else "确认操作",
                            "message": approval_prompt["message"] if approval_prompt else "是否确认执行此操作？",
                            "warning": approval_prompt["warning"] if approval_prompt else "",
                            "confirm_text": approval_prompt["confirm_text"] if approval_prompt else "确认",
                            "cancel_text": approval_prompt["cancel_text"] if approval_prompt else "取消",
                            "iteration": i + 1,
                        }

                        # 挂起等待用户确认
                        if conversation_id:
                            logger.info(f"[HITL] 等待用户确认高风险工具: {tool_name}")
                            approved = await suspension_manager.suspend(
                                conversation_id=conversation_id,
                                tool_name=tool_name,
                                tool_args=tool_args,
                            )
                        else:
                            # 无会话标识则无法挂起等待确认 → fail-closed，按用户拒绝处理
                            logger.warning(f"[HITL] 缺少 conversation_id，高危工具按拒绝处理: {tool_name}")
                            approved = False

                        if not approved:
                            # 用户拒绝或超时 → 返回结构化错误给 Agent
                            state_ref = suspension_manager.get_suspended_state(conversation_id)
                            reject_reason = state_ref.reject_reason if state_ref else "用户取消"
                            
                            tool_output = (
                                '{"error": true, "error_type": "user_rejected", '
                                f'"error_message": "用户拒绝执行此操作：{reject_reason}", '
                                '"hint": "用户拒绝了该工具调用，可以尝试换一种方式回答，或者建议用户咨询专业医生。"}'
                            )
                            
                            # 将错误加入消息历史
                            tool_call_id = tool_call.get("id", "")
                            if not tool_call_id:
                                tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
                            state["messages"].append(
                                ToolMessage(content=tool_output, tool_call_id=tool_call_id)
                            )
                            
                            # 记录工具调用历史
                            state["tool_calls_history"].append({
                                "tool_name": tool_name,
                                "arguments": tool_args,
                                "result_preview": f"用户拒绝: {reject_reason}",
                                "status": "rejected",
                                "risk_level": "high",
                                "duration": 0,
                                "iteration": i + 1
                            })
                            
                            # 发送工具结果事件（被拒绝）
                            yield {
                                "type": "tool_result",
                                "tool_name": tool_name,
                                "status": "rejected",
                                "risk_level": "high",
                                "result_preview": f"用户拒绝执行此操作",
                                "duration": 0,
                                "iteration": i + 1
                            }
                            
                            # 【拒绝即收口】用户取消后给出固定安全提示并结束本轮，
                            # 不再让模型用自己的知识把同一问题重新答一遍，避免绕过风险门禁
                            safe_answer = HIGH_RISK_DECLINED_ANSWER
                            yield {
                                "type": "answer",
                                "content": safe_answer,
                                "done": True
                            }
                            state["final_answer"] = safe_answer
                            rejected_exit = True
                            break
                        
                    #如果工具是低风险（比如搜个医院），或者用户点了“同意”
                    # 执行工具（Never-Throw 保证，内部自动重试+纠偏+降级）
                    # 执行器是同步实现（含 time.sleep 退避），放入线程池避免阻塞事件循环
                    exec_result: ToolExecutionResult = await asyncio.to_thread(
                        self.tool_executor.execute, tool_name, tool_args
                    )

                    # 工具结果（给 Agent 看的内容）
                    tool_output = exec_result.to_agent_message()

                    # 将工具结果加入消息历史
                    tool_call_id = tool_call.get("id", "")
                    if not tool_call_id:
                        tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
                    state["messages"].append(
                        ToolMessage(
                            content=tool_output,
                            tool_call_id=tool_call_id
                        )
                    )

                    # 记录工具调用历史（更丰富的信息）
                    state["tool_calls_history"].append({
                        "tool_name": tool_name,
                        "arguments": tool_args,
                        "result_preview": tool_output[:200] + "..." if len(tool_output) > 200 else tool_output,
                        "status": "success" if exec_result.success else "failed",
                        "error_type": exec_result.error_type,
                        "error_message": exec_result.error_message,
                        "retry_count": exec_result.retry_count,
                        "used_fallback": exec_result.used_fallback,
                        "duration": exec_result.duration,
                        "iteration": i + 1
                    })

                    # 发送工具结果事件
                    yield {
                        "type": "tool_result",
                        "tool_name": tool_name,
                        "status": "success" if exec_result.success else "failed",
                        "error_type": exec_result.error_type,
                        "retry_count": exec_result.retry_count,
                        "used_fallback": exec_result.used_fallback,
                        "result_preview": tool_output[:150] + "..." if len(tool_output) > 150 else tool_output,
                        "duration": exec_result.duration,
                        "iteration": i + 1
                    }

                # 用户取消高风险工具后，跳出 ReAct 循环，直接进入收尾
                if rejected_exit:
                    break
            else:
                # 没有工具调用，输出最终回答
                # 流式输出最终答案
                full_answer = response.content
                yield {
                    "type": "answer",
                    "content": full_answer,
                    "done": True
                }
                break
        else:
            # 达到最大迭代次数
            yield {
                "type": "answer",
                "content": "抱歉，经过多轮查询仍未能完全解答您的问题。建议您咨询专业医生获取更准确的建议。",
                "done": True
            }

        # 发送完成事件
        yield {
            "type": "done",
            "iteration_count": state["iteration_count"],
            "tool_calls_count": len(state["tool_calls_history"]),
            "tool_calls_history": state["tool_calls_history"]
        }

    def _get_tool_by_name(self, tool_name: str):
        """根据名称获取工具（委托给 tool_executor）"""
        return self.tool_executor.get_tool(tool_name)

    def _get_tool_description(self, tool_name: str) -> str:
        """获取工具描述"""
        tool_descriptions = {
            "symptom_analyzer": "症状分析 - 根据症状给出初步分析和就医建议",
            "hospital_search": "医院搜索 - 搜索附近的医院",
            "department_guide": "科室导诊 - 推荐应该就诊的科室",
            "medical_knowledge_search": "医疗知识 - 查询疾病、治疗、预防等健康知识",
            "route_to_hospital": "路线规划 - 规划去医院的交通路线",
            "medication_consultation": "用药咨询 - 查询药品用法用量和注意事项",
            "health_news_search": "健康资讯 - 搜索最新健康新闻和科普文章"
        }
        return tool_descriptions.get(tool_name, tool_name)


# ============ 单例模式 ============

_medical_agent_instance = None


def get_medical_agent() -> MedicalAgent:
    """获取智能就医助手 Agent 实例（单例）"""
    global _medical_agent_instance

    if _medical_agent_instance is None:
        print("🔄 初始化智能就医助手 Agent (ReAct + LangGraph)...")
        _medical_agent_instance = MedicalAgent()
        print(f"✅ 智能就医助手初始化成功")
        print(f"   可用工具数量: {len(_medical_agent_instance.tools)}")
        for tool in _medical_agent_instance.tools:
            print(f"     - {tool.name}")

    return _medical_agent_instance


def reset_medical_agent():
    """重置 Agent 实例"""
    global _medical_agent_instance
    _medical_agent_instance = None
