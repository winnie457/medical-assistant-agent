"""智能就医助手 ReAct Agent - 基于 LangGraph 实现"""

import asyncio
import json
import uuid
import logging
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.types import Command, interrupt
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
from .emergency import build_emergency_answer, detect_emergency
from .output_guard import enforce_final_answer
from .checkpointer import get_checkpointer_provider

logger = logging.getLogger(__name__)


def _emit(payload: Dict[str, Any]) -> None:
    """向 LangGraph 的 custom 流推送事件。

    仅在 stream_mode 包含 "custom" 时生效；非流式 graph.invoke 场景下
    静默忽略，保证两条链路共用同一套节点实现。
    """
    try:
        get_stream_writer()(payload)
    except Exception:  # 非流式或没有 custom 流时 writer 不可用
        return


def _approval_payload(tool_name: str, tool_args: Dict[str, Any], iteration: int) -> Dict[str, Any]:
    """构造高风险操作确认载荷（既作为 interrupt 的值，也直接作为 SSE 事件）"""
    prompt = get_approval_prompt(tool_name) or {}
    return {
        "type": "approval_required",
        "tool_name": tool_name,
        "tool_args": tool_args,
        "risk_level": "high",
        "title": prompt.get("title", "确认操作"),
        "message": prompt.get("message", "该操作属于高风险内容，是否继续？"),
        "warning": prompt.get("warning", ""),
        "confirm_text": prompt.get("confirm_text", "确认"),
        "cancel_text": prompt.get("cancel_text", "取消"),
        "iteration": iteration,
    }


def _final_ai_content(messages: List[BaseMessage]) -> str:
    """取最后一条不含 tool_calls 的 AI 消息正文，作为最终回答"""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not (getattr(msg, "tool_calls", None) or []):
            content = msg.content
            if isinstance(content, str) and content.strip():
                return content
    return ""


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
6. **急症优先**：判定为急症时不要调用任何工具、不要追问细节，直接给出拨打 120 的急救指引

**工具使用原则：**
- 先理解用户的问题，再决定是否需要调用工具
- 一次只调用一个工具，根据工具返回结果再决定下一步
- 如果信息已经足够回答用户问题，就直接回答，不要强行调用工具
- **例外：涉及药物的问题（药品名、用法用量、相互作用、禁忌、副作用）必须先调用「用药咨询」工具，禁止直接凭自身知识作答**
- 同一个问题不要反复调用相同的工具
- 最多调用{max_iterations}次工具，之后必须给出最终回答
- 工具数据有边界：医院搜索只返回名称/地址/电话等 POI 信息，不要补充科室排名、号源、专家、口碑；需要时直说未获取
- 路线工具失败或降级时，只说明未获取到实时路线，不要自行编排换乘方案、用时或距离
- 资讯检索为空时，明确说明未检索到，不要用通用知识补成"最新新闻"

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
    # 是否有交互确认通道：流式 True、非流式 False（用于 HITL 分流）
    interactive: bool
    # 本轮是否已在入口确认过高风险操作（避免同一轮重复询问）
    approved_high_risk: bool


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
        # 图与 checkpointer 懒构建：AsyncSqliteSaver 需要先 await 建表，
        # 不能在同步的 __init__ 里完成（见 checkpointer.py 的说明）。
        self._checkpointer_provider = get_checkpointer_provider()
        self.checkpointer = None
        self.graph = None
        self._graph_lock = asyncio.Lock()
        # Never-Throw 工具执行器（带降级方案）
        self.tool_executor = NeverThrowToolExecutor(
            tools=self.tools,
            fallback_providers=ToolFallbackProvider.get_all_fallbacks()
        )

    async def ensure_graph(self):
        """懒构建状态图。

        第一次真正跑图时才初始化 checkpointer（SQLite 建表是异步的），
        之后复用同一个 saver，保证同一进程内所有运行共享一份持久化状态。
        """
        if self.graph is not None:
            return self.graph
        async with self._graph_lock:
            if self.graph is None:
                self.checkpointer = await self._checkpointer_provider.get()
                self.graph = self._build_graph()
                logger.info(
                    "[Agent] 状态图就绪，checkpointer=%s（持久化=%s）",
                    self._checkpointer_provider.kind,
                    self._checkpointer_provider.persistent,
                )
        return self.graph

    async def _delete_thread(self, run_id: str) -> None:
        """释放某次运行的检查点（同步/异步 saver 都兼容）。"""
        try:
            if hasattr(self.checkpointer, "adelete_thread"):
                await self.checkpointer.adelete_thread(run_id)
            elif hasattr(self.checkpointer, "delete_thread"):
                self.checkpointer.delete_thread(run_id)
        except Exception as exc:  # 清理失败不影响主流程
            logger.debug("[Agent] 清理检查点失败(可忽略): %s", exc)

    async def aclose(self) -> None:
        """关闭持久化连接。

        aiosqlite 在后台线程里跑事件循环，连接不关则进程无法退出：
        CLI/压测脚本收尾、FastAPI shutdown 都要调用这里。
        """
        try:
            await self._checkpointer_provider.aclose()
        except Exception as exc:
            logger.debug("[Agent] 关闭 checkpointer 失败(可忽略): %s", exc)

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 状态图

        结构：precheck → agent ⇄ approval → tools → END
        - precheck：入口前置门禁，用药类问题在进入循环前先确认
        - approval：工具级风险门禁，模型选择高风险工具时挂起确认
        两条链路（非流式 invoke / 流式 astream）共用同一张图，
        交互能力由 state["interactive"] 决定，而非维护两套 ReAct 逻辑。
        """
        workflow = StateGraph(AgentState)

        workflow.add_node("precheck", self._precheck_node)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("approval", self._approval_gate_node)
        # 与非流式接口共用同一套容错执行器 + 高风险 fail-closed 门禁
        workflow.add_node("tools", self._safe_tools_node)

        workflow.set_entry_point("precheck")

        # precheck 已给出收口话术（拦截/拒绝）则直接结束，否则进入 ReAct 循环
        workflow.add_conditional_edges(
            "precheck",
            lambda state: "end" if state.get("final_answer") else "agent",
            {"agent": "agent", "end": END},
        )

        # agent 决定是否调用工具；调用前先经过风险门禁
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {"continue": "approval", "end": END},
        )

        # 门禁通过（或已确认/无需确认）→ 执行工具；被拒绝 → 收口结束
        workflow.add_conditional_edges(
            "approval",
            lambda state: "end" if state.get("final_answer") else "tools",
            {"tools": "tools", "end": END},
        )

        # tools → agent；若本轮因高危工具拦截已收口，则直接结束
        workflow.add_conditional_edges(
            "tools",
            lambda state: "end" if state.get("final_answer") else "continue",
            {"continue": "agent", "end": END},
        )

        return workflow.compile(checkpointer=self.checkpointer)

    def _precheck_node(self, state: AgentState) -> Dict[str, Any]:
        """入口前置门禁

        两级判定，顺序不能反：

        1. **急症优先**：命中急症规则（误服、卒中、抽搐、大出血、头孢配酒等）时
           直接给出 120 急救指引并结束本轮。急症问题里常常带"药"字，若先走用药
           门禁，用户会收到"请确认是否查看用药信息"——这是安全优先级错误。
        2. **用药门禁**：只拦工具调用是不够的，模型可以选择"直接作答"而绕过确认，
           因此在进入 ReAct 循环前再做一次确定性的用药意图识别。
           - 交互入口（流式）：interrupt 挂起，等待用户确认
           - 非交互入口（非流式）：没有确认通道 → fail-closed 直接收口
        """
        text = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                text = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        emergency = detect_emergency(text)
        if emergency:
            answer = build_emergency_answer(emergency)
            logger.info(f"[急症] 命中 {emergency}，跳过用药门禁直接给出急救指引")
            return {
                "messages": [AIMessage(content=answer)],
                "final_answer": answer,
                "tool_calls_history": [{
                    "tool_name": "emergency_triage",
                    "arguments": {"query": text},
                    "result_preview": f"急症规则命中: {emergency}",
                    "status": "emergency_shortcut",
                    "risk_level": "high",
                    "duration": 0,
                    "iteration": 1,
                }],
            }

        if not detect_medication_intent(text):
            return {}

        tool_args = {"query": text}

        if not state.get("interactive"):
            return {
                "messages": [AIMessage(content=HIGH_RISK_DECLINED_ANSWER)],
                "final_answer": HIGH_RISK_DECLINED_ANSWER,
                "tool_calls_history": [{
                    "tool_name": "medication_consultation",
                    "arguments": tool_args,
                    "result_preview": "非流式入口，用药类问题已拦截",
                    "status": "blocked_requires_approval",
                    "risk_level": "high",
                    "duration": 0,
                    "iteration": 1,
                }],
            }

        decision = interrupt(_approval_payload("medication_consultation", tool_args, 1))

        if (decision or {}).get("approved"):
            return {"approved_high_risk": True}

        reason = (decision or {}).get("reason") or "用户取消"
        return {
            "messages": [AIMessage(content=HIGH_RISK_DECLINED_ANSWER)],
            "final_answer": HIGH_RISK_DECLINED_ANSWER,
            "tool_calls_history": [{
                "tool_name": "medication_consultation",
                "arguments": tool_args,
                "result_preview": f"用户拒绝: {reason}",
                "status": "rejected",
                "risk_level": "high",
                "duration": 0,
                "iteration": 1,
            }],
        }

    def _approval_gate_node(self, state: AgentState) -> Dict[str, Any]:
        """工具级风险门禁

        模型决定调用高风险工具时挂起等待确认。interrupt() 必须位于节点
        的副作用之前：恢复时节点会从头重跑，前面的逻辑必须是幂等的。
        """
        if state.get("approved_high_risk") or not state.get("interactive"):
            return {}

        tool_calls = getattr(state["messages"][-1], "tool_calls", None) or []
        risky_calls = [tc for tc in tool_calls if is_high_risk(tc.get("name", ""))]
        if not risky_calls:
            return {}

        first = risky_calls[0]
        decision = interrupt(_approval_payload(
            first.get("name", ""),
            first.get("args", {}) or {},
            state.get("iteration_count", 1),
        ))

        if (decision or {}).get("approved"):
            return {"approved_high_risk": True}

        reason = (decision or {}).get("reason") or "用户取消"

        # 用户拒绝：为本轮所有 tool_call 补齐 ToolMessage，避免消息配对断裂；
        # 之后直接收口，不再让模型改用自身知识重新作答
        new_messages: List[BaseMessage] = []
        for tool_call in tool_calls:
            tool_call_id = tool_call.get("id") or f"call_{uuid.uuid4().hex[:8]}"
            new_messages.append(ToolMessage(
                content=json.dumps({
                    "error": True,
                    "error_type": "user_rejected",
                    "error_message": f"用户拒绝执行此操作：{reason}",
                    "hint": "用户已拒绝该工具调用，本轮直接结束，不要改用自身知识重新回答。",
                }, ensure_ascii=False),
                tool_call_id=tool_call_id,
            ))

        history = list(state.get("tool_calls_history", []))
        history.append({
            "tool_name": first.get("name", ""),
            "arguments": first.get("args", {}) or {},
            "result_preview": f"用户拒绝: {reason}",
            "status": "rejected",
            "risk_level": "high",
            "duration": 0,
            "iteration": state.get("iteration_count", 1),
        })

        return {
            "messages": new_messages,
            "tool_calls_history": history,
            "final_answer": HIGH_RISK_DECLINED_ANSWER,
        }

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

        # 推送推理进度（custom 流；非流式调用时静默忽略）
        _emit({
            "type": "thought",
            "content": f"第{iteration + 1}步：分析问题中...",
            "iteration": iteration + 1,
        })

        # 调用 LLM：用 stream 累积完整响应，
        # 既保持非流式聚合结果不变，又让流式场景能捕获 token 增量
        full_chunk: Optional[AIMessageChunk] = None
        emitted_delta = False
        reset_sent = False
        for chunk in self.llm_with_tools.stream([system_msg] + messages):
            full_chunk = chunk if full_chunk is None else full_chunk + chunk

            if getattr(chunk, "tool_call_chunks", None):
                # 一旦转入工具调用，之前误推的正文需要通知前端清空
                if emitted_delta and not reset_sent:
                    reset_sent = True
                    _emit({"type": "answer_reset"})
            elif chunk.content:
                text = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                emitted_delta = True
                _emit({"type": "answer_delta", "content": text})

        response = message_chunk_to_message(full_chunk) if full_chunk is not None else AIMessage(content="")

        # 推送本轮工具调用意图（含风险分级），供流式前端渲染
        for tool_call in getattr(response, "tool_calls", []) or []:
            tool_name = tool_call.get("name", "")
            _emit({
                "type": "tool_call",
                "tool_name": tool_name,
                "tool_description": self._get_tool_description(tool_name),
                "arguments": tool_call.get("args", {}) or {},
                "risk_level": get_tool_risk_level(tool_name).value,
                "risk_label": get_risk_label(tool_name),
                "iteration": iteration + 1,
            })

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

            # 未经确认的高风险工具一律 fail-closed；已由 precheck / approval 门禁确认的放行
            if is_high_risk(tool_name) and not state.get("approved_high_risk"):
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
                _emit({
                    "type": "tool_result",
                    "tool_name": tool_name,
                    "status": "blocked_requires_approval",
                    "risk_level": "high",
                    "result_preview": "非流式入口，高风险工具已拦截",
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
            _emit({
                "type": "tool_result",
                "tool_name": tool_name,
                "status": "success" if exec_result.success else "failed",
                "error_type": exec_result.error_type,
                "retry_count": exec_result.retry_count,
                "used_fallback": exec_result.used_fallback,
                "result_preview": tool_output[:150] + "..." if len(tool_output) > 150 else tool_output,
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

    async def chat(self, message: str, history: Optional[List[BaseMessage]] = None,
                   city: str = "北京", user_id: str = "default",
                   conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        非流式对话

        Args:
            message: 用户消息
            history: 对话历史
            city: 所在城市
            user_id: 用户标识（用于加载长期记忆）
            conversation_id: 会话 ID（仅用于日志与外部关联）
            interactive: 由入口固定为 False，高风险操作一律 fail-closed

        Returns:
            包含回答、工具调用历史等信息的字典
        """
        from ..memory.memory_manager import get_memory_manager

        # 非流式路径同样注入长期记忆，保持与流式一致
        profile = get_memory_manager().get_profile(user_id)

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
            # 非流式入口没有交互确认通道，precheck 会据此 fail-closed
            "interactive": False,
            "approved_high_risk": False,
        }

        # 每次请求使用独立的 run_id 作为 thread_id：
        # 历史消息由入口整表传入，避免同一会话多轮之间消息重复累积
        await self.ensure_graph()
        run_id = f"sync-{uuid.uuid4().hex[:12]}"
        config = {"configurable": {"thread_id": run_id}}

        try:
            result = await self.graph.ainvoke(initial_state, config)
        finally:
            # 非流式不会产生 interrupt，本轮结束后即可释放检查点
            await self._delete_thread(run_id)

        final_messages = result["messages"]
        final_answer = result.get("final_answer") or _final_ai_content(final_messages)
        # 与非流式保持同一套出口校验
        final_answer, guard_issues = enforce_final_answer(
            final_answer, result.get("tool_calls_history", [])
        )

        return {
            "answer": final_answer or "抱歉，我暂时无法回答这个问题。",
            "messages": final_messages,
            "tool_calls_history": result.get("tool_calls_history", []),
            "iteration_count": result.get("iteration_count", 0),
            "guard_issues": guard_issues,
        }

    async def chat_stream(self, message: str, history: Optional[List[BaseMessage]] = None,
                          city: str = "北京", conversation_id: str = None, user_id: str = "default"):
        """
        流式对话：驱动同一张 LangGraph，把 custom 事件与 interrupt 翻译成 SSE 事件

        Yields:
            thought / tool_call / tool_result / answer_delta / answer_reset /
            approval_required / answer / done
        """
        from ..memory.memory_manager import get_memory_manager

        # 加载用户健康画像（长期记忆）
        profile = get_memory_manager().get_profile(user_id)

        messages = history.copy() if history else []
        messages.append(HumanMessage(content=message))

        state: AgentState = {
            "messages": messages,
            "city": city,
            "iteration_count": 0,
            "max_iterations": self.settings.agent_max_iterations,
            "tool_calls_history": [],
            "final_answer": None,
            "user_profile": profile.format_for_prompt(),
            # 流式入口有交互通道，precheck / approval 节点可以 interrupt 挂起
            "interactive": True,
            "approved_high_risk": False,
        }

        await self.ensure_graph()
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        config = {"configurable": {"thread_id": run_id}}

        async for event in self._adapt_stream(
            self.graph.astream(state, config, stream_mode=["custom", "updates"]),
            run_id,
            conversation_id,
        ):
            yield event

    async def _adapt_stream(self, raw_stream, run_id: str, conversation_id: Optional[str]):
        """把 LangGraph 的多模式下原始流翻译为前端 SSE 事件协议"""
        async for mode, data in raw_stream:
            if mode == "custom":
                yield data
            elif mode == "updates":
                interrupts = data.get("__interrupt__") if isinstance(data, dict) else None
                if interrupts:
                    payload = dict(interrupts[0].value or {})
                    payload["run_id"] = run_id
                    payload["conversation_id"] = conversation_id
                    yield payload

        snapshot = await self.graph.aget_state({"configurable": {"thread_id": run_id}})
        values = snapshot.values or {}

        # 处于挂起状态：本轮尚未结束，不发 answer/done，
        # 等用户确认后由 resume_stream 继续推流
        if snapshot.interrupts:
            return

        history = values.get("tool_calls_history", [])
        final_answer = (
            values.get("final_answer")
            or _final_ai_content(values.get("messages", []))
            or "抱歉，经过多轮查询仍未能完全解答您的问题。建议您咨询专业医生获取更准确的建议。"
        )

        # 第三层护栏：工具没给的信息不许"补"，命中则在答案末尾追加限定说明
        final_answer, guard_issues = enforce_final_answer(final_answer, history)

        yield {"type": "answer", "content": final_answer, "done": True}
        yield {
            "type": "done",
            "conversation_id": conversation_id,
            "run_id": run_id,
            "iteration_count": values.get("iteration_count", 0),
            "tool_calls_count": len(history),
            "tool_calls_history": history,
            "guard_issues": guard_issues,
        }

        # 本轮已完成，释放本次运行的检查点，避免长期运行时堆积
        await self._delete_thread(run_id)

    async def resume_stream(self, run_id: str, decision: Dict[str, Any],
                            conversation_id: Optional[str] = None):
        """用户确认/拒绝后，用 Command(resume=...) 继续被中断的那次运行"""
        await self.ensure_graph()
        config = {"configurable": {"thread_id": run_id}}
        async for event in self._adapt_stream(
            self.graph.astream(Command(resume=decision), config, stream_mode=["custom", "updates"]),
            run_id,
            conversation_id,
        ):
            yield event

    async def get_pending_interrupt(self, run_id: str) -> Optional[Dict[str, Any]]:
        """读取某次运行中尚未处理的中断载荷；没有则返回 None。

        走 `aget_state`，因为 checkpointer 是异步实现；只要 checkpoint 落盘，
        换一个进程也能查到这次运行是否还在等确认。
        """
        await self.ensure_graph()
        snapshot = await self.graph.aget_state({"configurable": {"thread_id": run_id}})
        if not snapshot.interrupts:
            return None
        value = snapshot.interrupts[0].value
        return value if isinstance(value, dict) else {"value": value}


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
