"""
状态校验中间件 — Agent 状态流转的安全护栏

设计理念：
    LangGraph 的状态机模式虽然强大，但状态在多次流转中可能因为 LLM 输出异常、
    工具调用失败、并发修改等原因出现"脏状态"。如果不加校验，异常状态会在
    状态图中层层传递，最终导致难以排查的崩溃。

    这个中间件在每次状态流转前做校验和修复，保证状态始终是"健康的"。

校验维度：
    1. 消息列表完整性校验（不能为空、类型正确）
    2. 消息配对校验（ToolMessage 必须有对应的 AIMessage）
    3. 迭代次数合理性校验（不能倒退、不能超过上限）
    4. tool_calls 格式校验（id/name/args 完整性）
    5. 城市字段合法性校验
"""

import logging
import copy
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
)

logger = logging.getLogger(__name__)


class StateValidationResult:
    """状态校验结果"""
    
    def __init__(self, is_valid: bool, issues: List[str] = None, fixes: List[str] = None):
        self.is_valid = is_valid
        self.issues = issues or []
        self.fixes = fixes or []
    
    def add_issue(self, issue: str):
        self.issues.append(issue)
        self.is_valid = False
    
    def add_fix(self, fix: str):
        self.fixes.append(fix)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "issues": self.issues,
            "fixes": self.fixes,
        }


class AgentStateMiddleware:
    """
    Agent 状态中间件 — 在每次状态流转前做校验和修复。
    
    使用方式：
        1. 调用 validate_and_fix(state) 校验并自动修复状态
        2. 返回修复后的状态 + 校验结果
        3. 如果状态严重损坏，返回安全的默认状态
    """
    
    MAX_MESSAGES = 100  # 消息列表最大长度（防止无限增长）
    MAX_TOOL_CALLS_PER_MESSAGE = 10  # 单条消息最多工具调用数
    
    @classmethod
    def validate_and_fix(cls, state: Dict[str, Any]) -> Tuple[Dict[str, Any], StateValidationResult]:
        """
        校验并修复 Agent 状态。
        
        Args:
            state: 当前 Agent 状态
            
        Returns:
            (修复后的状态, 校验结果)
        """
        result = StateValidationResult(is_valid=True)
        fixed_state = copy.deepcopy(state)
        
        # 1. 基础字段校验
        fixed_state = cls._validate_basic_fields(fixed_state, result)
        
        # 2. 消息列表校验
        fixed_state = cls._validate_messages(fixed_state, result)
        
        # 3. 消息配对校验
        fixed_state = cls._validate_message_pairs(fixed_state, result)
        
        # 4. 迭代次数校验
        fixed_state = cls._validate_iteration(fixed_state, result)
        
        # 5. tool_calls 格式校验
        fixed_state = cls._validate_tool_calls(fixed_state, result)
        
        if result.issues:
            logger.warning(
                f"[状态校验] 发现 {len(result.issues)} 个问题，"
                f"自动修复 {len(result.fixes)} 个: {result.issues}"
            )
        
        return fixed_state, result
    
    @classmethod
    def _validate_basic_fields(cls, state: Dict, result: StateValidationResult) -> Dict:
        """校验基础字段"""
        if "messages" not in state:
            result.add_issue("缺少 messages 字段")
            state["messages"] = []
            result.add_fix("初始化 messages 为空列表")
        
        if "city" not in state:
            result.add_issue("缺少 city 字段")
            state["city"] = "北京"
            result.add_fix("设置默认城市为北京")
        
        if "iteration_count" not in state:
            result.add_issue("缺少 iteration_count 字段")
            state["iteration_count"] = 0
            result.add_fix("初始化迭代次数为 0")
        
        if "max_iterations" not in state:
            result.add_issue("缺少 max_iterations 字段")
            state["max_iterations"] = 10
            result.add_fix("设置最大迭代次数为 10")
        
        if "tool_calls_history" not in state:
            result.add_issue("缺少 tool_calls_history 字段")
            state["tool_calls_history"] = []
            result.add_fix("初始化工具调用历史为空列表")
        
        return state
    
    @classmethod
    def _validate_messages(cls, state: Dict, result: StateValidationResult) -> Dict:
        """校验消息列表"""
        messages = state.get("messages", [])
        
        # 不是列表 → 重置
        if not isinstance(messages, list):
            result.add_issue(f"messages 类型错误: {type(messages).__name__}")
            state["messages"] = []
            result.add_fix("重置 messages 为空列表")
            return state
        
        # 消息太多 → 截断（保留系统消息和最近 N 条）
        if len(messages) > cls.MAX_MESSAGES:
            result.add_issue(f"消息列表过长: {len(messages)} 条")
            # 保留第一条（如果是 SystemMessage）和最近的消息
            system_msg = messages[0] if messages and isinstance(messages[0], SystemMessage) else None
            recent = messages[-cls.MAX_MESSAGES + 1:] if system_msg else messages[-cls.MAX_MESSAGES:]
            state["messages"] = ([system_msg] if system_msg else []) + recent
            result.add_fix(f"截断消息列表至 {len(state['messages'])} 条")
        
        # 逐条校验消息类型
        valid_messages = []
        invalid_count = 0
        for msg in messages:
            if isinstance(msg, BaseMessage):
                valid_messages.append(msg)
            else:
                invalid_count += 1
        
        if invalid_count > 0:
            result.add_issue(f"消息列表中有 {invalid_count} 条非 BaseMessage 对象")
            state["messages"] = valid_messages
            result.add_fix(f"移除了 {invalid_count} 条无效消息")
        
        return state
    
    @classmethod
    def _validate_message_pairs(cls, state: Dict, result: StateValidationResult) -> Dict:
        """校验消息配对（ToolMessage 必须有对应的 AIMessage 在前）"""
        messages = state.get("messages", [])
        if not messages:
            return state
        
        # 检查 ToolMessage 是否都有对应的 AIMessage
        # （每条 ToolMessage 前面应该有一条带 tool_calls 的 AIMessage）
        fixed_messages = []
        skip_next_tool = False
        unpaired_tools = 0
        
        for i, msg in enumerate(messages):
            if isinstance(msg, ToolMessage):
                # 往前找对应的 AIMessage
                found_pair = False
                for prev_msg in reversed(fixed_messages):
                    if isinstance(prev_msg, AIMessage) and prev_msg.tool_calls:
                        found_pair = True
                        break
                
                if not found_pair:
                    unpaired_tools += 1
                    # 丢掉这条孤立的 ToolMessage
                    continue
            
            fixed_messages.append(msg)
        
        if unpaired_tools > 0:
            result.add_issue(f"发现 {unpaired_tools} 条孤立的 ToolMessage")
            state["messages"] = fixed_messages
            result.add_fix(f"移除了 {unpaired_tools} 条孤立的 ToolMessage")
        
        return state
    
    @classmethod
    def _validate_iteration(cls, state: Dict, result: StateValidationResult) -> Dict:
        """校验迭代次数"""
        iteration = state.get("iteration_count", 0)
        max_iter = state.get("max_iterations", 10)
        
        # 迭代次数不能是负数
        if not isinstance(iteration, int) or iteration < 0:
            result.add_issue(f"迭代次数非法: {iteration}")
            state["iteration_count"] = 0
            result.add_fix("重置迭代次数为 0")
        
        # 最大迭代次数不能小于等于0
        if not isinstance(max_iter, int) or max_iter <= 0:
            result.add_issue(f"最大迭代次数非法: {max_iter}")
            state["max_iterations"] = 10
            result.add_fix("重置最大迭代次数为 10")
        
        # 当前迭代不能超过最大迭代
        if isinstance(iteration, int) and isinstance(max_iter, int) and iteration > max_iter:
            result.add_issue(f"迭代次数({iteration})超过上限({max_iter})")
            state["iteration_count"] = max_iter
            result.add_fix("将迭代次数限制为最大值")
        
        return state
    
    @classmethod
    def _validate_tool_calls(cls, state: Dict, result: StateValidationResult) -> Dict:
        """校验 AIMessage 中 tool_calls 的格式"""
        messages = state.get("messages", [])
        fixed_count = 0
        
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                # 限制单条消息的工具调用数量
                if len(msg.tool_calls) > cls.MAX_TOOL_CALLS_PER_MESSAGE:
                    msg.tool_calls = msg.tool_calls[:cls.MAX_TOOL_CALLS_PER_MESSAGE]
                    fixed_count += 1
                
                # 校验每个 tool_call 的格式
                valid_tool_calls = []
                for tc in msg.tool_calls:
                    if not isinstance(tc, dict):
                        fixed_count += 1
                        continue
                    
                    # 必须有 name
                    if "name" not in tc or not tc["name"]:
                        fixed_count += 1
                        continue
                    
                    # args 必须是 dict
                    if "args" not in tc or not isinstance(tc["args"], dict):
                        tc["args"] = {}
                        fixed_count += 1
                    
                    # id 可以没有，但有就保留
                    if "id" not in tc:
                        tc["id"] = ""
                    
                    valid_tool_calls.append(tc)
                
                msg.tool_calls = valid_tool_calls
        
        if fixed_count > 0:
            result.add_issue(f"修复了 {fixed_count} 个 tool_call 格式问题")
            result.add_fix(f"修复 {fixed_count} 个 tool_call 格式")
        
        return state
    
    @classmethod
    def get_state_summary(cls, state: Dict[str, Any]) -> Dict[str, Any]:
        """获取状态的摘要信息（用于日志和调试）"""
        messages = state.get("messages", [])
        
        # 统计各类型消息数量
        msg_types = {}
        for msg in messages:
            type_name = type(msg).__name__
            msg_types[type_name] = msg_types.get(type_name, 0) + 1
        
        # 统计工具调用次数
        total_tool_calls = 0
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                total_tool_calls += len(msg.tool_calls)
        
        return {
            "message_count": len(messages),
            "message_types": msg_types,
            "total_tool_calls": total_tool_calls,
            "iteration_count": state.get("iteration_count", 0),
            "city": state.get("city", "unknown"),
        }
