"""
挂起会话管理器 — Human-in-the-loop 等待确认机制

设计理念：
    当 Agent 要调用高风险工具时，不能直接执行，必须挂起等待用户确认。
    这个管理器负责管理所有处于"挂起等待"状态的会话。

工作流程：
    1. Agent 遇到高风险工具 → 调用 suspend() 挂起会话
    2. 管理器创建一个 asyncio.Event，进入等待状态
    3. 用户在前端点"确认" → 调用 approve() → 设置 Event
    4. 用户在前端点"拒绝" → 调用 reject() → 设置 Event + 标记拒绝
    5. Agent 被唤醒，根据结果决定执行工具还是返回用户拒绝的错误

关键设计：
    - 用 asyncio.Event 实现异步等待，不阻塞整个服务
    - 挂起状态有超时（默认 5 分钟），防止用户不关页面导致内存泄漏
    - 每个会话同时只能有一个挂起状态
"""

import asyncio
import time
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 挂起超时时间（秒）
DEFAULT_SUSPEND_TIMEOUT = 300  # 5分钟


@dataclass
class SuspendedState:
    """挂起状态"""
    conversation_id: str                    # 会话ID
    tool_name: str                          # 等待确认的工具名
    tool_args: Dict[str, Any]               # 工具参数
    event: asyncio.Event                    # 等待事件
    approved: bool = False                  # 用户是否确认
    reject_reason: str = ""                 # 拒绝原因
    created_at: float = field(default_factory=time.time)  # 创建时间
    
    def is_expired(self, timeout: int = DEFAULT_SUSPEND_TIMEOUT) -> bool:
        """是否超时"""
        return time.time() - self.created_at > timeout


class SuspensionManager:
    """
    挂起会话管理器。
    
    管理所有等待用户确认的会话。
    单例模式，全局一个实例。
    """
    
    def __init__(self):
        self._suspended: Dict[str, SuspendedState] = {}
        self._lock = asyncio.Lock()  # 保护字典的并发访问
    
    async def suspend(self, conversation_id: str, tool_name: str, 
                      tool_args: Dict[str, Any]) -> bool:
        """
        挂起会话，等待用户确认。
        
        Args:
            conversation_id: 会话ID
            tool_name: 等待确认的工具名
            tool_args: 工具参数
            
        Returns:
            True = 用户确认了，False = 用户拒绝了或超时了
        """
        # 1. 创建挂起状态
        event = asyncio.Event()
        state = SuspendedState(
            conversation_id=conversation_id,
            tool_name=tool_name,
            tool_args=tool_args,
            event=event,
        )
        
        # 2. 存入字典（加锁）
        async with self._lock:
            # 如果这个会话已经有挂起状态了，先取消旧的
            if conversation_id in self._suspended:
                old_state = self._suspended[conversation_id]
                old_state.event.set()  # 唤醒旧的，让它超时退出
                logger.warning(f"[HITL] 会话 {conversation_id} 已有挂起状态，替换为新的")
            
            self._suspended[conversation_id] = state
        
        logger.info(f"[HITL] 会话 {conversation_id} 挂起，等待确认工具: {tool_name}")
        
        try:
            # 3. 等待用户确认（带超时）
            await asyncio.wait_for(event.wait(), timeout=DEFAULT_SUSPEND_TIMEOUT)
            
            # 4. 被唤醒了，返回结果
            return state.approved
            
        except asyncio.TimeoutError:
            # 超时了
            logger.warning(f"[HITL] 会话 {conversation_id} 等待确认超时")
            state.reject_reason = "用户长时间未确认，操作已超时"
            state.approved = False
            return False
        finally:
            # 5. 清理挂起状态
            async with self._lock:
                if conversation_id in self._suspended:
                    del self._suspended[conversation_id]
    
    async def approve(self, conversation_id: str) -> bool:
        """
        用户确认执行高风险工具。
        
        Args:
            conversation_id: 会话ID
            
        Returns:
            True = 成功唤醒，False = 会话不在挂起状态
        """
        async with self._lock:
            state = self._suspended.get(conversation_id)
        
        if state is None:
            logger.warning(f"[HITL] 确认失败：会话 {conversation_id} 不在挂起状态")
            return False
        
        state.approved = True
        state.event.set()  # 唤醒等待的协程
        
        logger.info(f"[HITL] 会话 {conversation_id} 用户已确认执行工具: {state.tool_name}")
        return True
    
    async def reject(self, conversation_id: str, reason: str = "用户取消") -> bool:
        """
        用户拒绝执行高风险工具。
        
        Args:
            conversation_id: 会话ID
            reason: 拒绝原因
            
        Returns:
            True = 成功唤醒，False = 会话不在挂起状态
        """
        async with self._lock:
            state = self._suspended.get(conversation_id)
        
        if state is None:
            logger.warning(f"[HITL] 拒绝失败：会话 {conversation_id} 不在挂起状态")
            return False
        
        state.approved = False
        state.reject_reason = reason
        state.event.set()  # 唤醒等待的协程
        
        logger.info(f"[HITL] 会话 {conversation_id} 用户拒绝执行工具: {state.tool_name}, 原因: {reason}")
        return True
    
    def get_suspended_state(self, conversation_id: str) -> Optional[SuspendedState]:
        """获取会话的挂起状态（如果有的话）"""
        return self._suspended.get(conversation_id)
    
    def is_suspended(self, conversation_id: str) -> bool:
        """会话是否处于挂起状态"""
        state = self._suspended.get(conversation_id)
        if state is None:
            return False
        if state.is_expired():
            return False
        return True
    
    def get_suspended_count(self) -> int:
        """获取当前挂起的会话数量"""
        # 清理过期的
        expired = [cid for cid, s in self._suspended.items() if s.is_expired()]
        for cid in expired:
            del self._suspended[cid]
        return len(self._suspended)


# ============ 单例模式 ============

_suspension_manager_instance = None


def get_suspension_manager() -> SuspensionManager:
    """获取挂起管理器实例（单例）"""
    global _suspension_manager_instance
    if _suspension_manager_instance is None:
        _suspension_manager_instance = SuspensionManager()
    return _suspension_manager_instance
