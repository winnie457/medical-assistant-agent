"""对话结束后的记忆更新任务（流式与非流式入口共用）"""

import asyncio
import logging

from ..memory.memory_extractor import get_memory_extractor
from ..memory.memory_manager import get_memory_manager

logger = logging.getLogger(__name__)


async def update_memory_after_chat(user_id: str, conversation_id: str, conversation_text: str) -> None:
    """对话结束后异步提取健康信息并更新用户画像"""
    try:
        extractor = get_memory_extractor()
        memory_manager = get_memory_manager()

        # 抽取内部是同步 LLM 调用，放入线程池避免阻塞事件循环
        memories = await asyncio.to_thread(
            extractor.extract_from_conversation,
            conversation_text=conversation_text,
            conversation_id=conversation_id,
        )

        if not memories:
            return

        results = memory_manager.batch_add_memories(user_id, memories)
        effective = sum(
            1 for r in results
            if r.get("action") in ("added", "updated", "confidence_boosted")
        )
        logger.info(f"[记忆] 对话后更新: {len(memories)} 条提取，{effective} 条生效")
    except Exception as e:
        logger.error(f"[记忆] 后台更新失败: {e}")
