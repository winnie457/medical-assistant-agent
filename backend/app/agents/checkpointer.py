"""Checkpointer 提供者：SQLite 持久化，缺依赖时降级为进程内。

背景：
    LangGraph 1.x 的 checkpoint 基类里，同步方法与异步方法是两套接口
    （`get_tuple` / `aget_tuple` 各自 NotImplemented），SqliteSaver 只实现同步侧、
    AsyncSqliteSaver 只实现异步侧。既然流式链路本来就是 async，
    统一用 AsyncSqliteSaver，并把 `chat()` 也改成 async —— 这样两类调用共享同一个
    连接，不会出现"同步写、异步读"两套 saver 抢同一个库的情况。

连接生命周期：
    AsyncSqliteSaver 需要一个 aiosqlite 连接，且必须先 `await setup()` 建表。
    这里做成懒加载：第一次真正跑图时创建连接并建表，之后复用。
    因此 MedicalAgent 的图也改成懒构建（见 `ensure_graph`）。
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

from ..storage.db import BASE_DIR

logger = logging.getLogger(__name__)

CHECKPOINT_PATH = BASE_DIR / "data" / "checkpoints.sqlite"


class CheckpointerProvider:
    """按需创建 checkpointer，并把"是否持久化"暴露给上层（指标/日志）。"""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else CHECKPOINT_PATH
        self._saver: Any = None
        self._kind = "uninitialized"
        self._lock = asyncio.Lock()

    @property
    def kind(self) -> str:
        """sqlite / memory / uninitialized"""
        return self._kind

    @property
    def persistent(self) -> bool:
        return self._kind == "sqlite"

    async def get(self) -> Any:
        if self._saver is not None:
            return self._saver
        async with self._lock:
            if self._saver is None:
                self._saver = await self._build()
        return self._saver

    async def aclose(self) -> None:
        """关闭底层连接。

        aiosqlite 在后台线程里跑事件循环，连接不关，进程退出时会挂住；
        所以测试收尾和 FastAPI shutdown 都要显式调用一次。
        """
        saver = self._saver
        if saver is None:
            return
        conn = getattr(saver, "conn", None)
        if conn is not None:
            try:
                await conn.close()
            except Exception as exc:  # 关闭失败不应阻断关停流程
                logger.debug("[checkpointer] 关闭连接失败(可忽略): %s", exc)
        self._saver = None
        self._kind = "uninitialized"

    async def _build(self) -> Any:
        try:
            import aiosqlite
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = await aiosqlite.connect(str(self.path))
            saver = AsyncSqliteSaver(conn)
            await saver.setup()
            self._kind = "sqlite"
            logger.info("[checkpointer] SQLite 持久化已启用: %s", self.path)
            return saver
        except Exception as exc:  # 缺包 / 版本不匹配 / 磁盘不可写
            from langgraph.checkpoint.memory import InMemorySaver

            self._kind = "memory"
            logger.warning(
                "[checkpointer] SQLite 不可用（%s），降级为进程内存储："
                "重启后中断态会丢失。安装方式: pip install langgraph-checkpoint-sqlite",
                exc,
            )
            return InMemorySaver()


_provider: Optional[CheckpointerProvider] = None


def get_checkpointer_provider() -> CheckpointerProvider:
    global _provider
    if _provider is None:
        _provider = CheckpointerProvider()
    return _provider
