"""存储层与运行记录的回归测试（不依赖模型与网络）。

运行方式（在 backend/ 下）：
    python -m unittest tests.test_storage -v

覆盖：
1. 会话与消息落 SQLite 后，"新连接"仍能读到 —— 即重启后可恢复；
2. 运行记录的状态流转（interrupted → completed）；
3. checkpointer 选型（有 sqlite 依赖则持久化，没有则降级为内存但明确报告）。
"""

from __future__ import annotations

import os
import tempfile
import unittest
import uuid
from pathlib import Path


class StorageBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # 直接用系统临时目录下的固定文件，避免 mkdtemp 新建子目录后写权限问题
        cls._db_path = Path(tempfile.gettempdir()) / f"med-store-{uuid.uuid4().hex}.db"
        os.environ["APP_DB_PATH"] = str(cls._db_path)
        from app.storage.db import reset_connection

        reset_connection()

    @classmethod
    def tearDownClass(cls) -> None:
        from app.storage.db import reset_connection

        reset_connection()
        os.environ.pop("APP_DB_PATH", None)
        for suffix in ("", "-wal", "-shm"):
            Path(str(cls._db_path) + suffix).unlink(missing_ok=True)


class TestConversationStore(StorageBase):
    def setUp(self) -> None:
        from app.services.conversation_service import ConversationManager

        self.manager = ConversationManager()
        self.conversation_id = self.manager.create_conversation()

    def test_create_and_add_message(self) -> None:
        self.assertTrue(self.manager.add_message(self.conversation_id, "user", "我头疼"))
        conv = self.manager.get_conversation(self.conversation_id)
        self.assertIsNotNone(conv)
        self.assertEqual(conv["message_count"], 1)
        self.assertEqual(conv["title"], "我头疼")

    def test_unknown_conversation(self) -> None:
        self.assertFalse(self.manager.add_message("not-exist", "user", "hi"))
        self.assertIsNone(self.manager.get_conversation("not-exist"))
        self.assertEqual(self.manager.get_langchain_messages("not-exist"), [])

    def test_langchain_history_types(self) -> None:
        from langchain_core.messages import AIMessage, HumanMessage

        self.manager.add_message(self.conversation_id, "user", "我头疼")
        self.manager.add_message(
            self.conversation_id, "assistant", "建议就诊", tool_calls=[]
        )
        messages = self.manager.get_langchain_messages(self.conversation_id)
        self.assertEqual(len(messages), 2)
        self.assertIsInstance(messages[0], HumanMessage)
        self.assertIsInstance(messages[1], AIMessage)

    def test_persist_across_new_connection(self) -> None:
        """模拟进程重启：换一个连接仍能读到会话与历史。"""
        self.manager.add_message(self.conversation_id, "user", "我对青霉素过敏")
        from app.storage.db import reset_connection
        from app.services.conversation_service import ConversationManager

        reset_connection()
        fresh = ConversationManager()
        conv = fresh.get_conversation(self.conversation_id)
        self.assertIsNotNone(conv)
        self.assertEqual(conv["messages"][0]["content"], "我对青霉素过敏")

    def test_list_and_delete(self) -> None:
        conversations = self.manager.list_conversations()
        self.assertTrue(any(c["conversation_id"] == self.conversation_id for c in conversations))
        self.assertTrue(self.manager.delete_conversation(self.conversation_id))
        self.assertIsNone(self.manager.get_conversation(self.conversation_id))
        self.assertFalse(self.manager.delete_conversation(self.conversation_id))


class TestRunRegistry(StorageBase):
    def test_record_and_update(self) -> None:
        from app.services import run_registry

        run_id = "run-test-1"
        run_registry.record_run(run_id, "conv-1", run_registry.INTERRUPTED, "等待确认")
        record = run_registry.get_run(run_id)
        self.assertEqual(record["status"], run_registry.INTERRUPTED)
        self.assertEqual(record["conversation_id"], "conv-1")

        run_registry.update_run(run_id, run_registry.COMPLETED, "本轮完成")
        record = run_registry.get_run(run_id)
        self.assertEqual(record["status"], run_registry.COMPLETED)
        self.assertEqual(record["conversation_id"], "conv-1")  # 不被覆盖

    def test_list_by_status(self) -> None:
        from app.services import run_registry

        run_registry.record_run("run-a", "c1", run_registry.INTERRUPTED)
        run_registry.record_run("run-b", "c2", run_registry.COMPLETED)
        pending = run_registry.list_runs(run_registry.INTERRUPTED)
        self.assertTrue(any(r["run_id"] == "run-a" for r in pending))
        self.assertFalse(any(r["run_id"] == "run-b" for r in pending))

    def test_unknown_run(self) -> None:
        from app.services import run_registry

        self.assertIsNone(run_registry.get_run("nope"))


class TestCheckpointer(unittest.TestCase):
    def test_provider_reports_implementation(self) -> None:
        import asyncio

        from app.agents.checkpointer import CheckpointerProvider

        provider = CheckpointerProvider(path=Path(tempfile.gettempdir()) / "ckpt-test.sqlite")

        async def build_and_close():
            saver = await provider.get()
            kind = provider.kind
            persistent = provider.persistent
            # aiosqlite 持有后台线程，必须显式关闭，否则进程不退出
            await provider.aclose()
            return saver, kind, persistent

        saver, kind, persistent = asyncio.run(build_and_close())
        self.assertIsNotNone(saver)
        self.assertIn(kind, {"sqlite", "memory"})
        # persistent 必须与 kind 一致，避免"以为存了其实没存"
        self.assertEqual(persistent, kind == "sqlite")


if __name__ == "__main__":
    unittest.main(verbosity=2)
