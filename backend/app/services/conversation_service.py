"""对话管理模块 — SQLite 持久化版。

原实现是进程内 dict：服务一重启，会话列表、历史消息、上下文全部消失，
"重启恢复"根本无从谈起。这里换成 SQLite，对外接口完全不变，
路由层与前端无需改动。

表结构见 app/storage/db.py；消息只追加不修改，删除会话依赖外键级联。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from ..storage.db import get_connection


class ConversationManager:
    """会话管理器（SQLite 持久化）"""

    # ---------- 内部工具 ----------

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat()

    @staticmethod
    def _row_to_message(row: Any) -> BaseMessage:
        role = row["role"]
        content = row["content"] or ""
        if role == "assistant":
            tool_calls = json.loads(row["tool_calls"]) if row["tool_calls"] else None
            return AIMessage(content=content, **({"tool_calls": tool_calls} if tool_calls else {}))
        return HumanMessage(content=content)

    # ---------- 会话 ----------

    def create_conversation(self, title: str = "新的咨询") -> str:
        conversation_id = str(uuid.uuid4())
        now = self._now()
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO conversations"
                " (conversation_id, title, created_at, updated_at, last_message, message_count)"
                " VALUES (?, ?, ?, ?, '', 0)",
                (conversation_id, title, now, now),
            )
        return conversation_id

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        tool_calls: Optional[List] = None,
    ) -> bool:
        conn = get_connection()
        row = conn.execute(
            "SELECT title, message_count FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            return False

        now = self._now()
        count = int(row["message_count"]) + 1
        title = row["title"]
        if count == 1 and role == "user":
            title = content[:20] + "..." if len(content) > 20 else content

        with conn:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, tool_calls, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    conversation_id,
                    role,
                    content,
                    json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None,
                    now,
                ),
            )
            conn.execute(
                "UPDATE conversations"
                " SET title = ?, message_count = ?, last_message = ?, updated_at = ?"
                " WHERE conversation_id = ?",
                (
                    title,
                    count,
                    content[:50] + "..." if len(content) > 50 else content,
                    now,
                    conversation_id,
                ),
            )
        return True

    def get_langchain_messages(self, conversation_id: str) -> List[BaseMessage]:
        rows = get_connection().execute(
            "SELECT role, content, tool_calls FROM messages"
            " WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
        return [self._row_to_message(row) for row in rows]

    def get_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        rows = get_connection().execute(
            "SELECT role, content, tool_calls, created_at FROM messages"
            " WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
        messages: List[Dict[str, Any]] = []
        for row in rows:
            item: Dict[str, Any] = {
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["created_at"],
            }
            if row["tool_calls"]:
                item["tool_calls"] = json.loads(row["tool_calls"])
            messages.append(item)
        return messages

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        row = get_connection().execute(
            "SELECT * FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["messages"] = self.get_messages(conversation_id)
        return data

    def list_conversations(self) -> List[Dict[str, Any]]:
        rows = get_connection().execute(
            "SELECT conversation_id, title, created_at, last_message, message_count"
            " FROM conversations ORDER BY created_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]

    def delete_conversation(self, conversation_id: str) -> bool:
        conn = get_connection()
        with conn:
            cursor = conn.execute(
                "DELETE FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )
        return cursor.rowcount > 0

    def clear(self) -> None:
        conn = get_connection()
        with conn:
            conn.execute("DELETE FROM messages")
            conn.execute("DELETE FROM conversations")


# 全局单例
conversation_manager = ConversationManager()


def get_conversation_manager() -> ConversationManager:
    """获取会话管理器实例"""
    return conversation_manager
