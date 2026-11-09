"""对话管理模块 - 会话存储和历史管理"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


class ConversationManager:
    """会话管理器（内存存储，生产环境建议使用数据库）"""

    def __init__(self):
        self.conversations: Dict[str, Dict] = {}

    def create_conversation(self, title: str = "新的咨询") -> str:
        """创建新会话"""
        conversation_id = str(uuid.uuid4())
        self.conversations[conversation_id] = {
            "conversation_id": conversation_id,
            "title": title,
            "created_at": datetime.now().isoformat(),
            "last_message": "",
            "message_count": 0,
            "messages": [],
            "langchain_messages": []
        }
        return conversation_id

    def add_message(self, conversation_id: str, role: str, content: str,
                    tool_calls: Optional[List] = None) -> bool:
        """添加消息到会话"""
        if conversation_id not in self.conversations:
            return False

        conv = self.conversations[conversation_id]
        timestamp = datetime.now().isoformat()

        # 存储普通消息
        msg = {
            "role": role,
            "content": content,
            "timestamp": timestamp
        }
        if tool_calls:
            msg["tool_calls"] = tool_calls

        conv["messages"].append(msg)
        conv["message_count"] += 1
        conv["last_message"] = content[:50] + "..." if len(content) > 50 else content

        # 构建 LangChain 消息对象
        if role == "user":
            lc_msg = HumanMessage(content=content)
        elif role == "assistant":
            additional = {}
            if tool_calls:
                additional["tool_calls"] = tool_calls
            lc_msg = AIMessage(content=content, **additional)
        else:
            lc_msg = HumanMessage(content=content)

        conv["langchain_messages"].append(lc_msg)

        # 更新标题（第一条用户消息作为标题）
        if conv["message_count"] == 1 and role == "user":
            conv["title"] = content[:20] + "..." if len(content) > 20 else content

        return True

    def get_langchain_messages(self, conversation_id: str) -> List[BaseMessage]:
        """获取 LangChain 格式的消息历史"""
        if conversation_id not in self.conversations:
            return []
        return self.conversations[conversation_id]["langchain_messages"].copy()

    def get_messages(self, conversation_id: str) -> List[Dict]:
        """获取普通格式的消息历史"""
        if conversation_id not in self.conversations:
            return []
        return self.conversations[conversation_id]["messages"].copy()

    def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """获取会话信息"""
        return self.conversations.get(conversation_id)

    def list_conversations(self) -> List[Dict]:
        """列出所有会话"""
        result = []
        for conv_id, conv in self.conversations.items():
            result.append({
                "conversation_id": conv_id,
                "title": conv["title"],
                "created_at": conv["created_at"],
                "last_message": conv["last_message"],
                "message_count": conv["message_count"]
            })
        # 按创建时间倒序
        result.sort(key=lambda x: x["created_at"], reverse=True)
        return result

    def delete_conversation(self, conversation_id: str) -> bool:
        """删除会话"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            return True
        return False

    def clear(self):
        """清空所有会话"""
        self.conversations.clear()


# 全局单例
conversation_manager = ConversationManager()


def get_conversation_manager() -> ConversationManager:
    """获取会话管理器实例"""
    return conversation_manager
