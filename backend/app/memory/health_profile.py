"""
用户健康画像 — 长期记忆系统

设计理念：
    普通的 Agent 只有短期记忆（当前对话历史），会话结束就忘了。
    长期记忆系统让 Agent 能记住用户的健康信息，跨会话提供更个性化的建议。

核心挑战：
    1. 怎么从对话里准确提取健康信息？
    2. 新旧信息冲突了怎么办？（用户说"我过敏"→后来又说"我不过敏了"）
    3. 怎么避免错误记忆污染对话（幻觉记忆）？

解决方案：
    - 每条记忆有置信度和时间戳
    - 冲突消解策略：新 > 旧，高置信度 > 低置信度，明确否定 > 肯定
    - 记忆来源可追溯（知道是从哪条对话提取的）
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import json
import uuid


class MemoryCategory(str, Enum):
    """记忆类别"""
    BASIC_INFO = "basic_info"           # 基本信息（年龄、性别）
    ALLERGY = "allergy"                 # 过敏史
    CHRONIC_DISEASE = "chronic_disease" # 慢性病
    SURGERY_HISTORY = "surgery_history" # 手术史
    MEDICATION = "medication"           # 用药史
    FAMILY_HISTORY = "family_history"   # 家族病史
    LIFESTYLE = "lifestyle"             # 生活习惯
    SYMPTOM_HISTORY = "symptom_history" # 症状历史
    OTHER = "other"                     # 其他健康信息


class MemoryConfidence(str, Enum):
    """记忆置信度"""
    LOW = "low"         # 低（推测的，比如从症状反推）
    MEDIUM = "medium"   # 中（用户间接提到）
    HIGH = "high"       # 高（用户明确陈述）
    CONFIRMED = "confirmed"  # 已确认（用户多次提到或明确确认）


class MemoryStatus(str, Enum):
    """记忆状态"""
    ACTIVE = "active"           # 有效
    SUPERSEDED = "superseded"   # 被新记忆替代
    REFUTED = "refuted"         # 被用户否定
    PENDING = "pending"         # 待确认


@dataclass
class MemoryItem:
    """单条记忆条目"""
    id: str
    category: MemoryCategory
    key: str                           # 记忆键名（如 "drug_allergy"、"hypertension"）
    value: str                         # 记忆内容
    confidence: MemoryConfidence       # 置信度
    status: MemoryStatus               # 状态
    created_at: str                    # 创建时间
    updated_at: str                    # 最后更新时间
    source_summary: str = ""           # 来源摘要（哪句话提取的）
    source_conversation_id: str = ""   # 来源会话ID
    notes: str = ""                    # 备注
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category.value,
            "key": self.key,
            "value": self.value,
            "confidence": self.confidence.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source_summary": self.source_summary,
            "source_conversation_id": self.source_conversation_id,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        return cls(
            id=data["id"],
            category=MemoryCategory(data["category"]),
            key=data["key"],
            value=data["value"],
            confidence=MemoryConfidence(data["confidence"]),
            status=MemoryStatus(data["status"]),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            source_summary=data.get("source_summary", ""),
            source_conversation_id=data.get("source_conversation_id", ""),
            notes=data.get("notes", ""),
        )


@dataclass
class UserHealthProfile:
    """用户健康画像（长期记忆集合）"""
    user_id: str
    memories: List[MemoryItem] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def get_active_memories(self) -> List[MemoryItem]:
        """获取所有有效记忆"""
        return [m for m in self.memories if m.status == MemoryStatus.ACTIVE]
    
    def get_memories_by_category(self, category: MemoryCategory) -> List[MemoryItem]:
        """按类别获取有效记忆"""
        return [m for m in self.get_active_memories() if m.category == category]
    
    def find_memory(self, key: str) -> Optional[MemoryItem]:
        """按 key 查找有效记忆"""
        for m in self.get_active_memories():
            if m.key == key:
                return m
        return None
    
    def add_memory(self, memory: MemoryItem):
        """添加一条记忆"""
        self.memories.append(memory)
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "memories": [m.to_dict() for m in self.memories],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "memory_count": len(self.get_active_memories()),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserHealthProfile':
        return cls(
            user_id=data["user_id"],
            memories=[MemoryItem.from_dict(m) for m in data.get("memories", [])],
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
        )
    
    def format_for_prompt(self) -> str:
        """
        格式化为系统提示词里的用户画像文本。
        只包含有效记忆，按类别分组。
        """
        active = self.get_active_memories()
        if not active:
            return "暂无已知健康信息"
        
        category_labels = {
            MemoryCategory.BASIC_INFO: "基本信息",
            MemoryCategory.ALLERGY: "过敏史",
            MemoryCategory.CHRONIC_DISEASE: "慢性病",
            MemoryCategory.SURGERY_HISTORY: "手术史",
            MemoryCategory.MEDICATION: "用药史",
            MemoryCategory.FAMILY_HISTORY: "家族病史",
            MemoryCategory.LIFESTYLE: "生活习惯",
            MemoryCategory.SYMPTOM_HISTORY: "症状历史",
            MemoryCategory.OTHER: "其他健康信息",
        }
        
        lines = []
        for category, label in category_labels.items():
            items = [m for m in active if m.category == category]
            if items:
                lines.append(f"【{label}】")
                for item in items:
                    lines.append(f"- {item.value}")
                lines.append("")
        
        return "\n".join(lines).strip()


def create_memory_item(
    category: MemoryCategory,
    key: str,
    value: str,
    confidence: MemoryConfidence = MemoryConfidence.MEDIUM,
    source_summary: str = "",
    source_conversation_id: str = "",
) -> MemoryItem:
    """创建一条新记忆"""
    now = datetime.now().isoformat()
    return MemoryItem(
        id=f"mem_{uuid.uuid4().hex[:12]}",
        category=category,
        key=key,
        value=value,
        confidence=confidence,
        status=MemoryStatus.ACTIVE,
        created_at=now,
        updated_at=now,
        source_summary=source_summary,
        source_conversation_id=source_conversation_id,
    )
