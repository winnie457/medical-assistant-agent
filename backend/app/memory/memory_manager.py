"""
记忆管理器 — 长期记忆的增删改查 + 冲突消解

核心难题：冲突消解
    用户的健康信息会变化，新记忆和旧记忆可能冲突。
    比如：
    - 旧记忆："对青霉素过敏"
    - 新记忆："我青霉素不过敏了"（去医院做了皮试）
    
    怎么处理？不能简单地"新覆盖旧"，因为新记忆可能是错的（LLM 提取错了）。

冲突消解策略：
    1. 同类同 key 才可能冲突（比如都是 drug_allergy）
    2. 明确否定 > 肯定（用户说"我不过敏了"比"我过敏"优先级高）
    3. 高置信度 > 低置信度
    4. 时间新 > 时间旧（同置信度下）
    5. 拿不准的设为 PENDING，不影响当前对话
"""

import os
import json
import logging
from typing import Dict, Optional, List, Any
from datetime import datetime

from .health_profile import (
    UserHealthProfile,
    MemoryItem,
    MemoryCategory,
    MemoryConfidence,
    MemoryStatus,
    create_memory_item,
)

logger = logging.getLogger(__name__)

# 持久化存储目录
MEMORY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "user_profiles")

# 置信度数值化，用于比较
CONFIDENCE_SCORE = {
    MemoryConfidence.LOW: 1,
    MemoryConfidence.MEDIUM: 2,
    MemoryConfidence.HIGH: 3,
    MemoryConfidence.CONFIRMED: 4,
}


class MemoryManager:
    """
    用户健康画像管理器。
    
    负责：
    - 画像的加载和持久化
    - 记忆的增删改查
    - 冲突消解
    """
    
    def __init__(self, storage_dir: str = MEMORY_DIR):
        self.storage_dir = storage_dir
        self._cache: Dict[str, UserHealthProfile] = {}
        self._ensure_storage_dir()
    
    def _ensure_storage_dir(self):
        """确保存储目录存在"""
        os.makedirs(self.storage_dir, exist_ok=True)
    
    def _get_profile_path(self, user_id: str) -> str:
        """获取用户画像文件路径"""
        return os.path.join(self.storage_dir, f"{user_id}.json")
    
    def get_profile(self, user_id: str) -> UserHealthProfile:
        """
        获取用户画像。
        先从缓存取，没有则从文件加载，再没有则创建新的。
        """
        if user_id in self._cache:
            return self._cache[user_id]
        
        profile_path = self._get_profile_path(user_id)
        if os.path.exists(profile_path):
            try:
                with open(profile_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                profile = UserHealthProfile.from_dict(data)
                self._cache[user_id] = profile
                logger.info(f"[记忆] 加载用户画像: {user_id}, 共 {len(profile.get_active_memories())} 条有效记忆")
                return profile
            except Exception as e:
                logger.error(f"[记忆] 加载用户画像失败 {user_id}: {e}")
        
        # 创建新画像
        profile = UserHealthProfile(user_id=user_id)
        self._cache[user_id] = profile
        self._save_profile(user_id)
        logger.info(f"[记忆] 创建新用户画像: {user_id}")
        return profile
    
    def _save_profile(self, user_id: str):
        """保存用户画像到文件"""
        if user_id not in self._cache:
            return
        
        profile = self._cache[user_id]
        profile.updated_at = datetime.now().isoformat()
        
        profile_path = self._get_profile_path(user_id)
        try:
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[记忆] 保存用户画像失败 {user_id}: {e}")
    
    def add_memory(
        self,
        user_id: str,
        category: MemoryCategory,
        key: str,
        value: str,
        confidence: MemoryConfidence = MemoryConfidence.MEDIUM,
        source_summary: str = "",
        source_conversation_id: str = "",
        is_negation: bool = False,
    ) -> Dict[str, any]:
        """
        添加一条记忆（带冲突消解）。
        
        Args:
            user_id: 用户ID
            category: 记忆类别
            key: 记忆键（同类同 key 视为同一记忆）
            value: 记忆内容
            confidence: 置信度
            source_summary: 来源摘要
            source_conversation_id: 来源会话ID
            is_negation: 是否为否定表述（如"我不过敏了"）
            
        Returns:
            操作结果字典，包含 action（新增/更新/消解/忽略）和记忆详情
        """
        profile = self.get_profile(user_id)
        
        # 查找是否已有同类同 key 的有效记忆
        existing = profile.find_memory(key)
        
        # 没有旧记忆 → 直接新增
        if existing is None:
            if is_negation:
                # 否定一个不存在的东西，忽略
                return {"action": "ignored", "reason": "negation_without_existing", "memory": None}
            
            new_memory = create_memory_item(
                category=category,
                key=key,
                value=value,
                confidence=confidence,
                source_summary=source_summary,
                source_conversation_id=source_conversation_id,
            )
            profile.add_memory(new_memory)
            self._save_profile(user_id)
            logger.info(f"[记忆] 新增记忆: {key} = {value}")
            return {"action": "added", "memory": new_memory.to_dict()}
        
        # 有旧记忆 → 冲突消解
        return self._resolve_conflict(
            user_id=user_id,
            profile=profile,
            existing=existing,
            new_category=category,
            new_key=key,
            new_value=value,
            new_confidence=confidence,
            is_negation=is_negation,
            source_summary=source_summary,
            source_conversation_id=source_conversation_id,
        )
    
    def _resolve_conflict(
        self,
        user_id: str,
        profile: UserHealthProfile,
        existing: MemoryItem,
        new_category: MemoryCategory,
        new_key: str,
        new_value: str,
        new_confidence: MemoryConfidence,
        is_negation: bool,
        source_summary: str,
        source_conversation_id: str,
    ) -> Dict[str, Any]:
        """
        冲突消解核心逻辑。
        
        策略优先级：
        1. 明确否定 → 废弃旧记忆
        2. 新置信度 >> 旧置信度 → 覆盖
        3. 新置信度 == 旧置信度 + 内容不同 → 新的替代旧的（时间新）
        4. 新置信度 < 旧置信度 → 保留旧的，新的做备注
        5. 内容相同 → 升级置信度（多次提到更可信）
        """
        old_conf_score = CONFIDENCE_SCORE[existing.confidence]
        new_conf_score = CONFIDENCE_SCORE[new_confidence]
        
        # 情况1：明确否定旧记忆
        if is_negation:
            if new_conf_score >= old_conf_score - 1:  # 新的置信度差不多就能否定
                existing.status = MemoryStatus.REFUTED
                existing.notes += f"\n被新记忆否定: {new_value} (来源: {source_summary})"
                existing.updated_at = datetime.now().isoformat()
                self._save_profile(user_id)
                logger.info(f"[记忆] 旧记忆被否定: {new_key}")
                return {"action": "refuted", "memory": existing.to_dict()}
            else:
                # 新记忆置信度太低，不足以否定旧记忆
                return {"action": "ignored", "reason": "negation_low_confidence", "memory": existing.to_dict()}
        
        # 情况2：内容完全相同 → 提升置信度
        if existing.value == new_value:
            if new_conf_score > old_conf_score:
                old_conf = existing.confidence
                existing.confidence = new_confidence
                existing.updated_at = datetime.now().isoformat()
                existing.notes += f"\n再次确认 (来源: {source_summary})"
                self._save_profile(user_id)
                logger.info(f"[记忆] 置信度提升: {new_key} ({old_conf.value} → {new_confidence.value})")
                return {"action": "confidence_boosted", "memory": existing.to_dict()}
            else:
                return {"action": "ignored", "reason": "duplicate", "memory": existing.to_dict()}
        
        # 情况3：内容不同，比较置信度
        if new_conf_score > old_conf_score:
            # 新记忆置信度更高 → 替代旧记忆
            existing.status = MemoryStatus.SUPERSEDED
            existing.notes += f"\n被新记忆替代: {new_value}"
            existing.updated_at = datetime.now().isoformat()
            
            new_memory = create_memory_item(
                category=new_category,
                key=new_key,
                value=new_value,
                confidence=new_confidence,
                source_summary=source_summary,
                source_conversation_id=source_conversation_id,
            )
            profile.add_memory(new_memory)
            self._save_profile(user_id)
            logger.info(f"[记忆] 记忆更新: {new_key} ({existing.value} → {new_value})")
            return {"action": "updated", "old_memory": existing.to_dict(), "new_memory": new_memory.to_dict()}
        
        elif new_conf_score == old_conf_score:
            # 置信度相同 → 新的替代旧的（时间更近）
            existing.status = MemoryStatus.SUPERSEDED
            existing.notes += f"\n被新记忆替代（同置信度，时间更新）: {new_value}"
            existing.updated_at = datetime.now().isoformat()
            
            new_memory = create_memory_item(
                category=new_category,
                key=new_key,
                value=new_value,
                confidence=new_confidence,
                source_summary=source_summary,
                source_conversation_id=source_conversation_id,
            )
            profile.add_memory(new_memory)
            self._save_profile(user_id)
            logger.info(f"[记忆] 记忆更新（同置信度）: {new_key}")
            return {"action": "updated", "old_memory": existing.to_dict(), "new_memory": new_memory.to_dict()}
        
        else:
            # 新记忆置信度更低 → 不替代，只做备注（供参考）
            existing.notes += f"\n低置信度新信息（未采纳）: {new_value} (来源: {source_summary})"
            existing.updated_at = datetime.now().isoformat()
            self._save_profile(user_id)
            logger.info(f"[记忆] 低置信度新信息，保留旧记忆: {new_key}")
            return {"action": "ignored", "reason": "lower_confidence", "memory": existing.to_dict()}
    
    def batch_add_memories(self, user_id: str, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量添加记忆（从 LLM 提取结果批量导入）。
        
        每条记忆格式:
        {
            "category": "allergy",
            "key": "penicillin_allergy",
            "value": "对青霉素过敏",
            "confidence": "high",
            "is_negation": false,
            "source_summary": ""
        }
        """
        results = []
        for mem_data in memories:
            try:
                category = MemoryCategory(mem_data.get("category", "other"))
                confidence = MemoryConfidence(mem_data.get("confidence", "medium"))
                result = self.add_memory(
                    user_id=user_id,
                    category=category,
                    key=mem_data.get("key", ""),
                    value=mem_data.get("value", ""),
                    confidence=confidence,
                    source_summary=mem_data.get("source_summary", ""),
                    source_conversation_id=mem_data.get("source_conversation_id", ""),
                    is_negation=mem_data.get("is_negation", False),
                )
                results.append(result)
            except Exception as e:
                logger.error(f"[记忆] 批量添加失败: {mem_data}, 错误: {e}")
                results.append({"action": "error", "memory": mem_data, "error": str(e)})
        
        return results
    
    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """删除一条记忆（软删除，改为 REFUTED 状态）"""
        profile = self.get_profile(user_id)
        for mem in profile.memories:
            if mem.id == memory_id:
                mem.status = MemoryStatus.REFUTED
                mem.notes += "\n用户手动删除"
                mem.updated_at = datetime.now().isoformat()
                self._save_profile(user_id)
                logger.info(f"[记忆] 用户删除记忆: {memory_id}")
                return True
        return False
    
    def clear_all_memories(self, user_id: str) -> int:
        """清空所有记忆（软删除）"""
        profile = self.get_profile(user_id)
        count = 0
        for mem in profile.memories:
            if mem.status == MemoryStatus.ACTIVE:
                mem.status = MemoryStatus.REFUTED
                mem.notes += "\n用户清空所有记忆"
                mem.updated_at = datetime.now().isoformat()
                count += 1
        self._save_profile(user_id)
        logger.info(f"[记忆] 用户清空所有记忆: {count} 条")
        return count
    
    def get_profile_summary(self, user_id: str) -> Dict[str, Any]:
        """获取画像摘要（给前端展示用）"""
        profile = self.get_profile(user_id)
        active = profile.get_active_memories()
        
        # 按类别统计
        category_counts = {}
        for mem in active:
            cat = mem.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        return {
            "user_id": profile.user_id,
            "memory_count": len(active),
            "category_counts": category_counts,
            "memories": [m.to_dict() for m in active],
            "created_at": profile.created_at,
            "updated_at": profile.updated_at,
        }


# ============ 单例模式 ============

_memory_manager_instance = None


def get_memory_manager() -> MemoryManager:
    """获取记忆管理器实例（单例）"""
    global _memory_manager_instance
    if _memory_manager_instance is None:
        _memory_manager_instance = MemoryManager()
        print("🧠 记忆管理器初始化成功")
    return _memory_manager_instance
