"""
记忆提取器 — 从对话中自动提取用户健康信息

工作原理：
    每轮对话结束后，把本轮对话丢给 LLM，让它提取其中的健康信息。
    LLM 按指定的 JSON 格式返回，我们再批量存入记忆管理器。

关键设计：
    - 只提取"用户主动透露"的信息，不提取 AI 说的
    - 提取结果有置信度评级（明确说的 high，间接提的 medium，推测的 low）
    - 否定表述也要提取（如"我不过敏了"）
    - 不确定的不提取（宁少勿滥，防止幻觉记忆）
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional

from ..services.llm_service import get_llm
from .health_profile import MemoryCategory

logger = logging.getLogger(__name__)


# 记忆提取提示词
MEMORY_EXTRACTION_PROMPT = """你是一个健康信息提取专家。你的任务是从用户和医生的对话中，提取出用户主动透露的健康相关信息，存入用户的健康档案。

## 提取规则

### 1. 只提取用户说的
- 只提取**用户**主动说的健康信息
- 不要提取 AI/医生说的（那是建议，不是用户事实）
- 不要提取你推测的（宁可不提取，也不要错误记忆）

### 2. 信息类别
只能提取以下类别（category）：
- `basic_info` — 基本信息：年龄、性别、身高、体重、职业等
- `allergy` — 过敏史：药物过敏、食物过敏等
- `chronic_disease` — 慢性病：高血压、糖尿病、胃病等
- `surgery_history` — 手术史：做过什么手术
- `medication` — 用药史：长期吃什么药
- `family_history` — 家族病史：直系亲属有什么病
- `lifestyle` — 生活习惯：抽烟、喝酒、作息、运动等
- `symptom_history` — 症状历史：之前有过什么症状/疾病
- `other` — 其他健康相关信息

### 3. 置信度（confidence）
- `high` — 用户明确说的（"我有高血压"、"我对青霉素过敏"）
- `medium` — 用户间接提到的（"我一直在吃降压药" → 推断有高血压）
- `low` — 推测的（用户说头疼 → 推测可能血压高，这个就不要提取了）

**只提取 high 和 medium 置信度的，low 的忽略。**

### 4. 否定表述也要提取（is_negation = true）
用户说"我现在不过敏了"、"我胃病好了"这种否定/治愈的表述，也要提取。
- is_negation = true
- value 里写清楚否定的内容（如"青霉素过敏已治愈"）
- 这种记忆会用来**废弃**旧的记忆

### 5. 记忆的 key 命名
- 用英文小写+下划线，格式统一
- 同类信息用同一个 key（方便冲突消解）
- 比如：drug_allergy_penicillin, chronic_hypertension, lifestyle_smoking

### 6. 什么都没提取到就返回空数组
不要强行凑数，没有新信息就返回空的。

## 输出格式
严格按以下 JSON 格式输出，不要加其他文字：

```json
[
  {{
    "category": "chronic_disease",
    "key": "chronic_hypertension",
    "value": "患有高血压，病史3年",
    "confidence": "high",
    "is_negation": false,
    "source_summary": "用户提到有高血压3年了"
  }},
  {{
    "category": "allergy",
    "key": "drug_allergy_penicillin",
    "value": "对青霉素过敏，小时候打针后出皮疹",
    "confidence": "high",
    "is_negation": false,
    "source_summary": "用户明确说对青霉素过敏"
  }}
]
```

## 当前对话内容

```
{conversation_text}
```

请提取用户的健康信息（只提取 high/medium 置信度的），输出 JSON 数组："""


class MemoryExtractor:
    """
    记忆提取器。
    
    从对话中提取用户健康信息，结构化后存入记忆管理器。
    """
    
    def __init__(self):
        self.llm = get_llm()
    
    def extract_from_conversation(
        self,
        conversation_text: str,
        conversation_id: str = "",
    ) -> List[Dict[str, Any]]:
        """
        从对话文本中提取健康信息。
        
        Args:
            conversation_text: 对话文本（用户和助手的对话）
            conversation_id: 会话ID（用于记录来源）
            
        Returns:
            提取到的记忆列表（可直接传给 batch_add_memories）
        """
        # 构造 prompt
        prompt = MEMORY_EXTRACTION_PROMPT.format(
            conversation_text=conversation_text[:5000]  # 限制长度
        )
        
        try:
            # 调用 LLM
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # 解析 JSON
            memories = self._parse_json_response(response_text)
            
            # 加上会话来源
            for mem in memories:
                mem["source_conversation_id"] = conversation_id
            
            logger.info(f"[记忆提取] 从对话中提取到 {len(memories)} 条健康信息")
            return memories
            
        except Exception as e:
            logger.error(f"[记忆提取] 失败: {e}")
            return []
    
    def _parse_json_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        解析 LLM 返回的 JSON。
        LLM 可能返回 ```json ... ``` 格式，需要做容错处理。
        """
        # 尝试直接解析
        try:
            data = json.loads(response_text)
            if isinstance(data, list):
                return self._validate_memories(data)
        except json.JSONDecodeError:
            pass
        
        # 尝试从 ```json 块中提取
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, list):
                    return self._validate_memories(data)
            except json.JSONDecodeError:
                pass
        
        # 尝试找第一个 [ 和最后一个 ]
        start = response_text.find('[')
        end = response_text.rfind(']')
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(response_text[start:end+1])
                if isinstance(data, list):
                    return self._validate_memories(data)
            except json.JSONDecodeError:
                pass
        
        logger.warning(f"[记忆提取] 无法解析 LLM 返回的 JSON: {response_text[:200]}")
        return []
    
    def _validate_memories(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """校验提取到的记忆格式，过滤无效的"""
        valid_categories = {c.value for c in MemoryCategory}
        valid_confidences = {"high", "medium", "low"}
        
        result = []
        for mem in memories:
            if not isinstance(mem, dict):
                continue
            
            # 必填字段
            if not mem.get("key") or not mem.get("value"):
                continue
            
            # 类别校验
            category = mem.get("category", "other")
            if category not in valid_categories:
                category = "other"
            mem["category"] = category
            
            # 置信度校验
            confidence = mem.get("confidence", "medium")
            if confidence not in valid_confidences:
                confidence = "medium"
            mem["confidence"] = confidence
            
            # 只保留 high 和 medium
            if confidence == "low":
                continue
            
            # is_negation 校验
            mem["is_negation"] = bool(mem.get("is_negation", False))
            
            # source_summary 默认值
            if "source_summary" not in mem:
                mem["source_summary"] = mem.get("value", "")[:50]
            
            result.append(mem)
        
        return result


# ============ 单例模式 ============

_memory_extractor_instance = None


def get_memory_extractor() -> MemoryExtractor:
    """获取记忆提取器实例（单例）"""
    global _memory_extractor_instance
    if _memory_extractor_instance is None:
        _memory_extractor_instance = MemoryExtractor()
    return _memory_extractor_instance
