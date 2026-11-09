"""
风险分级体系 — Human-in-the-loop 权限门禁

设计理念：
    医疗场景是高风险场景，AI 给出的建议直接关系到用户健康。
    不同类型的医疗建议风险等级不同，不能一概而论。
    风险分级门禁就是根据工具的风险等级，决定是否需要用户人工确认。

风险等级：
    LOW    低风险 — 信息查询类，不会直接影响健康决策
    MEDIUM 中风险 — 症状分析类，可能影响就医决策，需要提示但不强制阻断
    HIGH   高风险 — 用药建议类，直接涉及药物使用，必须人工确认

触发 HITL 的条件：
    - 工具风险等级为 HIGH → 强制阻断，等待用户确认
    - 工具风险等级为 MEDIUM → 不阻断，但回答中需加强警示
    - 工具风险等级为 LOW → 直接执行
"""

from enum import Enum
from typing import Dict, List, Optional


class RiskLevel(str, Enum):
    """风险等级枚举"""
    LOW = "low"           # 低风险：信息查询
    MEDIUM = "medium"     # 中风险：症状分析
    HIGH = "high"         # 高风险：用药建议


# 工具风险等级映射
TOOL_RISK_MAP: Dict[str, RiskLevel] = {
    # 低风险：纯信息查询
    "hospital_search": RiskLevel.LOW,           # 医院搜索
    "department_guide": RiskLevel.LOW,          # 科室导诊
    "medical_knowledge_search": RiskLevel.LOW,  # 医疗知识
    "route_to_hospital": RiskLevel.LOW,         # 路线规划
    "health_news_search": RiskLevel.LOW,        # 健康资讯
    
    # 中风险：症状分析（可能影响就医决策）
    "symptom_analyzer": RiskLevel.MEDIUM,       # 症状分析
    
    # 高风险：用药咨询（直接涉及药物使用）
    "medication_consultation": RiskLevel.HIGH,  # 用药咨询
}

# 风险等级描述（给用户看的）
RISK_LEVEL_LABEL = {
    RiskLevel.LOW: "低风险",
    RiskLevel.MEDIUM: "中风险",
    RiskLevel.HIGH: "高风险",
}

# 风险等级颜色（前端展示用）
RISK_LEVEL_COLOR = {
    RiskLevel.LOW: "green",
    RiskLevel.MEDIUM: "orange",
    RiskLevel.HIGH: "red",
}

# 各风险等级的确认提示文案
RISK_APPROVAL_PROMPT = {
    RiskLevel.HIGH: {
        "title": "⚠️ 高风险操作确认",
        "message": "用药建议属于高风险医疗信息，以下内容仅供参考，不能替代医生或药师的专业指导。您确认继续查看吗？",
        "warning": "请务必注意：自行用药可能存在风险，如有疑问请咨询专业医生或药师。",
        "confirm_text": "我已知晓风险，继续查看",
        "cancel_text": "取消",
    },
    RiskLevel.MEDIUM: {
        "title": "提示",
        "message": "",  # 中风险不阻断，只在回答中加警示
        "warning": "",
        "confirm_text": "",
        "cancel_text": "",
    },
}


def get_tool_risk_level(tool_name: str) -> RiskLevel:
    """获取工具的风险等级"""
    return TOOL_RISK_MAP.get(tool_name, RiskLevel.LOW)


def is_high_risk(tool_name: str) -> bool:
    """判断工具是否为高风险（需要人工确认）"""
    return get_tool_risk_level(tool_name) == RiskLevel.HIGH


# 用药类意图关键词（规则式粗筛）
# 拦截点如果只绑在“工具调用”上，模型选择直接作答时就会绕过确认门禁，
# 因此需要在入口先做一次确定性识别
MEDICATION_INTENT_KEYWORDS = (
    "药", "服用", "用法", "用量", "剂量", "副作用", "不良反应",
    "禁忌", "相互作用", "同服", "一起吃", "能不能吃", "停药",
    "抗生素", "消炎药", "头孢", "布洛芬", "阿莫西林", "对乙酰氨基酚",
    "感冒药", "退烧药", "止痛药", "哺乳期", "孕妇",
)


def detect_medication_intent(text: str) -> bool:
    """判断用户问题是否属于用药咨询类（高风险）"""
    if not text:
        return False
    return any(keyword in text for keyword in MEDICATION_INTENT_KEYWORDS)


def is_medium_risk(tool_name: str) -> bool:
    """判断工具是否为中风险"""
    return get_tool_risk_level(tool_name) == RiskLevel.MEDIUM


def get_risk_label(tool_name: str) -> str:
    """获取工具风险等级的中文标签"""
    level = get_tool_risk_level(tool_name)
    return RISK_LEVEL_LABEL.get(level, "未知")


def get_approval_prompt(tool_name: str) -> Optional[Dict]:
    """获取高风险工具的确认弹窗内容"""
    level = get_tool_risk_level(tool_name)
    if level != RiskLevel.HIGH:
        return None
    return RISK_APPROVAL_PROMPT.get(level, None)


def get_high_risk_tools() -> List[str]:
    """获取所有高风险工具列表"""
    return [name for name, level in TOOL_RISK_MAP.items() if level == RiskLevel.HIGH]
