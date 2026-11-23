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
# 旧版只要出现"药"字就判高危，实测把三类问题误拦了（评测 round2 里误拦 5 条全部来自这里）：
#   1) 急症："误服安眠药""头孢配酒"——应该走急救，而不是"请确认是否查看用药信息"；
#   2) 知识类："高血脂需要终身吃药吗"；
#   3) 资讯类："最近的减肥药新闻"。
# 新规则：有具体药名时看"问的是什么"，没有药名时只认剂量 / 安全性问题。

DRUG_NAMES = (
    "阿莫西林", "布洛芬", "头孢", "头孢克肟", "奥美拉唑", "蒙脱石散",
    "氯雷他定", "二甲双胍", "连花清瘟", "阿司匹林", "对乙酰氨基酚",
    "氨氯地平", "阿奇霉素", "左氧氟沙星", "甲硝唑", "青霉素",
    "感冒灵", "板蓝根", "退烧药", "止痛药", "感冒药", "消炎药", "抗生素",
)
DOSE_PATTERNS = ("剂量", "用量", "吃几", "一次吃", "一天吃", "几粒", "几片", "多少毫克", "毫克", "mg")
USAGE_PATTERNS = ("怎么吃", "怎么用", "饭前", "饭后", "空腹", "服用方法", "冲服", "温水送服")
SAFETY_PATTERNS = (
    "副作用", "不良反应", "禁忌", "相互作用", "一起吃", "一起喝", "一起",
    "同服", "停药", "喝酒", "饮酒", "过敏", "哺乳", "怀孕", "孕妇",
    "孩子", "儿童", "老人", "肝肾",
)
ASK_PATTERNS = ("能不能吃", "可以吃", "能吃", "吃吗", "该吃", "要不要吃", "需要吃")


def detect_medication_intent(text: str) -> bool:
    """判断用户问题是否属于用药咨询类（高风险）。

    判定顺序：
    1. 提到具体药名 → 看是否在问用法 / 剂量 / 安全性 / 能不能吃；
    2. 没提药名 → 只有明确问剂量或安全性、且确实在说"药"才算。
    单纯出现"药"字（减肥药新闻、高血脂要不要吃药）不再进入确认流程。
    """
    if not text:
        return False

    has_drug = any(name in text for name in DRUG_NAMES)
    has_dose = any(pattern in text for pattern in DOSE_PATTERNS)
    has_usage = any(pattern in text for pattern in USAGE_PATTERNS)
    has_safety = any(pattern in text for pattern in SAFETY_PATTERNS)
    asks_taking = any(pattern in text for pattern in ASK_PATTERNS)

    if has_drug:
        return has_dose or has_usage or has_safety or asks_taking

    mentions_medicine = "药" in text
    return mentions_medicine and (has_dose or has_safety)


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
