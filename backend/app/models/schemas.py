"""数据模型定义 - 智能就医助手"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


# ============ 对话相关模型 ============

class ChatMessage(BaseModel):
    """聊天消息"""
    role: str = Field(..., description="角色: user/assistant/tool")
    content: str = Field(..., description="消息内容")
    timestamp: Optional[str] = Field(default=None, description="时间戳")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(default=None, description="工具调用")
    tool_name: Optional[str] = Field(default=None, description="工具名称(tool消息)")


class ChatRequest(BaseModel):
    """对话请求"""
    message: str = Field(..., description="用户消息")
    conversation_id: Optional[str] = Field(default=None, description="会话ID")
    history: Optional[List[ChatMessage]] = Field(default=[], description="对话历史")
    city: Optional[str] = Field(default="北京", description="所在城市")
    user_id: Optional[str] = Field(default="default", description="用户ID（用于长期记忆）")


class ToolCallInfo(BaseModel):
    """工具调用信息（用于前端展示）"""
    tool_name: str
    tool_description: str
    arguments: Dict[str, Any]
    result_preview: str = ""
    status: str = "calling"  # calling/success/failed


class ConversationInfo(BaseModel):
    """会话信息"""
    conversation_id: str
    title: str
    created_at: str
    last_message: str
    message_count: int


# ============ 医院/医生相关模型 ============

class Location(BaseModel):
    """地理位置"""
    longitude: float
    latitude: float


class HospitalInfo(BaseModel):
    """医院信息"""
    name: str
    address: str
    location: Optional[Location] = None
    level: Optional[str] = None  # 三甲/三乙/二甲等
    type: Optional[str] = None  # 综合/专科
    phone: Optional[str] = None
    rating: Optional[float] = None
    distance: Optional[str] = None
    departments: Optional[List[str]] = None


class DoctorInfo(BaseModel):
    """医生信息"""
    name: str
    hospital: str
    department: str
    title: str  # 主任医师/副主任医师等
    specialty: str
    rating: Optional[float] = None


class DiseaseInfo(BaseModel):
    """疾病信息"""
    name: str
    category: str
    symptoms: List[str] = []
    description: str = ""
    common_treatments: List[str] = []
    severity: str = "普通"  # 轻微/普通/严重/紧急


# ============ 就医建议模型 ============

class MedicalAdvice(BaseModel):
    """就医建议"""
    urgency_level: str = Field(..., description="紧急程度: 立即就医/尽快就医/可自行观察")
    recommended_department: str = Field(..., description="建议科室")
    possible_conditions: List[DiseaseInfo] = Field(default_factory=list, description="可能的疾病")
    self_care_advice: List[str] = Field(default_factory=list, description="自我护理建议")
    warning_signs: List[str] = Field(default_factory=list, description="需要警惕的症状")
    preparation_tips: List[str] = Field(default_factory=list, description="就诊准备建议")


# ============ 工具调用结果模型 ============

class ToolResult(BaseModel):
    """工具执行结果"""
    tool_name: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    duration: float = 0.0


# ============ 评估相关模型 ============

class EvaluationMetric(BaseModel):
    """评估指标"""
    name: str
    description: str
    pipeline_score: float = 0.0
    react_score: float = 0.0
    improvement: float = 0.0


class EvaluationResult(BaseModel):
    """评估结果"""
    test_case_count: int
    metrics: List[EvaluationMetric] = []
    pipeline_avg_tools: float = 0.0
    react_avg_tools: float = 0.0
    pipeline_avg_time: float = 0.0
    react_avg_time: float = 0.0
    details: List[Dict[str, Any]] = []


# ============ 健康资讯模型 ============

class HealthArticle(BaseModel):
    """健康资讯文章"""
    title: str
    url: str
    source: str
    summary: str = ""
    published_date: Optional[str] = None
