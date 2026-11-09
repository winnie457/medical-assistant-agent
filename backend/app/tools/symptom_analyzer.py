"""症状分析工具 - 根据症状给出初步分析和就医建议"""

import json
from typing import Dict, Any, List
from langchain_core.tools import tool
from ..services.llm_service import get_llm


SYMPTOM_ANALYSIS_PROMPT = """你是一位经验丰富的全科医生助手。请根据用户描述的症状，给出专业的初步分析建议。

**重要提示：** 这只是初步参考建议，不能替代专业医生诊断。症状严重或持续不缓解请及时就医。

请严格按照以下JSON格式返回分析结果：
```json
{
  "urgency_level": "紧急程度：立即就医/尽快就医/可自行观察",
  "recommended_department": "建议就诊科室",
  "possible_conditions": [
    {
      "name": "可能的疾病名称",
      "probability": "高/中/低",
      "key_symptoms": ["主要症状1", "主要症状2"],
      "description": "简要说明"
    }
  ],
  "self_care_advice": ["自我护理建议1", "自我护理建议2"],
  "warning_signs": ["危险信号1，出现这些情况需要立即就医"],
  "preparation_tips": ["就诊前准备建议1", "就诊前准备建议2"]
}
```

用户症状描述：
{symptoms}

用户所在城市：{city}

请给出详细、专业且贴心的建议。确保JSON格式正确。"""


@tool
def symptom_analyzer(symptoms: str, city: str = "北京") -> str:
    """
    根据用户描述的症状进行初步分析，给出可能的疾病方向、建议就诊科室、
    紧急程度判断、自我护理建议和就诊提醒。
    
    适用场景：
    - 用户描述身体不适、疼痛、异常症状
    - 不确定该挂什么科室
    - 想了解症状是否严重
    
    Args:
        symptoms: 用户描述的症状，越详细越好
        city: 用户所在城市，用于给出本地化建议
    
    Returns:
        结构化的症状分析结果（JSON格式）
    """
    try:
        llm = get_llm()
        prompt = SYMPTOM_ANALYSIS_PROMPT.format(symptoms=symptoms, city=city)
        
        response = llm.invoke(prompt)
        content = response.content
        
        # 提取JSON
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            json_str = content[json_start:json_end].strip()
        elif "{" in content and "}" in content:
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            json_str = content[json_start:json_end]
        else:
            json_str = content
        
        result = json.loads(json_str)
        
        # 添加免责声明
        result["disclaimer"] = "以上为AI初步分析，仅供参考，不能替代专业医生诊断。如有不适请及时就医。"
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        error_result = {
            "urgency_level": "建议咨询专业医生",
            "recommended_department": "全科/内科",
            "possible_conditions": [],
            "self_care_advice": ["注意休息，多喝水", "如症状持续或加重，请及时就医"],
            "warning_signs": ["症状持续加重", "出现高热、呼吸困难等严重症状"],
            "preparation_tips": ["携带身份证、医保卡", "记录症状出现时间和变化"],
            "error": str(e),
            "disclaimer": "以上为AI初步分析，仅供参考，不能替代专业医生诊断。"
        }
        return json.dumps(error_result, ensure_ascii=False, indent=2)
