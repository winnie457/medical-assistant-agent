"""科室导诊工具 - 根据症状推荐科室和挂号建议"""

import json
from typing import Dict, Any
from langchain_core.tools import tool


# 科室知识库（简化版）
DEPARTMENT_KNOWLEDGE = {
    "内科": {
        "sub_departments": ["心内科", "呼吸内科", "消化内科", "神经内科", "肾内科", "内分泌科", "风湿免疫科", "血液科"],
        "common_symptoms": ["发热", "咳嗽", "胸闷", "腹痛", "头痛", "乏力", "体重变化"],
        "description": "处理内科系统疾病，无需手术的疾病大多先看内科"
    },
    "外科": {
        "sub_departments": ["普外科", "骨科", "神经外科", "心胸外科", "泌尿外科", "肝胆外科"],
        "common_symptoms": ["外伤", "骨折", "肿块", "疼痛持续加重", "需要手术评估"],
        "description": "需要手术或有创操作的疾病"
    },
    "儿科": {
        "sub_departments": ["小儿内科", "小儿外科", "新生儿科", "儿童保健科"],
        "common_symptoms": ["儿童发热", "儿童咳嗽", "儿童腹泻", "生长发育问题"],
        "description": "14岁以下儿童疾病"
    },
    "妇产科": {
        "sub_departments": ["妇科", "产科", "生殖内分泌科"],
        "common_symptoms": ["月经不调", "下腹痛", "阴道出血", "孕期问题", "白带异常"],
        "description": "女性生殖系统疾病和孕产相关问题"
    },
    "眼科": {
        "sub_departments": ["白内障科", "青光眼科", "眼底病科", "视光学科"],
        "common_symptoms": ["视力下降", "眼痛", "眼红", "视物模糊", "流泪"],
        "description": "眼部疾病和视力问题"
    },
    "耳鼻喉科": {
        "sub_departments": ["耳科", "鼻科", "咽喉科"],
        "common_symptoms": ["耳鸣", "耳聋", "鼻塞", "鼻出血", "咽痛", "声音嘶哑"],
        "description": "耳、鼻、咽喉部位疾病"
    },
    "口腔科": {
        "sub_departments": ["牙体牙髓科", "牙周科", "口腔外科", "正畸科", "种植科"],
        "common_symptoms": ["牙痛", "牙龈出血", "牙齿松动", "口腔溃疡", "颌面部疼痛"],
        "description": "口腔及牙齿相关疾病"
    },
    "皮肤科": {
        "sub_departments": ["皮肤病科", "性病科", "医学美容科"],
        "common_symptoms": ["皮疹", "瘙痒", "红斑", "脱发", "皮肤肿物"],
        "description": "皮肤及附属器疾病"
    },
    "急诊科": {
        "sub_departments": ["急诊内科", "急诊外科", "急诊儿科"],
        "common_symptoms": ["突发剧烈疼痛", "昏迷", "呼吸困难", "大出血", "严重外伤", "急性胸痛"],
        "description": "急危重症，需要立即处理"
    },
    "中医科": {
        "sub_departments": ["中医内科", "中医外科", "针灸科", "推拿科", "中医妇科"],
        "common_symptoms": ["体质调理", "慢性疾病", "亚健康", "术后康复"],
        "description": "中医辨证论治，针灸推拿等传统疗法"
    },
    "精神心理科": {
        "sub_departments": ["精神科", "临床心理科"],
        "common_symptoms": ["情绪低落", "焦虑不安", "失眠", "幻觉", "行为异常"],
        "description": "精神心理疾病"
    },
    "康复医学科": {
        "sub_departments": ["物理治疗科", "作业治疗科", "言语治疗科"],
        "common_symptoms": ["术后康复", "中风后康复", "运动损伤", "颈肩腰腿痛"],
        "description": "功能恢复和康复治疗"
    }
}


@tool
def department_guide(symptoms: str, patient_info: str = "") -> str:
    """
    根据症状描述，智能推荐应该就诊的科室，给出挂号建议。
    
    适用场景：
    - 不知道该挂什么科
    - 症状复杂，不确定看哪个科更好
    - 想了解科室就诊注意事项
    
    Args:
        symptoms: 症状描述，越详细越好
        patient_info: 患者信息，如年龄、性别、既往病史等
    
    Returns:
        科室推荐结果（JSON格式），包含推荐科室、备选科室、就诊建议等
    """
    try:
        # 简单的关键词匹配 + LLM 确认策略
        # 这里直接返回结构化的科室导诊信息
        from ..services.llm_service import get_llm
        
        llm = get_llm()
        
        prompt = f"""你是一位专业的医院导诊护士。请根据患者的症状描述，推荐最合适的就诊科室。

**医院科室列表：**
{json.dumps(DEPARTMENT_KNOWLEDGE, ensure_ascii=False, indent=2)}

**患者症状：** {symptoms}

**患者信息：** {patient_info if patient_info else "未提供"}

请严格按照以下JSON格式返回推荐结果：
```json
{{
  "primary_department": "首选科室名称",
  "primary_reason": "为什么推荐这个科室",
  "alternative_departments": [
    {{
      "department": "备选科室1",
      "reason": "什么情况下考虑这个科室"
    }}
  ],
  "urgency": "普通/较急/紧急",
  "booking_tips": ["挂号注意事项1", "挂号注意事项2"],
  "what_to_bring": ["就诊需要带的东西1", "就诊需要带的东西2"],
  "self_check": ["就诊前可以注意观察的要点"]
}}
```

请确保推荐准确、建议实用。"""
        
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
        result["disclaimer"] = "科室推荐仅供参考，具体以医院实际分诊为准。"
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "primary_department": "全科/内科",
            "primary_reason": f"症状分析暂时无法精准判断，建议先看全科或内科初诊（错误: {str(e)[:30]}）",
            "alternative_departments": [],
            "urgency": "普通",
            "booking_tips": ["建议提前预约挂号", "带上既往病历资料"],
            "what_to_bring": ["身份证", "医保卡", "既往病历"],
            "disclaimer": "科室推荐仅供参考，具体以医院实际分诊为准。"
        }, ensure_ascii=False, indent=2)
