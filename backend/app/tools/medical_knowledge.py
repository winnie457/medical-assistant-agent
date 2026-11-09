"""医疗健康知识搜索工具 - 搜索疾病、用药、健康科普知识"""

import json
from typing import Dict, Any, List
from langchain_core.tools import tool


# 内置的医学知识库（简化版，演示用）
MEDICAL_KNOWLEDGE_BASE = {
    "感冒": {
        "category": "呼吸系统疾病",
        "symptoms": ["鼻塞", "流涕", "咽痛", "咳嗽", "发热", "头痛", "乏力"],
        "causes": "病毒感染（鼻病毒、冠状病毒等）",
        "common_treatments": ["多休息", "多喝水", "对症使用感冒药", "发热可服退烧药"],
        "duration": "通常5-7天自愈",
        "when_to_see_doctor": ["高热超过3天", "呼吸困难", "胸痛", "症状持续超过1周"],
        "medication_advice": "对症治疗为主，避免滥用抗生素"
    },
    "发烧": {
        "category": "常见症状",
        "symptoms": ["体温升高", "畏寒", "乏力", "肌肉酸痛"],
        "causes": "感染、炎症、免疫反应等多种原因",
        "common_treatments": ["物理降温（温水擦浴）", "多饮水", "体温超过38.5°C可服退烧药", "查找病因"],
        "normal_temp": "腋温正常范围：36.0-37.2°C",
        "when_to_see_doctor": ["高热不退（>39°C）", "发热超过3天", "伴剧烈头痛/呕吐", "伴呼吸困难", "婴幼儿发热"],
        "medication_advice": "常用退烧药：对乙酰氨基酚、布洛芬。按说明书使用，避免过量。"
    },
    "头痛": {
        "category": "神经系统症状",
        "symptoms": ["头部疼痛", "可伴恶心呕吐", "可伴畏光畏声"],
        "types": ["偏头痛", "紧张性头痛", "丛集性头痛", "继发性头痛"],
        "common_treatments": ["休息", "避免诱因", "止痛药对症治疗"],
        "when_to_see_doctor": ["突发剧烈头痛（雷击样）", "头痛伴发热/颈部僵硬", "头痛伴意识障碍", "头痛进行性加重", "头部外伤后头痛"],
        "prevention": "规律作息、避免熬夜、减少咖啡因摄入、适度运动"
    },
    "胃痛": {
        "category": "消化系统症状",
        "symptoms": ["上腹部疼痛", "可伴反酸、嗳气", "可伴恶心呕吐", "可伴食欲不振"],
        "common_causes": ["胃炎", "胃溃疡", "消化不良", "胃食管反流"],
        "common_treatments": ["规律饮食", "避免辛辣刺激食物", "抑酸药", "胃黏膜保护剂"],
        "when_to_see_doctor": ["疼痛剧烈持续不缓解", "伴呕血/黑便", "伴体重明显下降", "吞咽困难", "贫血"],
        "diet_advice": "少食多餐，避免过饱，避免辛辣刺激、烟酒、浓茶咖啡"
    },
    "高血压": {
        "category": "心血管疾病",
        "symptoms": ["大多无症状", "可有头晕头痛", "可有颈项发紧", "可有疲劳心悸"],
        "diagnostic_criteria": "非同日3次测量，收缩压≥140mmHg和/或舒张压≥90mmHg",
        "common_treatments": ["生活方式干预", "降压药物治疗", "定期监测血压"],
        "lifestyle_changes": ["低盐饮食", "适量运动", "控制体重", "戒烟限酒", "减轻压力"],
        "complications": "脑卒中、心肌梗死、心衰、肾衰、眼底病变",
        "medication_advice": "需在医生指导下长期服药，不可自行停药"
    },
    "糖尿病": {
        "category": "内分泌代谢疾病",
        "types": ["1型糖尿病", "2型糖尿病", "妊娠期糖尿病"],
        "symptoms": ["多饮", "多食", "多尿", "体重下降", "乏力"],
        "diagnostic_criteria": "空腹血糖≥7.0mmol/L 或 餐后2小时≥11.1mmol/L",
        "common_treatments": ["饮食控制", "运动治疗", "口服降糖药", "胰岛素治疗"],
        "complications": "糖尿病肾病、糖尿病视网膜病变、糖尿病足、心血管疾病",
        "diet_advice": "控制总热量，均衡营养，定时定量，少食多餐"
    },
    "失眠": {
        "category": "睡眠障碍",
        "symptoms": ["入睡困难", "睡眠浅", "易醒", "早醒", "白天困倦"],
        "causes": ["心理压力", "环境因素", "生活习惯", "疾病因素", "药物影响"],
        "common_treatments": ["睡眠卫生教育", "认知行为疗法(CBT-I)", "必要时短期药物"],
        "sleep_hygiene": ["规律作息", "卧室安静舒适", "避免睡前使用电子设备", "避免睡前饮咖啡浓茶", "适度运动"],
        "when_to_see_doctor": ["失眠持续超过1个月", "严重影响白天功能", "伴抑郁焦虑", "伴打鼾/呼吸暂停"]
    },
    "皮肤过敏": {
        "category": "皮肤科疾病",
        "symptoms": ["皮肤瘙痒", "红斑", "丘疹", "风团", "可伴肿胀"],
        "common_allergens": ["食物（海鲜、坚果等）", "药物", "花粉", "尘螨", "动物毛发", "化妆品"],
        "common_treatments": ["避免接触过敏原", "抗组胺药物", "外用糖皮质激素", "严重时需就医"],
        "when_to_see_doctor": ["症状严重范围广", "伴呼吸困难/喉头水肿", "伴头晕心慌", "反复发作"]
    }
}


@tool
def medical_knowledge_search(query: str) -> str:
    """
    搜索医疗健康知识，包括疾病介绍、症状、治疗方法、用药建议、预防措施等。
    
    适用场景：
    - 想了解某种疾病的基本知识
    - 想知道某种症状可能是什么原因
    - 想了解用药注意事项
    - 想了解疾病的预防和康复知识
    
    Args:
        query: 搜索关键词，如"感冒怎么办"、"高血压饮食"、"失眠怎么调理"
    
    Returns:
        相关的医疗健康知识（JSON格式）
    """
    try:
        # 先在内置知识库中搜索
        results = []
        query_lower = query.lower()
        
        for disease_name, info in MEDICAL_KNOWLEDGE_BASE.items():
            # 简单匹配：关键词匹配疾病名或症状
            if disease_name in query or query in disease_name:
                results.append({
                    "name": disease_name,
                    "category": info.get("category", ""),
                    "summary": f"{disease_name}是{info.get('category', '常见疾病')}，主要表现为{', '.join(info.get('symptoms', [])[:3])}。",
                    "details": info,
                    "source": "内置医学知识库"
                })
            else:
                # 检查症状关键词匹配
                symptoms = info.get("symptoms", [])
                for symptom in symptoms:
                    if symptom in query:
                        results.append({
                            "name": disease_name,
                            "category": info.get("category", ""),
                            "summary": f"{disease_name}可能出现{symptom}等症状。",
                            "details": info,
                            "source": "内置医学知识库"
                        })
                        break
        
        if results:
            result = {
                "query": query,
                "results": results[:5],
                "total": min(len(results), 5),
                "disclaimer": "以上健康知识仅供参考，不能替代医生诊断。身体不适请及时就医。"
            }
            return json.dumps(result, ensure_ascii=False, indent=2)
        
        # 内置库没有，用LLM生成通用回答
        from ..services.llm_service import get_llm
        llm = get_llm()
        
        prompt = f"""请用通俗的语言回答以下医疗健康问题，内容要准确、实用。
注意：你只是提供健康科普知识，不能替代医生诊断。最后要加上免责声明。

问题：{query}

请按以下格式回答：
1. 简要介绍（1-2句话）
2. 常见原因/症状（3-5点）
3. 处理/缓解方法（3-5条实用建议）
4. 什么时候需要看医生（3-4种情况）
5. 预防建议（2-3条）"""
        
        response = llm.invoke(prompt)
        
        result = {
            "query": query,
            "answer": response.content,
            "source": "AI生成健康科普",
            "disclaimer": "以上内容为AI生成的健康科普，仅供参考，不能替代专业医生诊断。如有不适请及时就医。"
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "query": query,
            "answer": f"抱歉，暂时无法获取相关信息（{str(e)[:50]}）。建议咨询专业医生。",
            "disclaimer": "以上内容仅供参考，不能替代专业医生诊断。"
        }, ensure_ascii=False, indent=2)
