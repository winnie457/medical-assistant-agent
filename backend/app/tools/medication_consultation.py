"""用药咨询工具 - 提供用药建议和注意事项"""

import json
from typing import Dict, Any, List
from langchain_core.tools import tool


# 常用药品知识库（简化版，演示用）
MEDICATION_KNOWLEDGE = {
    "布洛芬": {
        "category": "非甾体抗炎药(NSAIDs)",
        "common_uses": ["退热", "止痛（头痛、牙痛、关节痛、痛经）", "抗炎"],
        "dosage": {
            "成人": "每次200-400mg，每日3-4次，每日最大剂量不超过2400mg",
            "儿童": "按体重计算，每次5-10mg/kg，每日3-4次"
        },
        "side_effects": ["胃肠道不适", "恶心", "头晕", "皮疹"],
        "contraindications": ["胃溃疡/消化道出血", "严重肝肾功能不全", "对NSAIDs过敏", "妊娠晚期"],
        "precautions": [
            "饭后服用可减少胃肠道刺激",
            "避免与其他NSAIDs同时使用",
            "长期使用需监测肝肾功能",
            "饮酒会增加胃肠道风险"
        ],
        "special_populations": {
            "孕妇": "妊娠早期慎用，妊娠晚期禁用",
            "哺乳期": "可使用，相对安全",
            "老年人": "减量使用，注意胃肠道和肾功能"
        }
    },
    "对乙酰氨基酚": {
        "category": "解热镇痛药",
        "common_uses": ["退热", "止痛（头痛、肌肉痛、关节痛）"],
        "dosage": {
            "成人": "每次300-600mg，每日3-4次，每日最大剂量不超过2000mg",
            "儿童": "按体重计算，每次10-15mg/kg，每4-6小时一次"
        },
        "side_effects": ["常规剂量副作用少", "过量可致肝损伤"],
        "contraindications": ["严重肝肾功能不全", "对本品过敏"],
        "precautions": [
            "严格按剂量服用，不可过量",
            "避免同时使用含对乙酰氨基酚的复方感冒药",
            "饮酒后不宜使用",
            "长期使用需监测肝功能"
        ],
        "special_populations": {
            "孕妇": "可使用，相对安全",
            "哺乳期": "可使用，相对安全",
            "老年人": "按常规剂量使用"
        }
    },
    "阿莫西林": {
        "category": "青霉素类抗生素",
        "common_uses": ["呼吸道感染", "泌尿道感染", "皮肤软组织感染", "耳鼻喉感染"],
        "dosage": {
            "成人": "每次0.5g，每6-8小时一次",
            "儿童": "按体重计算，每日50-100mg/kg，分3-4次"
        },
        "side_effects": ["恶心", "腹泻", "皮疹", "过敏反应"],
        "contraindications": ["青霉素过敏", "传染性单核细胞增多症"],
        "precautions": [
            "使用前需确认无青霉素过敏史",
            "按疗程服用，不可自行停药",
            "过敏体质者慎用",
            "严重过敏可致休克，需立即就医"
        ],
        "special_populations": {
            "孕妇": "可使用，相对安全",
            "哺乳期": "可使用",
            "老年人": "肾功能不全需调整剂量"
        }
    },
    "感冒灵": {
        "category": "复方感冒药（中西药结合）",
        "common_uses": ["感冒引起的头痛", "发热", "鼻塞", "流涕", "咽痛"],
        "ingredients": "含对乙酰氨基酚、马来酸氯苯那敏等",
        "side_effects": ["困倦", "嗜睡", "口干", "胃肠道不适"],
        "contraindications": ["严重肝肾功能不全", "对成分过敏"],
        "precautions": [
            "服药期间不得驾驶车辆或操作机械",
            "不可同时服用成分相似的其他感冒药",
            "避免饮酒",
            "肝肾功能不全者慎用"
        ],
        "special_populations": {
            "孕妇": "慎用",
            "哺乳期": "慎用",
            "老年人": "慎用"
        }
    },
    "奥美拉唑": {
        "category": "质子泵抑制剂(PPI)",
        "common_uses": ["胃溃疡", "十二指肠溃疡", "胃食管反流病", "幽门螺杆菌根除"],
        "dosage": {
            "成人": "每次20mg，每日1-2次，晨起空腹服用"
        },
        "side_effects": ["头痛", "腹泻", "恶心", "腹痛"],
        "contraindications": ["对本品过敏"],
        "precautions": [
            "晨起空腹服用效果最佳",
            "长期使用需监测骨密度和维生素B12",
            "可能掩盖胃癌症状，诊断明确前慎用"
        ],
        "special_populations": {
            "孕妇": "慎用",
            "哺乳期": "慎用",
            "老年人": "无需调整剂量"
        }
    }
}


@tool
def medication_consultation(medication_name: str, question: str = "") -> str:
    """
    查询药品的用法用量、副作用、禁忌症、注意事项等用药信息。
    
    适用场景：
    - 想了解某种药怎么吃
    - 想知道某种药有什么副作用
    - 想知道什么人不能吃某种药
    - 想了解药物相互作用
    
    Args:
        medication_name: 药品名称（通用名或商品名）
        question: 具体问题，如"怎么吃"、"副作用"、"孕妇能吃吗"（可选）
    
    Returns:
        用药咨询结果（JSON格式）
    """
    try:
        # 先在内置知识库中查找
        result_data = None
        matched_name = None
        
        for med_name, info in MEDICATION_KNOWLEDGE.items():
            if med_name in medication_name or medication_name in med_name:
                result_data = info
                matched_name = med_name
                break
        
        if result_data:
            # 如果有具体问题，过滤相关内容
            if question:
                from ..services.llm_service import get_llm
                llm = get_llm()
                
                prompt = f"""根据以下药品信息，回答用户的具体问题。

药品名称：{matched_name}
药品信息：{json.dumps(result_data, ensure_ascii=False, indent=2)}

用户问题：{question}

请用通俗的语言回答，确保准确易懂。最后加上"请仔细阅读药品说明书，在医生或药师指导下使用"的提示。"""
                
                response = llm.invoke(prompt)
                answer = response.content
            else:
                answer = f"以下是{matched_name}的基本用药信息"
            
            result = {
                "medication": matched_name,
                "question": question,
                "answer": answer,
                "details": result_data,
                "disclaimer": "以上信息仅供参考，不能替代医生或药师的专业建议。用药请仔细阅读说明书，遵医嘱使用。"
            }
            return json.dumps(result, ensure_ascii=False, indent=2)
        
        # 内置库没有，用LLM生成通用回答
        from ..services.llm_service import get_llm
        llm = get_llm()
        
        prompt = f"""请提供关于{medication_name}的用药信息。
用户的具体问题：{question if question else "请提供基本用药信息"}

请包含以下内容：
1. 药物类别和主要用途
2. 常用用法用量
3. 常见副作用
4. 禁忌症和注意事项
5. 特殊人群用药提示

请确保信息准确、通俗。
重要：最后必须加上免责声明："以上信息仅供参考，用药请遵医嘱，仔细阅读药品说明书。"
"""
        
        response = llm.invoke(prompt)
        
        result = {
            "medication": medication_name,
            "question": question,
            "answer": response.content,
            "source": "AI生成",
            "disclaimer": "以上信息仅供参考，不能替代医生或药师的专业建议。用药请遵医嘱。"
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "medication": medication_name,
            "question": question,
            "error": str(e),
            "disclaimer": "用药请咨询医生或药师，仔细阅读药品说明书。"
        }, ensure_ascii=False, indent=2)
