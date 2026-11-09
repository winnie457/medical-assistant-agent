"""健康资讯搜索工具 - 搜索最新的健康新闻和科普文章"""

import json
from typing import Dict, Any, List
from langchain_core.tools import tool


# 演示用的健康资讯数据
SAMPLE_HEALTH_NEWS = [
    {
        "title": "世卫组织发布最新流感疫苗接种指南",
        "source": "世界卫生组织",
        "category": "疾病预防",
        "summary": "世卫组织建议65岁以上老年人、慢性病患者、孕妇等高危人群优先接种流感疫苗。",
        "date": "2024-10-01",
        "url": "#"
    },
    {
        "title": "研究发现：每天步行8000步可显著降低心血管疾病风险",
        "source": "Circulation期刊",
        "category": "运动健康",
        "summary": "最新研究表明，每天步行8000步以上的人群，心血管疾病发病率降低约30%。",
        "date": "2024-09-28",
        "url": "#"
    },
    {
        "title": "国家医保局：更多罕见病药物纳入医保目录",
        "source": "国家医疗保障局",
        "category": "医疗政策",
        "summary": "2024年医保目录调整新增多种罕见病药物，患者用药负担进一步减轻。",
        "date": "2024-09-25",
        "url": "#"
    },
    {
        "title": "秋冬季节交替，注意预防呼吸道疾病",
        "source": "中国疾控中心",
        "category": "疾病预防",
        "summary": "秋冬交替气温变化大，是呼吸道疾病高发期，建议注意保暖、勤洗手、戴口罩。",
        "date": "2024-09-20",
        "url": "#"
    },
    {
        "title": "新研究：地中海饮食可降低糖尿病风险",
        "source": "Diabetes Care期刊",
        "category": "营养健康",
        "summary": "遵循地中海饮食模式的人群，2型糖尿病发病风险降低约25%。",
        "date": "2024-09-15",
        "url": "#"
    }
]


@tool
def health_news_search(keyword: str = "", category: str = "") -> str:
    """
    搜索最新的健康资讯、医学进展、健康科普文章。
    
    适用场景：
    - 想了解最新的健康新闻
    - 想了解某种疾病的最新研究进展
    - 想了解健康养生知识
    - 想了解医疗政策变化
    
    Args:
        keyword: 搜索关键词，如"高血压"、"新冠"、"健康饮食"
        category: 分类筛选，如"疾病预防"、"营养健康"、"运动健康"、"医疗政策"
    
    Returns:
        健康资讯列表（JSON格式）
    """
    try:
        # 过滤结果
        results = SAMPLE_HEALTH_NEWS
        
        if keyword:
            results = [
                article for article in results
                if keyword in article["title"] or keyword in article["summary"] or keyword in article["category"]
            ]
        
        if category:
            results = [
                article for article in results
                if category in article["category"]
            ]
        
        result = {
            "keyword": keyword or "全部",
            "category": category or "全部分类",
            "articles": results,
            "total": len(results),
            "note": "演示数据，实际可接入真实新闻API",
            "disclaimer": "资讯内容仅供参考，不构成医疗建议。"
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "keyword": keyword,
            "error": str(e),
            "articles": SAMPLE_HEALTH_NEWS[:3],
            "disclaimer": "资讯内容仅供参考。"
        }, ensure_ascii=False, indent=2)
