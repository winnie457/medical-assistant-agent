"""医院搜索工具 - 搜索附近的医院"""

import json
import httpx
from typing import Dict, Any, List
from langchain_core.tools import tool
from ..config import get_settings


@tool
def hospital_search(keyword: str, city: str = "北京", hospital_type: str = "") -> str:
    """
    搜索指定城市的医院，支持按科室、等级、类型筛选。
    
    适用场景：
    - 用户想找附近的医院
    - 想了解某个城市的三甲医院有哪些
    - 想找专科医院（如眼科医院、口腔医院）
    
    Args:
        keyword: 搜索关键词，如"三甲医院"、"眼科"、"协和医院"
        city: 城市名称
        hospital_type: 医院类型筛选，如"综合医院"、"专科医院"、"三甲"
    
    Returns:
        医院列表信息（JSON格式）
    """
    settings = get_settings()
    
    if not settings.amap_api_key:
        # 未配置高德API Key时直接报错，提示用户配置
        return json.dumps({
            "error": "高德地图API Key未配置",
            "message": "请在.env文件中配置AMAP_API_KEY以使用医院搜索功能",
            "hospitals": [],
            "total": 0
        }, ensure_ascii=False, indent=2)
    
    try:
        # 使用高德地图POI搜索
        search_keyword = f"{keyword} 医院" if hospital_type else keyword
        url = "https://restapi.amap.com/v3/place/text"
        params = {
            "key": settings.amap_api_key,
            "keywords": search_keyword,
            "city": city,
            "citylimit": "true",
            "offset": 10,
            "page": 1,
            "extensions": "all"
        }
        
        with httpx.Client(timeout=10) as client:
            response = client.get(url, params=params)
            data = response.json()
        
        pois = data.get("pois", [])
        hospitals = []
        
        for poi in pois[:8]:
            hospital = {
                "name": poi.get("name", ""),
                "address": poi.get("address", ""),
                "location": poi.get("location", ""),
                "type": poi.get("type", ""),
                "tel": poi.get("tel", ""),
                "rating": poi.get("rating", 0),
                "distance": poi.get("distance", "")
            }
            hospitals.append(hospital)
        
        result = {
            "hospitals": hospitals,
            "city": city,
            "keyword": keyword,
            "total": len(hospitals),
            "disclaimer": "医院信息来自高德地图，仅供参考，就诊前请确认医院信息。"
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        # API调用失败时返回错误信息
        return json.dumps({
            "error": "医院搜索失败",
            "message": f"高德地图API调用失败: {str(e)}",
            "hospitals": [],
            "total": 0
        }, ensure_ascii=False, indent=2)
