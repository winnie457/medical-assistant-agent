"""就医路线规划工具 - 规划去医院的交通路线"""

import json
import httpx
from typing import Dict, Any
from langchain_core.tools import tool
from ..config import get_settings


@tool
def route_to_hospital(origin: str, hospital_name: str, city: str = "北京", route_type: str = "transit") -> str:
    """
    规划从当前位置到医院的交通路线，支持公交、地铁、驾车、步行等方式。
    
    适用场景：
    - 想知道怎么去某个医院
    - 想了解去医院的交通方式和时间
    - 急诊时想知道最快的路线
    
    Args:
        origin: 起点地址或位置描述
        hospital_name: 医院名称
        city: 所在城市
        route_type: 交通方式：transit(公共交通)/driving(驾车)/walking(步行)
    
    Returns:
        路线规划结果（JSON格式）
    """
    settings = get_settings()
    
    if not settings.amap_api_key:
        # 返回演示数据
        route_type_names = {
            "transit": "公共交通",
            "driving": "驾车",
            "walking": "步行"
        }
        route_name = route_type_names.get(route_type, "公共交通")
        
        return json.dumps({
            "origin": origin,
            "destination": hospital_name,
            "city": city,
            "route_type": route_name,
            "routes": [
                {
                    "name": "推荐路线",
                    "distance": "约5.2公里",
                    "duration": "约35分钟",
                    "steps": [
                        f"从{origin}出发",
                        "步行至地铁站",
                        "乘坐地铁X号线（X方向）",
                        "在XX站下车，A口出",
                        f"步行约500米到达{hospital_name}"
                    ],
                    "traffic_condition": "畅通"
                }
            ],
            "tips": [
                "就诊建议提前出发，预留排队时间",
                "急诊建议优先选择驾车或打车",
                "可使用导航APP获取实时路线"
            ],
            "note": "演示数据（高德地图API未配置）",
            "disclaimer": "路线信息仅供参考，实际以导航软件为准。"
        }, ensure_ascii=False, indent=2)
    
    try:
        # 先搜索医院的位置
        url = "https://restapi.amap.com/v3/place/text"
        params = {
            "key": settings.amap_api_key,
            "keywords": hospital_name,
            "city": city,
            "citylimit": "true",
            "offset": 1,
            "page": 1
        }
        
        with httpx.Client(timeout=10) as client:
            poi_response = client.get(url, params=params)
            poi_data = poi_response.json()
        
        pois = poi_data.get("pois", [])
        if not pois:
            return json.dumps({
                "error": f"未找到{hospital_name}的位置信息",
                "disclaimer": "路线信息仅供参考。"
            }, ensure_ascii=False, indent=2)
        
        destination_location = pois[0].get("location", "")
        destination_name = pois[0].get("name", hospital_name)
        
        # 路线规划（这里简化处理，实际需要调用路线规划API）
        route_type_names = {
            "transit": "公共交通",
            "driving": "驾车",
            "walking": "步行"
        }
        
        result = {
            "origin": origin,
            "destination": destination_name,
            "destination_address": pois[0].get("address", ""),
            "city": city,
            "route_type": route_type_names.get(route_type, "公共交通"),
            "routes": [
                {
                    "name": "推荐路线",
                    "distance": "待获取",
                    "duration": "待获取",
                    "steps": ["使用高德地图API获取详细路线"]
                }
            ],
            "tips": [
                "就诊建议提前出发",
                "急诊建议拨打120"
            ],
            "disclaimer": "路线信息仅供参考，实际以导航软件为准。"
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "origin": origin,
            "destination": hospital_name,
            "city": city,
            "error": str(e),
            "tips": ["建议使用高德地图/百度地图导航前往医院"],
            "disclaimer": "路线信息仅供参考。"
        }, ensure_ascii=False, indent=2)
