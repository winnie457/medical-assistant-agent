"""
工具执行器 — Never-Throw 容错体系

核心设计理念：
    工具调用永远不会抛出异常，所有错误都被捕获并转化为结构化的错误信息返回给 Agent，
    由 Agent 自主决定如何处理（重试、换工具、或用自身知识兜底）。

能力清单：
    1. 指数退避重试（Exponential Backoff Retry）
    2. 参数语义化纠偏（Parameter Correction）
    3. Never-Throw 执行器（所有异常 → 结构化错误）
    4. 工具降级策略（主工具失败 → 备选方案）
"""

import time
import json
import logging
import re
from typing import Dict, Any, Optional, Callable, List
from functools import wraps
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


# ============================================================
# 1. 指数退避重试装饰器
# ============================================================

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 5.0,
    backoff_factor: float = 2.0,
    retry_on_exceptions: tuple = (Exception,),
):
    """
    指数退避重试装饰器。
    
    重试间隔 = min(base_delay * (backoff_factor ^ attempt), max_delay)
    
    Args:
        max_retries: 最大重试次数（不含首次调用）
        base_delay: 初始延迟（秒）
        max_delay: 最大延迟（秒）
        backoff_factor: 退避因子
        retry_on_exceptions: 哪些异常需要重试
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retry_on_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                        logger.warning(
                            f"[重试] {func.__name__} 第 {attempt + 1} 次失败: {e}, "
                            f"{delay:.1f}s 后重试 (共 {max_retries} 次机会)"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"[重试耗尽] {func.__name__} 重试 {max_retries} 次后仍然失败: {e}"
                        )
            raise last_exception
        return wrapper
    return decorator


# ============================================================
# 2. 参数语义化纠偏
# ============================================================

class ParameterCorrector:
    """
    参数语义化纠偏器。
    
    当 LLM 生成的工具参数不符合预期时，尝试自动修复：
    - 字符串转数字
    - 去除多余空格
    - 城市名标准化
    - JSON 字符串解析
    - 缺失必填参数时给出默认值
    """
    
    # 城市名别名映射（LLM 可能生成别名）
    CITY_ALIASES = {
        "首都": "北京", "京城": "北京", "北平": "北京",
        "魔都": "上海", "沪": "上海", "申城": "上海",
        "花城": "广州", "羊城": "广州",
        "鹏城": "深圳", "深": "深圳",
        "杭城": "杭州", "临安": "杭州",
        "蓉城": "成都", "锦官城": "成都",
        "江城": "武汉", "九省通衢": "武汉",
        "金陵": "南京", "石头城": "南京",
        "津门": "天津", "天津卫": "天津",
    }
    
    # 医院级别标准化
    HOSPITAL_LEVEL_MAP = {
        "三甲": "三级甲等", "三级甲": "三级甲等", "3甲": "三级甲等",
        "三乙": "三级乙等", "三级乙": "三级乙等",
        "二甲": "二级甲等", "二级甲": "二级甲等",
        "二乙": "二级乙等", "二级乙": "二级乙等",
        "一级": "一级甲等", "一甲": "一级甲等",
    }
    
    @classmethod
    def correct_parameters(cls, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        对工具参数进行语义化纠偏。
        
        Args:
            tool_name: 工具名称
            params: 原始参数
            
        Returns:
            纠偏后的参数字典
        """
        corrected = dict(params)
        
        # 通用纠偏：去除字符串两端空格
        for key, value in corrected.items():
            if isinstance(value, str):
                corrected[key] = value.strip()
        
        # 按工具类型做专项纠偏
        if tool_name == "hospital_search":
            corrected = cls._correct_hospital_search(corrected)
        elif tool_name == "symptom_analyzer":
            corrected = cls._correct_symptom_analyzer(corrected)
        elif tool_name == "department_guide":
            corrected = cls._correct_department_guide(corrected)
        elif tool_name == "route_to_hospital":
            corrected = cls._correct_route_planner(corrected)
        elif tool_name == "medication_consultation":
            corrected = cls._correct_medication(corrected)
        
        return corrected
    
    @classmethod
    def _correct_hospital_search(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """医院搜索参数纠偏"""
        # 城市名标准化
        if "city" in params and params["city"]:
            city = params["city"]
            params["city"] = cls.CITY_ALIASES.get(city, city)
        
        # 医院级别标准化
        if "hospital_type" in params and params["hospital_type"]:
            level = params["hospital_type"]
            params["hospital_type"] = cls.HOSPITAL_LEVEL_MAP.get(level, level)
        
        # keyword 为空时给默认值
        if "keyword" not in params or not params["keyword"]:
            params["keyword"] = "医院"
        
        return params
    
    @classmethod
    def _correct_symptom_analyzer(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """症状分析参数纠偏"""
        if "city" in params and params["city"]:
            city = params["city"]
            params["city"] = cls.CITY_ALIASES.get(city, city)
        
        # symptoms 为空时兜底
        if "symptoms" not in params or not params["symptoms"]:
            params["symptoms"] = "身体不适"
        
        return params
    
    @classmethod
    def _correct_department_guide(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """科室导诊参数纠偏"""
        if "symptoms" not in params or not params["symptoms"]:
            params["symptoms"] = params.get("symptom", "身体不适")
        
        return params
    
    @classmethod
    def _correct_route_planner(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """路线规划参数纠偏"""
        if "city" in params and params["city"]:
            city = params["city"]
            params["city"] = cls.CITY_ALIASES.get(city, city)
        
        # 起点终点不能为空
        if "origin" not in params or not params["origin"]:
            params["origin"] = "当前位置"
        if "destination" not in params or not params["destination"]:
            params["destination"] = "附近医院"
        
        return params
    
    @classmethod
    def _correct_medication(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """用药咨询参数纠偏"""
        if "drug_name" not in params or not params["drug_name"]:
            # 尝试从其他字段找
            if "medicine" in params:
                params["drug_name"] = params["medicine"]
            elif "drug" in params:
                params["drug_name"] = params["drug"]
            else:
                params["drug_name"] = "常用药物"
        
        return params
    
    @classmethod
    def try_parse_json_arg(cls, arg_str: str) -> Optional[Dict[str, Any]]:
        """
        尝试把字符串参数解析成 JSON。
        有时候 LLM 会把 JSON 对象塞到一个字符串字段里。
        """
        if not isinstance(arg_str, str):
            return None
        
        arg_str = arg_str.strip()
        
        # 看起来像 JSON
        if arg_str.startswith("{") and arg_str.endswith("}"):
            try:
                return json.loads(arg_str)
            except json.JSONDecodeError:
                # 试试把单引号换成双引号
                try:
                    fixed = re.sub(r"'([^']*)'\s*:", r'"\1":', arg_str)
                    fixed = fixed.replace("'", '"')
                    return json.loads(fixed)
                except:
                    pass
        
        return None


# ============================================================
# 3. Never-Throw 工具执行器
# ============================================================

class ToolExecutionResult:
    """工具执行结果（结构化）"""
    
    def __init__(
        self,
        success: bool,
        content: str = "",
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        retry_count: int = 0,
        used_fallback: bool = False,
        duration: float = 0.0,
    ):
        self.success = success
        self.content = content
        self.error_type = error_type
        self.error_message = error_message
        self.retry_count = retry_count
        self.used_fallback = used_fallback
        self.duration = duration
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "content": self.content,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "used_fallback": self.used_fallback,
            "duration": round(self.duration, 3),
        }
    
    def to_agent_message(self) -> str:
        """
        转化为给 Agent 看的消息内容。
        成功返回正常结果，失败返回结构化的错误信息，让 Agent 知道发生了什么。
        """
        if self.success:
            return self.content
        
        # 失败时，返回结构化的错误描述，Agent 可以据此决定下一步
        error_info = {
            "error": True,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "hint": self._get_hint(),
        }
        return json.dumps(error_info, ensure_ascii=False)
    
    def _get_hint(self) -> str:
        """给 Agent 的错误处理建议"""
        hints = {
            "api_error": "外部API调用失败，可尝试更换查询关键词或用其他工具替代。",
            "timeout": "调用超时，可简化查询条件后重试，或使用其他工具。",
            "param_error": "参数格式有误，请检查参数后重新调用。",
            "not_found": "未找到相关结果，可尝试更换关键词或扩大搜索范围。",
            "config_missing": "缺少必要的配置，请检查API Key是否正确设置。",
            "unknown": "未知错误，可尝试换一种方式提问或使用其他工具。",
        }
        return hints.get(self.error_type, "调用失败，请尝试其他方式。")


class NeverThrowToolExecutor:
    """
    Never-Throw 工具执行器。
    
    保证：无论工具内部发生什么异常，都不会向外抛出，
    而是返回一个结构化的 ToolExecutionResult。
    
    执行流程：
        1. 参数纠偏
        2. 调用工具（带指数退避重试）
        3. 失败时尝试降级方案
        4. 返回结构化结果
    """
    
    def __init__(self, tools: List[BaseTool], fallback_providers: Optional[Dict] = None):
        """
        Args:
            tools: 可用工具列表
            fallback_providers: 降级方案映射 {工具名: 降级函数}
        """
        self._tools = {tool.name: tool for tool in tools}
        self._fallback_providers = fallback_providers or {}
        self._corrector = ParameterCorrector()
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """根据名称获取工具"""
        return self._tools.get(tool_name)
    
    def has_tool(self, tool_name: str) -> bool:
        """检查工具是否存在"""
        return tool_name in self._tools
    
    def list_tools(self) -> List[str]:
        """列出所有工具名"""
        return list(self._tools.keys())
    
    def execute(self, tool_name: str, tool_args: Dict[str, Any]) -> ToolExecutionResult:
        """
        执行工具（Never-Throw 保证）。
        
        Args:
            tool_name: 工具名称
            tool_args: 工具参数
            
        Returns:
            ToolExecutionResult 结构化结果
        """
        start_time = time.time()
        retry_count = 0
        
        # 1. 检查工具是否存在
        if tool_name not in self._tools:
            duration = time.time() - start_time
            return ToolExecutionResult(
                success=False,
                error_type="param_error",
                error_message=f"工具 '{tool_name}' 不存在，可用工具: {self.list_tools()}",
                duration=duration,
            )
        
        tool = self._tools[tool_name]
        
        # 2. 参数纠偏
        try:
            corrected_args = self._corrector.correct_parameters(tool_name, dict(tool_args))
            if corrected_args != tool_args:
                logger.info(f"[参数纠偏] {tool_name}: {tool_args} → {corrected_args}")
        except Exception as e:
            logger.warning(f"[参数纠偏失败] {tool_name}: {e}")
            corrected_args = tool_args
        
        # 3. 执行工具（带重试）
        max_retries = 2  # 最多重试2次（共3次调用）
        last_error = None
        last_error_type = "unknown"
        
        for attempt in range(max_retries + 1):
            try:
                result = tool.invoke(corrected_args)
                duration = time.time() - start_time
                
                # 检查结果是不是"空结果"（有些工具失败时返回空字符串或空列表）
                if result is None or (isinstance(result, str) and not result.strip()):
                    last_error = "工具返回空结果"
                    last_error_type = "not_found"
                    if attempt < max_retries:
                        time.sleep(0.5 * (2 ** attempt))
                        retry_count += 1
                        continue
                
                return ToolExecutionResult(
                    success=True,
                    content=str(result),
                    retry_count=retry_count,
                    duration=duration,
                )
            
            except Exception as e:
                last_error = str(e)
                last_error_type = self._classify_error(e)
                retry_count = attempt + 1
                
                logger.warning(
                    f"[工具调用失败] {tool_name} 第{attempt+1}次: "
                    f"{last_error_type} - {e}"
                )
                
                if attempt < max_retries:
                    # 指数退避
                    delay = min(0.5 * (2 ** attempt), 3.0)
                    time.sleep(delay)
        
        # 4. 重试耗尽，尝试降级方案
        duration = time.time() - start_time
        
        if tool_name in self._fallback_providers:
            try:
                fallback_func = self._fallback_providers[tool_name]
                fallback_result = fallback_func(corrected_args)
                logger.info(f"[降级成功] {tool_name} 使用备选方案")
                
                return ToolExecutionResult(
                    success=True,
                    content=str(fallback_result),
                    error_type=last_error_type,
                    error_message=f"主工具失败，已使用降级方案: {last_error}",
                    retry_count=retry_count,
                    used_fallback=True,
                    duration=duration,
                )
            except Exception as fallback_err:
                logger.error(f"[降级失败] {tool_name}: {fallback_err}")
        
        # 5. 彻底失败，返回结构化错误
        return ToolExecutionResult(
            success=False,
            error_type=last_error_type,
            error_message=last_error,
            retry_count=retry_count,
            duration=duration,
        )
    
    def _classify_error(self, exception: Exception) -> str:
        """对异常进行分类，方便 Agent 理解错误类型"""
        error_msg = str(exception).lower()
        
        # 网络/API 错误
        if any(kw in error_msg for kw in ["timeout", "timed out", "连接超时", "请求超时"]):
            return "timeout"
        if any(kw in error_msg for kw in ["connection", "network", "网络", "连接", "http", "request"]):
            return "api_error"
        if any(kw in error_msg for kw in ["401", "403", "unauthorized", "key", "权限", "apikey"]):
            return "config_missing"
        if any(kw in error_msg for kw in ["404", "not found", "不存在"]):
            return "not_found"
        if any(kw in error_msg for kw in ["param", "参数", "invalid", "格式"]):
            return "param_error"
        if any(kw in error_msg for kw in ["rate limit", "限流", "频率", "429"]):
            return "rate_limit"
        
        return "unknown"


# ============================================================
# 4. 工具降级方案（备选 Provider）
# ============================================================

class ToolFallbackProvider:
    """
    工具降级方案提供者。
    
    当主工具（如高德API）调用失败时，提供降级方案：
    - 医院搜索：返回内置的知名医院数据
    - 路线规划：返回简化的路线描述
    - 健康资讯：返回通用健康建议
    """
    
    # 内置的知名医院数据（按城市）
    BUILTIN_HOSPITALS = {
        "北京": [
            {"name": "北京协和医院", "address": "东城区帅府园1号", "phone": "010-69156114",
             "level": "三级甲等", "type": "综合医院", "rating": 4.9,
             "specialties": ["综合内科", "外科", "妇产科", "疑难杂症"]},
            {"name": "北京大学第三医院", "address": "海淀区花园北路49号", "phone": "010-82266699",
             "level": "三级甲等", "type": "综合医院", "rating": 4.8,
             "specialties": ["骨科", "运动医学", "生殖医学", "神经内科"]},
            {"name": "北京大学人民医院", "address": "西城区西直门南大街11号", "phone": "010-88326666",
             "level": "三级甲等", "type": "综合医院", "rating": 4.7,
             "specialties": ["血液科", "骨科", "风湿免疫科", "消化内科"]},
            {"name": "首都医科大学附属北京友谊医院", "address": "西城区永安路95号", "phone": "010-63014411",
             "level": "三级甲等", "type": "综合医院", "rating": 4.7,
             "specialties": ["消化内科", "胃肠镜", "肝胆外科", "热带病"]},
            {"name": "中日友好医院", "address": "朝阳区樱花园东街2号", "phone": "010-84205566",
             "level": "三级甲等", "type": "综合医院", "rating": 4.6,
             "specialties": ["呼吸科", "中西医结合", "肿瘤科", "疼痛科"]},
        ],
        "上海": [
            {"name": "上海交通大学医学院附属瑞金医院", "address": "黄浦区瑞金二路197号", "phone": "021-64370045",
             "level": "三级甲等", "type": "综合医院", "rating": 4.9,
             "specialties": ["血液科", "内分泌科", "心血管内科", "普外科"]},
            {"name": "复旦大学附属中山医院", "address": "徐汇区枫林路180号", "phone": "021-64041990",
             "level": "三级甲等", "type": "综合医院", "rating": 4.8,
             "specialties": ["心内科", "肝外科", "呼吸科", "肾内科"]},
            {"name": "复旦大学附属华山医院", "address": "静安区乌鲁木齐中路12号", "phone": "021-52889999",
             "level": "三级甲等", "type": "综合医院", "rating": 4.8,
             "specialties": ["神经外科", "皮肤科", "神经内科", "手外科"]},
        ],
        "广州": [
            {"name": "中山大学附属第一医院", "address": "越秀区中山二路58号", "phone": "020-87755766",
             "level": "三级甲等", "type": "综合医院", "rating": 4.8,
             "specialties": ["肾内科", "普外科", "神经内科", "心血管内科"]},
            {"name": "南方医科大学南方医院", "address": "白云区广州大道北1838号", "phone": "020-61641888",
             "level": "三级甲等", "type": "综合医院", "rating": 4.7,
             "specialties": ["消化内科", "肾内科", "血液科", "创伤骨科"]},
        ],
        "默认": [
            {"name": "当地三甲综合医院", "address": "请查询当地卫生健康委员会官网", "phone": "120",
             "level": "三级甲等", "type": "综合医院", "rating": 4.5,
             "specialties": ["全科", "内科", "外科", "急诊"]},
        ],
    }
    
    @classmethod
    def hospital_search_fallback(cls, params: Dict[str, Any]) -> str:
        """医院搜索降级方案：返回内置医院数据"""
        city = params.get("city", "北京")
        keyword = params.get("keyword", "")
        
        hospitals = cls.BUILTIN_HOSPITALS.get(city, cls.BUILTIN_HOSPITALS["默认"])
        
        # 简单关键词过滤
        if keyword and keyword not in ("医院", "三甲", ""):
            filtered = []
            for h in hospitals:
                if (keyword in h["name"] or 
                    any(keyword in s for s in h["specialties"]) or
                    keyword in h["type"]):
                    filtered.append(h)
            if filtered:
                hospitals = filtered
        
        result = {
            "hospitals": hospitals[:6],
            "total": min(len(hospitals), 6),
            "city": city,
            "note": "（降级模式）数据来自内置医院库，非实时数据",
        }
        return json.dumps(result, ensure_ascii=False)
    
    @classmethod
    def route_planner_fallback(cls, params: Dict[str, Any]) -> str:
        """路线规划降级方案：返回简化路线描述"""
        origin = params.get("origin", "当前位置")
        destination = params.get("destination", "医院")
        city = params.get("city", "北京")
        strategy = params.get("strategy", "公交")
        
        result = {
            "route": {
                "origin": origin,
                "destination": destination,
                "city": city,
                "strategy": strategy,
                "duration": "约30-60分钟（估算）",
                "distance": "约10-20公里（估算）",
                "steps": [
                    f"从 {origin} 出发，前往 {destination}",
                    f"建议使用 {strategy} 方式出行",
                    "具体路线请使用地图导航APP查询",
                ],
                "note": "（降级模式）路线为估算，建议使用高德/百度地图获取精确导航",
            }
        }
        return json.dumps(result, ensure_ascii=False)
    
    @classmethod
    def health_news_fallback(cls, params: Dict[str, Any]) -> str:
        """健康资讯降级方案：返回通用健康建议"""
        topic = params.get("topic", "健康")
        
        result = {
            "news": [
                {
                    "title": f"秋季{topic}小贴士：注意增减衣物，预防感冒",
                    "summary": "秋季气温变化大，是呼吸道疾病高发期。建议注意保暖，勤洗手，多通风，适当运动增强免疫力。",
                    "source": "健康生活",
                    "date": "2024-10-15",
                },
                {
                    "title": "合理膳食，均衡营养——健康饮食的五大原则",
                    "summary": "1.食物多样，谷类为主；2.吃动平衡，健康体重；3.多吃蔬果、奶类、大豆；4.适量吃鱼、禽、蛋、瘦肉；5.少盐少油，控糖限酒。",
                    "source": "中国居民膳食指南",
                    "date": "2024-10-10",
                },
            ],
            "note": "（降级模式）资讯为内置通用健康建议，非实时新闻",
        }
        return json.dumps(result, ensure_ascii=False)
    
    @classmethod
    def get_all_fallbacks(cls) -> Dict[str, Callable]:
        """获取所有降级方案映射"""
        return {
            "hospital_search": cls.hospital_search_fallback,
            "route_to_hospital": cls.route_planner_fallback,
            "health_news_search": cls.health_news_fallback,
        }
