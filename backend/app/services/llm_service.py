"""LLM服务模块"""

from langchain_openai import ChatOpenAI
from ..config import get_settings

_llm_instance = None


def get_llm() -> ChatOpenAI:
    """获取LLM实例(单例模式)"""
    global _llm_instance

    if _llm_instance is None:
        settings = get_settings()

        _llm_instance = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0.3,
            streaming=True,
        )

        print(f"✅ LLM服务初始化成功")
        print(f"   模型: {settings.openai_model}")

    return _llm_instance


def reset_llm():
    global _llm_instance
    _llm_instance = None
