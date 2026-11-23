"""构建评测用的 Agent 实例。

与线上唯一的差别是 temperature：
    线上 llm_service 固定 0.3；评测必须可复现，所以这里用同一个模型/端点
    新建一个 temperature=0 的实例，并替换 llm_service 的单例，让 MedicalAgent
    在初始化时直接用上。业务代码一行不改。
"""

from __future__ import annotations

from typing import Any, Tuple


def build_agent(temperature: float = 0.0) -> Tuple[Any, Any]:
    """返回 (MedicalAgent, llm)。"""
    from langchain_openai import ChatOpenAI

    from app.agents.medical_agent import MedicalAgent
    from app.config import get_settings
    from app.services import llm_service

    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=temperature,
        streaming=True,
    )
    # 让 MedicalAgent.__init__ 里的 get_llm() 复用这个确定性实例
    llm_service._llm_instance = llm
    return MedicalAgent(), llm


def model_label() -> str:
    from app.config import get_settings

    settings = get_settings()
    return f"{settings.openai_model} @ {settings.openai_base_url}"
