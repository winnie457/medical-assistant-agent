"""配置管理模块"""

import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """应用配置"""

    app_name: str = "智能就医助手"
    app_version: str = "2.0.0"
    debug: bool = False

    host: str = "0.0.0.0"
    port: int = 8000

    cors_origins: str = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000"

    amap_api_key: str = ""

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    tavily_api_key: str = ""

    agent_max_iterations: int = 10

    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

    def get_cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(',')]


settings = Settings()


def get_settings() -> Settings:
    return settings


def validate_config():
    errors = []
    warnings = []

    if not settings.amap_api_key:
        warnings.append("AMAP_API_KEY未配置,地图工具可能无法使用")

    if not settings.openai_api_key:
        errors.append("OPENAI_API_KEY未配置")

    if errors:
        error_msg = "配置错误:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)

    if warnings:
        print("\n⚠️  配置警告:")
        for w in warnings:
            print(f"  - {w}")

    return True


def print_config():
    print(f"应用名称: {settings.app_name}")
    print(f"版本: {settings.app_version}")
    print(f"服务器: {settings.host}:{settings.port}")
    print(f"高德地图API Key: {'已配置' if settings.amap_api_key else '未配置'}")
    print(f"LLM API Key: {'已配置' if settings.openai_api_key else '未配置'}")
    print(f"LLM Base URL: {settings.openai_base_url}")
    print(f"LLM Model: {settings.openai_model}")
    print(f"Agent最大迭代次数: {settings.agent_max_iterations}")
    print(f"日志级别: {settings.log_level}")
