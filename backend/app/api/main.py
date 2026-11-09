"""FastAPI主应用 - 智能就医助手"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from ..config import get_settings, validate_config, print_config
from .routes import chat

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="基于 LangGraph ReAct Agent 的智能就医助手 API",
    docs_url=None,
    redoc_url="/redoc"
)

SWAGGER_UI_VERSION = "5.17.14"
SWAGGER_CDN = f"https://cdn.bootcdn.net/ajax/libs/swagger-ui/{SWAGGER_UI_VERSION}"


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """自定义 Swagger UI 页面，改用国内可访问的 CDN"""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
        swagger_js_url=f"{SWAGGER_CDN}/swagger-ui-bundle.js",
        swagger_css_url=f"{SWAGGER_CDN}/swagger-ui.css",
    )

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat.router, prefix="/api")


@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    print("\n" + "="*60)
    print(f"🏥 {settings.app_name} v{settings.app_version}")
    print("="*60)

    print_config()

    try:
        validate_config()
        print("\n✅ 配置验证通过")
    except ValueError as e:
        print(f"\n❌ 配置验证失败:\n{e}")
        print("\n请检查.env文件并确保所有必要的配置项都已设置")
        raise

    # 预初始化 Agent
    try:
        from ..agents.medical_agent import get_medical_agent
        get_medical_agent()
        print("\n✅ Agent 预加载完成")
    except Exception as e:
        print(f"\n⚠️  Agent 预加载失败: {e}")

    print("\n" + "="*60)
    print("📚 API文档: http://localhost:8000/docs")
    print("📖 ReDoc文档: http://localhost:8000/redoc")
    print("="*60 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    print("\n" + "="*60)
    print("👋 应用正在关闭...")
    print("="*60 + "\n")


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
        "features": [
            "ReAct Agent (LangGraph)",
            "7+ 医疗工具",
            "流式对话",
            "多轮对话记忆"
        ]
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version
    }
