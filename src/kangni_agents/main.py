from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from contextlib import asynccontextmanager
import logging
import sys
import asyncio
import os
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles

from .config import settings
from .api.routes import router
from .api.history_routes import router as history_router

# Import required services
from .services.rag_service import rag_service

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.get_log_level()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("kangni_agents.log")
    ]
)

logger = logging.getLogger(__name__)

async def check_service_availability():
    """检查所有必需服务的可用性"""
    logger.info("Checking service availability...")
    
    service_status = {}
    
    # 检查RAG服务
    try:
        from .services.rag_service import rag_service
        if await rag_service.check_availability():
            service_status["rag"] = "available"
            logger.info("RAG service available")
        else:
            service_status["rag"] = "unavailable"
            logger.error("RAG service unavailable")
    except Exception as e:
        service_status["rag"] = f"error: {e}"
        logger.error(f"RAG service error: {e}")
    
    # 检查数据库服务
    try:
        from .services.database_service import db_service
        await db_service.get_table_schema()
        service_status["database"] = "available"
        logger.info("Database service available")
    except Exception as e:
        service_status["database"] = f"error: {e}"
        logger.error(f"Database service error: {e}")
    
    # 检查LLM服务
    try:
        from .agents.react_agent import kangni_agent
        if kangni_agent.llm_available:
            service_status["llm"] = "available"
            logger.info("LLM service available")
        else:
            service_status["llm"] = "unavailable"
            logger.error("LLM service unavailable")
    except Exception as e:
        service_status["llm"] = f"error: {e}"
        logger.error(f"LLM service error: {e}")
    
    return service_status

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("Starting Kangni Agents application")
    
    # 启动时检查服务可用性
    try:
        service_status = await check_service_availability()
        unavailable_services = [name for name, status in service_status.items() if status != "available"]
        
        if unavailable_services:
            logger.warning(f"❌ Some services are unavailable: {unavailable_services}")
            # 可以选择是否在启动时失败，这里只是警告
        else:
            logger.info("✅ All services are available, application starting successfully")
    except Exception as e:
        logger.error(f"❌ Service availability check failed: {e}")
        raise RuntimeError(f"Application cannot start due to service unavailability: {e}")
    
    yield
    
    logger.info("Shutting down Kangni Agents application")
    
    # 关闭服务
    try:
        await rag_service.close()
    except Exception as e:
        logger.warning(f"Error closing RAG service: {e}")

def create_app() -> FastAPI:
    """创建FastAPI应用"""
    app = FastAPI(
        title="Kangni Agents",
        description="Industrial-grade FastAPI backend agent for Q&A with RAG and database integration",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None,  # Disable default docs to use custom
        redoc_url=None  # Disable redoc
    )

    # Mount static files directory
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        logger.info(f"Mounted static files from {static_dir}")
    
    # Custom Swagger UI with local assets
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=app.title + " - Swagger UI",
            swagger_js_url="/static/swagger-ui/swagger-ui-bundle.js",
            swagger_css_url="/static/swagger-ui/swagger-ui.css",
        )
    
    # CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境中应该限制具体域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 全局异常处理
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Global exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )
    
    # 注册路由
    app.include_router(router)
    app.include_router(history_router)
    
    @app.get("/")
    async def root():
        return {
            "message": "Welcome to Kangni Agents API",
            "version": "0.1.0",
            "docs": "/docs"
        }

    @app.get("/qomo/v1/health")
    async def health():
        """健康检查"""
        return {"status": "healthy", "service": "kangni-agents"}
    
    @app.get("/qomo/v1/healthCheck")
    async def health_check():
        """健康检查 - 验证所有服务是否可用"""
        try:
            service_status = await check_service_availability()
            unavailable_services = [name for name, status in service_status.items() if status != "available"]
            
            if unavailable_services:
                return {
                    "status": "degraded",
                    "message": f"Some services are unavailable: {unavailable_services}",
                    "services": service_status,
                    "timestamp": asyncio.get_event_loop().time()
                }
            else:
                return {
                    "status": "healthy",
                    "message": "All services are available",
                    "services": service_status,
                    "timestamp": asyncio.get_event_loop().time()
                }
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Service check failed: {str(e)}"
            )

    @router.get("/qomo/v1/config")
    async def get_config():
        """获取配置信息（用于调试）"""
        return {
            "ragflow_server": settings.ragflow_mcp_server_url,
            "default_dataset": settings.ragflow_default_dataset_id,
            "db_datasets": {
                "ddl": settings.db_ddl_dataset_id,
                "query_sql": settings.db_query_sql_dataset_id,
                "description": settings.db_description_dataset_id
            }
        }

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.kangni_agents.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )