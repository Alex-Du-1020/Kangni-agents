from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import sys
import asyncio

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
    
    # 检查RAG服务
    try:
        from .services.rag_service import rag_service
        logger.info("RAG service imported successfully")
        
        # 测试RAG服务连接
        if not await rag_service.check_availability():
            raise RuntimeError("RAG service connection test failed")
        logger.info("RAG service connection test successful")
        
    except ImportError as e:
        logger.error(f"Failed to import RAG service: {e}")
        raise RuntimeError(f"RAG service unavailable: {e}")
    
    # 检查数据库服务
    try:
        from .services.database_service import db_service
        logger.info("Database service imported successfully")
        
        # 测试数据库连接
        try:
            await db_service.get_table_schema()
            logger.info("Database connection test successful")
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            raise RuntimeError(f"Database service unavailable: {e}")
            
    except ImportError as e:
        logger.error(f"Failed to import database service: {e}")
        raise RuntimeError(f"Database service unavailable: {e}")
    
    # 检查LLM服务
    try:
        from .agents.react_agent import kangni_agent
        if not kangni_agent.llm_available:
            raise RuntimeError("LLM service not available")
        logger.info("LLM service available")
    except ImportError as e:
        logger.error(f"Failed to import LLM service: {e}")
        raise RuntimeError(f"LLM service unavailable: {e}")
    
    logger.info("All services are available")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("Starting Kangni Agents application")
    
    # 启动时检查服务可用性
    try:
        await check_service_availability()
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
        lifespan=lifespan
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
    
    @app.get("/health")
    async def health_check():
        """健康检查 - 验证所有服务是否可用"""
        try:
            await check_service_availability()
            return {
                "status": "healthy",
                "message": "All services are available",
                "timestamp": asyncio.get_event_loop().time()
            }
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Service unavailable: {str(e)}"
            )
    
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