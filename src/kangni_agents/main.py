from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import sys

from .config import settings
from .api.routes import router
from .services.rag_service import rag_service

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("kangni_agents.log")
    ]
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("Starting Kangni Agents application")
    yield
    logger.info("Shutting down Kangni Agents application")
    await rag_service.close()

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
    
    @app.get("/")
    async def root():
        return {
            "message": "Welcome to Kangni Agents API",
            "version": "0.1.0",
            "docs": "/docs"
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