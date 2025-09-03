from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional
import logging

from ..models import UserQuery, QueryResponse
from ..agents.react_agent import kangni_agent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["query"])

@router.post("/query", response_model=QueryResponse)
async def process_query(query: UserQuery):
    """处理用户查询"""
    try:
        logger.info(f"Processing query: {query.question[:100]}...")
        
        response = await kangni_agent.query(
            question=query.question,
            context=query.context
        )
        
        logger.info(f"Query processed successfully, type: {response.query_type}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "kangni-agents"}

@router.get("/config")
async def get_config():
    """获取配置信息（用于调试）"""
    from ..config import settings
    return {
        "ragflow_server": settings.ragflow_mcp_server_url,
        "default_dataset": settings.ragflow_default_dataset_id,
        "db_datasets": {
            "ddl": settings.db_ddl_dataset_id,
            "query_sql": settings.db_query_sql_dataset_id,
            "description": settings.db_description_dataset_id
        }
    }