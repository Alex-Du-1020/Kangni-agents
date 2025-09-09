from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional
import logging
import time

from ..models import UserQuery, QueryResponse
from ..services.history_service import history_service

# Try to import the full react_agent, fallback to minimal version
try:
    from ..agents.react_agent import kangni_agent
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import full react_agent ({e}), using fallback version")
    from ..agents.react_agent_fallback import kangni_agent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/qomo/v1", tags=["query"])

async def _save_query_history(
    query: UserQuery,
    response: Optional[QueryResponse],
    error_message: Optional[str],
    processing_time_ms: int
):
    """
    Save query history to database
    
    Args:
        query: The original user query
        response: The response if successful, None if failed
        error_message: Error message if failed, None if successful
        processing_time_ms: Processing time in milliseconds
    """
    # Extract SQL and sources from response if available
    sql_query = None
    sources = None
    query_type = None
    answer = None
    
    if response:
        # Try to extract SQL from the response
        if hasattr(response, 'sql_query'):
            sql_query = response.sql_query
        elif hasattr(response, 'data') and isinstance(response.data, dict):
            sql_query = response.data.get('sql')
        
        # Try to extract sources from the response
        if hasattr(response, 'sources'):
            sources = response.sources
        elif hasattr(response, 'data') and isinstance(response.data, dict):
            sources = response.data.get('sources')
        
        query_type = response.query_type
        answer = response.answer
    
    # Get LLM info from agent if available
    llm_provider = None
    model_name = None
    if hasattr(kangni_agent, 'llm_provider'):
        llm_provider = str(kangni_agent.llm_provider.value if hasattr(kangni_agent.llm_provider, 'value') else kangni_agent.llm_provider)
    if hasattr(kangni_agent, 'model_name'):
        model_name = kangni_agent.model_name
    
    # Save to history
    await history_service.save_query_history(
        session_id=query.session_id or "default",
        user_email=query.user_email,
        question=query.question,
        answer=answer,
        sql_query=sql_query,
        sources=sources,
        query_type=query_type,
        success=response is not None,
        error_message=error_message,
        processing_time_ms=processing_time_ms,
        llm_provider=llm_provider,
        model_name=model_name
    )

@router.post("/query", response_model=QueryResponse)
async def process_query(query: UserQuery):
    """
    处理用户查询
    
    Args:
        query: UserQuery object containing:
            - question (str): The user's question
            - user_email (str): User's email address (REQUIRED)
            - context (str, optional): Additional context
            - session_id (str, optional): Session identifier
    
    Returns:
        QueryResponse with answer and query type
    
    Raises:
        HTTPException: If user_email is not provided or query processing fails
    """
    # Validate required user_email
    if not query.user_email or not query.user_email.strip():
        raise HTTPException(
            status_code=400, 
            detail="user_email is required for query processing and history tracking"
        )
    
    start_time = time.time()
    response = None
    error_message = None
    
    try:
        logger.info(f"Processing query for user {query.user_email}: {query.question[:100]}...")
        
        response = await kangni_agent.query(
            question=query.question,
            context=query.context
        )
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Query processed successfully, type: {response.query_type}")
        
        return response
        
    except Exception as e:
        processing_time_ms = int((time.time() - start_time) * 1000)
        error_message = str(e)
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=error_message)
    
    finally:
        # Save to history (both success and failure cases)
        try:
            await _save_query_history(
                query=query,
                response=response,
                error_message=error_message,
                processing_time_ms=processing_time_ms
            )
        except Exception as history_error:
            logger.warning(f"Failed to save query history: {history_error}")
            # Don't fail the request if history saving fails

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