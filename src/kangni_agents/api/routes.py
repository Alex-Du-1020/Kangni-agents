from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional
import logging
import time
import uuid

from ..models import UserQuery, QueryResponse
from ..agents.react_agent import kangni_agent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/qomo/v1", tags=["query"])

@router.post("/query", response_model=QueryResponse)
async def process_query(query: UserQuery):
    """
    处理用户查询
    
    Args:
        query: UserQuery object containing:
            - question (str): The user's question
            - user_email (str): User's email address (REQUIRED)
            - context (str, optional): Additional context
            - session_id (str, optional): Session identifier (auto-generated if empty)
    
    Returns:
        QueryResponse with:
            - answer (str): The response answer
            - query_type (str): Type of query processed
            - session_id (str): Session ID used for this query (generated if not provided)
            - sources (list, optional): RAG search results
            - sql_query (str, optional): Generated SQL query
            - confidence (float, optional): Response confidence score
            - reasoning (str, optional): Reasoning process
    
    Raises:
        HTTPException: If user_email is not provided or query processing fails
    """
    # Validate required user_email
    if not query.user_email or not query.user_email.strip():
        raise HTTPException(
            status_code=400, 
            detail="user_email is required for query processing and history tracking"
        )
    
    # Generate session_id if not provided or empty
    session_id = query.session_id
    if not session_id or not session_id.strip():
        session_id = str(uuid.uuid4())
        logger.info(f"Generated new session_id: {session_id} for user {query.user_email}")
    
    start_time = time.time()
    response = None
    error_message = None
    
    try:
        logger.info(f"Processing query for user {query.user_email} (session: {session_id}): {query.question[:100]}...")
        
        response = await kangni_agent.query(
            question=query.question,
            context=query.context,
            user_email=query.user_email,
            session_id=session_id
        )
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Query processed successfully, type: {response.query_type}")
        
        # Add session_id to the response
        response.session_id = session_id
        
        return response
        
    except Exception as e:
        processing_time_ms = int((time.time() - start_time) * 1000)
        error_message = str(e)
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=error_message)
