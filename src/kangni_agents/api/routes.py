from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from typing import Optional
import logging
import time
import uuid

from ..models import UserQuery, QueryResponse
from ..agents.react_agent import kangni_agent
from ..services.history_service import history_service
from ..models.llm_implementations import llm_service
from ..models.llm_providers import LLMMessage

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
        # Add session_id to the response
        response.session_id = session_id
        
        return response
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=error_message)


@router.post("/llm-answer")
async def llm_answer(
    question: str = Query(..., description="Direct question to answer"),
    user_email: Optional[str] = Query(None, description="User email for context")
):
    """
    Answer a user's question using LLM only (no RAG, no database query)
    
    Args:
        question: Direct question to answer
        user_email: User email for context (optional)
    
    Returns:
        JSON response with LLM answer
        
    Raises:
        HTTPException: If processing fails
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing LLM-only question: {question}")
        
        # Prepare messages for LLM
        messages = [
            LLMMessage(role="system", content="You are a helpful assistant. Answer the user's question directly and concisely."),
            LLMMessage(role="user", content=question)
        ]
        
        # Get LLM response with fallback
        logger.info("Sending question to LLM for direct answer")
        try:
            llm_response = await llm_service.chat(messages)
            answer = llm_response.content
            llm_provider = llm_service.llm_provider
            model_name = llm_service.llm_client.get_model_info().get("model_name") if llm_service.llm_client else None
        except Exception as llm_error:
            logger.warning(f"LLM service failed: {llm_error}, providing fallback response")
            # Provide a fallback response when LLM is not available
            answer = f"I apologize, but I'm currently unable to process your question '{question}' due to LLM service unavailability. Please try again later or contact support."
            llm_provider = "fallback"
            model_name = "fallback"
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Prepare response
        response = {
            "answer": answer,
            "question": question,
            "user_email": user_email,
            "processing_time_ms": processing_time_ms,
            "llm_provider": llm_provider,
            "model_name": model_name,
            "query_type": "llm_only",
            "success": True
        }
        
        logger.info(f"LLM answer generated successfully in {processing_time_ms}ms")
        return response
        
    except Exception as e:
        processing_time_ms = int((time.time() - start_time) * 1000)
        error_message = str(e)
        logger.error(f"Error generating LLM answer: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate LLM answer: {error_message}"
        )


@router.get("/question-from-history")
async def get_question_from_history(
    history_id: int = Query(..., description="History ID to get question from")
):
    """
    Get question from history record (no LLM communication)
    
    Args:
        history_id: History ID to get question from
    
    Returns:
        JSON response with question from history
        
    Raises:
        HTTPException: If history ID not found
    """
    start_time = time.time()
    
    try:
        logger.info(f"Getting question from history ID: {history_id}")
        history_record = await history_service.get_history_by_id(history_id)
        
        if not history_record:
            raise HTTPException(
                status_code=404,
                detail=f"History record with ID {history_id} not found"
            )
        
        # Use rewritten question if available, otherwise use original question
        final_question = history_record.get('rewritten_question') or history_record.get('question')
        if not final_question:
            raise HTTPException(
                status_code=400,
                detail="No question found in the history record"
            )
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Prepare response
        response = {
            "question": final_question,
            "original_question": history_record.get('question'),
            "rewritten_question": history_record.get('rewritten_question'),
            "history_id": history_id,
            "user_email": history_record.get('user_email'),
            "session_id": history_record.get('session_id'),
            "created_at": history_record.get('created_at'),
            "processing_time_ms": processing_time_ms,
            "success": True
        }
        
        logger.info(f"Question retrieved successfully from history ID {history_id} in {processing_time_ms}ms")
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        processing_time_ms = int((time.time() - start_time) * 1000)
        error_message = str(e)
        logger.error(f"Error getting question from history: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get question from history: {error_message}"
        )
