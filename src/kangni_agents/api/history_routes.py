"""
API routes for query history and user feedback
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

from ..services.history_service import history_service
from ..models.history import QueryHistory

router = APIRouter(prefix="/api/v1/history", tags=["history"])


# Request/Response models
class HistoryResponse(BaseModel):
    id: int
    session_id: str
    user_email: Optional[str]
    question: str
    answer: Optional[str]
    sql_query: Optional[str]
    sources: Optional[List]
    query_type: Optional[str]
    success: bool
    error_message: Optional[str]
    created_at: datetime
    processing_time_ms: Optional[int]
    llm_provider: Optional[str]
    model_name: Optional[str]
    feedback: Optional[List]
    comments: Optional[List]


class FeedbackRequest(BaseModel):
    query_id: int
    user_email: str  # Changed from EmailStr to str
    feedback_type: str  # "like" or "dislike"


class CommentRequest(BaseModel):
    query_id: int
    user_email: str  # Changed from EmailStr to str
    comment: str


class FeedbackStatsResponse(BaseModel):
    query_id: int
    likes: int
    dislikes: int


class CommentResponse(BaseModel):
    id: int
    query_id: int
    user_email: str
    comment: str
    created_at: datetime


@router.get("/user/{user_email}", response_model=List[HistoryResponse])
async def get_user_history(
    user_email: str,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    include_feedback: bool = Query(default=True),
    include_comments: bool = Query(default=True)
):
    """
    Get query history for a specific user
    
    Args:
        user_email: User's email address
        limit: Maximum number of results (1-500)
        offset: Offset for pagination
        include_feedback: Include feedback data
        include_comments: Include comments
    
    Returns:
        List of query history items
    """
    try:
        history_items = await history_service.get_user_history(
            user_email=user_email,
            limit=limit,
            offset=offset,
            include_feedback=include_feedback,
            include_comments=include_comments
        )
        # history_items are already dictionaries, return them directly
        return history_items
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}", response_model=List[HistoryResponse])
async def get_session_history(
    session_id: str,
    limit: int = Query(default=50, ge=1, le=500)
):
    """
    Get query history for a specific session
    
    Args:
        session_id: Session identifier
        limit: Maximum number of results (1-500)
    
    Returns:
        List of query history items from the session
    """
    try:
        history_items = await history_service.get_history_by_session(
            session_id=session_id,
            limit=limit
        )
        # history_items are already dictionaries, return them directly
        return history_items
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search", response_model=List[HistoryResponse])
async def search_history(
    q: str = Query(..., description="Search term"),
    user_email: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500)
):
    """
    Search query history
    
    Args:
        q: Search term to look for in questions and answers
        user_email: Optional user email to filter by
        limit: Maximum number of results (1-500)
    
    Returns:
        List of matching query history items
    """
    try:
        history_items = await history_service.search_history(
            search_term=q,
            user_email=user_email,
            limit=limit
        )
        # history_items are already dictionaries, return them directly
        return history_items
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recent", response_model=List[HistoryResponse])
async def get_recent_queries(
    hours: int = Query(default=24, ge=1, le=168),  # Max 1 week
    limit: int = Query(default=100, ge=1, le=500)
):
    """
    Get recent queries within specified hours
    
    Args:
        hours: Number of hours to look back (1-168)
        limit: Maximum number of results (1-500)
    
    Returns:
        List of recent query history items
    """
    try:
        history_items = await history_service.get_recent_queries(
            hours=hours,
            limit=limit
        )
        # history_items are already dictionaries, return them directly
        return history_items
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def add_feedback(feedback: FeedbackRequest):
    """
    Add or update user feedback for a query
    
    Args:
        feedback: Feedback details including query_id, user_email, and feedback_type
    
    Returns:
        Success message
    """
    if feedback.feedback_type not in ["like", "dislike"]:
        raise HTTPException(
            status_code=400,
            detail="feedback_type must be 'like' or 'dislike'"
        )
    
    try:
        result = await history_service.add_feedback(
            query_id=feedback.query_id,
            user_email=feedback.user_email,
            feedback_type=feedback.feedback_type
        )
        return {
            "message": f"Feedback {feedback.feedback_type} added successfully",
            "feedback_id": result.id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/comments/{query_id}", response_model=List[CommentResponse])
async def get_query_comments(query_id: int):
    """
    Get all comments for a specific query
    
    Args:
        query_id: ID of the query
    
    Returns:
        List of comments for the query
    """
    try:
        comments = await history_service.get_query_comments(query_id)
        return comments
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/comment")
async def add_comment(comment: CommentRequest):
    """
    Add user comment for a query
    
    Args:
        comment: Comment details including query_id, user_email, and comment text
    
    Returns:
        Success message with comment ID
    """
    try:
        result = await history_service.add_comment(
            query_id=comment.query_id,
            user_email=comment.user_email,
            comment=comment.comment
        )
        return {
            "message": "Comment added successfully",
            "comment_id": result.id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CommentResponse(BaseModel):
    id: int
    query_id: int
    user_email: str
    comment: str
    created_at: datetime


@router.get("/feedback/stats/{query_id}", response_model=FeedbackStatsResponse)
async def get_feedback_stats(query_id: int):
    """
    Get feedback statistics for a specific query
    
    Args:
        query_id: ID of the query
    
    Returns:
        Feedback statistics including likes and dislikes counts
    """
    try:
        stats = await history_service.get_query_feedback_stats(query_id)
        return {
            "query_id": query_id,
            **stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))