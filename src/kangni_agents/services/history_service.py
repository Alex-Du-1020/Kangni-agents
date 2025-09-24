"""
History service for managing query history and user feedback
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy import desc, and_, or_
from sqlalchemy.orm import Session, joinedload
import logging
import time

from ..models.history import QueryHistory, UserFeedback, UserComment, FeedbackType
from ..models.database import get_db_config

logger = logging.getLogger(__name__)


class HistoryService:
    """Service for managing query history and user feedback"""
    
    def __init__(self):
        self.db_config = get_db_config()
    
    async def save_query_history(
        self,
        session_id: str,
        user_email: Optional[str],
        question: str,
        answer: Optional[str],
        rewritten_question: Optional[str] = None,
        sql_query: Optional[str] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
        query_type: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        processing_time_ms: Optional[int] = None,
        llm_provider: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> QueryHistory:
        """
        Save a query and its response to history
        
        Args:
            session_id: Session identifier
            user_email: User's email address
            question: The user's question
            answer: The generated answer
            sql_query: Generated SQL query if applicable
            sources: RAG sources used
            query_type: Type of query (rag, database, mixed)
            success: Whether the query was successful
            error_message: Error message if failed
            processing_time_ms: Time taken to process query
            llm_provider: LLM provider used
            model_name: Specific model name
        
        Returns:
            Created QueryHistory object
        """
        try:
            with self.db_config.session_scope() as session:
                history = QueryHistory(
                    session_id=session_id,
                    user_email=user_email,
                    question=question,
                    rewritten_question=rewritten_question,
                    answer=answer,
                    sql_query=sql_query,
                    sources=sources,
                    query_type=query_type,
                    success=success,
                    error_message=error_message,
                    processing_time_ms=processing_time_ms,
                    llm_provider=llm_provider,
                    model_name=model_name
                )
                session.add(history)
                session.flush()  # Get the ID
                history_id = history.id
                logger.info(f"Saved query history with ID: {history_id}")
                # Return a simple object with the ID to avoid session issues
                return type('QueryHistoryResult', (), {'id': history_id, 'history': history})()
        except Exception as e:
            logger.error(f"Failed to save query history: {e}")
            raise
    
    async def get_user_history(
        self,
        user_email: str,
        limit: int = 50,
        offset: int = 0,
        include_feedback: bool = True,
        include_comments: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get query history for a specific user
        
        Args:
            user_email: User's email address
            limit: Maximum number of results
            offset: Offset for pagination
            include_feedback: Whether to include feedback data
            include_comments: Whether to include comments
        
        Returns:
            List of query history dictionaries
        """
        try:
            with self.db_config.session_scope() as session:
                query = session.query(QueryHistory).filter(
                    QueryHistory.user_email == user_email
                ).order_by(desc(QueryHistory.created_at))
                
                if include_feedback:
                    query = query.options(joinedload(QueryHistory.feedback))
                
                if include_comments:
                    query = query.options(joinedload(QueryHistory.comments))
                
                results = query.limit(limit).offset(offset).all()
                
                # Convert to dictionaries within session scope
                history_dicts = []
                for item in results:
                    item_dict = {
                        "id": item.id,
                        "session_id": item.session_id,
                        "user_email": item.user_email,
                        "question": item.question,
                        "rewritten_question": item.rewritten_question,
                        "answer": item.answer,
                        "sql_query": item.sql_query,
                        "sources": item.sources,
                        "query_type": item.query_type,
                        "success": item.success,
                        "error_message": item.error_message,
                        "created_at": item.created_at.isoformat() if item.created_at else None,
                        "processing_time_ms": item.processing_time_ms,
                        "llm_provider": item.llm_provider,
                        "model_name": item.model_name
                    }
                    
                    if include_feedback and item.feedback:
                        item_dict["feedback"] = [
                            {
                                "id": f.id,
                                "feedback_type": f.feedback_type.value if f.feedback_type else None,
                                "user_email": f.user_email,
                                "created_at": f.created_at.isoformat() if f.created_at else None
                            } for f in item.feedback
                        ]
                    else:
                        item_dict["feedback"] = []
                    
                    if include_comments and item.comments:
                        item_dict["comments"] = [
                            {
                                "id": c.id,
                                "comment": c.comment,
                                "user_email": c.user_email,
                                "created_at": c.created_at.isoformat() if c.created_at else None
                            } for c in item.comments
                        ]
                    else:
                        item_dict["comments"] = []
                    
                    history_dicts.append(item_dict)
                
                logger.info(f"Retrieved {len(history_dicts)} history items for user: {user_email}")
                return history_dicts
        except Exception as e:
            logger.error(f"Failed to get user history: {e}")
            raise
    
    async def get_history_by_session(
        self,
        session_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get query history for a specific session
        
        Args:
            session_id: Session identifier
            limit: Maximum number of results
        
        Returns:
            List of query history dictionaries
        """
        try:
            with self.db_config.session_scope() as session:
                results = session.query(QueryHistory).filter(
                    QueryHistory.session_id == session_id
                ).order_by(desc(QueryHistory.created_at)).limit(limit).all()
                
                # Convert to dictionaries within session scope
                history_dicts = []
                for item in results:
                    history_dicts.append({
                        "id": item.id,
                        "session_id": item.session_id,
                        "user_email": item.user_email,
                        "question": item.question,
                        "answer": item.answer,
                        "sql_query": item.sql_query,
                        "sources": item.sources,
                        "query_type": item.query_type,
                        "success": item.success,
                        "error_message": item.error_message,
                        "created_at": item.created_at.isoformat() if item.created_at else None,
                        "processing_time_ms": item.processing_time_ms,
                        "llm_provider": item.llm_provider,
                        "model_name": item.model_name,
                        "feedback": [],
                        "comments": []
                    })
                
                logger.info(f"Retrieved {len(history_dicts)} history items for session: {session_id}")
                return history_dicts
        except Exception as e:
            logger.error(f"Failed to get session history: {e}")
            raise
    
    async def add_feedback(
        self,
        query_id: int,
        user_email: str,
        feedback_type: str
    ) -> UserFeedback:
        """
        Add user feedback for a query
        
        Args:
            query_id: ID of the query
            user_email: User's email address
            feedback_type: 'like' or 'dislike'
        
        Returns:
            Created UserFeedback object
        """
        try:
            with self.db_config.session_scope() as session:
                # Check if feedback already exists
                existing = session.query(UserFeedback).filter(
                    and_(
                        UserFeedback.query_id == query_id,
                        UserFeedback.user_email == user_email
                    )
                ).first()
                
                if existing:
                    # Update existing feedback
                    existing.feedback_type = FeedbackType(feedback_type)
                    session.flush()
                    feedback_id = existing.id
                    logger.info(f"Updated feedback for query {query_id}")
                    result = type('UserFeedback', (), {'id': feedback_id})()
                    return result
                else:
                    # Create new feedback
                    feedback = UserFeedback(
                        query_id=query_id,
                        user_email=user_email,
                        feedback_type=FeedbackType(feedback_type)
                    )
                    session.add(feedback)
                    session.flush()
                    feedback_id = feedback.id
                    logger.info(f"Added {feedback_type} feedback for query {query_id}")
                    result = type('UserFeedback', (), {'id': feedback_id})()
                    return result
        except Exception as e:
            logger.error(f"Failed to add feedback: {e}")
            raise
    
    async def add_comment(
        self,
        query_id: int,
        user_email: str,
        comment: str
    ) -> UserComment:
        """
        Add user comment for a query
        
        Args:
            query_id: ID of the query
            user_email: User's email address
            comment: Comment text
        
        Returns:
            Created UserComment object
        """
        try:
            with self.db_config.session_scope() as session:
                comment_obj = UserComment(
                    query_id=query_id,
                    user_email=user_email,
                    comment=comment
                )
                session.add(comment_obj)
                session.flush()
                comment_id = comment_obj.id
                logger.info(f"Added comment for query {query_id}")
                result = type('UserComment', (), {'id': comment_id})()
                return result
        except Exception as e:
            logger.error(f"Failed to add comment: {e}")
            raise
    
    async def get_query_feedback_stats(self, query_id: int) -> Dict[str, int]:
        """
        Get feedback statistics for a query
        
        Args:
            query_id: ID of the query
        
        Returns:
            Dictionary with 'likes' and 'dislikes' counts
        """
        try:
            with self.db_config.session_scope() as session:
                likes = session.query(UserFeedback).filter(
                    and_(
                        UserFeedback.query_id == query_id,
                        UserFeedback.feedback_type == FeedbackType.LIKE
                    )
                ).count()
                
                dislikes = session.query(UserFeedback).filter(
                    and_(
                        UserFeedback.query_id == query_id,
                        UserFeedback.feedback_type == FeedbackType.DISLIKE
                    )
                ).count()
                
                return {"likes": likes, "dislikes": dislikes}
        except Exception as e:
            logger.error(f"Failed to get feedback stats: {e}")
            raise
    
    async def search_history(
        self,
        search_term: str,
        user_email: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search query history
        
        Args:
            search_term: Term to search for in questions and answers
            user_email: Optional user email to filter by
            limit: Maximum number of results
        
        Returns:
            List of matching query history dictionaries
        """
        try:
            with self.db_config.session_scope() as session:
                query = session.query(QueryHistory).filter(
                    or_(
                        QueryHistory.question.contains(search_term),
                        QueryHistory.answer.contains(search_term)
                    )
                )
                
                if user_email:
                    query = query.filter(QueryHistory.user_email == user_email)
                
                results = query.order_by(desc(QueryHistory.created_at)).limit(limit).all()
                
                # Convert to dictionaries within session scope
                history_dicts = []
                for item in results:
                    history_dicts.append({
                        "id": item.id,
                        "session_id": item.session_id,
                        "user_email": item.user_email,
                        "question": item.question,
                        "answer": item.answer,
                        "sql_query": item.sql_query,
                        "sources": item.sources,
                        "query_type": item.query_type,
                        "success": item.success,
                        "error_message": item.error_message,
                        "created_at": item.created_at.isoformat() if item.created_at else None,
                        "processing_time_ms": item.processing_time_ms,
                        "llm_provider": item.llm_provider,
                        "model_name": item.model_name,
                        "feedback": [],
                        "comments": []
                    })
                
                logger.info(f"Found {len(history_dicts)} history items matching '{search_term}'")
                return history_dicts
        except Exception as e:
            logger.error(f"Failed to search history: {e}")
            raise
    
    async def get_query_comments(self, query_id: int) -> List[Dict]:
        """
        Get all comments for a specific query
        
        Args:
            query_id: ID of the query
        
        Returns:
            List of comment dictionaries
        """
        try:
            with self.db_config.session_scope() as session:
                comments = session.query(UserComment).filter(
                    UserComment.query_id == query_id
                ).order_by(UserComment.created_at.desc()).all()
                
                comment_dicts = []
                for comment in comments:
                    comment_dicts.append({
                        "id": comment.id,
                        "query_id": comment.query_id,
                        "user_email": comment.user_email,
                        "comment": comment.comment,
                        "created_at": comment.created_at.isoformat() if comment.created_at else None
                    })
                
                logger.info(f"Retrieved {len(comment_dicts)} comments for query {query_id}")
                return comment_dicts
        except Exception as e:
            logger.error(f"Failed to get query comments: {e}")
            raise
    
    async def get_history_by_id(
        self,
        history_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific query history record by ID
        
        Args:
            history_id: The ID of the history record to retrieve
        
        Returns:
            Dictionary containing the history record, or None if not found
        """
        try:
            with self.db_config.session_scope() as session:
                result = session.query(QueryHistory).filter(
                    QueryHistory.id == history_id
                ).first()
                
                if not result:
                    logger.warning(f"History record with ID {history_id} not found")
                    return None
                
                # Convert to dictionary
                history_dict = {
                    "id": result.id,
                    "session_id": result.session_id,
                    "user_email": result.user_email,
                    "question": result.question,
                    "rewritten_question": result.rewritten_question,
                    "answer": result.answer,
                    "sql_query": result.sql_query,
                    "sources": result.sources,
                    "query_type": result.query_type,
                    "success": result.success,
                    "error_message": result.error_message,
                    "created_at": result.created_at.isoformat() if result.created_at else None,
                    "processing_time_ms": result.processing_time_ms,
                    "llm_provider": result.llm_provider,
                    "model_name": result.model_name,
                    "memory_summary": result.memory_summary,
                    "context_used": result.context_used,
                    "importance_score": result.importance_score
                }
                
                logger.info(f"Retrieved history record {history_id}")
                return history_dict
        except Exception as e:
            logger.error(f"Failed to get history by ID {history_id}: {e}")
            raise

    async def get_recent_queries(
        self,
        hours: int = 24,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recent queries within specified hours
        
        Args:
            hours: Number of hours to look back
            limit: Maximum number of results
        
        Returns:
            List of recent query history dictionaries
        """
        try:
            with self.db_config.session_scope() as session:
                cutoff_time = datetime.utcnow() - timedelta(hours=hours)
                results = session.query(QueryHistory).filter(
                    QueryHistory.created_at >= cutoff_time
                ).order_by(desc(QueryHistory.created_at)).limit(limit).all()
                
                # Convert to dictionaries within session scope
                history_dicts = []
                for item in results:
                    history_dicts.append({
                        "id": item.id,
                        "session_id": item.session_id,
                        "user_email": item.user_email,
                        "question": item.question,
                        "answer": item.answer,
                        "sql_query": item.sql_query,
                        "sources": item.sources,
                        "query_type": item.query_type,
                        "success": item.success,
                        "error_message": item.error_message,
                        "created_at": item.created_at.isoformat() if item.created_at else None,
                        "processing_time_ms": item.processing_time_ms,
                        "llm_provider": item.llm_provider,
                        "model_name": item.model_name,
                        "feedback": [],
                        "comments": []
                    })
                
                logger.info(f"Retrieved {len(history_dicts)} queries from last {hours} hours")
                return history_dicts
        except Exception as e:
            logger.error(f"Failed to get recent queries: {e}")
            raise
    
    async def get_dislike_feedback_for_review(
        self,
        days: int = 7,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get queries with dislike feedback and their comments for admin review
        
        Args:
            days: Number of days to look back (1, 3, 5, 7, etc.)
            limit: Maximum number of results
        
        Returns:
            List of query history with dislike feedback and comments
        """
        try:
            with self.db_config.session_scope() as session:
                cutoff_time = datetime.utcnow() - timedelta(days=days)
                
                # Query for all queries that have dislike feedback within the time period
                query = session.query(QueryHistory).join(
                    UserFeedback,
                    QueryHistory.id == UserFeedback.query_id
                ).filter(
                    and_(
                        UserFeedback.feedback_type == FeedbackType.DISLIKE,
                        QueryHistory.created_at >= cutoff_time
                    )
                ).order_by(desc(QueryHistory.created_at))
                
                if limit:
                    query = query.limit(limit)
                
                results = query.all()
                
                # Convert to dictionaries and fetch related data
                review_data = []
                for item in results:
                    # Get dislike feedback for this query
                    dislike_feedback = session.query(UserFeedback).filter(
                        and_(
                            UserFeedback.query_id == item.id,
                            UserFeedback.feedback_type == FeedbackType.DISLIKE
                        )
                    ).all()
                    
                    # Get comments from users who gave dislike feedback
                    dislike_users = [f.user_email for f in dislike_feedback]
                    dislike_comments = []
                    
                    if dislike_users:
                        # Get comments from users who gave dislike feedback
                        comments = session.query(UserComment).filter(
                            and_(
                                UserComment.query_id == item.id,
                                UserComment.user_email.in_(dislike_users)
                            )
                        ).order_by(desc(UserComment.created_at)).all()
                        
                        dislike_comments = [
                            {
                                "user_email": c.user_email,
                                "comment": c.comment,
                                "created_at": c.created_at
                            } for c in comments
                        ]
                    
                    review_data.append({
                        "id": item.id,
                        "session_id": item.session_id,
                        "user_email": item.user_email,
                        "question": item.question,
                        "answer": item.answer,
                        "sql_query": item.sql_query,
                        "sources": item.sources,
                        "query_type": item.query_type,
                        "success": item.success,
                        "error_message": item.error_message,
                        "created_at": item.created_at,
                        "processing_time_ms": item.processing_time_ms,
                        "llm_provider": item.llm_provider,
                        "model_name": item.model_name,
                        "feedback_stats": dislike_comments
                    })
                
                logger.info(f"Retrieved {len(review_data)} queries with dislike feedback from last {days} days")
                return review_data
        except Exception as e:
            logger.error(f"Failed to get dislike feedback for review: {e}")
            raise


# Global history service instance
history_service = HistoryService()