"""
Database models for query history and user feedback
"""
from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum

Base = declarative_base()


class FeedbackType(enum.Enum):
    LIKE = "like"
    DISLIKE = "dislike"


class QueryHistory(Base):
    """
    Store user query history with answers, SQL, and sources
    """
    __tablename__ = "query_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), index=True)
    user_email = Column(String(255), index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text)
    sql_query = Column(Text)  # Store generated SQL if applicable
    sources = Column(JSON)  # Store RAG sources as JSON
    query_type = Column(String(50))  # 'rag', 'database', 'mixed'
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Processing metadata
    processing_time_ms = Column(Integer)  # Time taken to process query
    llm_provider = Column(String(50))  # Which LLM was used
    model_name = Column(String(100))  # Specific model name
    
    # Relationships
    feedback = relationship("UserFeedback", back_populates="query", cascade="all, delete-orphan")
    comments = relationship("UserComment", back_populates="query", cascade="all, delete-orphan")
    
    def to_dict(self):
        """Convert to dictionary for API response"""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "user_email": self.user_email,
            "question": self.question,
            "answer": self.answer,
            "sql_query": self.sql_query,
            "sources": self.sources,
            "query_type": self.query_type,
            "success": self.success,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "processing_time_ms": self.processing_time_ms,
            "llm_provider": self.llm_provider,
            "model_name": self.model_name,
            # Don't include related objects to avoid session issues
            # "feedback": [f.to_dict() for f in self.feedback] if self.feedback else [],
            # "comments": [c.to_dict() for c in self.comments] if self.comments else []
        }


class UserFeedback(Base):
    """
    Store user feedback (like/dislike) for query responses
    """
    __tablename__ = "user_feedback"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    query_id = Column(Integer, ForeignKey("query_history.id"), nullable=False)
    user_email = Column(String(255), nullable=False, index=True)
    feedback_type = Column(SQLEnum(FeedbackType), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    query = relationship("QueryHistory", back_populates="feedback")
    
    def to_dict(self):
        """Convert to dictionary for API response"""
        return {
            "id": self.id,
            "query_id": self.query_id,
            "user_email": self.user_email,
            "feedback_type": self.feedback_type.value if self.feedback_type else None,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class UserComment(Base):
    """
    Store user comments on query responses
    """
    __tablename__ = "user_comments"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    query_id = Column(Integer, ForeignKey("query_history.id"), nullable=False)
    user_email = Column(String(255), nullable=False, index=True)
    comment = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    query = relationship("QueryHistory", back_populates="comments")
    
    def to_dict(self):
        """Convert to dictionary for API response"""
        return {
            "id": self.id,
            "query_id": self.query_id,
            "user_email": self.user_email,
            "comment": self.comment,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }