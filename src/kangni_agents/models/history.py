"""
Database models for query history, user feedback, and memory system
"""
from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON, Enum as SQLEnum, Float, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum

Base = declarative_base()


class FeedbackType(enum.Enum):
    LIKE = "like"
    DISLIKE = "dislike"


class MemoryType(enum.Enum):
    SHORT_TERM = "short_term"  # Recent conversation context (last 5-10 interactions)
    LONG_TERM = "long_term"    # Important patterns and facts
    EPISODIC = "episodic"      # Specific memorable interactions
    SEMANTIC = "semantic"      # Facts and knowledge


class MemoryImportance(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class QueryHistory(Base):
    """
    Store user query history with answers, SQL, and sources
    """
    __tablename__ = "query_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), index=True)
    user_email = Column(String(255), index=True)
    question = Column(Text, nullable=False)
    rewritten_question = Column(Text)  # Store memory-rewritten question if applicable
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
    
    # Memory-related fields
    memory_summary = Column(Text)  # Summary for long-term memory
    context_used = Column(JSON)  # What memory context was used for this query
    importance_score = Column(Float, default=0.5)  # How important this interaction is (0-1)
    
    # Relationships
    feedback = relationship("UserFeedback", back_populates="query", cascade="all, delete-orphan")
    comments = relationship("UserComment", back_populates="query", cascade="all, delete-orphan")
    memories = relationship("Memory", back_populates="source_query", cascade="all, delete-orphan")
    
    def to_dict(self):
        """Convert to dictionary for API response"""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "user_email": self.user_email,
            "question": self.question,
            "rewritten_question": self.rewritten_question,
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
            "memory_summary": self.memory_summary,
            "context_used": self.context_used,
            "importance_score": self.importance_score,
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


class Memory(Base):
    """
    Store memories extracted from user interactions
    """
    __tablename__ = "memories"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_email = Column(String(255), nullable=False, index=True)
    session_id = Column(String(100), index=True)
    
    # Memory content
    content = Column(Text, nullable=False)  # The actual memory content
    embedding = Column(JSON)  # Vector embedding for similarity search (optional)
    
    # Memory metadata
    memory_type = Column(SQLEnum(MemoryType), nullable=False, default=MemoryType.LONG_TERM)
    importance = Column(SQLEnum(MemoryImportance), nullable=False, default=MemoryImportance.MEDIUM)
    relevance_score = Column(Float, default=0.5)  # Current relevance (0-1)
    access_count = Column(Integer, default=0)  # How often this memory is accessed
    
    # Temporal information
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)  # For short-term memories
    
    # Relationships and context
    source_query_id = Column(Integer, ForeignKey("query_history.id"))
    related_entities = Column(JSON)  # List of entities (people, projects, concepts)
    tags = Column(JSON)  # Tags for categorization
    
    # Relationship
    source_query = relationship("QueryHistory", back_populates="memories")
    
    # Indexes for efficient querying
    __table_args__ = (
        Index('idx_user_memory_type', 'user_email', 'memory_type'),
        Index('idx_session_memory', 'session_id', 'memory_type'),
        Index('idx_memory_importance', 'importance', 'relevance_score'),
    )
    
    def to_dict(self):
        """Convert to dictionary for API response"""
        return {
            "id": self.id,
            "user_email": self.user_email,
            "session_id": self.session_id,
            "content": self.content,
            "memory_type": self.memory_type.value if self.memory_type else None,
            "importance": self.importance.value if self.importance else None,
            "relevance_score": self.relevance_score,
            "access_count": self.access_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "source_query_id": self.source_query_id,
            "related_entities": self.related_entities,
            "tags": self.tags
        }