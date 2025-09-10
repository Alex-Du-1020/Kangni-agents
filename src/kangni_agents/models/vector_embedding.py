"""
Database models for vector embeddings.
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, UniqueConstraint, Index, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime
import os

Base = declarative_base()

# Determine database type
db_type = os.getenv('DB_TYPE', 'sqlite').lower()

if db_type == 'postgresql':
    try:
        # Try to use pgvector if available
        from pgvector.sqlalchemy import Vector
        # BGE-M3 typically generates 1024-dimensional embeddings
        EmbeddingColumn = Vector(1024)
    except ImportError:
        # Fallback to ARRAY if pgvector is not installed
        from sqlalchemy.dialects.postgresql import ARRAY
        EmbeddingColumn = ARRAY(Float)
else:
    # For SQLite, use Text to store JSON
    EmbeddingColumn = Text

class FieldValueEmbedding(Base):
    """
    Table for storing unique field values and their embeddings.
    This table saves distinct values to avoid duplicating embeddings.
    """
    __tablename__ = 'field_value_embeddings'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    table_name = Column(String(255), nullable=False)
    field_name = Column(String(255), nullable=False)
    field_value = Column(Text, nullable=False)
    embedding = Column(EmbeddingColumn, nullable=False)  # Vector embedding or JSON
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Unique constraint to prevent duplicate embeddings
    __table_args__ = (
        UniqueConstraint('table_name', 'field_name', 'field_value', name='uq_table_field_value'),
        Index('idx_table_field', 'table_name', 'field_name'),
    )

class RecordFieldMapping(Base):
    """
    Table for mapping actual records to their field values.
    This links business data rows to their embedded field values.
    """
    __tablename__ = 'record_field_mappings'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    record_id = Column(String(255), nullable=False)  # Business record ID
    table_name = Column(String(255), nullable=False)
    field_name = Column(String(255), nullable=False)
    field_value = Column(Text, nullable=False)
    embedding_id = Column(Integer, nullable=False)  # References FieldValueEmbedding.id
    created_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('idx_record_table', 'record_id', 'table_name'),
        Index('idx_embedding_ref', 'embedding_id'),
        Index('idx_table_field_mapping', 'table_name', 'field_name'),
    )