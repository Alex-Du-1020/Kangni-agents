from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from enum import Enum

class QueryType(str, Enum):
    RAG = "rag"
    DATABASE = "database" 
    HYBRID = "hybrid"

class UserQuery(BaseModel):
    question: str
    user_email: str  # Required for history tracking
    context: Optional[str] = None
    session_id: Optional[str] = None

class RAGSearchRequest(BaseModel):
    query: str
    dataset_id: str
    top_k: int = 5

class RAGSearchResult(BaseModel):
    content: str
    score: float
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def document_id(self) -> str:
        """获取文档ID"""
        return self.metadata.get("document_id", "") if self.metadata else ""
    
    @property
    def document_name(self) -> str:
        """获取文档名称"""
        return self.metadata.get("document_name", "") if self.metadata else ""
    
    @property
    def dataset_name(self) -> str:
        """获取数据集名称"""
        return self.metadata.get("dataset_name", "") if self.metadata else ""

class DatabaseQueryRequest(BaseModel):
    question: str
    sql_query: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    query_type: QueryType
    session_id: Optional[str] = None  # Return the session ID used for this query
    sources: Optional[List[RAGSearchResult]] = None
    sql_query: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None

# LLM相关导入
from .llm_providers import (
    LLMProvider, LLMConfig, LLMMessage, LLMResponse,
    OpenAIConfig, DeepSeekConfig, AlibabaConfig,
    BaseLLMProvider
)

__all__ = [
    # 原有模型
    "QueryType", "UserQuery", "QueryResponse", "RAGSearchRequest", 
    "RAGSearchResult", "DatabaseQueryRequest",
    
    # LLM相关模型
    "LLMProvider", "LLMConfig", "LLMMessage", "LLMResponse",
    "OpenAIConfig", "DeepSeekConfig", "AlibabaConfig",
    "BaseLLMProvider"
]