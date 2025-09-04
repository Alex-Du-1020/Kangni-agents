from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from enum import Enum

class QueryType(str, Enum):
    RAG = "rag"
    DATABASE = "database" 
    HYBRID = "hybrid"

class UserQuery(BaseModel):
    question: str
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

class DatabaseQueryRequest(BaseModel):
    question: str
    sql_query: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    query_type: QueryType
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
from .llm_manager import LLMManager, LLMFactory, create_simple_manager, create_multi_provider_manager

__all__ = [
    # 原有模型
    "QueryType", "UserQuery", "QueryResponse", "RAGSearchRequest", 
    "RAGSearchResult", "DatabaseQueryRequest",
    
    # LLM相关模型
    "LLMProvider", "LLMConfig", "LLMMessage", "LLMResponse",
    "OpenAIConfig", "DeepSeekConfig", "AlibabaConfig",
    "BaseLLMProvider", "LLMManager", "LLMFactory", 
    "create_simple_manager", "create_multi_provider_manager"
]