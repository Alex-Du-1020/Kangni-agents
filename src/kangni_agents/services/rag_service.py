import httpx
import json
from typing import List, Dict, Any
from ..config import settings
from ..models import RAGSearchRequest, RAGSearchResult
import logging

logger = logging.getLogger(__name__)

class RAGFlowService:
    def __init__(self):
        self.base_url = settings.ragflow_mcp_server_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def search_rag(self, query: str, dataset_id: str, top_k: int = 5) -> List[RAGSearchResult]:
        """调用RAGFlow MCP服务进行文档搜索"""
        try:
            # 构建MCP请求
            mcp_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "ragflow_retrieval",
                    "arguments": {
                        "query": query,
                        "dataset_id": dataset_id,
                        "top_k": top_k
                    }
                }
            }
            
            response = await self.client.post(
                self.base_url,
                json=mcp_request,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                logger.error(f"RAGFlow API error: {response.status_code} - {response.text}")
                return []
            
            result = response.json()
            
            if "error" in result:
                logger.error(f"RAGFlow MCP error: {result['error']}")
                return []
            
            # 解析结果
            content = result.get("result", {}).get("content", [])
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    content = [{"content": content, "score": 1.0}]
            
            search_results = []
            for item in content:
                if isinstance(item, dict):
                    search_results.append(RAGSearchResult(
                        content=item.get("content", ""),
                        score=item.get("score", 0.0),
                        metadata=item.get("metadata", {})
                    ))
                elif isinstance(item, str):
                    search_results.append(RAGSearchResult(
                        content=item,
                        score=1.0
                    ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching RAG: {e}")
            return []
    
    async def search_db_context(self, query: str) -> Dict[str, List[RAGSearchResult]]:
        """搜索数据库相关的上下文信息"""
        results = {}
        
        # 并行搜索三个数据集
        import asyncio
        tasks = [
            ("ddl", self.search_rag(query, settings.db_ddl_dataset_id)),
            ("query_sql", self.search_rag(query, settings.db_query_sql_dataset_id)),
            ("description", self.search_rag(query, settings.db_description_dataset_id))
        ]
        
        for name, task in tasks:
            try:
                results[name] = await task
            except Exception as e:
                logger.error(f"Error searching {name}: {e}")
                results[name] = []
        
        return results
    
    async def close(self):
        """关闭HTTP客户端"""
        await self.client.aclose()

# 全局实例
rag_service = RAGFlowService()