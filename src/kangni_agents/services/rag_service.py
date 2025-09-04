import json
from typing import List, Dict, Any
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from ..config import settings
from ..models import RAGSearchRequest, RAGSearchResult
import logging
import asyncio

logger = logging.getLogger(__name__)

class RAGFlowService:
    def __init__(self):
        # self.base_url = settings.ragflow_mcp_server_url
         self.base_url = "http://158.58.50.45:9382/mcp"
    
    async def check_availability(self) -> bool:
        """检查RAG服务是否可用"""
        try:
            async with streamablehttp_client(self.base_url) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    logger.info("RAG service connection test successful")
                    return True
        except Exception as e:
            logger.error(f"RAG service connection test failed: {e}")
            return False
    
    async def search_rag(self, query: str, dataset_id: str, top_k: int = 5) -> List[RAGSearchResult]:
        """调用RAGFlow MCP服务进行文档搜索"""
        try:
            async with streamablehttp_client(self.base_url) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    # 调用MCP工具
                    response = await session.call_tool(
                        name="ragflow_retrieval",
                        arguments={
                            "dataset_ids": [dataset_id],
                            "question": query,
                            "top_k": top_k
                        }
                    )
                    
                    # 解析结果 - 处理MCP TextContent响应格式
                    content = []
                    if hasattr(response, 'content') and response.content:
                        # 处理TextContent对象列表
                        if isinstance(response.content, list):
                            for text_content in response.content:
                                if hasattr(text_content, 'text'):
                                    try:
                                        # 解析JSON文本内容
                                        parsed_data = json.loads(text_content.text)
                                        if isinstance(parsed_data, dict) and 'chunks' in parsed_data:
                                            # 处理chunks数组
                                            chunks = parsed_data['chunks']
                                            for chunk in chunks:
                                                if isinstance(chunk, dict):
                                                    content.append(chunk)
                                        elif isinstance(parsed_data, list):
                                            # 直接是数组格式
                                            content.extend(parsed_data)
                                        else:
                                            # 单个对象
                                            content.append(parsed_data)
                                    except json.JSONDecodeError:
                                        # 如果不是JSON，作为普通文本处理
                                        content.append({"content": text_content.text, "score": 1.0})
                                else:
                                    # 直接是字符串
                                    content.append({"content": str(text_content), "score": 1.0})
                        elif isinstance(response.content, str):
                            try:
                                content = json.loads(response.content)
                                if not isinstance(content, list):
                                    content = [content]
                            except json.JSONDecodeError:
                                content = [{"content": response.content, "score": 1.0}]
                    
                    # Debug logging for RAG retrieve results
                    try:
                        logger.debug(f"RAG retrieve raw response: {json.dumps(content, ensure_ascii=False, indent=2)}")
                    except (TypeError, ValueError) as e:
                        logger.debug(f"RAG retrieve raw response (non-serializable): {content}")

                    search_results = []
                    for i, item in enumerate(content):
                        if isinstance(item, dict):
                            # 提取内容、评分和元数据
                            item_content = item.get("content", "")
                            item_score = item.get("similarity", item.get("score", 0.0))
                            item_metadata = {
                                "document_id": item.get("document_id", ""),
                                "document_name": item.get("document_name", ""),
                                "dataset_name": item.get("dataset_name", ""),
                                "highlight": item.get("highlight", ""),
                                "positions": item.get("positions", []),
                                "vector_similarity": item.get("vector_similarity", 0.0),
                                "term_similarity": item.get("term_similarity", 0.0)
                            }
                            
                            result = RAGSearchResult(
                                content=item_content,
                                score=float(item_score),
                                metadata=item_metadata
                            )
                            search_results.append(result)
                            logger.debug(f"RAG result {i+1}: score={result.score}, content_preview={result.content[:100]}...")
                        elif isinstance(item, str):
                            result = RAGSearchResult(
                                content=item,
                                score=1.0
                            )
                            search_results.append(result)
                            logger.debug(f"RAG result {i+1}: score={result.score}, content_preview={result.content[:100]}...")
                    
                    logger.debug(f"RAG retrieve completed: {len(search_results)} results found for query: {query}")
                    return search_results
                    
        except Exception as e:
            logger.error(f"Error searching RAG: {e}")
            raise RuntimeError(f"RAG service unavailable: {str(e)}")
    
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
                raise RuntimeError(f"Failed to search {name} dataset: {str(e)}")
        
        return results
    
    async def close(self):
        """关闭连接（使用streamable client无需手动关闭）"""
        pass

# 全局实例
rag_service = RAGFlowService()
if __name__ == "__main__":
    query = "内解锁接地线线束短，无法安装到紧固螺钉位置是那个项目发生的？"
    search_results = asyncio.run(rag_service.search_rag(query, "f3073258886911f08bc30242c0a82006", top_k=5))
    print(search_results)
