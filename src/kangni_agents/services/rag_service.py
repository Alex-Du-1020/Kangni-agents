import json
from typing import List, Dict, Any, Optional
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from ..config import settings
from ..models import RAGSearchRequest, RAGSearchResult
from ..models.llm_implementations import llm_service
from ..models.llm_providers import LLMMessage
import logging
import asyncio

logger = logging.getLogger(__name__)

class RAGFlowService:
    def __init__(self):
        self.base_url = settings.ragflow_mcp_server_url
    
    async def check_availability(self) -> bool:
        """检查RAG服务是否可用"""
        try:
            # Add timeout for connection check
            async def _check_connection():
                async with streamablehttp_client(self.base_url) as (read_stream, write_stream, _):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        logger.info("RAG service connection test successful")
                        return True
            
            # Wait max 5 seconds for connection
            return await asyncio.wait_for(_check_connection(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(f"RAG service connection test timed out after 5 seconds")
            return False
        except Exception as e:
            logger.error(f"RAG service connection test failed: {e}")
            return False
    
    async def _search_rag(self, query: str, dataset_id: str, top_k: int = 5) -> List[RAGSearchResult]:
        """调用RAGFlow MCP服务进行文档搜索"""
        try:
            async def _do_search():
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
                        return response
            
            # Add timeout for search operation (30 seconds)
            response = await asyncio.wait_for(_do_search(), timeout=30.0)
                    
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
                                    for i, chunk in enumerate(chunks):
                                        if isinstance(chunk, dict) and i < 5:
                                            if len(chunk) < 5000:
                                                content.append(chunk)
                                            else:
                                                content.append(chunk[:5000])
                                elif isinstance(parsed_data, list):
                                    # 直接是数组格式
                                    content.extend(parsed_data if len(parsed_data) < 5000 else parsed_data[:5000])
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
        except asyncio.TimeoutError:
            logger.error(f"RAG search timed out after 30 seconds for query: {query}")
            raise RuntimeError("RAG service timeout")
        except Exception as e:
            logger.error(f"Error searching RAG: {e}")
            raise RuntimeError(f"RAG service unavailable: {str(e)}")
    
    async def search_db_context(self, query: str) -> Dict[str, List[RAGSearchResult]]:
        """搜索数据库相关的上下文信息"""
        results = {}
        
        # 并行搜索三个数据集
        tasks = [
            ("ddl", self._search_rag(query, settings.db_ddl_dataset_id)),
            ("query_sql", self._search_rag(query, settings.db_query_sql_dataset_id)),
            ("description", self._search_rag(query, settings.db_description_dataset_id))
        ]
        
        for name, task in tasks:
            try:
                results[name] = await task
            except Exception as e:
                logger.error(f"Error searching {name}: {e}")
                raise RuntimeError(f"Failed to search {name} dataset: {str(e)}")
        
        return results
    
    async def generate_answer_with_llm(self, query: str, search_results: List[RAGSearchResult], memory_info: str = "") -> str:
        """使用LLM基于检索到的文档生成答案"""
        try:
            # 构建检索到的内容
            retrieval_content = ""
            for i, result in enumerate(search_results, 1):
                retrieval_content += f"文档 {i}:\n"
                retrieval_content += f"内容: {result.content}\n"
                if result.metadata and result.metadata.get('document_name'):
                    retrieval_content += f"来源: {result.metadata['document_name']}\n"
                retrieval_content += f"相关性评分: {result.score}\n\n"
            
            # 构建LLM提示词
            prompt = f"""角色
您是文档质量检查代理，一位专门的知识库助手，负责严格基于关联的文档存储库提供准确答案。

第一步需要理解用户的问题，根据下面的步骤理解用户的问题。
1. 先检查用户问题是否需要历史记忆来回答  
2. 如果需要，找到与问题最相关的记忆并引用  
3. 再根据当前输入给出最终答案

用户对话历史：{memory_info}
用户问题: {query}

知识库搜索答案： {retrieval_content}

核心原则
仅使用知识库：完全依据从关联知识库中检索到的信息回答问题。
不创造内容：绝不生成、推断或创建未在检索到的文档中明确出现的信息。
来源透明：始终明确标注信息是否来自知识库，或是否不可用。
准确性优于完整性：宁愿提供不完整但准确的答案，也不提供完整但可能不准确的信息。

回答指南
1. 当信息可用时
根据检索到的内容提供直接答案
在有帮助时引用相关部分
如果可用，标注来源文档/部分
使用诸如"根据文档1..."或"基于知识库2..."等短语告诉我们你的来源。
如果结果来自多个文档，就列出来所有结果的来源。
例如：
门板油漆损伤的原因是：
根据文档1，包装箱支撑防护不足
根据文档2，运输过程中存在磕碰

2. 当信息不可用时
明确声明："我在当前知识库中找不到此信息。"
不要尝试用通用知识填补空白
不要任何建议
使用诸如"文档未涵盖"或"此信息在知识库中不可用"等短语

回答格式
[严格基于知识库内容的回答]

**始终遵循以下要求：**  
- 对每个问题使用检索工具  
- 对信息的可用性保持透明  
- 仅遵循记录的事实  
- 承认知识库的局限性  
"""

            # 检查LLM服务是否可用
            if not llm_service.llm_available:
                logger.warning("LLM service not available, returning raw search results")
                return self._format_raw_results(search_results)
            
            # 调用LLM生成答案
            messages = [
                LLMMessage(role="user", content=prompt)
            ]
            
            logger.info(f"Generating answer with LLM for query: {query[:100]}...")
            response = await llm_service.chat(messages)
            
            if response and response.content:
                logger.info(f"RAG search LLM generated answer successfully: {response.content}")
                return response.content
            else:
                logger.warning("LLM returned empty response, falling back to raw results")
                return self._format_raw_results(search_results)
                
        except Exception as e:
            logger.error(f"Error generating answer with LLM: {e}")
            logger.info("Falling back to raw search results")
            return self._format_raw_results(search_results)
    
    def _format_raw_results(self, search_results: List[RAGSearchResult]) -> str:
        """格式化原始搜索结果作为答案"""
        if not search_results:
            return "我在当前知识库中找不到此信息。"
        
        formatted_answer = "基于知识库检索到的信息：\n\n"
        for i, result in enumerate(search_results, 1):
            formatted_answer += f"{i}. {result.content}\n"
            if result.metadata and result.metadata.get('document_name'):
                formatted_answer += f"   (来源: {result.metadata['document_name']})\n"
            formatted_answer += "\n"
        
        return formatted_answer
    
    async def search_rag_with_answer(self, query: str, dataset_id: str, memory_info: str = "", top_k: int = 5) -> Dict[str, Any]:
        """搜索RAG并生成答案"""
        try:
            # 首先进行文档搜索
            search_results = await self._search_rag(query, dataset_id, top_k)
            
            # 然后使用LLM生成答案
            answer = await self.generate_answer_with_llm(query, search_results, memory_info)

            if "文档未涵盖" in answer or "此信息在知识库中不可用" in answer:
                return {
                    "content": "未找到相关文档信息",
                    "rag_results": [],
                    "query": query,
                    "dataset_id": dataset_id,
                    "total_results": 0
                }
            
            # 检查答案中是否引用了文档
            referenced_docs = []
            
            # 检查答案中是否包含文档引用（如"根据文档1"、"基于知识库2"等）
            import re
            doc_refs = re.findall(r'文档\s*(\d+)', answer)
            knowledge_refs = re.findall(r'知识库\s*(\d+)', answer)
            
            # 收集被引用的文档
            all_refs = set()
            for ref in doc_refs + knowledge_refs:
                try:
                    doc_index = int(ref) - 1  # 转换为0-based索引
                    if 0 <= doc_index < len(search_results):
                        all_refs.add(doc_index)
                except ValueError:
                    continue
            
            # 只包含被引用的文档
            for doc_index in sorted(all_refs):
                if doc_index < len(search_results):
                    referenced_docs.append(search_results[doc_index])
            
            return {
                "content": answer,
                "rag_results": referenced_docs,
                "query": query,
                "dataset_id": dataset_id,
                "total_results": len(search_results)
            }
            
        except Exception as e:
            logger.error(f"Error in search_rag_with_answer: {e}")
            raise RuntimeError(f"RAG search with answer generation failed: {str(e)}")
    
    async def close(self):
        """关闭连接（使用streamable client无需手动关闭）"""
        pass

# 全局实例
rag_service = RAGFlowService()
