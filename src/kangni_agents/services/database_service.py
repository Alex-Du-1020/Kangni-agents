from typing import Optional, Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from ..config import settings
from ..models import RAGSearchResult
from .rag_service import rag_service
import logging
import json

logger = logging.getLogger(__name__)

class DatabaseService:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model="gpt-4",
            temperature=0
        )
    
    async def generate_sql_from_context(self, question: str, context_data: Dict[str, List[RAGSearchResult]]) -> Optional[str]:
        """基于RAG搜索结果生成SQL查询"""
        try:
            # 构建上下文信息
            ddl_context = "\n".join([r.content for r in context_data.get("ddl", [])])
            query_examples = "\n".join([r.content for r in context_data.get("query_sql", [])])
            db_description = "\n".join([r.content for r in context_data.get("description", [])])
            
            system_prompt = """你是一个专业的SQL生成助手。基于提供的数据库结构、查询示例和描述信息，为用户问题生成准确的SQL查询。

要求：
1. 只返回SQL查询语句，不要添加额外的解释
2. 确保SQL语法正确
3. 使用提供的表结构和字段名
4. 考虑查询性能，适当使用索引和限制条件
5. 如果问题不够明确或缺少必要信息，返回 "INSUFFICIENT_INFO"

数据库结构信息：
{ddl_context}

查询示例：
{query_examples}

数据库描述：
{db_description}
"""
            
            human_prompt = f"用户问题：{question}\n\n请生成对应的SQL查询："
            
            messages = [
                SystemMessage(content=system_prompt.format(
                    ddl_context=ddl_context,
                    query_examples=query_examples,
                    db_description=db_description
                )),
                HumanMessage(content=human_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            sql_query = response.content.strip()
            
            if sql_query == "INSUFFICIENT_INFO":
                return None
                
            return sql_query
            
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            return None
    
    async def execute_sql_query(self, sql_query: str) -> Optional[List[Dict[str, Any]]]:
        """执行SQL查询（这里需要集成你的数据库MCP服务）"""
        # TODO: 集成数据库MCP服务执行SQL查询
        # 这里暂时返回模拟结果
        logger.info(f"Would execute SQL: {sql_query}")
        return [{"result": "This is a placeholder for database query results"}]
    
    async def query_database(self, question: str) -> Dict[str, Any]:
        """完整的数据库查询流程"""
        try:
            # 1. 搜索数据库相关上下文
            context_data = await rag_service.search_db_context(question)
            
            # 2. 生成SQL查询
            sql_query = await self.generate_sql_from_context(question, context_data)
            
            if not sql_query:
                return {
                    "success": False,
                    "error": "无法生成SQL查询，问题可能不够明确或缺少相关数据库信息",
                    "context_data": context_data
                }
            
            # 3. 执行SQL查询
            query_results = await self.execute_sql_query(sql_query)
            
            return {
                "success": True,
                "sql_query": sql_query,
                "results": query_results,
                "context_data": context_data
            }
            
        except Exception as e:
            logger.error(f"Error in database query: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# 全局实例
db_service = DatabaseService()