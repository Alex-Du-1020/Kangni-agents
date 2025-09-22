from typing import Optional, Dict, Any, List
from ..config import settings
from ..models import RAGSearchResult
from ..models.llm_implementations import llm_service
from ..models.llm_providers import LLMMessage
from ..utils.query_preprocessor import query_preprocessor
from .rag_service import rag_service
import logging
import json
import os
from mysql.connector import connect, Error
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class DatabaseService:
    def __init__(self):
        # 加载环境变量
        load_dotenv()
        
        # 使用集中式LLM服务
        self.llm_available = llm_service.llm_available
        self.llm_provider = llm_service.llm_provider
        
        if self.llm_available:
            logger.info(f"Database service initialized with LLM provider: {self.llm_provider}")
        else:
            logger.warning("LLM service not available, database LLM features will be disabled")
    
    def get_db_config(self) -> Dict[str, Any]:
        """获取数据库配置"""
        config = {
            "host": os.getenv("MYSQL_HOST") or settings.mysql_host,
            "user": os.getenv("MYSQL_USER") or settings.mysql_user,
            "password": os.getenv("MYSQL_PASSWORD") or settings.mysql_password,
            "database": os.getenv("MYSQL_DATABASE") or settings.mysql_database,
            "port": int(os.getenv("MYSQL_PORT", settings.mysql_port)),
            "auth_plugin": "mysql_native_password",
            "charset": "utf8mb4",
            "use_unicode": True,
            "autocommit": True
        }
        
        # 检查关键配置
        if not all([config["user"], config["password"], config["database"]]):
            logger.error("Missing required database configuration. Please check environment variables:")
            logger.error("MYSQL_USER, MYSQL_PASSWORD, and MYSQL_DATABASE are required")
            raise ValueError("Missing required database configuration")
        
        return config
    
    async def get_table_schema(self) -> str:
        """获取数据库表结构信息"""
        config = self.get_db_config()
        try:
            with connect(**config) as conn:
                with conn.cursor() as cur:
                    cur.execute("SHOW TABLES")
                    tables = [row[0] for row in cur.fetchall()]
                    schema_info = []
                    for table in tables:
                        cur.execute(f"SHOW COLUMNS FROM {table}")
                        columns = [row[0] for row in cur.fetchall()]
                        schema_info.append(f"表 {table} 字段: {', '.join(columns)}")
                    return '\n'.join(schema_info)
        except Error as e:
            logger.error(f"Error getting schema info: {e}")
            raise RuntimeError(f"Database connection failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error getting schema info: {e}")
            raise RuntimeError(f"Database service error: {str(e)}")
    
    async def execute_sql_query(self, sql_query: str) -> List[Dict[str, Any]]:
        """执行SQL查询并返回结果"""
        config = self.get_db_config()
        try:
            with connect(**config) as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute(sql_query)
                    
                    if sql_query.strip().upper().startswith("SELECT"):
                        results = cursor.fetchall()
                        return results
                    else:
                        conn.commit()
                        return [{"rows_affected": cursor.rowcount, "message": "Query executed successfully"}]
                        
        except Error as e:
            logger.error(f"Error executing SQL '{sql_query}': {e}")
            raise RuntimeError(f"Database query failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error executing SQL: {e}")
            raise RuntimeError(f"Database service error: {str(e)}")
    
    async def generate_sql_from_context(self, question: str, 
        context_data: Dict[str, List[RAGSearchResult]]
        ) -> Optional[str]:

        """基于RAG搜索结果生成SQL查询"""
        if not self.llm_available:
            logger.warning("LLM not available, cannot generate SQL from context")
            return None
            
        try:
            # 1. 预处理查询，提取特殊标记
            preprocessed = await query_preprocessor.preprocess_query(question)
            logger.info(f"Query preprocessing completed: {len(preprocessed.entities)} entities found")
            
            # 2. 构建上下文信息
            ddl_context = "\n\n\n".join([r.content for r in context_data.get("ddl", [])])
            query_examples = "\n\n\n".join([r.content for r in context_data.get("query_sql", [])])
            db_description = "\n\n\n".join([r.content for r in context_data.get("description", [])])
            
            # 3. 构建基础系统提示
            base_system_prompt = f"""你是一个专业的SQL生成助手。
            
基于提供的数据库结构、查询示例和描述信息，为用户问题生成准确的SQL查询。

【关键要求 - 字段一致性验证】：
1. 在生成SQL之前，必须先选择一个完整的DDL语句作为参考
2. 确保SQL中使用的所有字段名都来自同一个DDL语句中的表结构
3. 如果多个DDL包含相似表名，必须明确选择其中一个DDL作为字段来源
4. 禁止混合使用不同DDL中的字段，即使字段名相同
5. 在生成SQL前，先列出选择的DDL和对应的表结构，确保字段一致性

必须遵守这些要求：
1. 只返回SQL查询语句，不要添加额外的解释
2. 确保SQL语法正确
3. 字段名必须全部来自同一个DDL语句，不能跨DDL混用字段
4. 考虑查询准确性尽量使用like，字段值需要使用%包裹，而不是 = 
5. 考虑查询性能，适当使用索引，限制条件，去重，分组等
6. 如果问题不够明确或缺少必要信息，返回 "INSUFFICIENT_INFO"
7. 如果无法确定使用哪个DDL，返回 "INSUFFICIENT_INFO"

特别注意：当用户提到"订单"但没有指定具体类型时，默认查询表 kn_quality_trace_prod_order（生产订单表）

数据库表结构信息（可能包含多个DDL，请选择一个完整的DDL使用）：
{ddl_context}

查询示例：
{query_examples}

数据库描述：
{db_description}

"""
            
            # 4. 使用预处理器增强提示词
            enhanced_system_prompt = query_preprocessor.build_enhanced_prompt(
                base_system_prompt.format(
                    ddl_context=ddl_context,
                    query_examples=query_examples,
                    db_description=db_description
                ),
                preprocessed
            )
            
            # 5. 构建人类提示，使用预处理后的查询
            human_prompt = f"""用户问题：{preprocessed.processed_query}

请按照以下步骤生成SQL查询：

步骤1：分析提供的DDL上下文，选择一个最相关的完整DDL语句
步骤2：列出选择的DDL中的表结构和字段信息
步骤3：确认所有要使用的字段都来自同一个DDL
步骤4：生成SQL查询

请直接返回SQL查询语句：
            """
            
            # 6. 调用集中式LLM服务生成SQL
            sql_query = await llm_service.chat_with_system_prompt(
                enhanced_system_prompt,
                human_prompt
            )
            sql_query = sql_query.strip()
            
            if sql_query == "INSUFFICIENT_INFO":
                return None
            
            # 7. 恢复占位符为原始值
            final_sql = query_preprocessor.restore_placeholders_in_sql(
                sql_query, 
                preprocessed.placeholders
            )
            
            logger.info(f"Generated SQL: {final_sql}")
            return final_sql
            
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            return None
    
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