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
        context_data: Dict[str, List[RAGSearchResult]], 
        memory_info : str = ""
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

第一步需要理解用户的问题，根据下面的步骤理解用户的问题。
1. 先检查用户问题是否需要历史记忆来回答  
2. 如果需要，找到与问题最相关的记忆并引用  
3. 再根据当前输入给出最终答案

用户对话历史：{memory_info}
            
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
4. 考虑查询性能，适当使用索引，限制条件，去重，分组等
5. 如果问题不够明确或缺少必要信息，返回 "INSUFFICIENT_INFO"
6. 如果无法确定使用哪个DDL，返回 "INSUFFICIENT_INFO"

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
                    memory_info=memory_info,
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
            
            # 8. 验证SQL中的字段是否来自同一个DDL
            if not self._validate_sql_field_consistency(final_sql, ddl_context):
                logger.warning(f"Generated SQL contains fields from different DDLs: {final_sql}")
                # 可以在这里添加重试逻辑或者返回更明确的错误信息
                return None
            
            logger.info(f"Generated SQL: {final_sql}")
            return final_sql
            
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            return None
    
    def _validate_sql_field_consistency(self, sql_query: str, ddl_context: str) -> bool:
        """验证SQL查询中的字段是否都来自同一个DDL"""
        try:
            import re
            
            # 提取SQL中的字段名（简化版本，主要检查SELECT和WHERE子句）
            field_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
            sql_fields = set(re.findall(field_pattern, sql_query.upper()))
            
            # 移除SQL关键字
            sql_keywords = {
                'SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'ORDER', 'BY', 'GROUP', 'HAVING',
                'INNER', 'LEFT', 'RIGHT', 'JOIN', 'ON', 'AS', 'ASC', 'DESC', 'LIMIT',
                'COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'DISTINCT', 'CASE', 'WHEN', 'THEN',
                'ELSE', 'END', 'IS', 'NULL', 'NOT', 'IN', 'LIKE', 'BETWEEN', 'EXISTS'
            }
            sql_fields = sql_fields - sql_keywords
            
            if not sql_fields:
                return True  # 没有字段需要验证
            
            # 将DDL上下文按表分割
            ddl_sections = ddl_context.split('\n\n\n')
            
            # 检查每个DDL部分是否包含所有字段
            for ddl_section in ddl_sections:
                if not ddl_section.strip():
                    continue
                    
                ddl_upper = ddl_section.upper()
                # 检查所有字段是否都在这个DDL中
                fields_in_ddl = all(field in ddl_upper for field in sql_fields)
                if fields_in_ddl:
                    logger.info(f"All SQL fields found in DDL section: {sql_fields}")
                    return True
            
            logger.warning(f"SQL fields not found in any single DDL: {sql_fields}")
            return False
            
        except Exception as e:
            logger.error(f"Error validating SQL field consistency: {e}")
            return True  # 验证失败时允许通过，避免阻塞正常流程
    
    async def query_database(self, question: str, memory_info: str = "") -> Dict[str, Any]:
        """完整的数据库查询流程"""
        try:
            
            # 1. 搜索数据库相关上下文
            context_data = await rag_service.search_db_context(question)
            
            # 2. 生成SQL查询
            sql_query = await self.generate_sql_from_context(question, context_data, memory_info)
            
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