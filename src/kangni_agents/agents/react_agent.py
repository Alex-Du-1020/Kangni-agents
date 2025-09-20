from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import time

from ..config import settings
from ..models import QueryType, RAGSearchResult, QueryResponse
from ..models.llm_implementations import llm_service
from ..models.llm_providers import LLMMessage
from ..services.rag_service import rag_service
from ..services.database_service import db_service
from ..services.vector_embedding_service import vector_service
from ..services.history_service import history_service
from ..services.memory_service import memory_service
from ..utils.intent_classifier import intent_classifier

import logging
import json

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    intent: Optional[QueryType]
    # RAG-related fields
    rag_results: Optional[List[RAGSearchResult]]
    rag_content: Optional[str]
    rag_has_results: bool
    # Database-related fields (also used for vector search results)
    db_results: Optional[Dict[str, Any]]
    sql_query: Optional[str]
    db_success: bool
    db_results_valid: bool
    needs_vector_search: bool
    formatted_db_results: Optional[str]
    result_too_large: bool
    retry_attempted: bool
    # Response fields
    final_answer: Optional[str]
    source_links: Optional[List[str]]
    # Memory-related fields
    memory_info: Optional[str]
    user_email: Optional[str]
    session_id: Optional[str]
    start_time: Optional[float]

@tool
async def rag_search_tool(query: str, memory_info: str = "") -> Dict[str, Any]:
    """搜索RAG文档库获取相关信息 - 支持多个数据集"""
    # 直接使用RAG服务，它内部会处理多个数据集
    result = await rag_service.search_rag_with_answer(query, memory_info, top_k=8)
    return result

@tool 
async def database_query_tool(question: str, memory_info: str = "",) -> Dict[str, Any]:
    """查询数据库获取统计信息"""
    # 直接执行数据库查询
    result = await db_service.query_database(question, memory_info)
    
    # 返回格式化结果
    if result.get("success"):
        return format_db_results(result)
    else:
        return {
            "content": f"数据库查询失败: {result.get('error', '未知错误')}",
            "sql_query": result.get("sql_query"),
            "results": [],
            "success": False,
            "error": result.get("error")
        }

@tool
async def vector_database_query_tool(question: str, failed_sql: str = None, memory_info: str = "") -> Dict[str, Any]:
    """使用向量搜索增强数据库查询，找到实际存在的值并重新生成SQL"""
    import yaml
    from pathlib import Path
    from ..utils.sql_parser import SQLParser
    
    logger.info(f"Starting vector-enhanced database query for: {question}")
    
    try:
        # Load vector search configuration
        config_path = Path(__file__).parent.parent / "config" / "vector_search_config.yaml"
        vector_config = {}
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                vector_config = yaml.safe_load(f)
        else:
            logger.warning("Vector search config not found, using defaults")
            return {
                "content": "向量搜索配置文件未找到",
                "success": False,
                "error": "Missing vector_search_config.yaml"
            }
        
        # Parse the failed SQL to identify tables and fields
        sql_parser = SQLParser()
        parsed_sql = {}
        if failed_sql:
            parsed_sql = sql_parser.parse_sql(failed_sql)
            logger.info(f"Parsed SQL - tables: {parsed_sql.get('tables')}, fields: {parsed_sql.get('fields')}")
        
        # Collect suggestions for all relevant fields
        all_suggestions = {}
        vector_fields = vector_config.get('vector_search_fields', [])
        
        for field_config in vector_fields:
            table = field_config['table']
            field = field_config['field']
            description = field_config.get('description', field)
            
            # Check if this field is relevant to the query
            should_search = False
            
            # If we have a parsed SQL, check if the table/field is mentioned
            if parsed_sql:
                if table in parsed_sql.get('tables', []) or field in parsed_sql.get('fields', []):
                    should_search = True
                    logger.info(f"Field {table}.{field} found in failed SQL")
            
            # Also check if keywords from the question match this field's keywords
            keywords = field_config.get('keywords', [])
            question_lower = question.lower()
            for keyword in keywords:
                if keyword.lower() in question_lower:
                    should_search = True
                    logger.info(f"Keyword '{keyword}' found in question for field {table}.{field}")
                    break
            
            if should_search:
                logger.info(f"Searching for similar values in {table}.{field} ({description})")
                
                # Get similar values from the database
                suggestions = await vector_service.search_similar_values(
                    search_text=question,
                    table_name=table,
                    field_name=field,
                    top_k=settings.max_suggestions or 3,  # This is already the max_suggestions value
                    similarity_threshold=settings.similarity_threshold or 0.3  # This is already the threshold value
                )
                
                if suggestions:
                    # Limit suggestions to max_suggestions
                    limited_suggestions = suggestions[:settings.max_suggestions]
                    # Extract just the field values for display
                    field_values = [suggestion['field_value'] for suggestion in limited_suggestions]
                    logger.info(f"Found {len(limited_suggestions)} suggestions for {description}: {field_values[:3]}...")
                    all_suggestions[f"{table}.{field}"] = {
                        "values": field_values,
                        "description": description,
                        "table": table,
                        "field": field
                    }
        
        if not all_suggestions:
            logger.info("No vector search suggestions found")
            return {
                "content": "未找到相似的数据库值",
                "success": False,
                "suggestions": {},
                "message": "向量搜索未找到匹配的值"
            }
        
        # Build enhanced prompt with suggestions
        suggestion_text = "\n\n基于向量搜索找到的数据库实际值：\n"
        for field_key, field_data in all_suggestions.items():
            suggestion_text += f"- {field_data['description']} ({field_data['table']}.{field_data['field']}): "
            suggestion_text += f"{', '.join(field_data['values'])}"
            if len(field_data['values']) > 3:
                suggestion_text += f" 等{len(field_data['values'])}个值"
            suggestion_text += "\n"
        
        enhanced_question = f"{question}{suggestion_text}\n请使用这些实际存在的值重新生成SQL查询。如果有多个匹配值，使用最相似的值来查询。"
        
        # Generate new SQL with enhanced context
        logger.info("Generating new SQL with vector search suggestions")
        enhanced_result = await db_service.query_database(enhanced_question, memory_info)
        
        if enhanced_result.get("success"):
            logger.info(f"Vector-enhanced query successful, got {len(enhanced_result.get('results', []))} results")
            enhanced_result["vector_enhanced"] = True
            enhanced_result["suggestions_used"] = all_suggestions
            return format_db_results(enhanced_result)
        else:
            logger.warning(f"Vector-enhanced query failed: {enhanced_result.get('error')}")
            return {
                "content": f"向量增强查询失败: {enhanced_result.get('error', '未知错误')}",
                "success": False,
                "suggestions": all_suggestions,
                "sql_query": enhanced_result.get("sql_query"),
                "error": enhanced_result.get("error")
            }
            
    except Exception as e:
        logger.error(f"Error in vector database query tool: {e}")
        return {
            "content": f"向量搜索出错: {str(e)}",
            "success": False,
            "error": str(e)
        }

def format_db_results(result: Dict[str, Any]) -> Dict[str, Any]:
    """格式化数据库查询结果"""
    # 处理日期序列化问题
    def serialize_dates(obj):
        """递归处理对象中的日期类型，转换为字符串"""
        if hasattr(obj, 'isoformat'):  # datetime, date 对象
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: serialize_dates(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [serialize_dates(item) for item in obj]
        else:
            return obj
    
    # 序列化结果中的日期对象
    serialized_results = serialize_dates(result.get("results", []))
    
    success = False
    try:
        success = True
        formatted_content = json.dumps({
            "sql_query": result.get("sql_query"),
            "results": serialized_results,
            "vector_enhanced": result.get("vector_enhanced", False),
            "suggestions_used": result.get("suggestions_used", [])
        }, ensure_ascii=False, indent=2)
    except (TypeError, ValueError) as e:
        # 如果仍然有序列化问题，使用更安全的格式化方法
        logger.warning(f"JSON serialization failed, using fallback formatting: {e}")
        formatted_content = f"SQL查询: {result.get('sql_query', 'N/A')}\n"
        formatted_content += f"结果数量: {len(serialized_results)}\n"
        formatted_content += "结果数据:\n"
        for i, row in enumerate(serialized_results[:5], 1):  # 只显示前5行
            formatted_content += f"  {i}. {row}\n"
        if len(serialized_results) > 5:
            formatted_content += f"  ... 还有 {len(serialized_results) - 5} 行数据"
    
    return {
        "success": success,
        "content": formatted_content,
        "sql_query": result.get("sql_query"),
        "results": serialized_results,
        "vector_enhanced": result.get("vector_enhanced", False),
        "suggestions_used": result.get("suggestions_used", [])
    }

class KangniReActAgent:
    def __init__(self):
        # 使用集中式LLM服务
        self.llm_available = llm_service.llm_available
        self.llm_provider = llm_service.llm_provider
        
        if self.llm_available:
            try:
                # 绑定工具 - 添加vector_database_query_tool
                self.tools = [rag_search_tool, database_query_tool, vector_database_query_tool]
                
                # 构建状态图
                self.workflow = self._build_workflow()

                # 保存状态图的可视化表示
                self._save_graph_visualization()
                
                logger.info(f"Agent initialized successfully with LLM provider: {self.llm_provider}")
                
            except Exception as e:
                logger.warning(f"Failed to initialize agent: {e}")
                self.llm_available = False
                self.workflow = None
        else:
            logger.warning("LLM service not available, agent features will be disabled")
            self.workflow = None
    
    def _save_graph_visualization(self, filename: str = "docs/workflow_graph.png") -> None:
        """保存状态图的可视化表示
        
        Args:
            filename: 保存文件路径
        """
        if not self.workflow:
            logger.warning("No workflow to visualize")
            return
            
        try:
            # 尝试生成 Mermaid 图
            graph = self.workflow.get_graph()
            
            # 首先尝试保存为 PNG（需要 graphviz）
            try:
                with open(filename, "wb") as f:
                    f.write(graph.draw_mermaid_png())
                logger.info(f"Graph visualization saved as PNG: {filename}")
                print(f"✅ Workflow graph saved as: {filename}")
            except Exception as png_error:
                logger.warning(f"Could not save as PNG (graphviz may not be installed): {png_error}")
                
                # 退而求其次，保存为 Mermaid 文本格式
                mermaid_filename = filename.replace('.png', '.mermaid')
                try:
                    mermaid_text = graph.draw_mermaid()
                    with open(mermaid_filename, "w", encoding="utf-8") as f:
                        f.write(mermaid_text)
                    logger.info(f"Graph saved as Mermaid text: {mermaid_filename}")
                    print(f"✅ Workflow graph saved as Mermaid text: {mermaid_filename}")
                    print(f"   You can visualize it at: https://mermaid.live/")
                    
                    # 同时打印图形结构
                    print("\n📊 Workflow Structure:")
                    print(mermaid_text)
                except Exception as mermaid_error:
                    logger.error(f"Could not save Mermaid text: {mermaid_error}")
                    
                    # 最后的备用方案：打印节点和边
                    print("\n📊 Workflow Nodes and Edges:")
                    print(f"Nodes: {graph.nodes}")
                    print(f"Edges: {graph.edges}")
                    
        except Exception as e:
            logger.error(f"Failed to save graph visualization: {e}")
            print(f"❌ Could not visualize workflow: {e}")
    
    def _build_workflow(self) -> StateGraph:
        """构建LangGraph工作流 - 新流程：1.加载记忆 2.RAG搜索 3.数据库查询 4.向量搜索 5.生成响应"""
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("load_memory", self.load_memory_info)
        workflow.add_node("rag_search", self.execute_rag_search)
        workflow.add_node("check_rag_results", self.check_rag_results)
        workflow.add_node("database_query", self.execute_database_query)
        workflow.add_node("validate_db_results", self.validate_database_results)
        workflow.add_node("database_retry", self.execute_database_retry)
        workflow.add_node("validate_retry_results", self.validate_database_results)
        workflow.add_node("vector_search", self.execute_vector_search)
        workflow.add_node("validate_vector_results", self.validate_database_results)
        workflow.add_node("generate_response", self.generate_response)
        workflow.add_node("save_memory", self.save_memory)
        
        # 设置入口点
        workflow.set_entry_point("load_memory")
        
        # 添加边
        workflow.add_edge("load_memory", "rag_search")
        workflow.add_edge("rag_search", "check_rag_results")
        
        # 条件边：检查RAG结果
        def check_rag_condition(state: AgentState) -> str:
            has_rag_results = state.get("rag_has_results", False)
            logger.info(f"Checking RAG results: {has_rag_results}")
            
            if has_rag_results:
                logger.info("RAG has results, routing to generate_response")
                return "generate_response"
            else:
                logger.info("No RAG results, routing to database_query")
                return "database_query"
        
        workflow.add_conditional_edges(
            "check_rag_results",
            check_rag_condition,
            {
                "generate_response": "generate_response",
                "database_query": "database_query"
            }
        )
        
        workflow.add_edge("database_query", "validate_db_results")
        
        # 条件边：检查数据库结果
        def check_db_condition(state: AgentState) -> str:
            needs_vector_search = state.get("needs_vector_search", False)
            db_results = state.get("db_results", [])
            result_too_large = state.get("result_too_large", False)
            
            logger.info(f"Checking database results - needs_vector: {needs_vector_search}, result_too_large: {result_too_large}")
            
            # 如果结果太大，先尝试重试优化SQL
            if result_too_large and not state.get("retry_attempted", False):
                logger.info("Result too large, routing to database_retry")
                return "database_retry"
            # 如果不需要向量搜索，直接生成响应
            elif not needs_vector_search:
                logger.info("Database has valid results, routing to generate_response")
                return "generate_response"
            # 否则进行向量搜索
            else:
                logger.info("No valid database results, routing to vector_search")
                return "vector_search"
        
        workflow.add_conditional_edges(
            "validate_db_results",
            check_db_condition,
            {
                "generate_response": "generate_response",
                "database_retry": "database_retry",
                "vector_search": "vector_search"
            }
        )
        
        workflow.add_edge("database_retry", "validate_retry_results")
        workflow.add_edge("validate_retry_results", "generate_response")
        workflow.add_edge("vector_search", "validate_vector_results")
        workflow.add_edge("validate_vector_results", "generate_response")
        workflow.add_edge("generate_response", "save_memory")
        workflow.add_edge("save_memory", END)
        
        return workflow.compile()
    
    async def load_memory_info(self, state: AgentState) -> AgentState:
        """Load memory context for the user"""
        user_email = state.get("user_email")
        session_id = state.get("session_id")
        query = state["query"]
        
        logger.info(f"Loading memory context for user: {user_email}, session: {session_id}")
        
        # Get memory context from memory service
        memory_info = ""
        if user_email:
            try:
                memory_context = await memory_service.get_memory_context_for_agent(
                    user_email=user_email,
                    question=query,
                    session_id=session_id
                )

                # Build memory context string
                memory_info = ""
                if memory_context:
                    # Add recent interactions
                    # recent_interactions = memory_context.get("recent_interactions", [])
                    # if recent_interactions:
                    #     memory_info += "\n最近的交互历史:\n"
                    #     for interaction in recent_interactions[:3]:
                    #         memory_info += f"- Q: {interaction['question']}...\n"
                    #         if interaction.get('answer'):
                    #             memory_info += f"  A: {interaction['answer']}...\n"
                    
                    # # Add long-term memories
                    # long_term = memory_context.get("long_term_memories", [])
                    # if long_term:
                    #     memory_info += "\n相关的长期记忆:\n"
                    #     for mem in long_term[:3]:
                    #         memory_info += f"- {mem['content'][:150]}... (重要性: {mem.get('importance', 'unknown')})\n"
                    
                    # Add short-term memories
                    short_term = memory_context.get("short_term_memories", [])
                    if short_term:
                        memory_info += "\n会话上下文:\n"
                        for mem in short_term[:3]:
                            memory_info += f"- {mem['content']}...\n"

                logger.info(f"Loaded memory context with {len(memory_context.get('short_term_memories', []))} short-term and {len(memory_context.get('long_term_memories', []))} long-term memories")
            except Exception as e:
                logger.error(f"Failed to load memory context: {e}")
        
        return {
            **state,
            "start_time": time.time(),
            "memory_info": memory_info
        }
    
    async def execute_rag_search(self, state: AgentState) -> AgentState:
        """执行RAG搜索"""
        query = state["query"]
        memory_info = state.get("memory_info", "")
        
        logger.info(f"Executing RAG search for query: {query}")
        
        try:
            result = await rag_search_tool.ainvoke({"query": query, "memory_info": memory_info})
            
            # 存储RAG结果
            rag_results = result.get("rag_results", [])
            content = result.get("content", "")
            
            # 检查是否有有效结果 - 更严格的条件
            has_results = bool(
                content and 
                content.strip() and 
                "未找到相关文档信息" not in content and
                "找不到" not in content and
                "没有找到" not in content and
                "无法找到" not in content and
                "没有相关" not in content and
                "无相关信息" not in content and
                "未包含所需信息" not in content and
                len(content.strip()) > 50  # 确保有足够的内容
            )
            
            # 创建工具消息
            tool_message = AIMessage(content=f"RAG搜索结果：\n{content}")
            
            logger.info(f"RAG search completed. Has results: {has_results}")
            
            return {
                **state,
                "rag_results": rag_results,
                "rag_content": content,
                "rag_has_results": has_results,
                "messages": [*state["messages"], tool_message]
            }
            
        except Exception as e:
            logger.error(f"RAG search error: {e}")
            error_message = AIMessage(content=f"RAG搜索时发生错误: {str(e)}")
            return {
                **state,
                "rag_has_results": False,
                "messages": [*state["messages"], error_message]
            }
    
    async def check_rag_results(self, state: AgentState) -> AgentState:
        """检查RAG搜索结果"""
        # 这个节点主要用于路由决策，状态已经在execute_rag_search中设置
        return state
    
    async def execute_database_query(self, state: AgentState) -> AgentState:
        """执行数据库查询"""
        query = state["query"]
        memory_info = state.get("memory_info", "")
        
        logger.info(f"Executing database query for: {query}")
        
        try:
            result = await database_query_tool.ainvoke({"question": query, "memory_info": memory_info})
            
            # 存储数据库结果
            db_results = result.get("results", [])
            sql_query = result.get("sql_query")
            success = result.get("success", False)
            
            # 创建工具消息
            tool_message = AIMessage(content=f"数据库查询结果：\n{result['content']}")
            
            logger.info(f"Database query completed. Success: {success}, Results count: {len(db_results) if db_results else 0}")
            
            return {
                **state,
                "db_results": db_results,
                "sql_query": sql_query,
                "db_success": success,
                "messages": [*state["messages"], tool_message]
            }
            
        except Exception as e:
            logger.error(f"Database query error: {e}")
            error_message = AIMessage(content=f"数据库查询时发生错误: {str(e)}")
            return {
                **state,
                "db_success": False,
                "messages": [*state["messages"], error_message]
            }
    
    async def validate_database_results(self, state: AgentState) -> AgentState:
        """验证工具执行结果的有效性 - 使用LLM智能判断"""

        db_results = state.get("db_results", [])
        sql_query = state.get("sql_query")
        db_success = state.get("db_success", False)
        query = state["query"]
        memory_info = state.get("memory_info", {})

        # 检查结果大小是否过大（仅在初次验证时检查）
        if not state.get("retry_attempted", False):
            result_too_large = self._check_result_size(db_results)
            if result_too_large:
                logger.info("Result too large, routing to database_retry")
                return {
                    **state,
                    "result_too_large": result_too_large,
                    "retry_attempted": False,  # 标记为需要重试
                }
            
        # 安全地序列化数据库结果，避免JSON转义问题
        def safe_serialize_results(results):
            if not results:
                return "空结果"
            try:
                # 使用json.dumps确保正确的JSON格式
                json_str = json.dumps(results, ensure_ascii=False, indent=2)
                return json_str
            except (TypeError, ValueError) as e:
                # 如果JSON序列化失败，使用简单的字符串表示
                logger.warning(f"JSON serialization failed: {e}, using fallback")
                if isinstance(results, list):
                    return f"结果列表，共{len(results)}条记录"
                elif isinstance(results, dict):
                    return f"结果字典，包含{len(results)}个字段"
                else:
                    return str(results)

        # 构建让LLM判断和格式化的提示
        validation_prompt = f"""请分析以下工具执行结果，判断查询是否成功。如果查询结果成功，直接格式化结果。

第一步需要理解用户的问题，根据下面的步骤理解用户的问题。
1. 先检查用户问题是否需要历史记忆来回答  
2. 如果需要，找到与问题最相关的记忆并引用  
3. 再根据当前输入给出最终答案

用户对话历史：{memory_info}
用户问题: {query}

生成的SQL: {sql_query if sql_query else "无"}

数据库查询结果: {safe_serialize_results(db_results)}

请按以下步骤处理：

1. 分析查询结果：
   - SQL查询是否成功生成？（是/否）
   - 数据库查询是否返回了有效数据？（是/否）
   - 如果SQL成功但返回0条记录，或者count(*)为0，说明用户输入可能不够准确，需要向量搜索来找到正确的数据库值（是/否）

2. 如果数据库查询返回了有效数据，请直接格式化结果：
   - 用自然语言回答用户的问题
   - 如果查询结果包含COUNT(*)或数量统计，请明确说出具体的数字
   - 对于数据列表，用清晰的方式展示关键信息
   - 如果数据量很大，只显示前几条记录并说明总数
   - 保持回答简洁明了，重点突出

判断标准：
- 如果SQL生成成功但返回0条记录，则认为失败
- 如果SQL运行报错，则认为失败
- 如果查询返回了数据，则认为成功，不需要额外操作

请按以下格式回答：
SQL生成成功: [是/否]
有效数据: [是/否]
需要向量搜索: [如果SQL成功但无数据，或用户输入不够准确，则填"是"，否则填"否"]
格式化结果: [如果有效数据，请直接给出格式化的回答，否则写"无"]"""

        # 转换消息格式为LLMMessage
        llm_messages = [
            LLMMessage(role="system", content="你是一个数据库查询结果分析助手，需要准确判断查询结果的有效性。"),
            LLMMessage(role="user", content=validation_prompt)
        ]
        
        # 不设置max_tokens，让服务器使用默认值
        response = await llm_service.chat(llm_messages)
        response_text = response.content.lower()
        
        # 解析LLM的判断结果
        has_valid_sql = "sql生成成功: 是" in response_text or "sql生成成功：是" in response_text
        has_valid_data = "有效数据: 是" in response_text or "有效数据：是" in response_text
        needs_vector = "需要向量搜索: 是" in response_text or "需要向量搜索：是" in response_text
        
        # 额外检查：如果SQL成功但结果为空，也需要向量搜索
        result_count = len(db_results) if db_results else 0
        sql_successful_but_empty = has_valid_sql and result_count == 0

        
        # 确定是否需要向量搜索
        needs_vector_search = not has_valid_data or sql_successful_but_empty or needs_vector
              
        # 提取格式化结果
        formatted_db_results = None
        format_match = response_text.find("格式化结果:")
        if format_match == -1:
            format_match = response_text.find("格式化结果：")
        
        if format_match != -1:
            # 提取格式化结果部分（包括多行）
            format_text = response_text[format_match:]
            if "格式化结果:" in format_text:
                formatted_db_results = format_text.split("格式化结果:")[1].strip()
            elif "格式化结果：" in format_text:
                formatted_db_results = format_text.split("格式化结果：")[1].strip()
            
            # 如果格式化结果是"无"或空，则设为None
            if formatted_db_results in ["无", "", "无数据", "无结果"]:
                formatted_db_results = None
        
        # 如果没有从LLM获取到格式化结果，但有有效数据，使用简单格式化
        if has_valid_data and db_results and not formatted_db_results:
            formatted_db_results = self._format_database_response(db_results, sql_query)
        
        # 对于重试后的结果，不再检查大小，直接使用LLM判断
        is_retry_validation = state.get("retry_attempted", False)
        
        logger.info(f"LLM validation result: sql_valid={has_valid_sql}, data_valid={has_valid_data}, result_count={result_count}, needs_vector_search={needs_vector_search}, is_retry={is_retry_validation}")
        logger.info(f"Formatted results: {'Yes' if formatted_db_results else 'No'}")
                
        return {
            **state,
            "db_results_valid": has_valid_data,
            "needs_vector_search": needs_vector_search,
            "result_too_large": False,  # 重试后不再标记为过大
            "retry_attempted": is_retry_validation,  # 保持重试状态
            "formatted_db_results": formatted_db_results
        }
    
    async def execute_database_retry(self, state: AgentState) -> AgentState:
        """执行数据库查询重试 - 优化SQL以减少结果大小"""
        query = state["query"]
        original_sql = state.get("sql_query")
        memory_info = state.get("memory_info", "")
        
        logger.info(f"Executing database retry for large results: {query}")
        
        try:
            # 使用LLM优化SQL
            optimized_sql = await self._optimize_sql_for_large_results(original_sql, query, memory_info)
            
            # 使用优化后的SQL重新查询数据库
            logger.info(f"Retrying with optimized SQL: {optimized_sql}")
            db_results = await db_service.execute_sql_query(optimized_sql)
            
            if db_results is not None:
                logger.info(f"Database retry successful, got {len(db_results)} results")
                
                # 格式化结果
                formatted_content = self._format_database_response(db_results, optimized_sql)
                
                # 创建工具消息
                tool_message = AIMessage(content=f"数据库重试结果（优化后）：\n{formatted_content}")
                
                return {
                    **state,
                    "db_results": db_results,
                    "sql_query": optimized_sql,
                    "db_success": True,
                    "retry_attempted": True,
                    "messages": [*state["messages"], tool_message]
                }
            else:
                logger.warning("Database retry failed: No results returned")
                error_message = AIMessage(content="数据库重试失败: 未返回结果")
                return {
                    **state,
                    "db_success": False,
                    "retry_attempted": True,
                    "messages": [*state["messages"], error_message]
                }
                
        except Exception as e:
            logger.error(f"Database retry error: {e}")
            error_message = AIMessage(content=f"数据库重试时发生错误: {str(e)}")
            return {
                **state,
                "db_success": False,
                "retry_attempted": True,
                "messages": [*state["messages"], error_message]
            }
    
    async def execute_vector_search(self, state: AgentState) -> AgentState:
        """执行向量搜索增强"""
        query = state["query"]
        failed_sql = state.get("sql_query")
        memory_info = state.get("memory_info", "")
        
        logger.info(f"Executing vector search enhancement for: {query}")
        
        try:
            result = await vector_database_query_tool.ainvoke({
                "question": query,
                "failed_sql": failed_sql or "",
                "memory_info": memory_info
            })
            
            # 使用数据库状态字段存储向量搜索结果
            db_success = result.get("success", False)
            results = result.get("results", [])
            
            # 创建工具消息
            tool_message = AIMessage(content=f"向量搜索增强结果：\n{result.get('content', '')}")
            
            logger.info(f"Vector search completed. Success: {db_success}, Results count: {len(results) if results else 0}")
            
            return {
                **state,
                "db_results": results,
                "sql_query": result.get("sql_query"),
                "db_success": db_success,
                "messages": [*state["messages"], tool_message]
            }
            
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            error_message = AIMessage(content=f"向量搜索时发生错误: {str(e)}")
            return {
                **state,
                "db_success": False,
                "messages": [*state["messages"], error_message]
            }
    
    
    async def generate_response(self, state: AgentState) -> AgentState:
        """生成最终回答 - 按照新流程：RAG -> 数据库 -> 向量搜索"""
        rag_has_results = state.get("rag_has_results", False)
        db_results_valid = state.get("db_results_valid", False)
        formatted_db_results = state.get("formatted_db_results")
        
        logger.info(f"Generating response - RAG: {rag_has_results}, DB: {db_results_valid}")
        
        # 1. 如果RAG有结果，直接返回RAG答案
        if rag_has_results:
            rag_content = state.get("rag_content", "")
            if rag_content:
                logger.info("Returning RAG answer")
                ai_message = AIMessage(content=rag_content)
                return {
                    **state,
                    "final_answer": rag_content,
                    "messages": [*state["messages"], ai_message]
                }
        
        # 2. 如果数据库查询有有效结果，返回格式化结果
        elif db_results_valid and formatted_db_results:
            logger.info("Returning formatted database/vector search answer")
            ai_message = AIMessage(content=formatted_db_results)
            return {
                **state,
                "final_answer": formatted_db_results,
                "messages": [*state["messages"], ai_message]
            }
        
        # 3. 如果都没有结果，返回无法找到信息的消息
        else:
            no_result_message = "抱歉，无法找到与您问题相关的信息。请尝试重新表述您的问题或提供更多详细信息。"
            logger.warning("No valid results found from any source")
            
            ai_message = AIMessage(content=no_result_message)
            return {
                **state,
                "final_answer": no_result_message,
                "messages": [*state["messages"], ai_message]
            }
    
    def _format_database_response(self, db_results: List[Dict], sql_query: str = None) -> str:
        """格式化数据库查询结果"""
        if not db_results:
            return "未找到相关数据"
        
        # 构建格式化回答
        answer_parts = []
        
        # 添加结果统计
        result_count = len(db_results)
        answer_parts.append(f"查询结果：共找到 {result_count} 条记录\n")
        
        # 添加结果数据
        if result_count > 0:
            answer_parts.append("数据详情：")
            
            # 显示前10条记录
            display_count = min(10, result_count)
            for i, row in enumerate(db_results[:display_count], 1):
                if isinstance(row, dict):
                    # 格式化字典数据
                    row_str = ", ".join([f"{k}: {v}" for k, v in row.items()])
                    answer_parts.append(f"{i}. {row_str}")
                else:
                    # 格式化其他类型数据
                    answer_parts.append(f"{i}. {row}")
            
            if result_count > display_count:
                answer_parts.append(f"... 还有 {result_count - display_count} 条记录")
        
        return "\n".join(answer_parts)
    
    def _check_result_size(self, db_results: List[Dict]) -> bool:
        """检查数据库结果大小是否过大"""
        if not db_results:
            return False
        
        # 计算结果的字符长度
        try:
            # 使用自定义序列化器处理Decimal等特殊类型
            def json_serializer(obj):
                if hasattr(obj, 'isoformat'):  # datetime, date 对象
                    return obj.isoformat()
                elif hasattr(obj, '__str__'):  # Decimal等对象
                    return str(obj)
                else:
                    return obj
            
            result_json = json.dumps(db_results, ensure_ascii=False, default=json_serializer)
            result_length = len(result_json)
            logger.info(f"Database result length: {result_length} characters")
            return result_length > 50000  # 超过10000字符认为过大
        except Exception as e:
            logger.warning(f"Failed to calculate result size: {e}")
            # 如果无法计算JSON长度，使用记录数量作为粗略估计
            return len(db_results) > 5000
    
    async def _optimize_sql_for_large_results(self, original_sql: str, question: str, memory_info: str = "") -> str:
        """使用LLM优化SQL查询以减少结果大小"""
        optimization_prompt = f"""请优化以下SQL查询以减少结果大小。原始查询返回了太多数据，需要添加DISTINCT或GROUP BY来聚合结果。

用户问题: {question}
原始SQL: {original_sql}

优化要求:
1. 如果查询返回重复数据，添加DISTINCT
2. 如果需要统计信息，使用GROUP BY和聚合函数(COUNT, SUM, AVG等)
3. 保持查询的核心逻辑不变
4. 优先使用聚合函数来提供统计信息而不是详细列表
5. 如果可能，添加LIMIT子句限制结果数量

请只返回优化后的SQL查询，不要包含其他解释。"""

        llm_messages = [
            LLMMessage(role="system", content="你是一个SQL优化专家，专门处理大数据量查询的优化。"),
            LLMMessage(role="user", content=optimization_prompt)
        ]
        
        try:
            response = await llm_service.chat(llm_messages)
            optimized_sql = response.content.strip()
            
            # 清理SQL，移除可能的markdown格式
            if optimized_sql.startswith("```sql"):
                optimized_sql = optimized_sql[6:]
            if optimized_sql.endswith("```"):
                optimized_sql = optimized_sql[:-3]
            optimized_sql = optimized_sql.strip()
            
            logger.info(f"SQL optimized: {original_sql} -> {optimized_sql}")
            return optimized_sql
            
        except Exception as e:
            logger.error(f"Failed to optimize SQL: {e}")
            return original_sql
    
    async def save_memory(self, state: AgentState) -> AgentState:
        """Save interaction to memory system"""
        user_email = state.get("user_email")
        session_id = state.get("session_id")
        query = state["query"]
        final_answer = state.get("final_answer", "")
        
        if not user_email:
            logger.info("No user email provided, skipping memory save")
            return state
        
        try:
            query_history_id = await self._save_query_history(state)
            # Extract and store memories from this interaction
            if query_history_id and final_answer:
                memory_ids = await memory_service.extract_and_store_memories(
                    query_id=query_history_id,
                    user_email=user_email,
                    question=query,
                    answer=final_answer,
                    session_id=session_id,
                    feedback_type=None  # Will be updated when user provides feedback
                )
                logger.info(f"Saved {len(memory_ids)} memories for interaction")
            
            # Periodically consolidate memories (every 10th query)
            if query_history_id and query_history_id % 10 == 0:
                consolidation_result = await memory_service.consolidate_memories(
                    user_email=user_email,
                    session_id=session_id
                )
                logger.info(f"Memory consolidation result: {consolidation_result}")
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
        
        return state

    async def _save_query_history(self, state: AgentState) -> int:
        """
        Save query history to database
        
        Args:
            state: The agent state containing all query information
            
        Returns:
            int: The query history ID if successful, None if failed
        """
        try:
            # Extract data from agent state
            query = state.get("query", "")
            user_email = state.get("user_email")
            session_id = state.get("session_id", "default")
            answer = state.get("final_answer", "")
            sql_query = state.get("sql_query")
            query_type = state.get("intent")
            rag_results = state.get("rag_results", [])
            
            # Determine success based on whether we have an answer
            success = bool(answer and answer.strip())
            
            # Get LLM info from agent if available
            llm_provider = ""
            model_name = ""
            if hasattr(self, 'llm_provider') and self.llm_provider:
                llm_provider = str(self.llm_provider.value if hasattr(self.llm_provider, 'value') else self.llm_provider)
            if hasattr(self, 'model_name') and self.model_name:
                model_name = self.model_name
            
            # Calculate processing time (simplified - could be enhanced with actual timing)
            processing_time_ms = (time.time() - state.get("start_time")) * 1000  # Default value, could be calculated from actual start/end times

            # 处理RAG结果，提取文档信息并去重
            unique_documents = {}
            if rag_results is not None:
                for result in rag_results:
                    if hasattr(result, 'metadata') and result.metadata:
                        doc_id = result.metadata.get('document_id')
                        if doc_id and doc_id not in unique_documents:
                            unique_documents[doc_id] = {
                                "document_id": doc_id,
                                "document_name": result.metadata.get('document_name', ''),
                                "dataset_name": result.metadata.get('dataset_name', '')
                            }
            
            # 转换为列表格式
            sources = list(unique_documents.values())

            # Save to history
            history = await history_service.save_query_history(
                session_id=session_id,
                user_email=user_email,
                question=query,
                answer=answer,
                sql_query=sql_query,
                sources=sources,
                query_type=query_type,
                success=success,
                processing_time_ms=processing_time_ms,
                llm_provider=llm_provider,
                model_name=model_name
            )
            
            logger.info(f"Saved query history with ID: {history.id}")
            return history.id
            
        except Exception as e:
            logger.error(f"Failed to save query history: {e}")
            return None
    
    async def query(self, question: str, context: Optional[str] = None, user_email: Optional[str] = None, session_id: Optional[str] = None) -> QueryResponse:
        """处理用户查询"""
        if not self.llm_available or not self.workflow:
            return QueryResponse(
                answer="抱歉，AI服务暂时不可用，请检查配置",
                query_type=QueryType.HYBRID,
                confidence=0.0
            )
            
        try:
            # 初始状态
            initial_state = {
                "messages": [HumanMessage(content=question)],
                "query": question,
                "intent": None,
                # RAG-related fields
                "rag_results": None,
                "rag_content": None,
                "rag_has_results": False,
                # Database-related fields (also used for vector search results)
                "db_results": None,
                "sql_query": None,
                "db_success": False,
                "db_results_valid": False,
                "needs_vector_search": False,
                "formatted_db_results": None,
                "result_too_large": False,
                "retry_attempted": False,
                # Response fields
                "final_answer": None,
                "source_links": None,
                # Memory-related fields
                "memory_info": None,
                "user_email": user_email,
                "session_id": session_id,
                "start_time": time.time()
            }
            
            # 运行工作流
            final_state = await self.workflow.ainvoke(
                initial_state,
                config=RunnableConfig(recursion_limit=10)
            )
            
            # 确保有有效的答案
            answer = final_state.get("final_answer")
            if not answer or answer.strip() == "":
                # 如果没有最终答案，尝试从消息中获取最后一个AI消息
                messages = final_state.get("messages", [])
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and msg.content and msg.content.strip():
                        answer = msg.content.strip()
                        break
                
                if not answer or answer.strip() == "":
                    answer = "抱歉，无法生成回答"
            
            # 确定查询类型
            query_type = final_state.get("intent")
            if not query_type:
                # 根据结果来源确定查询类型
                if final_state.get("rag_has_results", False):
                    query_type = QueryType.RAG
                elif final_state.get("db_results_valid", False):
                    query_type = QueryType.DATABASE
                else:
                    query_type = QueryType.HYBRID
            
            # 构建增强的响应
            response = QueryResponse(
                answer=answer,
                query_type=query_type,
                reasoning=None,  # 简化响应，不再包含reasoning
                confidence=0.8
            )
            
            # 添加SQL查询信息
            if final_state.get("sql_query"):
                response.sql_query = final_state["sql_query"]
            
            # 添加RAG源文件信息（如果来自RAG）
            if final_state.get("rag_results"):
                # 处理RAG结果，提取文档信息并去重
                unique_documents = {}
                for result in final_state["rag_results"]:
                    if hasattr(result, 'metadata') and result.metadata:
                        doc_id = result.metadata.get('document_id')
                        if doc_id and doc_id not in unique_documents:
                            unique_documents[doc_id] = {
                                "document_id": doc_id,
                                "document_name": result.metadata.get('document_name', ''),
                                "dataset_name": result.metadata.get('dataset_name', '')
                            }
                
                # 转换为列表格式
                response.sources = list(unique_documents.values())
            
            # 记录查询完成情况
            rag_used = final_state.get("rag_has_results", False)
            db_used = final_state.get("db_results_valid", False)
            
            logger.info(f"Query completed - RAG: {rag_used}, DB: {db_used}, SQL: {bool(response.sql_query)}, Sources: {len(response.sources) if response.sources else 0}")
            
            return response
            
        except Exception as e:
            logger.error(f"Agent query error: {e}")
            return QueryResponse(
                answer=f"处理查询时发生错误: {str(e)}",
                query_type=QueryType.HYBRID,
                confidence=0.0
            )

# 全局实例
kangni_agent = KangniReActAgent()