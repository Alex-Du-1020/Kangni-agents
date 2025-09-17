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
    rag_results: Optional[List[RAGSearchResult]]
    db_results: Optional[Dict[str, Any]]
    sql_query: Optional[str]
    source_links: Optional[List[str]]
    final_answer: Optional[str]
    reasoning: Optional[str]
    needs_tools: bool
    tool_to_use: Optional[str]
    # 新增字段支持验证和fallback流程
    db_results_valid: bool
    needs_fallback: bool
    needs_vector_search: bool  # Added this missing field!
    fallback_executed: bool
    has_mixed_results: bool
    vector_enhanced: Optional[bool]  # Also add this for tracking
    suggestions_used: Optional[Dict[str, Any]]  # And this for tracking suggestions
    validation_reason: Optional[str]  # And validation reason
    # Memory-related fields
    memory_info: Optional[str]  # Memory context from memory service
    user_email: Optional[str]  # User email for memory retrieval
    session_id: Optional[str]  # Session ID for memory tracking
    start_time: Optional[float]  # Start time for query
    # Data formatting fields
    formatted_db_results: Optional[str]  # LLM-formatted database results

@tool
async def rag_search_tool(query: str, memory_info: str = "", dataset_id: Optional[str] = None) -> Dict[str, Any]:
    """搜索RAG文档库获取相关信息"""
    if not dataset_id:
        dataset_id = settings.ragflow_default_dataset_id
    
    result = await rag_service.search_rag_with_answer(query, dataset_id, memory_info)
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
        """构建LangGraph工作流"""
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("load_memory", self.load_memory_info)  # New memory node
        workflow.add_node("classify_intent", self.classify_intent)
        workflow.add_node("agent_reasoning", self.agent_reasoning)
        workflow.add_node("tool_execution", self.execute_tools)
        workflow.add_node("validate_results", self.validate_results)
        workflow.add_node("vector_search", self.vector_search_enhancement)  # 新增向量搜索节点
        workflow.add_node("fallback_search", self.fallback_search)
        workflow.add_node("generate_response", self.generate_response)
        workflow.add_node("save_memory", self.save_memory)  # New save memory node
        
        # 设置入口点
        workflow.set_entry_point("load_memory")
        
        # 添加基础边
        workflow.add_edge("load_memory", "classify_intent")
        workflow.add_edge("classify_intent", "agent_reasoning")
        
        # 直接边：总是执行工具
        workflow.add_edge("agent_reasoning", "tool_execution")
        workflow.add_edge("tool_execution", "validate_results")
        
        # 条件边：检查是否需要向量搜索增强
        def should_use_vector_search(state: AgentState) -> str:
            # Debug logging
            logger.info(f"Checking vector search condition - needs_vector_search: {state.get('needs_vector_search')}, needs_fallback: {state.get('needs_fallback')}")
            
            # 如果数据库查询返回0结果，使用向量搜索
            if state.get("needs_vector_search", False):
                logger.info("Routing to vector_search")
                return "vector_search"
            # 否则检查是否需要fallback
            elif state.get("needs_fallback", False):
                logger.info("Routing to fallback_search")
                return "fallback_search"
            else:
                logger.info("Routing to generate_response")
                return "generate_response"
        
        workflow.add_conditional_edges(
            "validate_results", 
            should_use_vector_search,
            {
                "vector_search": "vector_search",
                "fallback_search": "fallback_search",
                "generate_response": "generate_response"
            }
        )
        workflow.add_edge("vector_search", "generate_response")
        workflow.add_edge("fallback_search", "generate_response")
        workflow.add_edge("generate_response", "save_memory")  # Save memory after generating response
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
    
    async def classify_intent(self, state: AgentState) -> AgentState:
        """意图分类节点"""
        query = state["query"]
        intent = intent_classifier.classify_intent(query)
        explanation = intent_classifier.get_classification_explanation(query, intent)
        
        logger.info(f"Intent classified as: {intent} - {explanation}")
        
        return {
            **state,
            "intent": intent,
            "reasoning": explanation
        }
    
    async def agent_reasoning(self, state: AgentState) -> AgentState:
        """Agent推理节点"""
        query = state["query"]
        intent = state["intent"]
        memory_info = state.get("memory_info", {})
        
        # 构建系统提示
        system_prompt = f"""你是一个智能助手，需要分析用户问题并决定使用哪些工具。你有两个工具可用：

1. rag_search_tool: 用于搜索文档和知识库，适合回答概念、原因、方法等问题
2. database_query_tool: 用于查询数据库，适合回答统计、数据分析等问题
   - 特别注意：当用户提到"订单"但没有指定具体类型时，系统会默认查询 kn_quality_trace_prod_order（生产订单表）

当前问题意图分类为: {intent}
分类原因: {state.get('reasoning', '')}

用户问题: {query}

请分析问题并决定使用哪些工具：
1. 如果问题明确需要数据库查询（如统计、计数、具体数据），选择 database_query_tool
2. 如果问题明确需要文档搜索（如概念解释、原因分析），选择 rag_search_tool  
3. 如果问题意图不明确或需要综合信息，选择 both（同时使用两个工具）
4. 考虑用户的历史交互模式，如果用户通常询问某类问题，可以参考历史偏好

请按以下格式回答：
工具选择: [rag_search_tool/database_query_tool/both]
理由: [简要说明为什么选择这些工具]
"""
        
        # 转换消息格式为LLMMessage
        llm_messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=query)
        ]
        
        # Debug logging for LLM input
        logger.debug(f"LLM reasoning input: {system_prompt}")
        
        try:
            response = await llm_service.chat(llm_messages)
            
            # Debug logging for LLM output
            logger.debug(f"LLM reasoning output: {response.content}")
            
            # 解析响应，确定工具选择
            response_text = response.content.lower()
            
            # 确定工具类型
            tool_to_use = None
            needs_tools = True  # 总是需要工具，不再直接回答
            
            if "工具选择: both" in response_text or "工具选择：both" in response_text:
                tool_to_use = "both"
            elif "rag_search_tool" in response_text:
                tool_to_use = "rag_search_tool"
            elif "database_query_tool" in response_text:
                tool_to_use = "database_query_tool"
            else:
                # 默认使用both工具
                tool_to_use = "both"
            
            logger.info(f"Tool analysis: needs_tools={needs_tools}, tool_to_use={tool_to_use}")
            
            # 创建AIMessage响应
            ai_message = AIMessage(content=response.content)
            
            return {
                **state,
                "needs_tools": needs_tools,
                "tool_to_use": tool_to_use,
                "messages": [*state["messages"], ai_message]
            }
            
        except Exception as e:
            logger.error(f"LLM reasoning error: {e}")
            error_message = AIMessage(content=f"抱歉，分析您的问题时发生错误: {str(e)}")
            return {
                **state,
                "needs_tools": False,
                "messages": [*state["messages"], error_message]
            }
    
    async def execute_tools(self, state: AgentState) -> AgentState:
        """工具执行节点"""
        tool_to_use = state.get("tool_to_use")
        query = state["query"]
        memory_info = state.get("memory_info", "")
        
        if not tool_to_use:
            logger.warning("No tool specified for execution")
            return state
            
        logger.info(f"Executing tool: {tool_to_use} with query: {query}")
        
        try:
            messages = state["messages"]
            
            if tool_to_use == "rag_search_tool":
                result = await rag_search_tool.ainvoke({"query": query, "memory_info": memory_info})
                
                # 存储RAG结果
                state["rag_results"] = result.get("rag_results", [])
                
                # 创建工具消息 - 直接使用LLM生成的答案
                tool_message = AIMessage(content=f"RAG搜索结果：\n{result['content']}")
                messages.append(tool_message)
                
            elif tool_to_use == "database_query_tool":
                result = await database_query_tool.ainvoke({"question": query, "memory_info": memory_info})
                
                # 存储数据库结果
                state["db_results"] = result.get("results", [])
                state["sql_query"] = result.get("sql_query")
                
                # 创建工具消息
                tool_message = AIMessage(content=f"数据库查询结果：\n{result['content']}")
                messages.append(tool_message)
                
            elif tool_to_use == "both":
                # 同时执行两个工具
                logger.info("Executing both RAG and database tools")
                
                # 执行RAG搜索
                rag_result = await rag_search_tool.ainvoke({"query": query, "memory_info": memory_info})
                state["rag_results"] = rag_result.get("rag_results", [])
                rag_message = AIMessage(content=f"RAG搜索结果：\n{rag_result['content']}")
                messages.append(rag_message)
                
                if "未找到相关文档信息" in rag_result['content']:
                    # 执行数据库查询
                    db_result = await database_query_tool.ainvoke({"question": query, "memory_info": memory_info})
                    state["db_results"] = db_result.get("results", [])
                    state["sql_query"] = db_result.get("sql_query")
                    db_message = AIMessage(content=f"数据库查询结果：\n{db_result['content']}")
                    messages.append(db_message)
                    
                    logger.info("Both tools executed successfully")
                else:
                    tool_to_use = "rag_search_tool"
                    logger.info("RAG search successfully, skipping database query")
                
            else:
                tool_message = AIMessage(content=f"未知工具: {tool_to_use}")
                messages.append(tool_message)
            
            logger.debug(f"Tool {tool_to_use} execution completed")
            
            return {
                **state,
                "tool_to_use": tool_to_use,
                "messages": messages
            }
            
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            error_message = AIMessage(content=f"工具执行错误: {str(e)}")
            return {
                **state,
                "messages": [*state["messages"], error_message]
            }
    
    
    async def validate_results(self, state: AgentState) -> AgentState:
        """验证工具执行结果的有效性 - 使用LLM智能判断"""
        tool_to_use = state.get("tool_to_use")
        
        if tool_to_use in ["database_query_tool", "both"]:
            # 获取数据库查询结果
            db_results = state.get("db_results", [])
            sql_query = state.get("sql_query")
            query = state["query"]
            
            # 获取RAG结果（如果使用了both工具）
            rag_results = state.get("rag_results", [])
            memory_info = state.get("memory_info", {})
            
            # 构建让LLM判断和格式化的提示
            validation_prompt = f"""请分析以下工具执行结果，判断查询是否成功。如果查询结果成功，直接格式化结果。如果失败，判断是否需要使用向量搜索增强。

第一步需要理解用户的问题，根据下面的步骤理解用户的问题。
1. 先检查用户问题是否需要历史记忆来回答  
2. 如果需要，找到与问题最相关的记忆并引用  
3. 再根据当前输入给出最终答案

用户对话历史：{memory_info}
用户问题: {query}

生成的SQL: {sql_query if sql_query else "无"}

数据库查询结果: {json.dumps(db_results, ensure_ascii=False, indent=2) if db_results else "空结果"}

RAG搜索结果: {"找到相关文档" if rag_results else "无RAG结果"}

请按以下步骤处理：

1. 分析查询结果：
   - SQL查询是否成功生成？（是/否）
   - 数据库查询是否返回了有效数据？（是/否）
   - 如果没有返回数据，是否应该使用向量搜索来找到正确的数据库值？（是/否）
   - 如果向量搜索也不适用，是否应该使用RAG文档搜索作为备用方案？（是/否）

2. 如果数据库查询返回了有效数据，请直接格式化结果：
   - 用自然语言回答用户的问题
   - 如果查询结果包含COUNT(*)或数量统计，请明确说出具体的数字
   - 对于数据列表，用清晰的方式展示关键信息
   - 如果数据量很大，只显示前几条记录并说明总数
   - 保持回答简洁明了，重点突出

判断标准：
- 如果SQL生成成功但返回0条记录，很可能是查询条件中的值不准确（如项目名称拼写错误），应该使用向量搜索
- 如果SQL生成失败或查询出错，应该使用RAG作为备用方案
- 如果查询返回了数据，则认为成功，不需要额外操作
- 如果使用了both工具且RAG有结果，优先使用数据库结果，RAG作为补充

请按以下格式回答：
SQL生成成功: [是/否]
有效数据: [是/否]  
需要向量搜索: [是/否]
需要RAG备用: [是/否]
原因: [简要说明判断理由]
格式化结果: [如果有效数据，请直接给出格式化的回答，否则写"无"]"""

            # 转换消息格式为LLMMessage
            llm_messages = [
                LLMMessage(role="system", content="你是一个数据库查询结果分析助手，需要准确判断查询结果的有效性。"),
                LLMMessage(role="user", content=validation_prompt)
            ]
            
            try:
                response = await llm_service.chat(llm_messages)
                response_text = response.content.lower()
                
                # 解析LLM的判断结果
                has_valid_sql = "sql生成成功: 是" in response_text or "sql生成成功：是" in response_text
                has_valid_data = "有效数据: 是" in response_text or "有效数据：是" in response_text
                needs_vector = "需要向量搜索: 是" in response_text or "需要向量搜索：是" in response_text
                needs_rag = "需要rag备用: 是" in response_text or "需要rag备用：是" in response_text
                
                # 提取原因
                reason_match = response_text.find("原因:")
                if reason_match == -1:
                    reason_match = response_text.find("原因：")
                reason = response_text[reason_match:].split('\n')[0] if reason_match != -1 else "未提供原因"
                
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
                
                logger.info(f"LLM validation result: sql_valid={has_valid_sql}, data_valid={has_valid_data}, needs_vector={needs_vector}, needs_rag={needs_rag}")
                logger.info(f"Validation reason: {reason}")
                logger.info(f"Formatted results: {'Yes' if formatted_db_results else 'No'}")
                
                return {
                    **state,
                    "db_results_valid": has_valid_data,
                    "needs_vector_search": needs_vector,  # Changed from needs_vector_search to match what we check
                    "needs_fallback": needs_rag,
                    "fallback_executed": False,
                    "validation_reason": reason,
                    "formatted_db_results": formatted_db_results
                }
                
            except Exception as e:
                logger.error(f"LLM validation error: {e}, falling back to simple check")
                # 如果LLM判断失败，使用简单的备用逻辑
                # 检查是否有SQL和结果
                has_sql = bool(sql_query)
                has_results = False
                
                # 尝试多种方式检查是否有结果
                if db_results:
                    if isinstance(db_results, list) and len(db_results) > 0:
                        has_results = True
                    elif isinstance(db_results, dict) and db_results:
                        has_results = True
                    elif isinstance(db_results, str) and db_results.strip():
                        has_results = True
                
                # 简单判断逻辑
                needs_vector_search = has_sql and not has_results and state.get("intent") == QueryType.DATABASE
                needs_fallback = not has_sql or (not has_results and state.get("intent") != QueryType.DATABASE)
                
                # If database results are valid, format them using simple formatting
                formatted_db_results = None
                if has_sql and has_results and db_results:
                    formatted_db_results = self._format_database_response(db_results, sql_query)
                    logger.info("Database results formatted using simple formatting (fallback)")
                
                logger.info(f"Fallback validation: has_sql={has_sql}, has_results={has_results}, needs_vector={needs_vector_search}, needs_fallback={needs_fallback}")
                
                return {
                    **state,
                    "db_results_valid": has_sql and has_results,
                    "needs_vector_search": needs_vector_search,
                    "needs_fallback": needs_fallback,
                    "fallback_executed": False,
                    "validation_reason": "使用备用验证逻辑",
                    "formatted_db_results": formatted_db_results
                }
        else:
            # RAG搜索或其他情况，直接认为有效
            return {
                **state,
                "db_results_valid": True,
                "needs_vector_search": False,
                "needs_fallback": False,
                "fallback_executed": False,
                "validation_reason": "RAG搜索，无需数据库验证"
            }

    async def vector_search_enhancement(self, state: AgentState) -> AgentState:
        """使用向量搜索增强数据库查询"""
        query = state["query"]
        failed_sql = state.get("sql_query")
        memory_info = state.get("memory_info", "")

        logger.info(f"Starting vector search enhancement for query: {query}")
        
        try:
            # 调用向量数据库查询工具
            result = await vector_database_query_tool.ainvoke({
                "question": query,
                "failed_sql": failed_sql,
                "memory_info": memory_info
            })
            
            # 检查向量搜索是否成功
            # 注意：对于COUNT查询，即使结果是0也是成功的查询
            vector_success = result.get("success", False)
            
            if vector_success:
                # 向量搜索成功，更新状态
                results = result.get("results", [])
                logger.info(f"Vector search successful, found {len(results)} results")
                
                # 创建工具消息，显示具体的查询结果
                tool_message = AIMessage(
                    content=f"向量搜索增强结果：\n{result.get('content', '')}"
                )
                
                return {
                    **state,
                    "db_results": results,
                    "sql_query": result.get("sql_query"),
                    "vector_enhanced": True,
                    "suggestions_used": result.get("suggestions_used", {}),
                    "db_results_valid": True,  # 即使COUNT是0，也是有效的查询结果
                    "messages": [*state["messages"], tool_message]
                }
            else:
                # 向量搜索失败（比如SQL生成失败）
                logger.warning("Vector search enhancement failed")
                
                no_result_message = AIMessage(
                    content="向量搜索未能找到匹配的数据库值，可能需要检查查询条件"
                )
                
                # 如果向量搜索也失败，可能需要尝试RAG
                return {
                    **state,
                    "vector_enhanced": False,
                    "needs_fallback": True,  # 尝试RAG作为最后的手段
                    "messages": [*state["messages"], no_result_message]
                }
                
        except Exception as e:
            logger.error(f"Vector search enhancement error: {e}")
            error_message = AIMessage(content=f"向量搜索增强时发生错误: {str(e)}")
            
            return {
                **state,
                "vector_enhanced": False,
                "needs_fallback": True,  # 出错时尝试RAG
                "messages": [*state["messages"], error_message]
            }
    
    async def fallback_search(self, state: AgentState) -> AgentState:
        """当数据库查询无效结果时，执行RAG搜索作为fallback"""
        query = state["query"]
        memory_info = state.get("memory_info", "")
        
        logger.info(f"Executing RAG fallback search for query: {query}")
        
        try:
            # 执行RAG搜索
            result = await rag_search_tool.ainvoke({"query": query, "memory_info": memory_info})
            
            # 存储RAG结果
            rag_results = result.get("rag_results", [])
            source_links = result.get("source_links", [])
            
            # 检查RAG是否有有效结果
            has_rag_results = bool(result.get("content") and result["content"].strip())
            
            if has_rag_results:
                # 标记为混合结果（包含SQL和RAG）
                has_mixed_results = bool(state.get("sql_query"))
                
                # 创建工具消息
                tool_message = AIMessage(content=f"RAG搜索结果（作为补充）：\n{result['content']}")
                
                logger.info(f"Fallback search successful: found RAG answer")
                
                return {
                    **state,
                    "rag_results": rag_results,
                    "source_links": source_links,
                    "fallback_executed": True,
                    "has_mixed_results": has_mixed_results,
                    "messages": [*state["messages"], tool_message]
                }
            else:
                # RAG也没有找到结果
                no_result_message = AIMessage(content="RAG搜索也未找到相关信息")
                
                logger.warning("Fallback search found no results")
                
                return {
                    **state,
                    "fallback_executed": True,
                    "has_mixed_results": False,
                    "messages": [*state["messages"], no_result_message]
                }
                
        except Exception as e:
            logger.error(f"Fallback search error: {e}")
            error_message = AIMessage(content=f"补充搜索时发生错误: {str(e)}")
            return {
                **state,
                "fallback_executed": True,
                "messages": [*state["messages"], error_message]
            }
    
    async def generate_response(self, state: AgentState) -> AgentState:
        """生成最终回答"""
        # 检查工具执行结果类型
        tool_to_use = state.get("tool_to_use")
        rag_results = state.get("rag_results", [])
        db_results = state.get("db_results", [])
        sql_query = state.get("sql_query")
        
        # 如果只有RAG结果，直接返回RAG答案
        if tool_to_use == "rag_search_tool" and rag_results:
            # 从RAG工具消息中提取答案
            rag_answer = ""
            for msg in state["messages"]:
                if isinstance(msg, AIMessage) and "RAG搜索结果" in msg.content:
                    # 提取RAG答案（去掉"RAG搜索结果："前缀）
                    content = msg.content
                    if "RAG搜索结果：\n" in content:
                        rag_answer = content.split("RAG搜索结果：\n", 1)[1]
                    else:
                        rag_answer = content
                    break
            
            if rag_answer:
                logger.info("Returning RAG answer directly")
                ai_message = AIMessage(content=rag_answer)
                return {
                    **state,
                    "final_answer": rag_answer,
                    "messages": [*state["messages"], ai_message]
                }
        
        # 如果只有数据库结果，使用预格式化的结果或格式化数据
        elif tool_to_use == "database_query_tool" and db_results:
            # 优先使用LLM预格式化的结果
            if state.get("formatted_db_results"):
                formatted_answer = state["formatted_db_results"]
                logger.info("Returning LLM-formatted database response")
            else:
                # 回退到简单格式化
                formatted_answer = self._format_database_response(db_results, sql_query)
                logger.info("Returning simple formatted database response")
            
            ai_message = AIMessage(content=formatted_answer)
            return {
                **state,
                "final_answer": formatted_answer,
                "messages": [*state["messages"], ai_message]
            }
        
        # 如果有混合结果，优先使用RAG答案
        elif state.get("has_mixed_results", False) and rag_results:
            # 从RAG工具消息中提取答案
            rag_answer = ""
            for msg in state["messages"]:
                if isinstance(msg, AIMessage) and "RAG搜索结果" in msg.content:
                    content = msg.content
                    if "RAG搜索结果：\n" in content:
                        rag_answer = content.split("RAG搜索结果：\n", 1)[1]
                    else:
                        rag_answer = content
                    break
            
            if rag_answer:
                logger.info("Returning RAG answer from mixed results")
                ai_message = AIMessage(content=rag_answer)
                return {
                    **state,
                    "final_answer": rag_answer,
                    "messages": [*state["messages"], ai_message]
                }
            else:
                return {
                    **state,
                    "final_answer": "抱歉，没有找到相关信息",
                    "messages": [*state["messages"], AIMessage(content="抱歉，没有找到相关信息")]
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
                "rag_results": None,
                "db_results": None,
                "sql_query": None,
                "source_links": None,
                "final_answer": None,
                "reasoning": None,
                "needs_tools": False,
                "tool_to_use": None,
                # 新增字段的默认值
                "db_results_valid": False,
                "needs_fallback": False,
                "needs_vector_search": False,  # Added
                "fallback_executed": False,
                "has_mixed_results": False,
                "vector_enhanced": None,  # Added
                "suggestions_used": None,  # Added
                "validation_reason": None,  # Added
                # Memory-related fields
                "memory_info": None,
                "user_email": user_email,
                "session_id": session_id,
                # Data formatting fields
                "formatted_db_results": None
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
            
            # 构建增强的响应
            response = QueryResponse(
                answer=answer,
                query_type=final_state.get("intent", QueryType.HYBRID),
                reasoning=final_state.get("reasoning"),
                confidence=0.8
            )
            
            # 添加SQL查询信息（如果来自数据库）
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
            
            logger.info(f"Query completed successfully. SQL: {bool(response.sql_query)}, Sources: {len(response.sources) if response.sources else 0}")
            
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