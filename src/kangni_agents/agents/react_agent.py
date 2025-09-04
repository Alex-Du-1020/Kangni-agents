from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from ..config import settings
from ..models import QueryType, RAGSearchResult, QueryResponse
from ..models.llm_implementations import llm_service
from ..models.llm_providers import LLMMessage
from ..services.rag_service import rag_service
from ..services.database_service import db_service
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

@tool
async def rag_search_tool(query: str, dataset_id: Optional[str] = None) -> Dict[str, Any]:
    """搜索RAG文档库获取相关信息"""
    if not dataset_id:
        dataset_id = settings.ragflow_default_dataset_id
    
    results = await rag_service.search_rag(query, dataset_id)
    
    if not results:
        return {
            "content": "未找到相关文档信息",
            "sources": [],
            "source_links": []
        }
    
    # 格式化结果 - 只返回核心信息
    formatted_results = []
    source_links = []
    
    for i, result in enumerate(results[:5], 1):
        # 简化响应，只包含必要信息
        doc_info = {
            "document_id": getattr(result, 'document_id', f'doc_{i}'),
            "dataset_id": dataset_id,
            "document_name": getattr(result, 'document_name', f'Document {i}')
        }
        
        # 构建简洁的显示文本
        formatted_results.append(f"{i}. [{doc_info['document_name']}] (ID: {doc_info['document_id']})")
        
        if result.metadata and 'source' in result.metadata:
            source_links.append(result.metadata['source'])
    
    return {
        "content": "\n".join(formatted_results),
        "sources": results[:5],
        "source_links": list(set(source_links))  # 去重
    }

@tool 
async def database_query_tool(question: str) -> Dict[str, Any]:
    """查询数据库获取统计信息"""
    result = await db_service.query_database(question)
    
    if not result.get("success"):
        return {
            "content": f"数据库查询失败: {result.get('error', '未知错误')}",
            "sql_query": None,
            "results": []
        }
    
    formatted_content = json.dumps({
        "sql_query": result.get("sql_query"),
        "results": result.get("results", [])
    }, ensure_ascii=False, indent=2)
    
    return {
        "content": formatted_content,
        "sql_query": result.get("sql_query"),
        "results": result.get("results", [])
    }

class KangniReActAgent:
    def __init__(self):
        # 使用集中式LLM服务
        self.llm_available = llm_service.llm_available
        self.llm_provider = llm_service.llm_provider
        
        if self.llm_available:
            try:
                # 绑定工具
                self.tools = [rag_search_tool, database_query_tool]
                
                # 构建状态图
                self.workflow = self._build_workflow()
                
                logger.info(f"Agent initialized successfully with LLM provider: {self.llm_provider}")
                
            except Exception as e:
                logger.warning(f"Failed to initialize agent: {e}")
                self.llm_available = False
                self.workflow = None
        else:
            logger.warning("LLM service not available, agent features will be disabled")
            self.workflow = None
    
    def _build_workflow(self) -> StateGraph:
        """构建LangGraph工作流"""
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("classify_intent", self.classify_intent)
        workflow.add_node("agent_reasoning", self.agent_reasoning)
        workflow.add_node("tool_execution", self.execute_tools)
        workflow.add_node("generate_response", self.generate_response)
        
        # 设置入口点
        workflow.set_entry_point("classify_intent")
        
        # 添加边
        workflow.add_edge("classify_intent", "agent_reasoning")
        
        # 条件边：检查是否需要工具
        def should_use_tools(state: AgentState) -> str:
            return "tool_execution" if state.get("needs_tools", False) else "generate_response"
        
        workflow.add_conditional_edges("agent_reasoning", should_use_tools)
        workflow.add_edge("tool_execution", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()
    
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
        
        # 构建系统提示
        system_prompt = f"""你是一个智能助手，需要分析用户问题并决定是否需要使用工具。你有两个工具可用：

1. rag_search_tool: 用于搜索文档和知识库，适合回答概念、原因、方法等问题
2. database_query_tool: 用于查询数据库，适合回答统计、数据分析等问题

当前问题意图分类为: {intent}
分类原因: {state.get('reasoning', '')}

用户问题: {query}

请分析问题并回答以下两个问题：
1. 这个问题是否需要使用工具？(回答：是 或 否)
2. 如果需要工具，应该使用哪个工具？为什么？
3. 如果不需要工具，请直接回答用户的问题。

请按以下格式回答：
需要工具: [是/否]
工具选择: [rag_search_tool/database_query_tool/无]
理由: [简要说明]
回答: [如果不需要工具，请直接回答；如果需要工具，请说明工具调用计划]
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
            
            # 解析响应，确定是否需要工具
            response_text = response.content.lower()
            needs_tools = "需要工具: 是" in response_text or "需要工具：是" in response_text
            
            # 确定工具类型
            tool_to_use = None
            if needs_tools:
                if "rag_search_tool" in response_text:
                    tool_to_use = "rag_search_tool"
                elif "database_query_tool" in response_text:
                    tool_to_use = "database_query_tool"
            
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
        
        if not tool_to_use:
            logger.warning("No tool specified for execution")
            return state
            
        logger.info(f"Executing tool: {tool_to_use} with query: {query}")
        
        try:
            if tool_to_use == "rag_search_tool":
                result = await rag_search_tool.ainvoke({"query": query})
                
                # 存储RAG结果
                state["rag_results"] = result.get("sources", [])
                state["source_links"] = result.get("source_links", [])
                
                # 创建工具消息
                tool_message = AIMessage(content=f"RAG搜索结果：\n{result['content']}")
                
            elif tool_to_use == "database_query_tool":
                result = await database_query_tool.ainvoke({"question": query})
                
                # 存储数据库结果
                state["db_results"] = result.get("results", [])
                state["sql_query"] = result.get("sql_query")
                
                # 创建工具消息
                tool_message = AIMessage(content=f"数据库查询结果：\n{result['content']}")
                
            else:
                tool_message = AIMessage(content=f"未知工具: {tool_to_use}")
            
            logger.debug(f"Tool {tool_to_use} execution result: {tool_message.content[:200]}...")
            
            return {
                **state,
                "messages": [*state["messages"], tool_message]
            }
            
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            error_message = AIMessage(content=f"工具执行错误: {str(e)}")
            return {
                **state,
                "messages": [*state["messages"], error_message]
            }
    
    async def generate_response(self, state: AgentState) -> AgentState:
        """生成最终回答"""
        # 检查是否有工具结果
        has_tool_results = False
        for msg in state["messages"]:
            if isinstance(msg, ToolMessage):
                has_tool_results = True
                break
        
        if has_tool_results:
            # 如果有工具结果，基于工具结果生成回答
            final_prompt = """请基于以上工具执行结果生成最终回答。回答要求：
1. 准确、完整地回答用户问题
2. 合理整合工具结果信息
3. 保持回答的逻辑性和可读性
4. 如果工具结果不足，要明确说明

请直接给出最终答案，不要重复工具调用过程。
"""
        else:
            # 如果没有工具结果，直接回答
            final_prompt = """请基于以上对话生成最终回答。回答要求：
1. 准确、完整地回答用户问题
2. 保持回答的逻辑性和可读性
3. 如果信息不足，要明确说明

请直接给出最终答案。
"""
        
        # 转换消息格式为LLMMessage
        llm_messages = []
        for msg in state["messages"]:
            if hasattr(msg, 'content'):
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                llm_messages.append(LLMMessage(role=role, content=msg.content))
        
        # 添加最终提示
        llm_messages.append(LLMMessage(role="user", content=final_prompt))
        
        # Debug logging for final response generation
        logger.debug(f"Final response generation input: {[{'role': msg.role, 'content': msg.content} for msg in llm_messages]}")
        
        try:
            response = await llm_service.chat(llm_messages)
            
            # Debug logging for final response output
            logger.debug(f"Final response output: {response.content}")
            
            # 创建AIMessage响应
            ai_message = AIMessage(content=response.content)
            
            return {
                **state,
                "final_answer": response.content,
                "messages": [*state["messages"], ai_message]
            }
            
        except Exception as e:
            logger.error(f"LLM final response error: {e}")
            error_message = AIMessage(content=f"抱歉，生成最终回答时发生错误: {str(e)}")
            return {
                **state,
                "final_answer": f"抱歉，生成最终回答时发生错误: {str(e)}",
                "messages": [*state["messages"], error_message]
            }
    
    async def query(self, question: str, context: Optional[str] = None) -> QueryResponse:
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
                "tool_to_use": None
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
                response.sources = final_state["rag_results"]
            
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