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
    final_answer: Optional[str]
    reasoning: Optional[str]

@tool
async def rag_search_tool(query: str, dataset_id: Optional[str] = None) -> str:
    """搜索RAG文档库获取相关信息"""
    if not dataset_id:
        dataset_id = settings.ragflow_default_dataset_id
    
    results = await rag_service.search_rag(query, dataset_id)
    
    if not results:
        return "未找到相关文档信息"
    
    # 格式化结果
    formatted_results = []
    for i, result in enumerate(results[:5], 1):
        formatted_results.append(f"{i}. {result.content[:500]}...")
    
    return "\n".join(formatted_results)

@tool 
async def database_query_tool(question: str) -> str:
    """查询数据库获取统计信息"""
    result = await db_service.query_database(question)
    
    if not result.get("success"):
        return f"数据库查询失败: {result.get('error', '未知错误')}"
    
    return json.dumps({
        "sql_query": result.get("sql_query"),
        "results": result.get("results", [])
    }, ensure_ascii=False, indent=2)

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
        workflow.add_edge("agent_reasoning", "tool_execution")
        
        # 条件边：如果有工具调用，继续到生成响应；否则直接结束
        def should_continue(state: AgentState) -> str:
            # 检查是否有工具调用
            messages = state["messages"]
            for msg in reversed(messages):
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    return "generate_response"
            return END
        
        workflow.add_conditional_edges("tool_execution", should_continue)
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
        system_prompt = f"""你是一个智能助手，需要回答用户问题。你有两个工具可用：

1. rag_search_tool: 用于搜索文档和知识库，适合回答概念、原因、方法等问题
2. database_query_tool: 用于查询数据库，适合回答统计、数据分析等问题

当前问题意图分类为: {intent}
分类原因: {state.get('reasoning', '')}

请根据问题内容决定是否需要使用工具：
- 如果问题需要搜索文档或知识库信息，使用 rag_search_tool
- 如果问题需要查询数据库或统计数据，使用 database_query_tool
- 如果问题很简单或你已经知道答案，可以直接回答，不需要使用工具

用户问题: {query}

请分析问题并决定是否需要使用工具。如果需要，请调用相应的工具；如果不需要，请直接回答。
"""
        
        # 转换消息格式为LLMMessage
        llm_messages = []
        for msg in state["messages"]:
            if hasattr(msg, 'content'):
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                llm_messages.append(LLMMessage(role=role, content=msg.content))
        
        # 添加系统提示
        llm_messages.append(LLMMessage(role="user", content=system_prompt))
        
        # Debug logging for LLM input
        logger.debug(f"LLM input messages: {[{'role': msg.role, 'content': msg.content} for msg in llm_messages]}")
        
        try:
            response = await llm_service.chat(llm_messages)
            
            # Debug logging for LLM output
            logger.debug(f"LLM output: {response.content}")
            
            # 创建AIMessage响应
            ai_message = AIMessage(content=response.content)
            
            return {
                **state,
                "messages": [*state["messages"], ai_message]
            }
            
        except Exception as e:
            logger.error(f"LLM chat error: {e}")
            error_message = AIMessage(content=f"抱歉，处理您的问题时发生错误: {str(e)}")
            return {
                **state,
                "messages": [*state["messages"], error_message]
            }
    
    async def execute_tools(self, state: AgentState) -> AgentState:
        """工具执行节点"""
        last_message = state["messages"][-1]
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            tool_messages = []
            
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_call_id = tool_call["id"]
                
                logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
                
                try:
                    if tool_name == "rag_search_tool":
                        result = await rag_search_tool.ainvoke(tool_args)
                    elif tool_name == "database_query_tool":
                        result = await database_query_tool.ainvoke(tool_args)
                    else:
                        result = f"Unknown tool: {tool_name}"
                    
                    # 创建正确的ToolMessage
                    tool_message = ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call_id
                    )
                    tool_messages.append(tool_message)
                    
                    # Debug logging for tool execution result
                    logger.debug(f"Tool {tool_name} execution result: {str(result)[:200]}...")
                    
                except Exception as e:
                    logger.error(f"Tool execution error: {e}")
                    # 创建错误消息
                    tool_message = ToolMessage(
                        content=f"工具执行错误: {str(e)}",
                        tool_call_id=tool_call_id
                    )
                    tool_messages.append(tool_message)
            
            return {
                **state,
                "messages": [*state["messages"], *tool_messages]
            }
        
        return state
    
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
                "final_answer": None,
                "reasoning": None
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
            
            return QueryResponse(
                answer=answer,
                query_type=final_state.get("intent", QueryType.HYBRID),
                reasoning=final_state.get("reasoning"),
                confidence=0.8
            )
            
        except Exception as e:
            logger.error(f"Agent query error: {e}")
            return QueryResponse(
                answer=f"处理查询时发生错误: {str(e)}",
                query_type=QueryType.HYBRID,
                confidence=0.0
            )

# 全局实例
kangni_agent = KangniReActAgent()