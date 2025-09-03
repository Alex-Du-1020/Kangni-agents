from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langgraph import StateGraph, END
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

from ..config import settings
from ..models import QueryType, RAGSearchResult, QueryResponse
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
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model="gpt-4",
            temperature=0.1
        )
        
        # 绑定工具
        self.tools = [rag_search_tool, database_query_tool]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # 构建状态图
        self.workflow = self._build_workflow()
    
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
        system_prompt = f"""你是一个智能助手，需要回答用户问题。你有两个工具可用：

1. rag_search_tool: 用于搜索文档和知识库，适合回答概念、原因、方法等问题
2. database_query_tool: 用于查询数据库，适合回答统计、数据分析等问题

当前问题意图分类为: {intent}
分类原因: {state.get('reasoning', '')}

请根据问题内容决定使用哪些工具，并进行推理回答。如果需要使用工具，请先调用相应工具获取信息。

用户问题: {query}
"""
        
        messages = [
            *state["messages"],
            HumanMessage(content=system_prompt)
        ]
        
        response = await self.llm_with_tools.ainvoke(messages)
        
        return {
            **state,
            "messages": [*state["messages"], response]
        }
    
    async def execute_tools(self, state: AgentState) -> AgentState:
        """工具执行节点"""
        last_message = state["messages"][-1]
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            tool_results = []
            
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
                
                try:
                    if tool_name == "rag_search_tool":
                        result = await rag_search_tool.ainvoke(tool_args)
                    elif tool_name == "database_query_tool":
                        result = await database_query_tool.ainvoke(tool_args)
                    else:
                        result = f"Unknown tool: {tool_name}"
                    
                    tool_results.append({
                        "tool": tool_name,
                        "result": result
                    })
                    
                except Exception as e:
                    logger.error(f"Tool execution error: {e}")
                    tool_results.append({
                        "tool": tool_name,
                        "result": f"工具执行错误: {str(e)}"
                    })
            
            # 将工具结果添加到消息中
            tool_message = HumanMessage(content=f"工具执行结果:\n{json.dumps(tool_results, ensure_ascii=False, indent=2)}")
            
            return {
                **state,
                "messages": [*state["messages"], tool_message]
            }
        
        return state
    
    async def generate_response(self, state: AgentState) -> AgentState:
        """生成最终回答"""
        # 添加最终回答的提示
        final_prompt = """请基于以上信息生成最终回答。回答要求：
1. 准确、完整地回答用户问题
2. 如果使用了工具结果，要合理整合信息
3. 保持回答的逻辑性和可读性
4. 如果信息不足，要明确说明

请直接给出最终答案，不要重复工具调用过程。
"""
        
        messages = [
            *state["messages"],
            HumanMessage(content=final_prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        
        return {
            **state,
            "final_answer": response.content,
            "messages": [*state["messages"], response]
        }
    
    async def query(self, question: str, context: Optional[str] = None) -> QueryResponse:
        """处理用户查询"""
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
            
            return QueryResponse(
                answer=final_state.get("final_answer", "抱歉，无法生成回答"),
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