"""
Test the enhanced validate_results function with LLM data formatting
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from kangni_agents.agents.react_agent import KangniReActAgent, AgentState
from kangni_agents.models import QueryType
from langchain_core.messages import HumanMessage, AIMessage


class TestEnhancedValidation:
    """Test enhanced validation with LLM data formatting"""
    
    @pytest.fixture
    def agent(self):
        """Create agent instance for testing"""
        with patch('kangni_agents.agents.react_agent.llm_service') as mock_llm:
            mock_llm.llm_available = True
            mock_llm.llm_provider = "test"
            mock_llm.chat = AsyncMock()
            return KangniReActAgent()
    
    @pytest.mark.asyncio
    async def test_validate_results_with_llm_formatting(self, agent):
        """Test that validate_results formats data with LLM when SQL results are valid"""
        # Mock LLM response that includes both validation and formatting
        combined_response = AIMessage(content="""
        SQL生成成功: 是
        有效数据: 是
        需要向量搜索: 否
        需要RAG备用: 否
        原因: 查询成功返回了数据
        格式化结果: 根据查询结果，共找到 2 条订单记录：
        1. 订单ID: 12345, 状态: 已完成, 金额: 1000元
        2. 订单ID: 12346, 状态: 进行中, 金额: 500元
        """)
        
        # Mock the LLM service
        with patch('kangni_agents.agents.react_agent.llm_service') as mock_llm:
            mock_llm.chat = AsyncMock(return_value=combined_response)
            
            # Test state with valid database results
            state = AgentState(
                messages=[HumanMessage(content="查询订单信息")],
                query="查询订单信息",
                intent=QueryType.DATABASE,
                rag_results=None,
                db_results=[
                    {"order_id": 12345, "status": "已完成", "amount": 1000},
                    {"order_id": 12346, "status": "进行中", "amount": 500}
                ],
                sql_query="SELECT * FROM orders",
                source_links=None,
                final_answer=None,
                reasoning=None,
                needs_tools=True,
                tool_to_use="database_query_tool",
                db_results_valid=False,
                needs_fallback=False,
                needs_vector_search=False,
                fallback_executed=False,
                has_mixed_results=False,
                vector_enhanced=None,
                suggestions_used=None,
                validation_reason=None,
                memory_info="",
                user_email="test@example.com",
                session_id="test_session",
                start_time=None,
                formatted_db_results=None
            )
            
            # Call validate_results
            result = await agent.validate_results(state)
            
            # Verify that validation passed
            assert result["db_results_valid"] is True
            assert result["needs_vector_search"] is False
            assert result["needs_fallback"] is False
            
            # Verify that data was formatted by LLM
            assert result["formatted_db_results"] is not None
            assert "共找到 2 条订单记录" in result["formatted_db_results"]
            assert "订单id: 12345" in result["formatted_db_results"]
            
            # Verify LLM was called only once (combined validation + formatting)
            assert mock_llm.chat.call_count == 1
    
    @pytest.mark.asyncio
    async def test_validate_results_fallback_formatting(self, agent):
        """Test that fallback validation formats data with simple formatting"""
        # Mock LLM to raise an exception during validation, triggering fallback
        with patch('kangni_agents.agents.react_agent.llm_service') as mock_llm:
            # LLM validation raises exception, triggering fallback
            mock_llm.chat = AsyncMock(side_effect=Exception("LLM validation error"))
            
            # Test state with valid database results
            state = AgentState(
                messages=[HumanMessage(content="查询项目信息")],
                query="查询项目信息",
                intent=QueryType.DATABASE,
                rag_results=None,
                db_results=[{"project_name": "测试项目", "status": "进行中"}],
                sql_query="SELECT * FROM projects",
                source_links=None,
                final_answer=None,
                reasoning=None,
                needs_tools=True,
                tool_to_use="database_query_tool",
                db_results_valid=False,
                needs_fallback=False,
                needs_vector_search=False,
                fallback_executed=False,
                has_mixed_results=False,
                vector_enhanced=None,
                suggestions_used=None,
                validation_reason=None,
                memory_info="",
                user_email="test@example.com",
                session_id="test_session",
                start_time=None,
                formatted_db_results=None
            )
            
            # Call validate_results (this will trigger fallback validation)
            result = await agent.validate_results(state)
            
            # Verify that validation passed
            assert result["db_results_valid"] is True
            
            # Verify that data was formatted using simple formatting
            assert result["formatted_db_results"] is not None
            assert "查询结果：共找到 1 条记录" in result["formatted_db_results"]
            assert "project_name: 测试项目" in result["formatted_db_results"]
    
    @pytest.mark.asyncio
    async def test_generate_response_uses_formatted_results(self, agent):
        """Test that generate_response uses pre-formatted database results"""
        # Test state with pre-formatted results
        state = AgentState(
            messages=[HumanMessage(content="查询订单信息")],
            query="查询订单信息",
            intent=QueryType.DATABASE,
            rag_results=None,
            db_results=[{"order_id": 12345, "status": "已完成"}],
            sql_query="SELECT * FROM orders",
            source_links=None,
            final_answer=None,
            reasoning=None,
            needs_tools=True,
            tool_to_use="database_query_tool",
            db_results_valid=True,
            needs_fallback=False,
            needs_vector_search=False,
            fallback_executed=False,
            has_mixed_results=False,
            vector_enhanced=None,
            suggestions_used=None,
            validation_reason="查询成功",
            memory_info="",
            user_email="test@example.com",
            session_id="test_session",
            query_history_id=None,
            start_time=None,
            formatted_db_results="根据查询结果，找到 1 条订单记录：订单ID 12345 状态为已完成"
        )
        
        # Call generate_response
        result = await agent.generate_response(state)
        
        # Verify that pre-formatted results are used
        assert result["final_answer"] == "根据查询结果，找到 1 条订单记录：订单ID 12345 状态为已完成"
        assert len(result["messages"]) == 2  # Original message + AI response
    
    @pytest.mark.asyncio
    async def test_validate_results_error_handling(self, agent):
        """Test error handling in validate_results when LLM fails"""
        # Mock LLM to raise an exception during validation
        with patch('kangni_agents.agents.react_agent.llm_service') as mock_llm:
            mock_llm.chat = AsyncMock(side_effect=Exception("LLM error"))
            
            # Test state with valid database results
            state = AgentState(
                messages=[HumanMessage(content="查询测试信息")],
                query="查询测试信息",
                intent=QueryType.DATABASE,
                rag_results=None,
                db_results=[{"id": 1, "name": "test"}],
                sql_query="SELECT * FROM test",
                source_links=None,
                final_answer=None,
                reasoning=None,
                needs_tools=True,
                tool_to_use="database_query_tool",
                db_results_valid=False,
                needs_fallback=False,
                needs_vector_search=False,
                fallback_executed=False,
                has_mixed_results=False,
                vector_enhanced=None,
                suggestions_used=None,
                validation_reason=None,
                memory_info="",
                user_email="test@example.com",
                session_id="test_session",
                start_time=None,
                formatted_db_results=None
            )
            
            # Call validate_results (this will trigger fallback validation)
            result = await agent.validate_results(state)
            
            # Should fallback to simple validation and formatting
            assert result["db_results_valid"] is True
            assert "查询结果：共找到 1 条记录" in result["formatted_db_results"]
            assert "id: 1, name: test" in result["formatted_db_results"]


if __name__ == "__main__":
    pytest.main([__file__])
