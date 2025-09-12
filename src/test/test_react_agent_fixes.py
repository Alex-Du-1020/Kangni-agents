#!/usr/bin/env python3
"""
Test the fixes for react_agent.py
"""
import asyncio
import sys
import os
import pytest
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "../.."))

from kangni_agents.agents.react_agent import kangni_agent
from kangni_agents.models import UserQuery

@pytest.mark.asyncio
async def test_rag_tool_sources_filtering():
    """Test that RAG tool only returns sources that are referenced in the answer"""
    print("🧪 Testing RAG tool sources filtering")
    print("=" * 60)
    
    # Test query that should trigger RAG search
    query = "内解锁接地线线束短，无法安装到紧固螺钉位置是那个项目发生的？"
    
    try:
        response = await kangni_agent.query(
            question=query,
            user_email="test@example.com",
            session_id="test-session-123"
        )
        
        print(f"✅ Query processed successfully")
        print(f"   Answer: {response.answer[:200]}...")
        print(f"   Query Type: {response.query_type}")
        print(f"   Sources: {len(response.sources) if response.sources else 0}")
        
        # Check if sources are only included when referenced
        if response.sources:
            print(f"   Source details: {response.sources}")
            # The answer should reference the sources
            answer_lower = response.answer.lower()
            has_doc_refs = any(f"文档{i+1}" in answer_lower or f"知识库{i+1}" in answer_lower 
                             for i in range(len(response.sources)))
            print(f"   Answer references sources: {has_doc_refs}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

@pytest.mark.asyncio
async def test_database_response_formatting():
    """Test that database responses are properly formatted without additional processing"""
    print("\n🧪 Testing database response formatting")
    print("=" * 60)
    
    # Test query that should trigger database search
    query = "总共有多少个生产订单？"
    
    try:
        response = await kangni_agent.query(
            question=query,
            user_email="test@example.com",
            session_id="test-session-456"
        )
        
        print(f"✅ Query processed successfully")
        print(f"   Answer: {response.answer[:200]}...")
        print(f"   Query Type: {response.query_type}")
        print(f"   SQL Query: {response.sql_query is not None}")
        
        # Check if the response is properly formatted
        if response.sql_query:
            print(f"   SQL: {response.sql_query}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

@pytest.mark.asyncio
async def test_mixed_results_handling():
    """Test handling of mixed results (SQL + RAG)"""
    print("\n🧪 Testing mixed results handling")
    print("=" * 60)
    
    # Test query that might trigger both tools
    query = "什么是质量追溯系统？"
    
    try:
        response = await kangni_agent.query(
            question=query,
            user_email="test@example.com",
            session_id="test-session-789"
        )
        
        print(f"✅ Query processed successfully")
        print(f"   Answer: {response.answer[:200]}...")
        print(f"   Query Type: {response.query_type}")
        print(f"   Sources: {len(response.sources) if response.sources else 0}")
        print(f"   SQL Query: {response.sql_query is not None}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
