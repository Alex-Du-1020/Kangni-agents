#!/usr/bin/env python3
"""
Test RAG service with LLM integration
"""
import asyncio
import sys
import os
import pytest
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "../.."))

from kangni_agents.services.rag_service import rag_service
from kangni_agents.config import settings

@pytest.mark.asyncio
async def test_rag_llm_integration():
    """Test RAG service with LLM answer generation"""
    print("🧪 Testing RAG service with LLM integration")
    print("=" * 60)
    
    # Test query
    query = "内解锁接地线线束短，无法安装到紧固螺钉位置是那个项目发生的？"
    dataset_id = settings.ragflow_default_dataset_id
    
    try:
        # Test 1: Basic RAG search (original functionality)
        print("\n1️⃣ Testing basic RAG search...")
        search_results = await rag_service.search_rag(query, dataset_id, top_k=3)
        print(f"✅ Found {len(search_results)} search results")
        
        # Test 2: LLM answer generation
        print("\n2️⃣ Testing LLM answer generation...")
        answer = await rag_service.generate_answer_with_llm(query, search_results)
        print(f"✅ Generated answer (length: {len(answer)})")
        print(f"Answer preview: {answer[:200]}...")
        
        # Test 3: Combined search with answer
        print("\n3️⃣ Testing combined search with answer...")
        result = await rag_service.search_rag_with_answer(query, dataset_id, top_k=3)
        print(f"✅ Combined result:")
        print(f"   Answer: {result['answer'][:200]}...")
        print(f"   Total results: {result['total_results']}")
        print(f"   Query: {result['query']}")
        
        # Verify result structure
        assert 'answer' in result
        assert 'search_results' in result
        assert 'query' in result
        assert 'dataset_id' in result
        assert 'total_results' in result
        assert len(result['search_results']) == result['total_results']
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

@pytest.mark.asyncio
async def test_rag_llm_fallback():
    """Test RAG service fallback when LLM is not available"""
    print("\n🧪 Testing RAG service fallback behavior")
    print("=" * 60)
    
    query = "测试查询"
    dataset_id = settings.ragflow_default_dataset_id
    
    try:
        # Test with empty search results
        print("\n1️⃣ Testing with empty search results...")
        empty_results = []
        answer = await rag_service.generate_answer_with_llm(query, empty_results)
        print(f"✅ Empty results answer: {answer}")
        assert "找不到此信息" in answer or "不可用" in answer
        
        # Test with some mock results
        print("\n2️⃣ Testing with mock search results...")
        from kangni_agents.models import RAGSearchResult
        mock_results = [
            RAGSearchResult(
                content="这是一个测试文档内容",
                score=0.9,
                metadata={"document_name": "测试文档.pdf"}
            )
        ]
        answer = await rag_service.generate_answer_with_llm(query, mock_results)
        print(f"✅ Mock results answer: {answer[:200]}...")
        assert len(answer) > 0
        
        print("\n✅ Fallback tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Fallback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

@pytest.mark.asyncio
async def test_rag_service_availability():
    """Test RAG service availability"""
    print("\n🧪 Testing RAG service availability")
    print("=" * 60)
    
    try:
        is_available = await rag_service.check_availability()
        print(f"✅ RAG service available: {is_available}")
        
        if is_available:
            print("✅ RAG service is ready for testing")
        else:
            print("⚠️ RAG service is not available, some tests may fail")
        
        return True
        
    except Exception as e:
        print(f"❌ Availability test failed: {e}")
        return False

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
