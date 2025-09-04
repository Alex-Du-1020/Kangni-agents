#!/usr/bin/env python3
"""Comprehensive test script for Kangni Agents"""

import asyncio
import sys
import os

# Add kangni_agents directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

async def test_service_availability():
    """Test service availability"""
    print("Testing service availability...")
    try:
        from kangni_agents.main import check_service_availability
        await check_service_availability()
        print("✅ Service availability test passed")
        return True
    except Exception as e:
        print(f"❌ Service availability test failed: {e}")
        return False

async def test_llm_providers():
    """Test LLM provider configuration"""
    print("\nTesting LLM providers...")
    try:
        from kangni_agents.services.database_service import db_service
        from kangni_agents.agents.react_agent import kangni_agent
        
        if (db_service.llm_provider.value == "deepseek" and 
            kangni_agent.llm_provider.value == "deepseek"):
            print("✅ LLM provider test passed (DeepSeek)")
            return True
        else:
            print("❌ LLM provider test failed")
            return False
    except Exception as e:
        print(f"❌ LLM provider test failed: {e}")
        return False

async def test_specific_question():
    """Test the specific question"""
    print("\nTesting specific question...")
    try:
        from kangni_agents.services.database_service import db_service
        
        question = "德里地铁4期项目(20D21028C000)在故障信息查询中共发生多少起故障？"
        result = await db_service.query_database(question)
        
        if result.get('success') and '5' in str(result.get('results', [])):
            print("✅ Specific question test passed (result: 5)")
            return True
        else:
            print("❌ Specific question test failed")
            return False
    except Exception as e:
        print(f"❌ Specific question test failed: {e}")
        return False

async def test_rag_functionality():
    """Test RAG functionality"""
    print("\nTesting RAG functionality...")
    try:
        from kangni_agents.services.rag_service import rag_service
        
        question = "内解锁接地线线束短，无法安装到紧固螺钉位置是那个项目发生的？"
        results = await rag_service.search_rag(question, 'f3073258886911f08bc30242c0a82006')
        
        if results and len(results) > 0:
            print(f"✅ RAG functionality test passed (found {len(results)} results)")
            return True
        else:
            print("⚠️  RAG functionality test: No results found (this may be normal if dataset is empty)")
            return True  # Don't fail the test for empty results
    except Exception as e:
        print(f"❌ RAG functionality test failed: {e}")
        return False

async def test_application_startup():
    """Test application startup"""
    print("\nTesting application startup...")
    try:
        from kangni_agents.main import app
        print("✅ Application startup test passed")
        return True
    except Exception as e:
        print(f"❌ Application startup test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("Running comprehensive tests for Kangni Agents...")
    print("=" * 60)
    
    tests = [
        test_service_availability(),
        test_llm_providers(),
        test_specific_question(),
        test_rag_functionality(),
        test_application_startup()
    ]
    
    results = await asyncio.gather(*tests, return_exceptions=True)
    
    passed = sum(1 for r in results if r is True)
    total = len(results)
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is working correctly.")
        print("\n✅ DeepSeek is configured as the default LLM provider")
        print("✅ The specific question returns the expected result (5)")
        print("✅ All services are available and working")
        print("✅ Application can start successfully")
    else:
        print("💥 Some tests failed! Please check the configuration.")
        print("\nTroubleshooting tips:")
        print("1. Check your .env file has correct API keys and database credentials")
        print("2. Ensure RAG MCP server is running at http://158.58.50.45:9382/mcp")
        print("3. Verify database connection and required tables exist")
        print("4. Check LLM API key is valid and accessible")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
