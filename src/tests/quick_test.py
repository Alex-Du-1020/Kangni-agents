#!/usr/bin/env python3
"""Quick test script for basic functionality verification"""

import asyncio
import sys
import os

# Add kangni_agents directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

async def quick_test():
    """Quick test of basic functionality"""
    print("🚀 Quick Test for Kangni Agents")
    print("=" * 40)
    
    try:
        # Test 1: Configuration
        print("1. Testing configuration...")
        from kangni_agents.config import settings
        print(f"   ✅ Base URL: {settings.openai_base_url}")
        print(f"   ✅ Model: {settings.llm_chat_model}")
        print(f"   ✅ API Key: {'Set' if settings.openai_api_key else 'Not set'}")
        
        # Test 2: LLM Provider
        print("\n2. Testing LLM provider...")
        from kangni_agents.services.database_service import db_service
        print(f"   ✅ Provider: {db_service.llm_provider}")
        
        # Test 3: Specific Question
        print("\n3. Testing specific question...")
        question = "德里地铁4期项目(20D21028C000)在故障信息查询中共发生多少起故障？"
        result = await db_service.query_database(question)
        
        if result.get('success') and '5' in str(result.get('results', [])):
            print(f"   ✅ Answer: {result.get('answer')}")
            print("   ✅ Expected result (5) found!")
        else:
            print("   ❌ Expected result not found")
            return False
        
        # Test 4: Application
        print("\n4. Testing application startup...")
        from kangni_agents.main import app
        print("   ✅ FastAPI app created successfully")
        
        print("\n🎉 Quick test passed! System is ready to use.")
        print("\nNext steps:")
        print("- Run 'python src/tests/test_all.py' for comprehensive testing")
        print("- Start the server with './dev_server.sh'")
        print("- Access the API at http://localhost:8000/docs")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Quick test failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check your .env file configuration")
        print("2. Ensure all dependencies are installed: pip install -e .")
        print("3. Verify database and RAG services are accessible")
        return False

if __name__ == "__main__":
    success = asyncio.run(quick_test())
    sys.exit(0 if success else 1)
