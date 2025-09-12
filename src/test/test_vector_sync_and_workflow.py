#!/usr/bin/env python3
"""
Test script for vector sync and improved agent workflow
"""

import asyncio
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.kangni_agents.services.database_service import db_service
from src.kangni_agents.services.vector_embedding_service import vector_service
from src.kangni_agents.agents.react_agent import kangni_agent
from src.kangni_agents.models.history import QueryHistory, UserFeedback, UserComment, Memory
from src.kangni_agents.models.database import get_db_config

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def clear_test_data(test_emails=None):
    """Clear test data from the database for specific users
    
    Args:
        test_emails: List of email addresses to clear data for. If None, clears all data.
    """
    if test_emails is None:
        test_emails = ["test@example.com"]
    
    print(f"🧹 Clearing test data for emails: {test_emails}")
    
    try:
        db_config = get_db_config()
        with db_config.session_scope() as session:
            
            # 1. Clear memories for test users
            session.query(Memory).filter(Memory.user_email.in_(test_emails)).delete()
            
            # 2. Clear comments for test users
            session.query(UserComment).filter(UserComment.user_email.in_(test_emails)).delete()
            
            # 3. Clear feedback for test users
            session.query(UserFeedback).filter(UserFeedback.user_email.in_(test_emails)).delete()
            
            # 4. Clear query history for test users
            session.query(QueryHistory).filter(QueryHistory.user_email.in_(test_emails)).delete()
            
            session.commit()
            print("✅ Test data cleared successfully")
            
    except Exception as e:
        print(f"❌ Error clearing test data: {e}")
        raise


@pytest.mark.asyncio


async def test_vector_sync():
    """Test the vector sync functionality"""
    print("\n" + "="*50)
    print("Testing Vector Sync Functionality")
    print("="*50)
    
    try:
        
        # Test syncing production orders with a small limit
        print("\nSyncing production order project names...")
        result = await vector_service.sync_table_field_data(
            table_name="kn_quality_trace_prod_order",
            field_name="projectname_s",
            limit=10  # Start with a small number for testing
        )
        
        if result['success']:
            print(f"✅ Sync successful!")
            print(f"   - Synced: {result['synced']} records")
            print(f"   - Skipped: {result['skipped']} records")
            print(f"   - Errors: {result['errors']} records")
            print(f"   - Total: {result['total']} records")
            if 'message' in result:
                print(f"   - Message: {result['message']}")
        else:
            print(f"❌ Sync failed: {result.get('error', 'Unknown error')}")
            return False
            
        # Test vector search to verify embeddings were created
        print("\n\nTesting vector search for similar project names...")
        test_query = "德里地铁项目"
        suggestions = await vector_service.get_field_values_for_query(
            query_text=test_query,
            table_name="kn_quality_trace_prod_order",
            field_name="projectname_s"
        )
        
        if suggestions:
            print(f"✅ Vector search found {len(suggestions)} similar values:")
            for i, suggestion in enumerate(suggestions[:5], 1):
                print(f"   {i}. {suggestion}")
        else:
            print("⚠️ No similar values found (this might be expected if no similar data exists)")
            
        return True
        
    except Exception as e:
        print(f"❌ Error during vector sync test: {e}")
        logger.error(f"Vector sync test error: {e}", exc_info=True)
        return False

@pytest.mark.asyncio

async def test_agent_workflow():
    """Test the improved agent workflow with vector fallback"""
    print("\n" + "="*50)
    print("Testing Agent Workflow with Vector Fallback")
    print("="*50)
    
    try:
        # Test case 1: Query that might need vector enhancement
        test_queries = [
            {
                "query": "德里地铁项目有多少订单？",
                "description": "Testing with potentially fuzzy project name"
            },
            {
                "query": "生产订单总数是多少？",
                "description": "Testing simple count query"
            },
            {
                "query": "地铁项目的故障统计",
                "description": "Testing partial match that needs vector search"
            }
        ]
        
        for test_case in test_queries:
            print(f"\n\nTest: {test_case['description']}")
            print(f"Query: {test_case['query']}")
            print("-" * 40)
            
            # Execute query through the agent
            response = await kangni_agent.query(test_case['query'])
            
            print(f"Answer: {response.answer[:200]}...")
            print(f"Query Type: {response.query_type}")
            
            if response.sql_query:
                print(f"SQL Generated: {response.sql_query}")
            
            if hasattr(response, 'sources') and response.sources:
                print(f"Sources: {len(response.sources)} documents")
                
        return True
        
    except Exception as e:
        print(f"❌ Error during agent workflow test: {e}")
        logger.error(f"Agent workflow test error: {e}", exc_info=True)
        return False

async def main():
    """Main test function"""
    print("\n" + "="*60)
    print("TESTING VECTOR SYNC AND IMPROVED AGENT WORKFLOW")
    print("="*60)
    
    try:
        # Clear any existing test data
        await clear_test_data()
        
        # Test 1: Vector Sync
        sync_success = await test_vector_sync()
        
        # Test 2: Agent Workflow  
        workflow_success = await test_agent_workflow()
        
        # Summary
        print("\n" + "="*50)
        print("TEST SUMMARY")
        print("="*50)
        print(f"Vector Sync Test: {'✅ PASSED' if sync_success else '❌ FAILED'}")
        print(f"Agent Workflow Test: {'✅ PASSED' if workflow_success else '❌ FAILED'}")
        
        if sync_success and workflow_success:
            print("\n🎉 All tests passed successfully!")
            return 0
        else:
            print("\n⚠️ Some tests failed. Please check the logs above.")
            return 1
            
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        return 1
    finally:
        # Clean up test data
        print("\n🧹 Cleaning up test data...")
        await clear_test_data()

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)