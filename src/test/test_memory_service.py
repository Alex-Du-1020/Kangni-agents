#!/usr/bin/env python3
"""
Memory Service Test Suite for Kangni Agents
Tests all memory-related functionality including creation, retrieval, consolidation, and search
"""
import asyncio
import pytest
import sys
import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "../.."))

from kangni_agents.models.history import QueryHistory, UserFeedback, UserComment, Memory
from kangni_agents.models.database import get_db_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test configuration
TEST_EMAIL = "memory_test@example.com"
TEST_SESSION = "memory-test-session-001"
TEST_EMAIL_2 = "memory_test_2@example.com"
TEST_SESSION_2 = "memory-test-session-002"


async def clear_test_data(test_emails=None):
    """Clear test data from the database for specific users
    
    Args:
        test_emails: List of email addresses to clear data for. If None, clears all data.
    """
    if test_emails is None:
        test_emails = [
            "memory_test@example.com", "memory_test_2@example.com", 
            "nonexistent@example.com"
        ]
    
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


class MemoryServiceTests:
    """Test suite for memory service functionality"""
    
    def __init__(self):
        self.memory_service = None
        self.history_service = None
        self.test_query_ids: List[int] = []
        self.test_memory_ids: List[int] = []
        self.results = {}
        
    async def setup_test_data(self):
        """Create initial test data for memory tests"""
        print("\n📝 Setting up memory test data...")
        try:
            from kangni_agents.services.memory_service import memory_service
            from kangni_agents.services.history_service import history_service
            from kangni_agents.models.history import MemoryType, MemoryImportance
            
            self.memory_service = memory_service
            self.history_service = history_service
            
            # Create test query history entries first
            test_queries = [
                {
                    "session_id": TEST_SESSION,
                    "user_email": TEST_EMAIL,
                    "question": "What is the total number of users in the system?",
                    "answer": "There are 1,234 users in the system. This includes both active and inactive users.",
                    "sql_query": "SELECT COUNT(*) FROM users;",
                    "sources": [{"content": "Database", "score": 0.95}],
                    "query_type": "database",
                    "success": True,
                    "processing_time_ms": 150,
                    "llm_provider": "deepseek",
                    "model_name": "deepseek-chat"
                },
                {
                    "session_id": TEST_SESSION,
                    "user_email": TEST_EMAIL,
                    "question": "How do I configure the database connection?",
                    "answer": "To configure the database connection, you need to set the DB_TYPE environment variable and provide connection details in the config file.",
                    "sources": [{"content": "Documentation", "score": 0.88}],
                    "query_type": "rag",
                    "success": True,
                    "processing_time_ms": 200,
                    "llm_provider": "deepseek",
                    "model_name": "deepseek-chat"
                },
                {
                    "session_id": TEST_SESSION_2,
                    "user_email": TEST_EMAIL_2,
                    "question": "What are the best practices for memory management?",
                    "answer": "Best practices include regular consolidation, proper importance scoring, and cleanup of expired memories.",
                    "sources": [{"content": "Best Practices Guide", "score": 0.92}],
                    "query_type": "rag",
                    "success": True,
                    "processing_time_ms": 180,
                    "llm_provider": "deepseek",
                    "model_name": "deepseek-chat"
                }
            ]
            
            # Save test queries
            for query_data in test_queries:
                history = await history_service.save_query_history(**query_data)
                self.test_query_ids.append(history.id)
            
            print(f"✅ Created {len(test_queries)} test queries")
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup test data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    @pytest.mark.asyncio
    
    async def test_create_memory(self):
        """Test creating different types of memories"""
        print("\n1️⃣ Testing memory creation...")
        
        try:
            from kangni_agents.models.history import MemoryType, MemoryImportance
            
            # Test 1: Create short-term memory
            short_memory = await self.memory_service.create_memory(
                user_email=TEST_EMAIL,
                content="User asked about user count and got answer about 1,234 users",
                memory_type=MemoryType.SHORT_TERM,
                importance=MemoryImportance.MEDIUM,
                session_id=TEST_SESSION,
                source_query_id=self.test_query_ids[0] if self.test_query_ids else None,
                related_entities=["users", "count", "database"],
                tags=["conversation", "recent"]
            )
            self.test_memory_ids.append(short_memory.id)
            print(f"✅ Created short-term memory with ID: {short_memory.id}")
            
            # Test 2: Create long-term memory
            long_memory = await self.memory_service.create_memory(
                user_email=TEST_EMAIL,
                content="Database configuration requires DB_TYPE environment variable",
                memory_type=MemoryType.LONG_TERM,
                importance=MemoryImportance.HIGH,
                session_id=TEST_SESSION,
                source_query_id=self.test_query_ids[1] if len(self.test_query_ids) > 1 else None,
                related_entities=["database", "configuration", "environment"],
                tags=["knowledge", "technical"]
            )
            self.test_memory_ids.append(long_memory.id)
            print(f"✅ Created long-term memory with ID: {long_memory.id}")
            
            # Test 3: Create episodic memory
            episodic_memory = await self.memory_service.create_memory(
                user_email=TEST_EMAIL,
                content="Important interaction: User learned about memory management best practices",
                memory_type=MemoryType.EPISODIC,
                importance=MemoryImportance.CRITICAL,
                session_id=TEST_SESSION,
                source_query_id=self.test_query_ids[2] if len(self.test_query_ids) > 2 else None,
                related_entities=["memory", "management", "best practices"],
                tags=["important", "episodic"]
            )
            self.test_memory_ids.append(episodic_memory.id)
            print(f"✅ Created episodic memory with ID: {episodic_memory.id}")
            
            # Test 4: Create semantic memory
            semantic_memory = await self.memory_service.create_memory(
                user_email=TEST_EMAIL,
                content="Memory consolidation helps maintain system performance",
                memory_type=MemoryType.SEMANTIC,
                importance=MemoryImportance.MEDIUM,
                session_id=TEST_SESSION,
                related_entities=["memory", "consolidation", "performance"],
                tags=["fact", "knowledge"]
            )
            self.test_memory_ids.append(semantic_memory.id)
            print(f"✅ Created semantic memory with ID: {semantic_memory.id}")
            
            self.results["create_memory"] = {"success": True, "count": 4}
            return True
            
        except Exception as e:
            print(f"❌ Error creating memories: {e}")
            import traceback
            traceback.print_exc()
            self.results["create_memory"] = {"success": False, "error": str(e)}
            return False
    
    @pytest.mark.asyncio
    
    async def test_get_relevant_memories(self):
        """Test retrieving relevant memories for a query"""
        print("\n2️⃣ Testing memory retrieval...")
        
        try:
            # Test 1: Get memories for user query about database
            short_term, long_term = await self.memory_service.get_relevant_memories(
                user_email=TEST_EMAIL,
                query="How do I configure the database?",
                session_id=TEST_SESSION,
                limit=5
            )
            
            print(f"✅ Retrieved {len(short_term)} short-term and {len(long_term)} long-term memories")
            
            # Test 2: Get memories for different user
            short_term_2, long_term_2 = await self.memory_service.get_relevant_memories(
                user_email=TEST_EMAIL_2,
                query="What are memory management practices?",
                session_id=TEST_SESSION_2,
                limit=5
            )
            
            print(f"✅ Retrieved {len(short_term_2)} short-term and {len(long_term_2)} long-term memories for user 2")
            
            # Test 3: Get memories with expired filter
            short_term_expired, long_term_expired = await self.memory_service.get_relevant_memories(
                user_email=TEST_EMAIL,
                query="database configuration",
                session_id=TEST_SESSION,
                limit=5,
                include_expired=True
            )
            
            print(f"✅ Retrieved {len(short_term_expired)} short-term and {len(long_term_expired)} long-term memories (including expired)")
            
            self.results["get_relevant_memories"] = {
                "success": True, 
                "user1_short": len(short_term),
                "user1_long": len(long_term),
                "user2_short": len(short_term_2),
                "user2_long": len(long_term_2)
            }
            return True
            
        except Exception as e:
            print(f"❌ Error retrieving memories: {e}")
            import traceback
            traceback.print_exc()
            self.results["get_relevant_memories"] = {"success": False, "error": str(e)}
            return False
    
    @pytest.mark.asyncio
    
    async def test_extract_and_store_memories(self):
        """Test extracting and storing memories from query interactions"""
        print("\n3️⃣ Testing memory extraction and storage...")
        
        try:
            if not self.test_query_ids:
                print("⚠️ No test query IDs available")
                self.results["extract_memories"] = {"success": False, "error": "No query IDs"}
                return False
            
            # Test extracting memories from a query interaction
            memory_ids = await self.memory_service.extract_and_store_memories(
                query_id=self.test_query_ids[0],
                user_email=TEST_EMAIL,
                question="What is the current system status?",
                answer="The system is running normally with 1,234 active users and all services operational.",
                session_id=TEST_SESSION,
                feedback_type="like"
            )
            
            print(f"✅ Extracted and stored {len(memory_ids)} memories from query interaction")
            
            # Test with different feedback type
            memory_ids_2 = await self.memory_service.extract_and_store_memories(
                query_id=self.test_query_ids[1] if len(self.test_query_ids) > 1 else self.test_query_ids[0],
                user_email=TEST_EMAIL,
                question="How to troubleshoot connection issues?",
                answer="Check network connectivity and verify database credentials.",
                session_id=TEST_SESSION,
                feedback_type="dislike"
            )
            
            print(f"✅ Extracted and stored {len(memory_ids_2)} memories with dislike feedback")
            
            self.results["extract_memories"] = {
                "success": True, 
                "first_extraction": len(memory_ids),
                "second_extraction": len(memory_ids_2)
            }
            return True
            
        except Exception as e:
            print(f"❌ Error extracting memories: {e}")
            import traceback
            traceback.print_exc()
            self.results["extract_memories"] = {"success": False, "error": str(e)}
            return False
    
    @pytest.mark.asyncio
    
    async def test_consolidate_memories(self):
        """Test memory consolidation process"""
        print("\n4️⃣ Testing memory consolidation...")
        
        try:
            # Test consolidation for user 1
            consolidation_result = await self.memory_service.consolidate_memories(
                user_email=TEST_EMAIL,
                session_id=TEST_SESSION
            )
            
            print(f"✅ Consolidated memories for {TEST_EMAIL}:")
            print(f"   - Consolidated: {consolidation_result['consolidated']}")
            print(f"   - Expired deleted: {consolidation_result['expired_deleted']}")
            print(f"   - Total processed: {consolidation_result['total_processed']}")
            
            # Test consolidation for user 2
            consolidation_result_2 = await self.memory_service.consolidate_memories(
                user_email=TEST_EMAIL_2,
                session_id=TEST_SESSION_2
            )
            
            print(f"✅ Consolidated memories for {TEST_EMAIL_2}:")
            print(f"   - Consolidated: {consolidation_result_2['consolidated']}")
            print(f"   - Expired deleted: {consolidation_result_2['expired_deleted']}")
            print(f"   - Total processed: {consolidation_result_2['total_processed']}")
            
            self.results["consolidate_memories"] = {
                "success": True,
                "user1": consolidation_result,
                "user2": consolidation_result_2
            }
            return True
            
        except Exception as e:
            print(f"❌ Error consolidating memories: {e}")
            import traceback
            traceback.print_exc()
            self.results["consolidate_memories"] = {"success": False, "error": str(e)}
            return False
    
    @pytest.mark.asyncio
    
    async def test_search_memories(self):
        """Test searching memories by content"""
        print("\n5️⃣ Testing memory search...")
        
        try:
            from kangni_agents.models.history import MemoryType
            
            # Test 1: Search for "database" related memories
            search_results = await self.memory_service.search_memories(
                user_email=TEST_EMAIL,
                search_term="database",
                memory_types=[MemoryType.LONG_TERM, MemoryType.SEMANTIC],
                limit=10
            )
            
            print(f"✅ Found {len(search_results)} memories containing 'database'")
            
            # Test 2: Search for "memory" related memories
            search_results_2 = await self.memory_service.search_memories(
                user_email=TEST_EMAIL,
                search_term="memory",
                limit=5
            )
            
            print(f"✅ Found {len(search_results_2)} memories containing 'memory'")
            
            # Test 3: Search for non-existent term
            search_results_3 = await self.memory_service.search_memories(
                user_email=TEST_EMAIL,
                search_term="nonexistent_term_xyz",
                limit=5
            )
            
            print(f"✅ Found {len(search_results_3)} memories containing 'nonexistent_term_xyz' (should be 0)")
            
            # Display some search results
            if search_results:
                print("   Sample search results:")
                for i, result in enumerate(search_results[:3], 1):
                    print(f"   {i}. {result['content'][:100]}... (Type: {result['memory_type']})")
            
            self.results["search_memories"] = {
                "success": True,
                "database_results": len(search_results),
                "memory_results": len(search_results_2),
                "nonexistent_results": len(search_results_3)
            }
            return True
            
        except Exception as e:
            print(f"❌ Error searching memories: {e}")
            import traceback
            traceback.print_exc()
            self.results["search_memories"] = {"success": False, "error": str(e)}
            return False
    
    @pytest.mark.asyncio
    
    async def test_get_memory_context_for_agent(self):
        """Test getting formatted memory context for agent"""
        print("\n6️⃣ Testing memory context for agent...")
        
        try:
            # Test getting memory context for user 1
            context = await self.memory_service.get_memory_context_for_agent(
                user_email=TEST_EMAIL,
                question="How do I configure the database connection?",
                session_id=TEST_SESSION
            )
            
            print(f"✅ Retrieved memory context for agent:")
            print(f"   - Short-term memories: {len(context['short_term_memories'])}")
            print(f"   - Long-term memories: {len(context['long_term_memories'])}")
            print(f"   - Recent interactions: {len(context['recent_interactions'])}")
            print(f"   - User profile: {context['user_profile']['email']}")
            
            # Test getting memory context for user 2
            context_2 = await self.memory_service.get_memory_context_for_agent(
                user_email=TEST_EMAIL_2,
                question="What are the best practices?",
                session_id=TEST_SESSION_2
            )
            
            print(f"✅ Retrieved memory context for user 2:")
            print(f"   - Short-term memories: {len(context_2['short_term_memories'])}")
            print(f"   - Long-term memories: {len(context_2['long_term_memories'])}")
            print(f"   - Recent interactions: {len(context_2['recent_interactions'])}")
            
            # Test with non-existent user (should return empty context)
            context_empty = await self.memory_service.get_memory_context_for_agent(
                user_email="nonexistent@example.com",
                question="Test question",
                session_id="nonexistent-session"
            )
            
            print(f"✅ Retrieved empty context for non-existent user:")
            print(f"   - Short-term memories: {len(context_empty['short_term_memories'])}")
            print(f"   - Long-term memories: {len(context_empty['long_term_memories'])}")
            
            self.results["memory_context"] = {
                "success": True,
                "user1_context": {
                    "short_term": len(context['short_term_memories']),
                    "long_term": len(context['long_term_memories']),
                    "recent": len(context['recent_interactions'])
                },
                "user2_context": {
                    "short_term": len(context_2['short_term_memories']),
                    "long_term": len(context_2['long_term_memories']),
                    "recent": len(context_2['recent_interactions'])
                }
            }
            return True
            
        except Exception as e:
            print(f"❌ Error getting memory context: {e}")
            import traceback
            traceback.print_exc()
            self.results["memory_context"] = {"success": False, "error": str(e)}
            return False
    
    @pytest.mark.asyncio
    
    async def test_memory_cleanup(self):
        """Test memory cleanup and expiry"""
        print("\n7️⃣ Testing memory cleanup...")
        
        try:
            from kangni_agents.models.history import MemoryType, MemoryImportance
            from datetime import datetime, timedelta
            
            # Create a memory that should expire soon
            expired_memory = await self.memory_service.create_memory(
                user_email=TEST_EMAIL,
                content="This memory should expire soon",
                memory_type=MemoryType.SHORT_TERM,
                importance=MemoryImportance.LOW,
                session_id=TEST_SESSION,
                related_entities=["test", "expiry"],
                tags=["temporary"]
            )
            
            print(f"✅ Created memory with ID: {expired_memory.id} (will expire)")
            
            # Test getting memories before expiry
            short_term, long_term = await self.memory_service.get_relevant_memories(
                user_email=TEST_EMAIL,
                query="expiry test",
                session_id=TEST_SESSION,
                limit=10,
                include_expired=False
            )
            
            print(f"✅ Retrieved {len(short_term)} short-term memories (excluding expired)")
            
            # Test getting memories including expired
            short_term_all, long_term_all = await self.memory_service.get_relevant_memories(
                user_email=TEST_EMAIL,
                query="expiry test",
                session_id=TEST_SESSION,
                limit=10,
                include_expired=True
            )
            
            print(f"✅ Retrieved {len(short_term_all)} short-term memories (including expired)")
            
            # Test consolidation (should clean up expired memories)
            consolidation_result = await self.memory_service.consolidate_memories(
                user_email=TEST_EMAIL,
                session_id=TEST_SESSION
            )
            
            print(f"✅ Cleanup result: {consolidation_result['expired_deleted']} expired memories deleted")
            
            self.results["memory_cleanup"] = {
                "success": True,
                "before_expiry": len(short_term),
                "after_expiry": len(short_term_all),
                "expired_deleted": consolidation_result['expired_deleted']
            }
            return True
            
        except Exception as e:
            print(f"❌ Error testing memory cleanup: {e}")
            import traceback
            traceback.print_exc()
            self.results["memory_cleanup"] = {"success": False, "error": str(e)}
            return False
    
    @pytest.mark.asyncio
    
    async def test_chinese_text_processing(self):
        """Test Chinese text processing functionality"""
        print("\n8️⃣ Testing Chinese text processing...")
        
        try:
            from kangni_agents.models.history import MemoryType, MemoryImportance
            
            # Test Chinese text samples
            chinese_texts = [
                "数据库配置需要设置DB_TYPE环境变量，这是系统运行的基础要求。",
                "用户经常询问关于订单统计的问题，特别是生产订单表kn_quality_trace_prod_order的数据。",
                "内存管理系统需要定期清理过期数据，以提高系统性能。"
            ]
            
            # Test keyword extraction
            print("测试中文关键词提取:")
            for i, text in enumerate(chinese_texts, 1):
                keywords = self.memory_service._extract_keywords(text)
                print(f"  文本 {i}: {keywords}")
            
            # Test entity extraction
            print("测试中文实体提取:")
            for i, text in enumerate(chinese_texts, 1):
                entities = self.memory_service._extract_entities(text)
                print(f"  文本 {i}: {entities}")
            
            # Test fact extraction
            print("测试中文事实提取:")
            for i, text in enumerate(chinese_texts, 1):
                facts = self.memory_service._extract_facts(text)
                print(f"  文本 {i}: {facts}")
            
            # Test creating memories with Chinese content
            chinese_memory = await self.memory_service.create_memory(
                user_email=TEST_EMAIL,
                content="用户经常询问数据库配置问题，特别是DB_TYPE环境变量的设置方法。",
                memory_type=MemoryType.LONG_TERM,
                importance=MemoryImportance.HIGH,
                session_id=TEST_SESSION,
                related_entities=["数据库", "配置", "环境变量"],
                tags=["技术", "常见问题"]
            )
            print(f"✅ Created Chinese memory with ID: {chinese_memory.id}")
            
            # Test searching Chinese memories
            chinese_search = await self.memory_service.search_memories(
                user_email=TEST_EMAIL,
                search_term="数据库",
                limit=5
            )
            print(f"✅ Chinese search returned {len(chinese_search)} results")
            
            self.results["chinese_text_processing"] = {"success": True}
            return True
            
        except Exception as e:
            print(f"❌ Error testing Chinese text processing: {e}")
            import traceback
            traceback.print_exc()
            self.results["chinese_text_processing"] = {"success": False, "error": str(e)}
            return False
    
    @pytest.mark.asyncio
    
    async def test_memory_edge_cases(self):
        """Test edge cases and error handling"""
        print("\n9️⃣ Testing memory edge cases...")
        
        try:
            # Test 1: Create memory with minimal data
            minimal_memory = await self.memory_service.create_memory(
                user_email=TEST_EMAIL,
                content="Minimal memory test"
            )
            print(f"✅ Created minimal memory with ID: {minimal_memory.id}")
            
            # Test 2: Search with empty search term
            empty_search = await self.memory_service.search_memories(
                user_email=TEST_EMAIL,
                search_term="",
                limit=5
            )
            print(f"✅ Empty search returned {len(empty_search)} results")
            
            # Test 3: Get memories for non-existent user
            non_existent_memories = await self.memory_service.get_relevant_memories(
                user_email="nonexistent@example.com",
                query="test query",
                limit=5
            )
            print(f"✅ Non-existent user returned {len(non_existent_memories[0])} short-term and {len(non_existent_memories[1])} long-term memories")
            
            # Test 4: Extract memories with None values
            try:
                memory_ids = await self.memory_service.extract_and_store_memories(
                    query_id=None,
                    user_email=TEST_EMAIL,
                    question="Test question",
                    answer="Test answer",
                    session_id=None,
                    feedback_type=None
                )
                print(f"✅ Extracted memories with None values: {len(memory_ids)} memories")
            except Exception as e:
                print(f"⚠️ Expected error with None values: {e}")
            
            self.results["edge_cases"] = {"success": True}
            return True
            
        except Exception as e:
            print(f"❌ Error testing edge cases: {e}")
            import traceback
            traceback.print_exc()
            self.results["edge_cases"] = {"success": False, "error": str(e)}
            return False
    
    async def run_all_tests(self):
        """Run all memory service tests"""
        print("🚀 Starting Memory Service Tests")
        print("=" * 60)
        
        try:
            # Clear any existing test data
            await clear_test_data()
            
            # Setup test data
            if not await self.setup_test_data():
                print("❌ Failed to setup test data, aborting tests")
                return False
        
            # Run all tests
            test_methods = [
                self.test_create_memory,
                self.test_get_relevant_memories,
                self.test_extract_and_store_memories,
                self.test_consolidate_memories,
                self.test_search_memories,
                self.test_get_memory_context_for_agent,
                self.test_memory_cleanup,
                self.test_chinese_text_processing,
                self.test_memory_edge_cases
            ]
            
            passed = 0
            failed = 0
            
            for test_method in test_methods:
                try:
                    if await test_method():
                        passed += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"❌ Test {test_method.__name__} failed with error: {e}")
                    import traceback
                    traceback.print_exc()
                    failed += 1
            
            # Print summary
            print("\n" + "=" * 60)
            print("📊 MEMORY SERVICE TEST SUMMARY")
            print("=" * 60)
            print(f"Total tests: {passed + failed}")
            print(f"Passed: {passed}")
            print(f"Failed: {failed}")
            print(f"Success rate: {(passed/(passed+failed)*100):.1f}%")
            
            if failed == 0:
                print("\n✅ All memory service tests passed!")
            else:
                print(f"\n❌ {failed} test(s) failed")
            
            return failed == 0
            
        except Exception as e:
            print(f"❌ Test suite failed with error: {e}")
            return False
        finally:
            # Clean up test data
            print("\n🧹 Cleaning up test data...")
            await clear_test_data()


async def main():
    """Main test function"""
    # Initialize database if needed
    print("Initializing database...")
    try:
        from kangni_agents.models.database import get_db_config
        db_config = get_db_config()
        print(f"✅ Database initialized")
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        return 1
    
    # Run tests
    tests = MemoryServiceTests()
    success = await tests.run_all_tests()
    
    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
