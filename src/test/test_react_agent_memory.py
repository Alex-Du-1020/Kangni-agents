#!/usr/bin/env python3
"""
React Agent Memory Integration Test Suite for Kangni Agents
Tests the react agent with memory functionality integration
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test configuration
TEST_EMAIL = "agent_memory_test@example.com"
TEST_SESSION = "agent-memory-test-session-111"
TEST_EMAIL_2 = "agent_memory_test_2@example.com"
TEST_SESSION_2 = "agent-memory-test-session-222"
TEST_EMAIL_3 = "agent_memory_test_3@example.com"
TEST_SESSION_3 = "agent-memory-test-session-333"


class ReactAgentMemoryTests:
    """Test suite for react agent with memory functionality"""
    
    def __init__(self):
        self.agent = None
        self.memory_service = None
        self.history_service = None
        self.test_query_ids: List[int] = []
        self.results = {}
        
    async def setup_test_data(self):
        """Create initial test data for agent memory tests"""
        from kangni_agents.agents.react_agent import kangni_agent
            
        self.agent = kangni_agent
        from kangni_agents.services.memory_service import memory_service
        self.memory_service = memory_service
        from kangni_agents.models.history import MemoryType, MemoryImportance
        
        await self.memory_service.create_memory(
            user_email=TEST_EMAIL,
            content="""Q: 内解锁接地线线束短，无法安装到紧固螺钉位置是那个项目发生的？
A: 内解锁接地线线束短、无法安装到紧固螺钉位置的故障发生在 **东莞1号线项目**。

解决方案：
1. 检查线束长度是否符合设计要求
2. 重新测量紧固螺钉位置
3. 更换合适长度的接地线线束
4. 调整安装位置或使用延长线束

来源：文档 1（东莞1号线项目故障分析报告）。""",
            memory_type=MemoryType.SHORT_TERM,
            importance=MemoryImportance.HIGH,
            session_id=TEST_SESSION
        )

        await self.memory_service.create_memory(
            user_email=TEST_EMAIL_2,
            content="""Q: 合肥S1号线项目乘客室门这个项目一共有多少个生产订单？
A: 合肥s1号线项目乘客室门项目共有 **209** 个生产订单。
            """,
            memory_type=MemoryType.SHORT_TERM,
            importance=MemoryImportance.HIGH,
            session_id=TEST_SESSION_2
        )


        await self.memory_service.create_memory(
            user_email=TEST_EMAIL_3,
            content="""Q: 上海1号线项目的发生了多少起故障，他们的故障模式分别是什么
A: 上海1号线项目共发生了 **4 起** 故障，故障模式为 **“其他”**。
            """,
            memory_type=MemoryType.SHORT_TERM,
            importance=MemoryImportance.HIGH,
            session_id=TEST_SESSION_3
        )
            
        # Check if agent is available
        if not self.agent.llm_available:
            print("⚠️ LLM service not available, will test memory functionality only")
        
        return True
    
    @pytest.mark.asyncio
    
    async def test_agent_with_memory_context(self):
        """Test agent query with memory context"""
        print("\n1️⃣ Testing agent query with memory context...")
        
        try:
            if not self.agent.llm_available:
                print("⚠️ LLM not available, skipping test")
                self.results["agent_memory_context"] = {"success": False, "error": "LLM not available"}
                return False
            
            test_question = "这个项目问题怎么解决？"
            
            response = await self.agent.query(
                question=test_question,
                user_email=TEST_EMAIL,
                session_id=TEST_SESSION
            )
            
            print(f"✅ Agent response received:")
            print(f"   - Answer: {response}")
            print(f"   - Query type: {response.query_type}")
            print(f"   - Confidence: {response.confidence}")
            print(f"   - Has SQL: {response.sql_query}")
            print(f"   - Has sources: {len(response.sources) if response.sources else 0}")
            
            # Check if response contains memory-related information
            if response.answer and ("东莞1号线" in response.answer or "接地线" in response.answer or "线束" in response.answer):
                assert True, "Response contains memory context"
            else:
                assert False, "Response does not contain memory context"
            
            # Check if memories were saved to database
            await self.check_memories_saved(TEST_EMAIL)
            
            self.results["agent_memory_context"] = {
                "success": True,
                "answer_length": len(response.answer) if response.answer else 0,
                "query_type": str(response.query_type),
                "confidence": response.confidence,
                "has_sql": bool(response.sql_query),
                "has_sources": len(response.sources) if response.sources else 0
            }
            return True
            
        except Exception as e:
            print(f"❌ Error testing agent with memory: {e}")
            import traceback
            traceback.print_exc()
            self.results["agent_memory_context"] = {"success": False, "error": str(e)}
            return False
    
    @pytest.mark.asyncio
    async def test_agent_with_memory_sql_query(self):
        """Test agent query with memory context"""
        print("\n1️⃣ Testing agent query with memory context...")
        
        try:
            if not self.agent.llm_available:
                print("⚠️ LLM not available, skipping test")
                self.results["agent_memory_context"] = {"success": False, "error": "LLM not available"}
                return False
            
            test_question = "列出这个项目所有的生产线？"
            
            response = await self.agent.query(
                question=test_question,
                user_email=TEST_EMAIL_2,
                session_id=TEST_SESSION_2
            )
            
            print(f"✅ Agent response received:")
            print(f"   - Answer: {response}")
            print(f"   - Query type: {response.query_type}")
            print(f"   - Confidence: {response.confidence}")
            print(f"   - Has SQL: {response.sql_query}")
            print(f"   - Has sources: {len(response.sources) if response.sources else 0}")

            # Check if response contains memory-related information
            if response.answer and ("东莞1号线" in response.answer or "接地线" in response.answer or "线束" in response.answer):
                assert True
            else:
                print("⚠️ Response may not be using memory context effectively")
                assert False
            
            # Check if memories were saved to database
            await self.check_memories_saved(TEST_EMAIL_3)
            
            self.results["agent_memory_context"] = {
                "success": True,
                "answer_length": len(response.answer) if response.answer else 0,
                "query_type": str(response.query_type),
                "confidence": response.confidence,
                "has_sql": bool(response.sql_query),
                "has_sources": len(response.sources) if response.sources else 0
            }
            return True
            
        except Exception as e:
            print(f"❌ Error testing agent with memory: {e}")
            import traceback
            traceback.print_exc()
            self.results["agent_memory_context"] = {"success": False, "error": str(e)}
            return False
   
    @pytest.mark.asyncio
    async def test_agent_with_memory_fault(self):
        """Test agent query with memory context"""
        print("\n1️⃣ Testing agent query with memory context...")
        
        try:
            if not self.agent.llm_available:
                print("⚠️ LLM not available, skipping test")
                self.results["agent_memory_context"] = {"success": False, "error": "LLM not available"}
                return False
            
            test_question = "当前项目故障件发生什么问题了?"
            
            response = await self.agent.query(
                question=test_question,
                user_email=TEST_EMAIL_3,
                session_id=TEST_SESSION_3
            )
            
            print(f"✅ Agent response received:")
            print(f"   - Answer: {response}")
            print(f"   - Query type: {response.query_type}")
            print(f"   - Confidence: {response.confidence}")
            print(f"   - Has SQL: {response.sql_query}")
            print(f"   - Has sources: {len(response.sources) if response.sources else 0}")
            
            # Check if response contains memory-related information
            if response.answer and ("上海1号线" in response.answer or "基础部件" in response.answer or "其他零部件" in response.answer):
                assert True
            else:
                print("⚠️ Response may not be using memory context effectively")
                assert False
            
            # Check if memories were saved to database
            await self.check_memories_saved(TEST_EMAIL_2)
            
            self.results["agent_memory_context"] = {
                "success": True,
                "answer_length": len(response.answer) if response.answer else 0,
                "query_type": str(response.query_type),
                "confidence": response.confidence,
                "has_sql": bool(response.sql_query),
                "has_sources": len(response.sources) if response.sources else 0
            }
            return True
            
        except Exception as e:
            print(f"❌ Error testing agent with memory: {e}")
            import traceback
            traceback.print_exc()
            self.results["agent_memory_context"] = {"success": False, "error": str(e)}
            return False
   

    async def check_memories_saved(self, user_email: str):
        """Check if memories are being saved to the database"""
        try:
            from kangni_agents.models.history import Memory
            from kangni_agents.models.database import get_db_config
            
            db_config = get_db_config()
            
            with db_config.session_scope() as session:
                # Count memories for test user
                memory_count = session.query(Memory).filter(
                    Memory.user_email == user_email
                ).count()
                
                print(f"📊 Database check:")
                print(f"   - Memories in database for {user_email}: {memory_count}")
                
                if memory_count > 0:
                    print("✅ Memories are being saved to database")
                    
                    # Show some sample memories
                    recent_memories = session.query(Memory).filter(
                        Memory.user_email == user_email
                    ).order_by(Memory.created_at.desc()).limit(3).all()
                    
                    print("   Recent memories:")
                    for i, memory in enumerate(recent_memories, 1):
                        print(f"   {i}. {memory.content[:60]}... (Type: {memory.memory_type.value})")
                else:
                    print("⚠️ No memories found in database - memory saving may not be working")
                
                return memory_count > 0
                
        except Exception as e:
            print(f"❌ Error checking memories: {e}")
            return False
    
    async def cleanup_test_data(self):
        """Clean up test data from database"""
        print("\n🧹 Cleaning up test data...")
        
        try:
            from kangni_agents.models.history import QueryHistory, Memory
            from kangni_agents.models.database import get_db_config
            
            db_config = get_db_config()
            
            with db_config.session_scope() as session:
                # Clear memories for test users
                memories_deleted = session.query(Memory).filter(
                    Memory.user_email.in_([TEST_EMAIL, TEST_EMAIL_2])
                ).delete()
                
                # Clear query history for test users
                history_deleted = session.query(QueryHistory).filter(
                    QueryHistory.user_email.in_([TEST_EMAIL, TEST_EMAIL_2])
                ).delete()
                
                session.commit()
                
                print(f"✅ Cleanup completed:")
                print(f"   - Memories deleted: {memories_deleted}")
                print(f"   - Query history deleted: {history_deleted}")
                
                return True
                
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def run_all_tests(self):
        """Run all react agent memory tests"""
        print("🚀 Starting React Agent Memory Tests")
        print("=" * 60)
        
        # Setup test data
        if not await self.setup_test_data():
            print("❌ Failed to setup test data, aborting tests")
            return False
        
        # Run all tests
        test_methods = [
            # self.test_agent_with_memory_context,
            self.test_agent_with_memory_sql_query,
            # self.test_agent_with_memory_fault
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
        print("📊 REACT AGENT MEMORY TEST SUMMARY")
        print("=" * 60)
        print(f"Total tests: {passed + failed}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success rate: {(passed/(passed+failed)*100):.1f}%")
        
        if failed == 0:
            print("\n✅ All react agent memory tests passed!")
        else:
            print(f"\n❌ {failed} test(s) failed")
        
        # Clean up test data after tests complete
        # Comment out the next line when doing manual testing to keep test data
        await self.cleanup_test_data()  # <-- Comment this line for manual testing
        
        return failed == 0


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
    tests = ReactAgentMemoryTests()
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
