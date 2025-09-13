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
TEST_SESSION = "agent-memory-test-session-005"
TEST_EMAIL_2 = "agent_memory_test_2@example.com"
TEST_SESSION_2 = "agent-memory-test-session-002"


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
        print("\n📝 Setting up agent memory test data...")
        try:
            from kangni_agents.agents.react_agent import kangni_agent
            from kangni_agents.services.memory_service import memory_service
            from kangni_agents.services.history_service import history_service
            from kangni_agents.models.history import MemoryType, MemoryImportance
            
            self.agent = kangni_agent
            self.memory_service = memory_service
            self.history_service = history_service
            
            # Check if agent is available
            if not self.agent.llm_available:
                print("⚠️ LLM service not available, will test memory functionality only")
                # Continue with setup even without LLM
            
            # Load test cases from JSON file
            import json
            test_cases_path = Path(__file__).parent / "data" / "test_cases.json"
            with open(test_cases_path, 'r', encoding='utf-8') as f:
                test_cases = json.load(f)
            
            # Create test query history entries from test cases
            test_queries = []
            
            # Database query test cases (with SQL)
            db_test_cases = [case for case in test_cases if 'SQL' in case]
            for i, case in enumerate(db_test_cases[:3]):  # Take first 3 database cases
                test_queries.append({
                    "session_id": TEST_SESSION,
                    "user_email": TEST_EMAIL,
                    "question": case["question"],
                    "answer": f"根据查询结果，{case['question'].split('？')[0]}的答案是相关的统计数据。",
                    "sql_query": case["SQL"],
                    "sources": [{"content": "Database", "score": 0.95}],
                    "query_type": "database",
                    "success": True,
                    "processing_time_ms": 150 + i * 20,
                    "llm_provider": "deepseek",
                    "model_name": "deepseek-chat"
                })
            
            # RAG query test cases (with keywords)
            rag_test_cases = [case for case in test_cases if 'keywords' in case]
            for i, case in enumerate(rag_test_cases[:3]):  # Take first 3 RAG cases
                keywords_str = "、".join(case["keywords"])
                test_queries.append({
                    "session_id": TEST_SESSION,
                    "user_email": TEST_EMAIL,
                    "question": case["question"],
                    "answer": f"根据文档查询，{case['question'].split('？')[0]}的答案是：{keywords_str}。",
                    "sources": [{"content": "Documentation", "score": 0.88}],
                    "query_type": "rag",
                    "success": True,
                    "processing_time_ms": 200 + i * 30,
                    "llm_provider": "deepseek",
                    "model_name": "deepseek-chat"
                })
            
            # Add one test case for second user
            if len(test_cases) > 6:
                case = test_cases[6]
                keywords_str = "、".join(case.get("keywords", ["相关信息"]))
                test_queries.append({
                    "session_id": TEST_SESSION_2,
                    "user_email": TEST_EMAIL_2,
                    "question": case["question"],
                    "answer": f"根据查询结果，{case['question'].split('？')[0]}的答案是：{keywords_str}。",
                    "sources": [{"content": "Best Practices Guide", "score": 0.92}],
                    "query_type": "rag",
                    "success": True,
                    "processing_time_ms": 180,
                    "llm_provider": "deepseek",
                    "model_name": "deepseek-chat"
                })
            
            # Save test queries
            for query_data in test_queries:
                history = await history_service.save_query_history(**query_data)
                self.test_query_ids.append(history.id)
            
            # Create test memories based on Chinese test cases
            test_memories = [
                {
                    "user_email": TEST_EMAIL,
                    "content": "用户经常询问生产订单统计和项目相关信息，特别是合肥S1号线项目乘客室门相关数据",
                    "memory_type": MemoryType.LONG_TERM,
                    "importance": MemoryImportance.HIGH,
                    "session_id": TEST_SESSION,
                    "related_entities": ["生产订单", "合肥S1号线", "乘客室门", "统计"],
                    "tags": ["pattern", "user_behavior", "项目查询"]
                },
                {
                    "user_email": TEST_EMAIL,
                    "content": "用户关注质量工程师信息和团队人员配置，如深圳14号线门扇外观问题团队",
                    "memory_type": MemoryType.SEMANTIC,
                    "importance": MemoryImportance.HIGH,
                    "session_id": TEST_SESSION,
                    "related_entities": ["质量工程师", "深圳14号线", "门扇外观", "团队"],
                    "tags": ["knowledge", "人员查询"]
                },
                {
                    "user_email": TEST_EMAIL,
                    "content": "用户经常查询BOM物料信息和供应商信息，如0128000064物料的供应商",
                    "memory_type": MemoryType.LONG_TERM,
                    "importance": MemoryImportance.MEDIUM,
                    "session_id": TEST_SESSION,
                    "related_entities": ["BOM", "物料", "供应商", "0128000064"],
                    "tags": ["pattern", "物料查询"]
                },
                {
                    "user_email": TEST_EMAIL_2,
                    "content": "用户对故障信息查询和NCR问题性质分析感兴趣",
                    "memory_type": MemoryType.LONG_TERM,
                    "importance": MemoryImportance.MEDIUM,
                    "session_id": TEST_SESSION_2,
                    "related_entities": ["故障信息", "NCR", "问题性质", "分析"],
                    "tags": ["pattern", "质量分析"]
                }
            ]
            
            # Create test memories
            for memory_data in test_memories:
                memory = await memory_service.create_memory(**memory_data)
                print(f"✅ Created test memory with ID: {memory.id}")
            
            print(f"✅ Created {len(test_queries)} test queries and {len(test_memories)} test memories")
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup test data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    @pytest.mark.asyncio
    
    async def test_memory_functionality_only(self):
        """Test memory functionality without requiring LLM"""
        print("\n0️⃣ Testing memory functionality (no LLM required)...")
        
        try:
            # Test memory creation
            from kangni_agents.models.history import MemoryType, MemoryImportance
            
            test_memory = await self.memory_service.create_memory(
                user_email=TEST_EMAIL,
                content="测试记忆：用户经常查询生产订单信息",
                memory_type=MemoryType.LONG_TERM,
                importance=MemoryImportance.HIGH,
                session_id=TEST_SESSION,
                related_entities=["生产订单", "测试"],
                tags=["test", "memory"]
            )
            
            print(f"✅ Created test memory with ID: {test_memory.id}")
            
            # Test memory retrieval
            short_term, long_term = await self.memory_service.get_relevant_memories(
                user_email=TEST_EMAIL,
                query="生产订单查询",
                session_id=TEST_SESSION,
                limit=5
            )
            
            print(f"✅ Retrieved memories: {len(short_term)} short-term, {len(long_term)} long-term")
            
            # Test memory search
            search_results = await self.memory_service.search_memories(
                user_email=TEST_EMAIL,
                search_term="生产订单",
                limit=5
            )
            
            print(f"✅ Memory search found {len(search_results)} results")
            
            # Test memory context generation
            context = await self.memory_service.get_memory_context_for_agent(
                user_email=TEST_EMAIL,
                question="测试问题",
                session_id=TEST_SESSION
            )
            
            print(f"✅ Memory context generated: {len(context['long_term_memories'])} long-term memories")
            
            # Check if memories are saved to database
            await self.check_memories_saved()
            
            self.results["memory_functionality"] = {"success": True}
            return True
            
        except Exception as e:
            print(f"❌ Error testing memory functionality: {e}")
            import traceback
            traceback.print_exc()
            self.results["memory_functionality"] = {"success": False, "error": str(e)}
            return False
    
    @pytest.mark.asyncio
    
    async def test_agent_with_memory_context(self):
        """Test agent query with memory context"""
        print("\n1️⃣ Testing agent query with memory context...")
        
        try:
            if not self.agent.llm_available:
                print("⚠️ LLM not available, skipping test")
                self.results["agent_memory_context"] = {"success": False, "error": "LLM not available"}
                return False
            
            # Test query that should benefit from memory context
            test_question = "#合肥S1号线项目乘客室门#这个项目一共有多少个去重后生产订单？"
            # test_question = "这个项目主要的生产线有哪几个？"
            
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
            if response.answer and ("生产订单" in response.answer or "合肥S1号线" in response.answer):
                print("✅ Response contains memory context (project information mentioned)")
            else:
                print("⚠️ Response may not be using memory context effectively")
            
            # Check if memories were saved to database
            await self.check_memories_saved()
            
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
    
    async def test_agent_memory_workflow(self):
        """Test the complete memory workflow in agent"""
        print("\n2️⃣ Testing agent memory workflow...")
        
        try:
            if not self.agent.llm_available:
                print("⚠️ LLM not available, skipping test")
                self.results["agent_memory_workflow"] = {"success": False, "error": "LLM not available"}
                return False
            
            # Test a series of related queries to see memory accumulation
            test_questions = [
                "深圳14号线门扇外观问题团队中的质量工程师是谁？",
                "BOM中0128000064物料的供应商是谁？",
                "刚才查询的质量工程师信息是什么？"
            ]
            
            responses = []
            for i, question in enumerate(test_questions):
                print(f"   Query {i+1}: {question}")
                
                response = await self.agent.query(
                    question=question,
                    user_email=TEST_EMAIL,
                    session_id=TEST_SESSION
                )
                
                responses.append({
                    "question": question,
                    "answer": response.answer,
                    "query_type": response.query_type,
                    "has_memory_context": "刚才" in response.answer or "之前" in response.answer if response.answer else False
                })
                
                print(f"   Response: {response.answer[:100] if response.answer else 'No answer'}...")
                
                # Small delay to simulate real usage
                await asyncio.sleep(0.5)
            
            # Check if later responses show memory of earlier ones
            memory_usage_detected = any(r["has_memory_context"] for r in responses)
            
            print(f"✅ Completed {len(responses)} queries in sequence")
            print(f"   Memory usage detected: {memory_usage_detected}")
            
            self.results["agent_memory_workflow"] = {
                "success": True,
                "queries_processed": len(responses),
                "memory_usage_detected": memory_usage_detected,
                "responses": responses
            }
            return True
            
        except Exception as e:
            print(f"❌ Error testing agent memory workflow: {e}")
            import traceback
            traceback.print_exc()
            self.results["agent_memory_workflow"] = {"success": False, "error": str(e)}
            return False
    
    @pytest.mark.asyncio
    async def test_agent_memory_consolidation(self):
        """Test memory consolidation after agent interactions"""
        print("\n3️⃣ Testing agent memory consolidation...")
        
        try:
            # First, create some memories through agent interactions
            if self.agent.llm_available:
                # Make a few queries to generate memories
                test_queries = [
                    "德里地铁4期项目(20D21028C000)在故障信息查询中共发生多少起故障？",
                    "故障编号SH-202508-0899的故障后果是什么？",
                    "NCR编号NCR-202508-1070的问题性质是什么？"
                ]
                
                for question in test_queries:
                    await self.agent.query(
                        question=question,
                        user_email=TEST_EMAIL,
                        session_id=TEST_SESSION
                    )
                    await asyncio.sleep(0.2)  # Small delay
            
            # Now test memory consolidation
            consolidation_result = await self.memory_service.consolidate_memories(
                user_email=TEST_EMAIL,
                session_id=TEST_SESSION
            )
            
            print(f"✅ Memory consolidation completed:")
            print(f"   - Consolidated: {consolidation_result['consolidated']}")
            print(f"   - Expired deleted: {consolidation_result['expired_deleted']}")
            print(f"   - Total processed: {consolidation_result['total_processed']}")
            
            # Test getting memory context after consolidation
            context = await self.memory_service.get_memory_context_for_agent(
                user_email=TEST_EMAIL,
                question="你还记得我们之前关于故障信息的对话吗？",
                session_id=TEST_SESSION
            )
            
            print(f"✅ Memory context after consolidation:")
            print(f"   - Short-term memories: {len(context['short_term_memories'])}")
            print(f"   - Long-term memories: {len(context['long_term_memories'])}")
            print(f"   - Recent interactions: {len(context['recent_interactions'])}")
            
            self.results["agent_memory_consolidation"] = {
                "success": True,
                "consolidation_result": consolidation_result,
                "context_after_consolidation": {
                    "short_term": len(context['short_term_memories']),
                    "long_term": len(context['long_term_memories']),
                    "recent": len(context['recent_interactions'])
                }
            }
            return True
            
        except Exception as e:
            print(f"❌ Error testing agent memory consolidation: {e}")
            import traceback
            traceback.print_exc()
            self.results["agent_memory_consolidation"] = {"success": False, "error": str(e)}
            return False
    
    @pytest.mark.asyncio
    
    async def test_agent_memory_search(self):
        """Test memory search functionality in agent context"""
        print("\n4️⃣ Testing agent memory search...")
        
        try:
            # Test searching for specific memories
            search_terms = ["生产订单", "质量工程师", "BOM", "故障信息"]
            
            search_results = {}
            for term in search_terms:
                results = await self.memory_service.search_memories(
                    user_email=TEST_EMAIL,
                    search_term=term,
                    limit=5
                )
                search_results[term] = len(results)
                print(f"   Search '{term}': {len(results)} results")
            
            # Test getting relevant memories for a specific query
            relevant_short, relevant_long = await self.memory_service.get_relevant_memories(
                user_email=TEST_EMAIL,
                query="生产订单统计和BOM物料信息查询",
                session_id=TEST_SESSION,
                limit=5
            )
            
            print(f"✅ Relevant memories for complex query:")
            print(f"   - Short-term: {len(relevant_short)}")
            print(f"   - Long-term: {len(relevant_long)}")
            
            # Display some sample memories
            if relevant_long:
                print("   Sample long-term memories:")
                for i, memory in enumerate(relevant_long[:2], 1):
                    print(f"   {i}. {memory['content'][:80]}...")
            
            self.results["agent_memory_search"] = {
                "success": True,
                "search_results": search_results,
                "relevant_short": len(relevant_short),
                "relevant_long": len(relevant_long)
            }
            return True
            
        except Exception as e:
            print(f"❌ Error testing agent memory search: {e}")
            import traceback
            traceback.print_exc()
            self.results["agent_memory_search"] = {"success": False, "error": str(e)}
            return False
    
    @pytest.mark.asyncio
    
    async def test_agent_memory_persistence(self):
        """Test memory persistence across agent sessions"""
        print("\n5️⃣ Testing agent memory persistence...")
        
        try:
            from kangni_agents.models.history import MemoryType, MemoryImportance
            
            # Create memories in one "session"
            memory_1 = await self.memory_service.create_memory(
                user_email=TEST_EMAIL,
                content="用户偏好详细的技术解释和具体的项目信息",
                memory_type=MemoryType.LONG_TERM,
                importance=MemoryImportance.HIGH,
                session_id=TEST_SESSION,
                related_entities=["用户偏好", "技术解释", "项目信息"],
                tags=["behavior", "preference"]
            )
            
            memory_2 = await self.memory_service.create_memory(
                user_email=TEST_EMAIL,
                content="用户经常询问生产订单统计和BOM物料信息",
                memory_type=MemoryType.SEMANTIC,
                importance=MemoryImportance.MEDIUM,
                session_id=TEST_SESSION,
                related_entities=["生产订单", "BOM", "物料信息"],
                tags=["pattern", "frequent"]
            )
            
            print(f"✅ Created memories with IDs: {memory_1.id}, {memory_2.id}")
            
            # Test retrieving memories in a "new session" (same user, different session)
            new_session_context = await self.memory_service.get_memory_context_for_agent(
                user_email=TEST_EMAIL,
                question="你知道我的查询偏好吗？",
                session_id="new-session-123"  # Different session
            )
            
            print(f"✅ Memory context in new session:")
            print(f"   - Short-term memories: {len(new_session_context['short_term_memories'])}")
            print(f"   - Long-term memories: {len(new_session_context['long_term_memories'])}")
            print(f"   - Recent interactions: {len(new_session_context['recent_interactions'])}")
            
            # Test that long-term memories persist across sessions
            long_term_persisted = len(new_session_context['long_term_memories']) > 0
            
            # Test agent query with cross-session memory
            if self.agent.llm_available:
                response = await self.agent.query(
                    question="你知道我的查询偏好吗？",
                    user_email=TEST_EMAIL,
                    session_id="new-session-123"
                )
                
                print(f"✅ Agent response in new session:")
                print(f"   - Answer: {response.answer[:100] if response.answer else 'No answer'}...")
                print(f"   - Uses memory: {'偏好' in response.answer or '习惯' in response.answer if response.answer else False}")
            
            self.results["agent_memory_persistence"] = {
                "success": True,
                "long_term_persisted": long_term_persisted,
                "new_session_context": {
                    "short_term": len(new_session_context['short_term_memories']),
                    "long_term": len(new_session_context['long_term_memories']),
                    "recent": len(new_session_context['recent_interactions'])
                }
            }
            return True
            
        except Exception as e:
            print(f"❌ Error testing agent memory persistence: {e}")
            import traceback
            traceback.print_exc()
            self.results["agent_memory_persistence"] = {"success": False, "error": str(e)}
            return False
    
    @pytest.mark.asyncio
    
    async def test_agent_memory_error_handling(self):
        """Test error handling in agent memory functionality"""
        print("\n6️⃣ Testing agent memory error handling...")
        
        try:
            # Test 1: Agent query with invalid user email
            if self.agent.llm_available:
                try:
                    response = await self.agent.query(
                        question="Test question",
                        user_email=None,  # Invalid email
                        session_id=TEST_SESSION
                    )
                    print(f"✅ Agent handled None email gracefully: {response.answer[:50] if response.answer else 'No answer'}...")
                except Exception as e:
                    print(f"⚠️ Agent failed with None email: {e}")
            
            # Test 2: Memory service with invalid parameters
            try:
                context = await self.memory_service.get_memory_context_for_agent(
                    user_email="",  # Empty email
                    question="Test question",
                    session_id=None
                )
                print(f"✅ Memory service handled empty email gracefully")
            except Exception as e:
                print(f"⚠️ Memory service failed with empty email: {e}")
            
            # Test 3: Search with invalid parameters
            try:
                results = await self.memory_service.search_memories(
                    user_email=TEST_EMAIL,
                    search_term=None,  # None search term
                    limit=5
                )
                print(f"✅ Memory search handled None search term gracefully: {len(results)} results")
            except Exception as e:
                print(f"⚠️ Memory search failed with None search term: {e}")
            
            # Test 4: Consolidation with non-existent user
            try:
                result = await self.memory_service.consolidate_memories(
                    user_email="nonexistent@example.com",
                    session_id="nonexistent-session"
                )
                print(f"✅ Memory consolidation handled non-existent user gracefully")
            except Exception as e:
                print(f"⚠️ Memory consolidation failed with non-existent user: {e}")
            
            self.results["agent_memory_error_handling"] = {"success": True}
            return True
            
        except Exception as e:
            print(f"❌ Error testing agent memory error handling: {e}")
            import traceback
            traceback.print_exc()
            self.results["agent_memory_error_handling"] = {"success": False, "error": str(e)}
            return False
    
    @pytest.mark.asyncio
    
    async def test_agent_memory_performance(self):
        """Test memory performance with multiple operations"""
        print("\n7️⃣ Testing agent memory performance...")
        
        try:
            import time
            
            # Test memory context retrieval performance
            start_time = time.time()
            
            contexts = []
            for i in range(5):  # Test 5 concurrent memory context retrievals
                context = await self.memory_service.get_memory_context_for_agent(
                    user_email=TEST_EMAIL,
                    question=f"Performance test query {i}",
                    session_id=TEST_SESSION
                )
                contexts.append(context)
            
            context_time = time.time() - start_time
            print(f"✅ Retrieved 5 memory contexts in {context_time:.2f} seconds")
            
            # Test memory search performance
            start_time = time.time()
            
            search_terms = ["database", "configuration", "users", "memory", "system"]
            search_results = []
            
            for term in search_terms:
                results = await self.memory_service.search_memories(
                    user_email=TEST_EMAIL,
                    search_term=term,
                    limit=10
                )
                search_results.append(len(results))
            
            search_time = time.time() - start_time
            print(f"✅ Completed 5 memory searches in {search_time:.2f} seconds")
            print(f"   Search results: {search_results}")
            
            # Test agent query performance with memory
            if self.agent.llm_available:
                start_time = time.time()
                
                agent_queries = [
                    "合肥S1号线项目乘客室门这个项目一共有多少个去重后生产订单？",
                    "深圳14号线门扇外观问题团队中的质量工程师是谁？",
                    "BOM中0128000064物料的供应商是谁？",
                    "德里地铁4期项目在故障信息查询中共发生多少起故障？",
                    "故障编号SH-202508-0899的故障后果是什么？"
                ]
                
                agent_responses = []
                for query in agent_queries:
                    response = await self.agent.query(
                        question=query,
                        user_email=TEST_EMAIL,
                        session_id=TEST_SESSION
                    )
                    agent_responses.append(len(response.answer) if response.answer else 0)
                
                agent_time = time.time() - start_time
                print(f"✅ Completed 5 agent queries in {agent_time:.2f} seconds")
                print(f"   Response lengths: {agent_responses}")
            
            self.results["agent_memory_performance"] = {
                "success": True,
                "context_time": context_time,
                "search_time": search_time,
                "agent_time": agent_time if self.agent.llm_available else None,
                "search_results": search_results,
                "agent_responses": agent_responses if self.agent.llm_available else []
            }
            return True
            
        except Exception as e:
            print(f"❌ Error testing agent memory performance: {e}")
            import traceback
            traceback.print_exc()
            self.results["agent_memory_performance"] = {"success": False, "error": str(e)}
            return False
    
    async def check_memories_saved(self):
        """Check if memories are being saved to the database"""
        try:
            from kangni_agents.models.history import Memory
            from kangni_agents.models.database import get_db_config
            
            db_config = get_db_config()
            
            with db_config.session_scope() as session:
                # Count memories for test user
                memory_count = session.query(Memory).filter(
                    Memory.user_email == TEST_EMAIL
                ).count()
                
                print(f"📊 Database check:")
                print(f"   - Memories in database for {TEST_EMAIL}: {memory_count}")
                
                if memory_count > 0:
                    print("✅ Memories are being saved to database")
                    
                    # Show some sample memories
                    recent_memories = session.query(Memory).filter(
                        Memory.user_email == TEST_EMAIL
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
            # self.test_memory_functionality_only,
            self.test_agent_with_memory_context,
            # self.test_agent_memory_workflow,
            # self.test_agent_memory_consolidation,
            # self.test_agent_memory_search,
            # self.test_agent_memory_persistence,
            # self.test_agent_memory_error_handling,
            # self.test_agent_memory_performance
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
        # await self.cleanup_test_data()  # <-- Comment this line for manual testing
        
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
