#!/usr/bin/env python3
"""
History Endpoint Test Suite for Kangni Agents
Tests all history-related API endpoints with mocked data
"""
import asyncio
import pytest
import sys
import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

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
BASE_URL = "http://localhost:8000"
TEST_EMAIL = "test@example.com"
TEST_SESSION = "test-session-001"


async def clear_test_data(test_emails=None):
    """Clear test data from the database for specific users
    
    Args:
        test_emails: List of email addresses to clear data for. If None, clears all data.
    """
    if test_emails is None:
        test_emails = [
            "test@example.com", "other@example.com", "alice@example.com", 
            "bob@example.com", "nonexistent@example.com"
        ]
    
    print(f"🧹 Clearing test data for emails: {test_emails}")
    
    try:
        db_config = get_db_config()
        with db_config.session_scope() as session:
            
            # 2. Clear memories for test users
            session.query(Memory).filter(Memory.user_email.in_(test_emails)).delete()
            
            # 3. Clear comments for test users
            session.query(UserComment).filter(UserComment.user_email.in_(test_emails)).delete()
            
            # 4. Clear feedback for test users
            session.query(UserFeedback).filter(UserFeedback.user_email.in_(test_emails)).delete()
            
            # 5. Clear query history for test users
            session.query(QueryHistory).filter(QueryHistory.user_email.in_(test_emails)).delete()
            
            session.commit()
            print("✅ Test data cleared successfully")
            
    except Exception as e:
        print(f"❌ Error clearing test data: {e}")
        raise


class HistoryEndpointTests:
    """Test suite for history API endpoints"""
    
    def __init__(self):
        self.test_query_id: Optional[int] = None
        self.results = {}
        
    async def setup_test_data(self):
        """Create initial test data using history service"""
        print("\n📝 Setting up test data...")
        try:
            from kangni_agents.services.history_service import history_service
            
            # Create test queries
            test_queries = [
                {
                    "session_id": TEST_SESSION,
                    "user_email": TEST_EMAIL,
                    "question": "How many users are in the system?",
                    "answer": "There are 1,234 users in the system.",
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
                    "question": "What is the total number of orders?",
                    "answer": "The total number of orders is 5,678.",
                    "sql_query": "SELECT COUNT(*) FROM orders;",
                    "sources": [{"content": "Database", "score": 0.92}],
                    "query_type": "database",
                    "success": True,
                    "processing_time_ms": 120,
                    "llm_provider": "deepseek",
                    "model_name": "deepseek-chat"
                },
                {
                    "session_id": "different-session",
                    "user_email": "other@example.com",
                    "question": "What is the revenue?",
                    "answer": "The total revenue is $100,000.",
                    "sql_query": "SELECT SUM(amount) FROM revenue;",
                    "sources": None,
                    "query_type": "database",
                    "success": True,
                    "processing_time_ms": 200,
                    "llm_provider": "deepseek",
                    "model_name": "deepseek-chat"
                },
                {
                    "session_id": TEST_SESSION,
                    "user_email": TEST_EMAIL,
                    "question": "Invalid query",
                    "answer": None,
                    "sql_query": None,
                    "sources": None,
                    "query_type": None,
                    "success": False,
                    "error_message": "Database connection timeout",
                    "processing_time_ms": 5000,
                    "llm_provider": None,
                    "model_name": None
                }
            ]
            
            # Save test queries
            for query_data in test_queries:
                history = await history_service.save_query_history(**query_data)
                if self.test_query_id is None and query_data["success"]:
                    self.test_query_id = history.id
                    
            print(f"✅ Created {len(test_queries)} test queries")
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup test data: {e}")
            return False
    
    @pytest.mark.asyncio
    
    async def test_get_user_history(self):
        """Test GET /qomo/v1/history/user/{email}"""
        print("\n1️⃣ Testing GET /qomo/v1/history/user/{email}...")
        
        try:
            from kangni_agents.services.history_service import history_service
            
            # Test with valid user
            history = await history_service.get_user_history(
                user_email=TEST_EMAIL,
                limit=10,
                offset=0,
                include_feedback=True,
                include_comments=True
            )
            
            if history and len(history) > 0:
                print(f"✅ Retrieved {len(history)} items for {TEST_EMAIL}")
                print(f"   Latest query: {history[0]['question'][:50]}...")
                self.results["user_history"] = {"success": True, "count": len(history)}
                
                # Store query ID for later tests
                if not self.test_query_id and history[0].get('id'):
                    self.test_query_id = history[0]['id']
                return True
            else:
                print(f"⚠️ No history found for {TEST_EMAIL}")
                self.results["user_history"] = {"success": False, "error": "No history found"}
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            self.results["user_history"] = {"success": False, "error": str(e)}
            return False
    
    @pytest.mark.asyncio
    
    async def test_get_session_history(self):
        """Test GET /qomo/v1/history/session/{session_id}"""
        print("\n2️⃣ Testing GET /qomo/v1/history/session/{session_id}...")
        
        try:
            from kangni_agents.services.history_service import history_service
            
            history = await history_service.get_history_by_session(
                session_id=TEST_SESSION,
                limit=10
            )
            
            if history and len(history) > 0:
                print(f"✅ Retrieved {len(history)} items for session {TEST_SESSION}")
                self.results["session_history"] = {"success": True, "count": len(history)}
                return True
            else:
                print(f"⚠️ No history found for session {TEST_SESSION}")
                self.results["session_history"] = {"success": False, "error": "No history found"}
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            self.results["session_history"] = {"success": False, "error": str(e)}
            return False
    
    @pytest.mark.asyncio
    
    async def test_search_history(self):
        """Test GET /qomo/v1/history/search"""
        print("\n3️⃣ Testing GET /qomo/v1/history/search...")
        
        try:
            from kangni_agents.services.history_service import history_service
            
            # Search for "users" in history
            results = await history_service.search_history(
                search_term="users",
                user_email=TEST_EMAIL,
                limit=10
            )
            
            print(f"✅ Search found {len(results)} items containing 'users'")
            self.results["search"] = {"success": True, "count": len(results)}
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            self.results["search"] = {"success": False, "error": str(e)}
            return False
    
    @pytest.mark.asyncio
    
    async def test_recent_queries(self):
        """Test GET /qomo/v1/history/recent"""
        print("\n4️⃣ Testing GET /qomo/v1/history/recent...")
        
        try:
            from kangni_agents.services.history_service import history_service
            
            results = await history_service.get_recent_queries(
                hours=24,
                limit=100
            )
            
            print(f"✅ Found {len(results)} queries in the last 24 hours")
            self.results["recent"] = {"success": True, "count": len(results)}
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            self.results["recent"] = {"success": False, "error": str(e)}
            return False
    
    @pytest.mark.asyncio
    
    async def test_add_feedback(self):
        """Test POST /qomo/v1/history/feedback"""
        print("\n5️⃣ Testing POST /qomo/v1/history/feedback...")
        
        if not self.test_query_id:
            print("⚠️ No query ID available for feedback test")
            self.results["feedback"] = {"success": False, "error": "No query ID"}
            return False
        
        try:
            from kangni_agents.services.history_service import history_service
            
            # Add like feedback
            feedback = await history_service.add_feedback(
                query_id=self.test_query_id,
                user_email=TEST_EMAIL,
                feedback_type="like"
            )
            
            print(f"✅ Added 'like' feedback with ID: {feedback.id}")
            
            # Add dislike feedback from another user
            feedback2 = await history_service.add_feedback(
                query_id=self.test_query_id,
                user_email="other@example.com",
                feedback_type="dislike"
            )
            
            print(f"✅ Added 'dislike' feedback with ID: {feedback2.id}")
            self.results["feedback"] = {"success": True}
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            self.results["feedback"] = {"success": False, "error": str(e)}
            return False
    
    @pytest.mark.asyncio
    
    async def test_add_comment(self):
        """Test POST /qomo/v1/history/comment"""
        print("\n6️⃣ Testing POST /qomo/v1/history/comment...")
        
        if not self.test_query_id:
            print("⚠️ No query ID available for comment test")
            self.results["comment"] = {"success": False, "error": "No query ID"}
            return False
        
        try:
            from kangni_agents.services.history_service import history_service
            
            comment = await history_service.add_comment(
                query_id=self.test_query_id,
                user_email=TEST_EMAIL,
                comment="This is a test comment. Very helpful answer!"
            )
            
            print(f"✅ Added comment with ID: {comment.id}")
            self.results["comment"] = {"success": True}
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            self.results["comment"] = {"success": False, "error": str(e)}
            return False
    
    @pytest.mark.asyncio
    
    async def test_get_comments(self):
        """Test GET /qomo/v1/history/comments/{query_id}"""
        print("\n7️⃣ Testing GET /qomo/v1/history/comments/{query_id}...")
        
        if not self.test_query_id:
            print("⚠️ No query ID available for get comments test")
            self.results["get_comments"] = {"success": False, "error": "No query ID"}
            return False
        
        try:
            from kangni_agents.services.history_service import history_service
            
            # Add multiple comments first
            test_comments = [
                {"user": "alice@example.com", "text": "Great answer!"},
                {"user": "bob@example.com", "text": "The answer is 42, obviously."},
                {"user": "alice@example.com", "text": "I agree with Bob!"}
            ]
            
            for comment_data in test_comments:
                await history_service.add_comment(
                    query_id=self.test_query_id,
                    user_email=comment_data["user"],
                    comment=comment_data["text"]
                )
            
            # Now test getting comments
            comments = await history_service.get_query_comments(self.test_query_id)
            
            print(f"✅ Retrieved {len(comments)} comments for query {self.test_query_id}")
            
            # Display comments
            for idx, comment in enumerate(comments[:3], 1):  # Show first 3 comments
                print(f"   Comment {idx}: {comment['comment'][:50]}...")
            
            # Test with non-existent query
            empty_comments = await history_service.get_query_comments(999999)
            if len(empty_comments) == 0:
                print("✅ Correctly returned empty list for non-existent query")
            
            self.results["get_comments"] = {"success": True, "count": len(comments)}
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            self.results["get_comments"] = {"success": False, "error": str(e)}
            return False
    
    @pytest.mark.asyncio
    
    async def test_feedback_stats(self):
        """Test GET /qomo/v1/history/feedback/stats/{query_id}"""
        print("\n8️⃣ Testing GET /qomo/v1/history/feedback/stats/{query_id}...")
        
        if not self.test_query_id:
            print("⚠️ No query ID available for stats test")
            self.results["stats"] = {"success": False, "error": "No query ID"}
            return False
        
        try:
            from kangni_agents.services.history_service import history_service
            
            stats = await history_service.get_query_feedback_stats(self.test_query_id)
            
            print(f"✅ Feedback stats for query {self.test_query_id}:")
            print(f"   Likes: {stats['likes']}")
            print(f"   Dislikes: {stats['dislikes']}")
            self.results["stats"] = {"success": True, "stats": stats}
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            self.results["stats"] = {"success": False, "error": str(e)}
            return False
    
    @pytest.mark.asyncio
    
    async def test_api_endpoints(self):
        """Test actual API endpoints if server is running"""
        print("\n9️⃣ Testing API Endpoints (requires server running)...")
        
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                # Test health endpoint
                response = await client.get(f"{BASE_URL}/qomo/v1/health", timeout=5.0)
                if response.status_code != 200:
                    print("⚠️ Server not healthy or not running")
                    return False
                
                print("✅ Server is running")
                
                # Test user history endpoint
                response = await client.get(
                    f"{BASE_URL}/qomo/v1/history/user/{TEST_EMAIL}",
                    params={"limit": 10, "offset": 0},
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ API: Retrieved {len(data)} items via user history endpoint")
                else:
                    print(f"❌ API: User history endpoint returned {response.status_code}")
                
                # Test session history endpoint
                response = await client.get(
                    f"{BASE_URL}/qomo/v1/history/session/{TEST_SESSION}",
                    params={"limit": 10},
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ API: Retrieved {len(data)} items via session history endpoint")
                else:
                    print(f"❌ API: Session history endpoint returned {response.status_code}")
                
                # Test comments endpoint if we have a query ID
                if self.test_query_id:
                    response = await client.get(
                        f"{BASE_URL}/qomo/v1/history/comments/{self.test_query_id}",
                        timeout=10.0
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        print(f"✅ API: Retrieved {len(data)} comments via comments endpoint")
                    else:
                        print(f"❌ API: Comments endpoint returned {response.status_code}")
                
                return True
                
        except httpx.ConnectError:
            print("⚠️ Server not running - skipping API endpoint tests")
            print("   Run './dev_server.sh' to start the server")
            return False
        except Exception as e:
            print(f"❌ API test error: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all history endpoint tests"""
        print("🚀 Starting History Endpoint Tests")
        print("=" * 60)
        
        try:
            # Clear any existing test data
            await clear_test_data()
            
            # Setup test data
            if not await self.setup_test_data():
                print("❌ Failed to setup test data, aborting tests")
                return
        
            # Run all tests
            test_methods = [
                self.test_get_user_history,
                self.test_get_session_history,
                self.test_search_history,
                self.test_recent_queries,
                self.test_add_feedback,
                self.test_add_comment,
                self.test_get_comments,
                self.test_feedback_stats,
                self.test_api_endpoints
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
                    failed += 1
            
            # Print summary
            print("\n" + "=" * 60)
            print("📊 TEST SUMMARY")
            print("=" * 60)
            print(f"Total tests: {passed + failed}")
            print(f"Passed: {passed}")
            print(f"Failed: {failed}")
            print(f"Success rate: {(passed/(passed+failed)*100):.1f}%")
            
            if failed == 0:
                print("\n✅ All history endpoint tests passed!")
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
    tests = HistoryEndpointTests()
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