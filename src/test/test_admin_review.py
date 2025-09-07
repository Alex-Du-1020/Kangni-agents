#!/usr/bin/env python3
"""
Test the admin review endpoint for dislike feedback
"""
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Set SQLite for testing
os.environ["DB_TYPE"] = "sqlite"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "../.."))

from src.kangni_agents.services.history_service import history_service
from src.kangni_agents.models.history import FeedbackType


async def setup_test_data():
    """Create test data with some dislike feedback"""
    print("📝 Setting up test data...")
    
    # Create queries with various feedback
    test_queries = [
        {
            "session_id": "admin-test-001",
            "user_email": "user1@example.com",
            "question": "Why is the system slow?",
            "answer": "The system may be slow due to various reasons...",
            "sql_query": None,
            "sources": None,
            "query_type": "general",
            "success": True,
            "processing_time_ms": 500,
            "llm_provider": "deepseek",
            "model_name": "deepseek-chat"
        },
        {
            "session_id": "admin-test-002",
            "user_email": "user2@example.com",
            "question": "How many orders are pending?",
            "answer": "There are 42 pending orders.",
            "sql_query": "SELECT COUNT(*) FROM orders WHERE status='pending'",
            "sources": None,
            "query_type": "database",
            "success": True,
            "processing_time_ms": 200,
            "llm_provider": "deepseek",
            "model_name": "deepseek-chat"
        },
        {
            "session_id": "admin-test-003",
            "user_email": "user3@example.com",
            "question": "What is the revenue for Q1?",
            "answer": "The Q1 revenue is $1.5M.",
            "sql_query": "SELECT SUM(amount) FROM revenue WHERE quarter='Q1'",
            "sources": None,
            "query_type": "database",
            "success": True,
            "processing_time_ms": 300,
            "llm_provider": "deepseek",
            "model_name": "deepseek-chat"
        }
    ]
    
    query_ids = []
    for query_data in test_queries:
        result = await history_service.save_query_history(**query_data)
        query_ids.append(result.id)
        print(f"  ✓ Created query {result.id}: {query_data['question'][:30]}...")
    
    # Add feedback (some likes, some dislikes)
    print("\n💔 Adding dislike feedback...")
    
    # Query 1: 2 dislikes, 1 like
    await history_service.add_feedback(query_ids[0], "user1@example.com", "dislike")
    await history_service.add_feedback(query_ids[0], "admin@example.com", "dislike")
    await history_service.add_feedback(query_ids[0], "user4@example.com", "like")
    print(f"  ✓ Query {query_ids[0]}: 2 dislikes, 1 like")
    
    # Query 2: 1 dislike
    await history_service.add_feedback(query_ids[1], "user2@example.com", "dislike")
    print(f"  ✓ Query {query_ids[1]}: 1 dislike")
    
    # Query 3: only likes (shouldn't appear in dislike review)
    await history_service.add_feedback(query_ids[2], "user3@example.com", "like")
    await history_service.add_feedback(query_ids[2], "user5@example.com", "like")
    print(f"  ✓ Query {query_ids[2]}: 2 likes (won't appear in dislike review)")
    
    # Add comments to queries with dislikes
    print("\n💬 Adding comments...")
    await history_service.add_comment(query_ids[0], "user1@example.com", 
                                     "The answer is too vague and doesn't provide specific solutions.")
    await history_service.add_comment(query_ids[0], "admin@example.com", 
                                     "This needs more technical details about performance optimization.")
    await history_service.add_comment(query_ids[1], "user2@example.com", 
                                     "The SQL query seems incorrect, it's missing some conditions.")
    print("  ✓ Added comments to queries with dislikes")
    
    return query_ids


async def test_admin_review_endpoint():
    """Test the admin review endpoint"""
    print("\n" + "=" * 60)
    print("🔍 Testing Admin Review Endpoint")
    print("=" * 60)
    
    # Test different time periods
    time_periods = [1, 3, 5, 7]
    
    for days in time_periods:
        print(f"\n📅 Getting dislike feedback from last {days} day(s)...")
        
        try:
            review_data = await history_service.get_dislike_feedback_for_review(
                days=days,
                limit=10
            )
            
            print(f"  Found {len(review_data)} queries with dislikes")
            
            for item in review_data:
                print(f"\n  Query ID: {item['id']}")
                print(f"  Question: {item['question'][:50]}...")
                print(f"  User: {item['user_email']}")
                print(f"  Created: {item['created_at']}")
                print(f"  Feedback: {item['feedback_stats']['likes']} likes, "
                      f"{item['feedback_stats']['dislikes']} dislikes")
                print(f"  Disliked by: {', '.join(item['feedback_stats']['dislike_users'])}")
                
                if item['comments']:
                    print(f"  Comments ({len(item['comments'])}):")
                    for comment in item['comments']:
                        print(f"    - {comment['user_email']}: {comment['comment'][:60]}...")
                else:
                    print("  No comments")
                    
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Admin Review Endpoint Test Complete!")


async def test_api_endpoint():
    """Test via actual API if server is running"""
    print("\n" + "=" * 60)
    print("🌐 Testing via API (requires server)")
    print("=" * 60)
    
    try:
        import httpx
        
        async with httpx.AsyncClient() as client:
            # Test with different day parameters
            for days in [1, 7]:
                response = await client.get(
                    f"http://localhost:8000/api/v1/history/admin/dislike-review",
                    params={"days": days, "limit": 10},
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✓ API: Retrieved {len(data)} items for {days} day(s)")
                    
                    if data:
                        print(f"  Sample: {data[0]['question'][:40]}...")
                        print(f"  Dislikes: {data[0]['feedback_stats']['dislikes']}")
                else:
                    print(f"❌ API returned status {response.status_code}")
                    print(f"   Response: {response.text}")
                    
    except httpx.ConnectError:
        print("⚠️ Server not running - skipping API test")
        print("   Run './dev_server.sh' to test the API endpoint")
    except Exception as e:
        print(f"❌ API test error: {e}")


async def main():
    """Main test function"""
    print("🚀 Admin Review Endpoint Test Suite")
    print("Testing dislike feedback review functionality")
    
    # Setup test data
    query_ids = await setup_test_data()
    
    # Test the service method
    await test_admin_review_endpoint()
    
    # Test via API if available
    await test_api_endpoint()
    
    print("\n" + "=" * 60)
    print("📊 Summary:")
    print("The admin review endpoint allows filtering by:")
    print("  • Days: 1, 3, 5, 7 (or any value 1-30)")
    print("  • Returns: Queries with dislikes, ordered by created time")
    print("  • Includes: Feedback stats and all comments")
    print("\nEndpoint: GET /api/v1/history/admin/dislike-review")
    print("Parameters: ?days=7&limit=100")


if __name__ == "__main__":
    asyncio.run(main())