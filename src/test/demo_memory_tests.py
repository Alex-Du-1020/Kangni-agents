#!/usr/bin/env python3
"""
Memory Tests Demo Script
Demonstrates how to run memory tests and shows expected output
"""
import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "../.."))

async def demo_memory_service():
    """Demo memory service functionality"""
    print("🧠 Memory Service Demo")
    print("=" * 40)
    
    try:
        from kangni_agents.services.memory_service import memory_service
        from kangni_agents.models.history import MemoryType, MemoryImportance
        
        # Create a test memory
        print("Creating a test memory...")
        memory = await memory_service.create_memory(
            user_email="demo@example.com",
            content="This is a demo memory for testing purposes",
            memory_type=MemoryType.LONG_TERM,
            importance=MemoryImportance.MEDIUM,
            session_id="demo-session",
            related_entities=["demo", "testing"],
            tags=["example", "test"]
        )
        print(f"✅ Created memory with ID: {memory.id}")
        
        # Search for the memory
        print("Searching for memories...")
        results = await memory_service.search_memories(
            user_email="demo@example.com",
            search_term="demo",
            limit=5
        )
        print(f"✅ Found {len(results)} memories containing 'demo'")
        
        # Get memory context
        print("Getting memory context...")
        context = await memory_service.get_memory_context_for_agent(
            user_email="demo@example.com",
            question="What do you know about testing?",
            session_id="demo-session"
        )
        print(f"✅ Retrieved memory context with {len(context['long_term_memories'])} long-term memories")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

async def demo_react_agent_memory():
    """Demo react agent with memory"""
    print("\n🤖 React Agent Memory Demo")
    print("=" * 40)
    
    try:
        from kangni_agents.agents.react_agent import kangni_agent
        
        if not kangni_agent.llm_available:
            print("⚠️ LLM service not available, skipping agent demo")
            return True
        
        # Test agent query with memory
        print("Testing agent query with memory...")
        response = await kangni_agent.query(
            question="What do you know about memory management?",
            user_email="demo@example.com",
            session_id="demo-session"
        )
        
        print(f"✅ Agent response received:")
        print(f"   - Answer length: {len(response.answer) if response.answer else 0} characters")
        print(f"   - Query type: {response.query_type}")
        print(f"   - Confidence: {response.confidence}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent demo failed: {e}")
        return False

async def demo_chinese_processing():
    """Demo Chinese text processing capabilities"""
    print("🧠 中文记忆处理演示")
    print("=" * 50)
    
    try:
        from kangni_agents.services.memory_service import memory_service
        from kangni_agents.models.history import MemoryType, MemoryImportance
        
        # Test Chinese text samples
        test_texts = [
            "数据库配置需要设置DB_TYPE环境变量，这是系统运行的基础要求。",
            "用户经常询问关于订单统计的问题，特别是生产订单表kn_quality_trace_prod_order的数据。",
            "内存管理系统需要定期清理过期数据，以提高系统性能。",
            "根据用户反馈，系统应该支持中文查询和回答。",
            "数据库连接超时是一个常见问题，通常是由于网络配置不当造成的。"
        ]
        
        print("测试文本样本:")
        for i, text in enumerate(test_texts, 1):
            print(f"{i}. {text}")
        
        print("\n" + "=" * 50)
        print("关键词提取演示:")
        print("=" * 50)
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n文本 {i}: {text}")
            keywords = memory_service._extract_keywords(text)
            print(f"关键词: {keywords}")
        
        print("\n" + "=" * 50)
        print("实体提取演示:")
        print("=" * 50)
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n文本 {i}: {text}")
            entities = memory_service._extract_entities(text)
            print(f"实体: {entities}")
        
        print("\n" + "=" * 50)
        print("事实提取演示:")
        print("=" * 50)
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n文本 {i}: {text}")
            facts = memory_service._extract_facts(text)
            print(f"事实: {facts}")
        
        print("\n" + "=" * 50)
        print("记忆创建和搜索演示:")
        print("=" * 50)
        
        # Create memories with Chinese content
        test_memories = [
            {
                "user_email": "chinese_demo@example.com",
                "content": "用户经常询问数据库配置问题，特别是DB_TYPE环境变量的设置方法。",
                "memory_type": MemoryType.LONG_TERM,
                "importance": MemoryImportance.HIGH,
                "session_id": "chinese-demo-session",
                "related_entities": ["数据库", "配置", "环境变量"],
                "tags": ["技术", "常见问题"]
            },
            {
                "user_email": "chinese_demo@example.com",
                "content": "系统支持中文查询和回答，这是用户的重要需求。",
                "memory_type": MemoryType.SEMANTIC,
                "importance": MemoryImportance.MEDIUM,
                "session_id": "chinese-demo-session",
                "related_entities": ["中文", "查询", "回答"],
                "tags": ["功能", "用户需求"]
            }
        ]
        
        created_memories = []
        for memory_data in test_memories:
            memory = await memory_service.create_memory(**memory_data)
            created_memories.append(memory)
            print(f"✅ 创建记忆: {memory.id} - {memory_data['content'][:30]}...")
        
        # Test memory search with Chinese terms
        print("\n中文记忆搜索演示:")
        search_terms = ["数据库", "配置", "中文", "用户", "系统"]
        for term in search_terms:
            results = await memory_service.search_memories(
                user_email="chinese_demo@example.com",
                search_term=term,
                limit=5
            )
            print(f"搜索 '{term}': 找到 {len(results)} 条记忆")
            for result in results:
                print(f"  - {result['content'][:50]}...")
        
        # Test memory context generation
        print("\n记忆上下文生成演示:")
        context = await memory_service.get_memory_context_for_agent(
            user_email="chinese_demo@example.com",
            question="如何配置数据库连接？",
            session_id="chinese-demo-session"
        )
        
        print(f"上下文信息:")
        print(f"  - 短期记忆: {len(context['short_term_memories'])} 条")
        print(f"  - 长期记忆: {len(context['long_term_memories'])} 条")
        print(f"  - 最近交互: {len(context['recent_interactions'])} 条")
        
        if context['long_term_memories']:
            print("长期记忆示例:")
            for memory in context['long_term_memories'][:2]:
                print(f"  - {memory['content'][:60]}...")
        
        print("\n✅ 中文记忆处理演示完成!")
        return True
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run memory demos"""
    print("🚀 Memory System Demo")
    print("=" * 50)
    
    # Initialize database
    print("Initializing database...")
    try:
        from kangni_agents.models.database import get_db_config
        db_config = get_db_config()
        print(f"✅ Database initialized")
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        return 1
    
    # Run demos
    memory_demo_success = await demo_memory_service()
    agent_demo_success = await demo_react_agent_memory()
    chinese_demo_success = await demo_chinese_processing()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Demo Summary")
    print("=" * 50)
    print(f"Memory Service Demo: {'✅ SUCCESS' if memory_demo_success else '❌ FAILED'}")
    print(f"React Agent Demo: {'✅ SUCCESS' if agent_demo_success else '❌ FAILED'}")
    
    if memory_demo_success and agent_demo_success and chinese_demo_success:
        print("\n🎉 All demos completed successfully!")
        print("\nTo run the full test suite:")
        print("  python run_memory_tests.py")
    else:
        print("\n❌ Some demos failed. Check the error messages above.")
    
    return 0 if (memory_demo_success and agent_demo_success) else 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
