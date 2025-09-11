#!/usr/bin/env python3
"""
Test script for enhanced vector search with configuration
"""
import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Set environment variables
os.environ['DB_TYPE'] = 'postgresql'
os.environ['POSTGRES_HOST'] = 'localhost'
os.environ['POSTGRES_PORT'] = '5432'
os.environ['POSTGRES_USER'] = 'postgres'
os.environ['POSTGRES_PASSWORD'] = 'postgres'
os.environ['POSTGRES_DATABASE'] = 'kangni_ai_chatbot'

from src.kangni_agents.agents.react_agent import database_query_tool
from src.kangni_agents.services.vector_embedding_service import vector_service
from src.kangni_agents.utils.sql_parser import SQLParser
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_sql_parser():
    """Test SQL parser functionality"""
    print("\n=== Testing SQL Parser ===")
    parser = SQLParser()
    
    test_queries = [
        "SELECT COUNT(*) FROM kn_quality_trace_prod_order WHERE projectname_s LIKE '%合肥S1号线%' AND partname_s LIKE '%乘客室门%'",
        "SELECT * FROM kn_quality_trace_prod_order WHERE projectname_s = '南京项目'",
        "SELECT projectname_s, COUNT(*) FROM kn_quality_trace_prod_order GROUP BY projectname_s"
    ]
    
    for sql in test_queries:
        print(f"\nSQL: {sql[:80]}...")
        parsed = parser.parse_sql(sql)
        print(f"  Tables: {parsed['tables']}")
        print(f"  Fields: {parsed['fields']}")
        print(f"  Conditions: {parsed['conditions']}")

async def test_vector_config():
    """Test loading vector search configuration"""
    print("\n=== Testing Vector Config Loading ===")
    import yaml
    from pathlib import Path
    
    config_path = Path(__file__).parent.parent.parent / "src" / "kangni_agents" / "config" / "vector_search_config.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"Found {len(config.get('vector_search_fields', []))} configured fields:")
        for field in config.get('vector_search_fields', []):
            print(f"  - {field['table']}.{field['field']} ({field['description']})")
        
        settings = config.get('settings', {})
        print(f"\nSettings:")
        print(f"  Similarity threshold: {settings.get('similarity_threshold', 'not set')}")
        print(f"  Max suggestions: {settings.get('max_suggestions', 'not set')}")
    else:
        print(f"Config file not found at: {config_path}")

async def test_enhanced_query():
    """Test enhanced database query with vector search"""
    print("\n=== Testing Enhanced Database Query ===")
    
    test_projects = [
        "合肥S1号线项目乘客室门",
        "南京地铁3号线屏蔽门项目",
        "上海高铁座椅项目",
        "北京地铁门控系统"
    ]
    
    print("\nStoring test project names in vector database...")
    for project in test_projects:
        try:
            await vector_service.store_field_embedding(
                "kn_quality_trace_prod_order",
                "projectname_s",
                project
            )
            print(f"  ✓ Stored: {project}")
        except Exception as e:
            print(f"  ✗ Failed to store {project}: {e}")
    
    # Test queries that should trigger vector search
    test_queries = [
        "合肥S1号线项目有多少个生产订单？",
        "查询南京地铁的订单数量",
        "北京门控系统的生产情况"
    ]
    
    print("\nTesting queries with vector search enhancement:")
    for query in test_queries:
        print(f"\n Query: {query}")
        try:
            # Use invoke method instead of direct call
            result = await database_query_tool.ainvoke({"question": query, "use_vector_search": True})
            
            if result.get("vector_enhanced"):
                print("  ✓ Vector search was used")
                suggestions = result.get("suggestions_used", {})
                if suggestions:
                    print("  Suggestions found:")
                    for field, data in suggestions.items():
                        print(f"    - {data['description']}: {data['values'][:3]}")
            else:
                print("  ℹ Vector search was not triggered")
            
            if result.get("sql_query"):
                print(f"  SQL: {result['sql_query'][:100]}...")
            
            if result.get("results"):
                print(f"  Results: {len(result['results'])} rows")
            else:
                print(f"  No results returned")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")

async def main():
    """Run all tests"""
    print("=" * 80)
    print("Testing Enhanced Vector Search Implementation")
    print("=" * 80)
    
    await test_sql_parser()
    await test_vector_config()
    await test_enhanced_query()
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())