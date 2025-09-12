"""测试RAG搜索功能"""

import sys
import pytest
import os
import asyncio
import logging

# Add kangni_agents directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from kangni_agents.services.rag_service import rag_service
    from kangni_agents.config import settings
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Configure logging to reduce verbosity and hide HTTPS info
logging.basicConfig(
    level=logging.INFO,  # Only show warnings and errors
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Suppress specific loggers that might print HTTPS info
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

@pytest.mark.asyncio

async def test_rag_search():
    """测试RAG搜索功能"""
    print("=" * 60)
    print("Testing RAG Search Functionality")
    print("=" * 60)
    
    # Print environment information
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries
    
    # 测试服务可用性
    print("1. Testing RAG service availability...")
    try:
        is_available = await rag_service.check_availability()
        if is_available:
            print("✅ RAG service is available")
        else:
            print("❌ RAG service is not available")
            return
    except Exception as e:
        print(f"❌ Error checking RAG service availability: {e}")
        return
    
    # # 测试用户问题
    query = "内解锁接地线线束短，无法安装到紧固螺钉位置是那个项目发生的？"
    print(f"\n2. Testing RAG search with query: {query}")
    
    try:
        # 使用默认数据集进行搜索
        results = await rag_service.search_rag(query, settings.ragflow_default_dataset_id, top_k=5)
        assert len(results) == 5
        
        # 确保results不为None
        if results is None:
            results = []
        
        print(f"搜索到 {len(results)} 条相关记录:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. 评分: {result.score}")
            print(f"   内容预览: {result.content[:200]}...")
            if result.metadata:
                print(f"   元数据: {result.metadata}")

        # 检查是否找到相关结果
        if results:
            print(f"\n✅ 测试通过：找到 {len(results)} 条相关记录")
            
            # 检查是否包含预期的项目信息
            content_text = " ".join([result.content for result in results])
            if "东莞1号线项目" in content_text:
                print("✅ 测试通过：找到'东莞1号线项目'相关信息")
            else:
                print("⚠️  警告：未找到'东莞1号线项目'相关信息")
        else:
            print("⚠️  警告：未找到相关记录")
            
    except Exception as e:
        print(f"❌ RAG搜索测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试数据库上下文搜索
    print(f"\n3. Testing database context search...")
    query = "物料号为0102010010对应的供应商是谁"
    try:
        db_results = await rag_service.search_db_context(query)

        # 确保results不为None
        if db_results is None:
            db_results = {}
        
        print(f"搜索到数据库上下文结果，包含 {len(db_results)} 个类别:")
        
        # 检查是否找到相关结果
        if db_results:
            total_results = 0
            db_content_text = ""
            
            # 遍历每个类别的结果
            for category, results in db_results.items():
                if results and len(results) > 0:
                    print(f"\n{category.upper()} 类别: {len(results)} 条记录")
                    total_results += len(results)
                    
                    # 显示前几条结果
                    for i, result in enumerate(results[:2], 1):  # 只显示前2条
                        print(f"  {i}. 评分: {result.score}")
                        print(f"     内容预览: {result.content[:100]}...")
                        if result.metadata:
                            print(f"     文档: {result.metadata.get('document_name', 'N/A')}")
                    
                    # 收集所有内容用于关键词检查
                    db_content_text += " ".join([result.content for result in results])
            assert total_results > 10
            print(f"\n✅ 测试通过：总共找到 {total_results} 条相关记录")
            
            # 检查是否包含预期的项目信息
            if "kn_quality_trace_bom_data" in db_content_text:
                print("✅ 测试通过：找到'物料bom表：kn_quality_trace_bom_data'相关信息")
            else:
                print("⚠️  警告：未找到'物料bom表：kn_quality_trace_bom_data'相关信息")
                
            # 检查是否包含SQL查询信息
            if "query_sql" in db_results and db_results["query_sql"]:
                print("✅ 测试通过：找到SQL查询相关信息")
            else:
                print("⚠️  警告：未找到SQL查询相关信息")
                
        else:
            print("⚠️  警告：未找到相关记录")
        
        print("✅ 数据库上下文搜索测试完成")
        
    except Exception as e:
        print(f"❌ 数据库上下文搜索测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试LLM答案生成
    print(f"\n4. Testing LLM answer generation...")
    try:
        # 使用之前搜索到的结果进行LLM答案生成
        if results:
            answer = await rag_service.generate_answer_with_llm(query, results)
            print(f"✅ LLM生成的答案:")
            print(f"   长度: {len(answer)} 字符")
            print(f"   预览: {answer[:200]}...")
        else:
            print("⚠️ 没有搜索结果可用于LLM答案生成")
        
        # 测试组合搜索和答案生成
        print(f"\n5. Testing combined search with answer generation...")
        combined_result = await rag_service.search_rag_with_answer(query, settings.ragflow_default_dataset_id, top_k=3)
        print(f"✅ 组合结果:")
        print(f"   答案长度: {len(combined_result['answer'])} 字符")
        print(f"   搜索结果数量: {combined_result['total_results']}")
        print(f"   答案预览: {combined_result['answer'][:200]}...")
        
    except Exception as e:
        print(f"❌ LLM答案生成测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    try:
        asyncio.run(test_rag_search())
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()