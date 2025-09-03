"""测试RAG搜索功能"""

from kangni_agents.rag_searcher import RAGSearcher

def test_rag_search():
    searcher = RAGSearcher()
    
    # 测试用户问题
    query = "内解锁接地线线束短，无法安装到紧固螺钉位置是那个项目发生的？"
    answer = searcher.get_answer(query)
    
    print(f"问题: {query}")
    print(f"答案: {answer}")
    
    # 验证答案是否包含"东莞1号线项目"
    assert "东莞1号线项目" in answer, f"答案中应该包含'东莞1号线项目'，但实际答案是: {answer}"
    print("✅ 测试通过：答案包含'东莞1号线项目'")
    
    # 测试搜索结果
    results = searcher.search(query)
    print(f"\n搜索到 {len(results)} 条相关记录:")
    for i, result in enumerate(results, 1):
        print(f"{i}. 项目: {result['project']}")
        print(f"   问题: {result['issue']}")
        print(f"   描述: {result['description']}")
        print()

if __name__ == "__main__":
    test_rag_search()