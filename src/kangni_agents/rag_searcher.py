"""RAG搜索模块"""

class RAGSearcher:
    def __init__(self):
        # 模拟知识库数据
        self.knowledge_base = [
            {
                "project": "东莞1号线项目",
                "issue": "内解锁接地线线束短，无法安装到紧固螺钉位置",
                "description": "在东莞1号线项目中，发现内解锁接地线线束过短，导致无法正确安装到指定的紧固螺钉位置。这影响了设备的正常安装和接地安全。",
                "solution": "需要重新制作更长的接地线线束，确保能够到达紧固螺钉位置。"
            },
            {
                "project": "深圳3号线项目", 
                "issue": "信号灯接线故障",
                "description": "信号灯接线出现松动，导致信号不稳定。",
                "solution": "重新紧固接线端子，检查线材质量。"
            },
            {
                "project": "广州2号线项目",
                "issue": "控制柜门锁故障",
                "description": "控制柜门锁机制失效，无法正常开启。", 
                "solution": "更换门锁机构，调整门框间隙。"
            }
        ]
    
    def search(self, query: str) -> list:
        """
        RAG搜索功能
        """
        results = []
        query_lower = query.lower()
        
        for item in self.knowledge_base:
            # 简单的关键词匹配
            if any(keyword in item["issue"].lower() or keyword in item["description"].lower() 
                   for keyword in ["接地线", "线束", "安装", "螺钉"]):
                if "接地线" in query_lower and "线束" in query_lower:
                    results.append(item)
        
        return results
    
    def get_answer(self, query: str) -> str:
        """
        获取问题答案
        """
        results = self.search(query)
        
        if not results:
            return "抱歉，没有找到相关信息。"
        
        # 返回第一个匹配结果的项目信息
        result = results[0]
        return f"该问题发生在{result['project']}。具体情况：{result['description']}"