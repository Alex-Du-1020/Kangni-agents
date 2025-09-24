import re
from typing import Dict, List
from ..models import QueryType
import logging

logger = logging.getLogger(__name__)

class IntentClassifier:
    def __init__(self):
        # RAG相关关键词
        self.rag_keywords = [
            "总结", "summarize", "summary", "概述", "概要",
            "如何", "怎么", "how", "方法", "步骤",
            "什么是", "what is", "定义", "definition", "解释", "explain",
            "背景", "background", "历史", "history",
            "优势", "advantage", "缺点", "disadvantage",
            "比较", "compare", "对比", "区别", "difference"
        ]
        
        # 数据库查询相关关键词  
        self.database_keywords = [
            "统计", "statistics", "stat", "count", "总数", "数量",
            "平均", "average", "avg", "均值", "mean",
            "最大", "最小", "max", "min", "maximum", "minimum",
            "排序", "排名", "rank", "top", "前", "后",
            "查询", "query", "select", "搜索记录",
            "报表", "report", "数据", "data",
            "增长", "growth", "趋势", "trend",
            "占比", "比例", "percentage", "percent", "率",
            "分组", "group", "分类", "category",
            "时间段", "时间范围", "date range", "period",
            "NCR", "历史故障", "变更", "变更流程", "生产过程变更"
        ]
        
        # 明确的数据库操作词汇
        self.explicit_db_keywords = [
            "多少", "几个", "多少个", "有多少",
            "总共", "一共", "总计", "合计", "统计"
        ]
    
    def classify_intent(self, question: str) -> QueryType:
        """分类用户意图"""
        question_lower = question.lower()
        
        # 计算关键词匹配分数
        rag_score = self._calculate_keyword_score(question_lower, self.rag_keywords)
        db_score = self._calculate_keyword_score(question_lower, self.database_keywords)
        explicit_db_score = self._calculate_keyword_score(question_lower, self.explicit_db_keywords)
        
        logger.info(f"Intent scores - RAG: {rag_score}, DB: {db_score}, Explicit DB: {explicit_db_score}")
        
        # 如果有明确的数据库操作词汇，优先考虑数据库查询
        if explicit_db_score > 0:
            db_score += explicit_db_score * 2
        
        # 检查SQL相关词汇
        if any(word in question_lower for word in ["数据库", "database", "sql", "数据库表", "table", "字段", "field"]):
            db_score += 1
        
        # 判断意图
        if db_score > rag_score and db_score > 0:
            return QueryType.DATABASE
        elif rag_score == db_score and rag_score > 0:
            return QueryType.HYBRID
        else:
            # 默认情况：如果都没有明确关键词，使用混合模式
            return QueryType.HYBRID
    
    def _calculate_keyword_score(self, text: str, keywords: List[str]) -> float:
        """计算关键词匹配分数"""
        score = 0
        for keyword in keywords:
            if keyword.lower() in text:
                # 完整词匹配得更高分
                if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text):
                    score += 1
                else:
                    score += 0.5
        return score
    
    def get_classification_explanation(self, question: str, intent: QueryType) -> str:
        """获取分类解释"""
        explanations = {
            QueryType.RAG: "检测到概念解释或原因分析相关词汇，将使用文档搜索",
            QueryType.DATABASE: "检测到统计或数据查询相关词汇，将使用数据库查询",
            QueryType.HYBRID: "问题意图不够明确，将同时使用文档搜索和数据库查询"
        }
        return explanations.get(intent, "未知分类")

# 全局实例
intent_classifier = IntentClassifier()