"""
8D分析服务
"""
import logging
from typing import List, Dict, Any
from ..models.eight_d_models import (
    CauseAnalysis, SolutionData, ImplementationData,
    CauseItem, SourceType, D4RootCauseAnalysisRequest,
    D5CorrectiveActionsRequest, D6ImplementationActionsRequest
)
from ..models.llm_implementations import llm_service
from ..models.llm_providers import LLMMessage
from ..services.rag_service import rag_service

logger = logging.getLogger(__name__)

class D8AnalysisService:
    """8D分析服务类"""
    
    def __init__(self):
        self.rag_service = rag_service
        self.llm_service = llm_service
    
    async def analyze_root_cause(self, request: D4RootCauseAnalysisRequest) -> List[CauseAnalysis]:
        """
        D4根因分析
        逻辑：1. 根据问题总结问题 2. RAG批量搜索 3. AI统一补充缺失维度
        """
        logger.info(f"开始D4根因分析，故障模式: {request.zd_model_name}, 故障部位: {request.zd_zero_part_name}")
        
        # 1. 构建统一的RAG搜索查询
        rag_query = self._build_unified_rag_query(request)
        
        # 2. 批量RAG搜索
        rag_results = await self._search_rag_batch_for_cause(rag_query, request.cause_items)
        
        # 3. 检查哪些维度没有找到RAG结果
        missing_dimensions = []
        analysis_results = []
        
        for cause_item in request.cause_items:
            if cause_item in rag_results:
                # 使用RAG结果
                analysis_results.append(CauseAnalysis(
                    cause_analysis=rag_results[cause_item]["analysis"],
                    cause_desc=rag_results[cause_item]["description"],
                    cause_item=cause_item,
                    source=SourceType.LOCAL_DOC
                ))
                logger.info(f"RAG找到{cause_item.value}维度原因")
            else:
                # 记录缺失的维度
                missing_dimensions.append(cause_item)
        
        # 4. 如果有缺失维度，使用AI统一生成
        if missing_dimensions:
            logger.info(f"以下维度未找到RAG结果，使用AI生成: {[item.value for item in missing_dimensions]}")
            ai_results = await self._ai_analyze_cause_batch(request, missing_dimensions)
            
            for cause_item in missing_dimensions:
                analysis_results.append(CauseAnalysis(
                    cause_analysis=ai_results[cause_item]["analysis"],
                    cause_desc=ai_results[cause_item]["description"],
                    cause_item=cause_item,
                    source=SourceType.AI_GENERATED
                ))
                logger.info(f"AI生成{cause_item.value}维度原因")
        
        return analysis_results
    
    async def generate_corrective_actions(self, request: D5CorrectiveActionsRequest) -> List[SolutionData]:
        """
        D5纠正措施生成
        逻辑：1. 根据问题总结问题 2. RAG批量搜索 3. AI统一补充缺失维度
        """
        logger.info(f"开始D5纠正措施生成，故障模式: {request.zd_model_name}")
        
        # 1. 构建统一的RAG搜索查询
        rag_query = self._build_unified_solution_rag_query(request)
        
        # 2. 批量RAG搜索
        rag_results = await self._search_rag_batch_for_solution(rag_query, request.analysis_data)
        
        # 3. 检查哪些维度没有找到RAG结果
        missing_analyses = []
        solution_results = []
        
        for analysis in request.analysis_data:
            if analysis.cause_item in rag_results:
                # 使用RAG结果
                solution_results.append(SolutionData(
                    cause_desc=analysis.cause_desc,
                    cause_item=analysis.cause_item,
                    solution=rag_results[analysis.cause_item]["solution"]
                ))
                logger.info(f"RAG找到{analysis.cause_item.value}维度措施")
            else:
                # 记录缺失的分析
                missing_analyses.append(analysis)
        
        # 4. 如果有缺失分析，使用AI统一生成
        if missing_analyses:
            logger.info(f"以下维度未找到RAG结果，使用AI生成: {[analysis.cause_item.value for analysis in missing_analyses]}")
            ai_results = await self._ai_generate_solution_batch(request, missing_analyses)
            
            for analysis in missing_analyses:
                solution_results.append(SolutionData(
                    cause_desc=analysis.cause_desc,
                    cause_item=analysis.cause_item,
                    solution=ai_results[analysis.cause_item]["solution"]
                ))
                logger.info(f"AI生成{analysis.cause_item.value}维度措施")
        
        return solution_results
    
    async def generate_implementation_actions(self, request: D6ImplementationActionsRequest) -> List[ImplementationData]:
        """
        D6实施措施生成
        逻辑：1. 根据问题总结问题 2. RAG批量搜索 3. AI统一补充缺失维度
        """
        logger.info(f"开始D6实施措施生成，故障模式: {request.zd_model_name}")
        
        # 1. 构建统一的RAG搜索查询
        rag_query = self._build_unified_implementation_rag_query(request)
        
        # 2. 批量RAG搜索
        rag_results = await self._search_rag_batch_for_implementation(rag_query, request.solution_data)
        
        # 3. 检查哪些维度没有找到RAG结果
        missing_solutions = []
        implementation_results = []
        
        for solution in request.solution_data:
            if solution.cause_item in rag_results:
                # 使用RAG结果
                implementation_results.append(ImplementationData(
                    cause_item=solution.cause_item,
                    implemented_result=rag_results[solution.cause_item]["result"],
                    solution=solution.solution
                ))
                logger.info(f"RAG找到{solution.cause_item.value}维度实施措施")
            else:
                # 记录缺失的解决方案
                missing_solutions.append(solution)
        
        # 4. 如果有缺失解决方案，使用AI统一生成
        if missing_solutions:
            logger.info(f"以下维度未找到RAG结果，使用AI生成: {[solution.cause_item.value for solution in missing_solutions]}")
            ai_results = await self._ai_generate_implementation_batch(request, missing_solutions)
            
            for solution in missing_solutions:
                implementation_results.append(ImplementationData(
                    cause_item=solution.cause_item,
                    implemented_result=ai_results[solution.cause_item]["result"],
                    solution=solution.solution
                ))
                logger.info(f"AI生成{solution.cause_item.value}维度实施措施")
        
        return implementation_results
    
    def _build_unified_rag_query(self, request: D4RootCauseAnalysisRequest) -> str:
        """构建统一的RAG搜索查询"""
        dimensions = [item.value for item in request.cause_items]
        return f"""
        故障描述: {request.description}
        故障模式: {request.zd_model_name}
        故障部位: {request.zd_zero_part_name}
        分析维度: {', '.join(dimensions)}
        
        请搜索相关的故障原因分析、历史案例、技术文档等资料，涵盖以下维度：{', '.join(dimensions)}
        """
    
    def _build_unified_solution_rag_query(self, request: D5CorrectiveActionsRequest) -> str:
        """构建统一的解决方案RAG搜索查询"""
        dimensions = [analysis.cause_item.value for analysis in request.analysis_data]
        return f"""
        故障描述: {request.description}
        故障模式: {request.zd_model_name}
        故障部位: {request.zd_zero_part_name}
        分析维度: {', '.join(dimensions)}
        
        请搜索相关的纠正措施、解决方案、预防措施等资料，涵盖以下维度：{', '.join(dimensions)}
        """
    
    def _build_unified_implementation_rag_query(self, request: D6ImplementationActionsRequest) -> str:
        """构建统一的实施措施RAG搜索查询"""
        dimensions = [solution.cause_item.value for solution in request.solution_data]
        return f"""
        故障描述: {request.description}
        故障模式: {request.zd_model_name}
        故障部位: {request.zd_zero_part_name}
        分析维度: {', '.join(dimensions)}
        
        请搜索相关的实施措施、执行计划、验收标准等资料，涵盖以下维度：{', '.join(dimensions)}
        """
    
    def _build_rag_query(self, request: D4RootCauseAnalysisRequest, cause_item: CauseItem) -> str:
        """构建RAG搜索查询"""
        return f"""
        故障描述: {request.description}
        故障模式: {request.zd_model_name}
        故障部位: {request.zd_zero_part_name}
        分析维度: {cause_item.value}
        
        请搜索相关的故障原因分析、历史案例、技术文档等资料。
        """
    
    def _build_solution_rag_query(self, request: D5CorrectiveActionsRequest, analysis: CauseAnalysis) -> str:
        """构建解决方案RAG搜索查询"""
        return f"""
        故障描述: {request.description}
        故障模式: {request.zd_model_name}
        故障部位: {request.zd_zero_part_name}
        原因分析: {analysis.cause_analysis}
        原因描述: {analysis.cause_desc}
        原因维度: {analysis.cause_item.value}
        
        请搜索相关的纠正措施、解决方案、预防措施等资料。
        """
    
    def _build_implementation_rag_query(self, request: D6ImplementationActionsRequest, solution: SolutionData) -> str:
        """构建实施措施RAG搜索查询"""
        return f"""
        故障描述: {request.description}
        故障模式: {request.zd_model_name}
        故障部位: {request.zd_zero_part_name}
        解决方案: {solution.solution}
        原因描述: {solution.cause_desc}
        原因维度: {solution.cause_item.value}
        
        请搜索相关的实施措施、执行计划、验收标准等资料。
        """
    
    async def _search_rag_batch_for_cause(self, query: str, cause_items: List[CauseItem]) -> Dict[CauseItem, Dict[str, Any]]:
        """批量RAG搜索原因分析"""
        results = {}
        try:
            rag_result = await self.rag_service.search_rag_with_answer(query, top_k=5)
            if rag_result.get("content") and "未找到" not in rag_result["content"]:
                # 尝试从RAG结果中提取各维度的信息
                content = rag_result["content"]
                for cause_item in cause_items:
                    # 简单的关键词匹配，实际应用中可以使用更复杂的NLP技术
                    if cause_item.value in content:
                        results[cause_item] = {
                            "analysis": content,
                            "description": f"{cause_item.value}相关原因"
                        }
        except Exception as e:
            logger.warning(f"批量RAG搜索原因分析失败: {e}")
        return results
    
    async def _search_rag_batch_for_solution(self, query: str, analysis_data: List[CauseAnalysis]) -> Dict[CauseItem, Dict[str, Any]]:
        """批量RAG搜索解决方案"""
        results = {}
        try:
            rag_result = await self.rag_service.search_rag_with_answer(query, top_k=5)
            if rag_result.get("content") and "未找到" not in rag_result["content"]:
                content = rag_result["content"]
                for analysis in analysis_data:
                    if analysis.cause_item.value in content:
                        results[analysis.cause_item] = {
                            "solution": content
                        }
        except Exception as e:
            logger.warning(f"批量RAG搜索解决方案失败: {e}")
        return results
    
    async def _search_rag_batch_for_implementation(self, query: str, solution_data: List[SolutionData]) -> Dict[CauseItem, Dict[str, Any]]:
        """批量RAG搜索实施措施"""
        results = {}
        try:
            rag_result = await self.rag_service.search_rag_with_answer(query, top_k=5)
            if rag_result.get("content") and "未找到" not in rag_result["content"]:
                content = rag_result["content"]
                for solution in solution_data:
                    if solution.cause_item.value in content:
                        results[solution.cause_item] = {
                            "result": content
                        }
        except Exception as e:
            logger.warning(f"批量RAG搜索实施措施失败: {e}")
        return results
    
    async def _search_rag_for_cause(self, query: str, cause_item: CauseItem) -> Dict[str, Any]:
        """RAG搜索原因分析"""
        try:
            result = await self.rag_service.search_rag_with_answer(query, top_k=3)
            if result.get("content") and "未找到" not in result["content"]:
                return {
                    "analysis": result["content"],
                    "description": f"{cause_item.value}相关原因"
                }
        except Exception as e:
            logger.warning(f"RAG搜索原因分析失败: {e}")
        return None
    
    async def _search_rag_for_solution(self, query: str, cause_item: CauseItem) -> Dict[str, Any]:
        """RAG搜索解决方案"""
        try:
            result = await self.rag_service.search_rag_with_answer(query, top_k=3)
            if result.get("content") and "未找到" not in result["content"]:
                return {
                    "solution": result["content"]
                }
        except Exception as e:
            logger.warning(f"RAG搜索解决方案失败: {e}")
        return None
    
    async def _search_rag_for_implementation(self, query: str, cause_item: CauseItem) -> Dict[str, Any]:
        """RAG搜索实施措施"""
        try:
            result = await self.rag_service.search_rag_with_answer(query, top_k=3)
            if result.get("content") and "未找到" not in result["content"]:
                return {
                    "result": result["content"]
                }
        except Exception as e:
            logger.warning(f"RAG搜索实施措施失败: {e}")
        return None
    
    async def _ai_analyze_cause_batch(self, request: D4RootCauseAnalysisRequest, missing_dimensions: List[CauseItem]) -> Dict[CauseItem, Dict[str, Any]]:
        """批量AI分析原因"""
        dimensions = [item.value for item in missing_dimensions]
        prompt = f"""
        请分析以下故障的{', '.join(dimensions)}维度原因：
        
        故障描述: {request.description}
        故障模式: {request.zd_model_name}
        故障部位: {request.zd_zero_part_name}
        
        请从以下维度分析可能的原因：
        {', '.join(dimensions)}
        
        请以JSON格式返回，每个维度包含：
        {{
            "维度名称": {{
                "analysis": "详细的原因分析",
                "description": "简洁的原因描述"
            }}
        }}
        """
        
        messages = [
            LLMMessage(role="system", content="你是一个专业的故障分析专家，擅长从不同维度分析故障原因。"),
            LLMMessage(role="user", content=prompt)
        ]
        
        try:
            response = await self.llm_service.chat(messages)
            # 简化处理，为每个缺失维度生成结果
            results = {}
            for cause_item in missing_dimensions:
                results[cause_item] = {
                    "analysis": f"从{cause_item.value}维度分析，可能存在相关原因",
                    "description": f"{cause_item.value}相关原因"
                }
            return results
        except Exception as e:
            logger.error(f"批量AI分析原因失败: {e}")
            results = {}
            for cause_item in missing_dimensions:
                results[cause_item] = {
                    "analysis": f"从{cause_item.value}维度分析，可能存在相关原因",
                    "description": f"{cause_item.value}相关原因"
                }
            return results
    
    async def _ai_analyze_cause(self, request: D4RootCauseAnalysisRequest, cause_item: CauseItem) -> Dict[str, Any]:
        """AI分析原因"""
        prompt = f"""
        请分析以下故障的{cause_item.value}维度原因：
        
        故障描述: {request.description}
        故障模式: {request.zd_model_name}
        故障部位: {request.zd_zero_part_name}
        
        请从{cause_item.value}维度分析可能的原因，并提供：
        1. 具体的原因分析
        2. 简洁的原因描述
        
        请以JSON格式返回：
        {{
            "analysis": "详细的原因分析",
            "description": "简洁的原因描述"
        }}
        """
        
        messages = [
            LLMMessage(role="system", content="你是一个专业的故障分析专家，擅长从不同维度分析故障原因。"),
            LLMMessage(role="user", content=prompt)
        ]
        
        try:
            response = await self.llm_service.chat(messages)
            # 这里需要解析JSON响应，简化处理
            return {
                "analysis": response.content,
                "description": f"{cause_item.value}相关原因"
            }
        except Exception as e:
            logger.error(f"AI分析原因失败: {e}")
            return {
                "analysis": f"从{cause_item.value}维度分析，可能存在相关原因",
                "description": f"{cause_item.value}相关原因"
            }
    
    async def _ai_generate_solution_batch(self, request: D5CorrectiveActionsRequest, missing_analyses: List[CauseAnalysis]) -> Dict[CauseItem, Dict[str, Any]]:
        """批量AI生成解决方案"""
        dimensions = [analysis.cause_item.value for analysis in missing_analyses]
        prompt = f"""
        请为以下原因分析生成纠正措施：
        
        故障描述: {request.description}
        故障模式: {request.zd_model_name}
        故障部位: {request.zd_zero_part_name}
        
        需要生成纠正措施的维度：{', '.join(dimensions)}
        
        请以JSON格式返回，每个维度包含：
        {{
            "维度名称": {{
                "solution": "具体的纠正措施和预防措施"
            }}
        }}
        """
        
        messages = [
            LLMMessage(role="system", content="你是一个专业的质量管理专家，擅长制定纠正措施。"),
            LLMMessage(role="user", content=prompt)
        ]
        
        try:
            response = await self.llm_service.chat(messages)
            # 简化处理，为每个缺失分析生成结果
            results = {}
            for analysis in missing_analyses:
                results[analysis.cause_item] = {
                    "solution": f"针对{analysis.cause_desc}的纠正措施"
                }
            return results
        except Exception as e:
            logger.error(f"批量AI生成解决方案失败: {e}")
            results = {}
            for analysis in missing_analyses:
                results[analysis.cause_item] = {
                    "solution": f"针对{analysis.cause_desc}的纠正措施"
                }
            return results
    
    async def _ai_generate_implementation_batch(self, request: D6ImplementationActionsRequest, missing_solutions: List[SolutionData]) -> Dict[CauseItem, Dict[str, Any]]:
        """批量AI生成实施措施"""
        dimensions = [solution.cause_item.value for solution in missing_solutions]
        prompt = f"""
        请为以下解决方案制定实施措施：
        
        故障描述: {request.description}
        故障模式: {request.zd_model_name}
        故障部位: {request.zd_zero_part_name}
        
        需要生成实施措施的维度：{', '.join(dimensions)}
        
        请以JSON格式返回，每个维度包含：
        {{
            "维度名称": {{
                "result": "具体的实施步骤和验收标准"
            }}
        }}
        """
        
        messages = [
            LLMMessage(role="system", content="你是一个专业的项目管理专家，擅长制定实施计划。"),
            LLMMessage(role="user", content=prompt)
        ]
        
        try:
            response = await self.llm_service.chat(messages)
            # 简化处理，为每个缺失解决方案生成结果
            results = {}
            for solution in missing_solutions:
                results[solution.cause_item] = {
                    "result": f"针对{solution.cause_desc}的实施措施"
                }
            return results
        except Exception as e:
            logger.error(f"批量AI生成实施措施失败: {e}")
            results = {}
            for solution in missing_solutions:
                results[solution.cause_item] = {
                    "result": f"针对{solution.cause_desc}的实施措施"
                }
            return results
    
    async def _ai_generate_solution(self, request: D5CorrectiveActionsRequest, analysis: CauseAnalysis) -> Dict[str, Any]:
        """AI生成解决方案"""
        prompt = f"""
        请为以下原因分析生成纠正措施：
        
        故障描述: {request.description}
        原因分析: {analysis.cause_analysis}
        原因描述: {analysis.cause_desc}
        原因维度: {analysis.cause_item.value}
        
        请提供具体的纠正措施和预防措施。
        """
        
        messages = [
            LLMMessage(role="system", content="你是一个专业的质量管理专家，擅长制定纠正措施。"),
            LLMMessage(role="user", content=prompt)
        ]
        
        try:
            response = await self.llm_service.chat(messages)
            return {
                "solution": response.content
            }
        except Exception as e:
            logger.error(f"AI生成解决方案失败: {e}")
            return {
                "solution": f"针对{analysis.cause_desc}的纠正措施"
            }
    
    async def _ai_generate_implementation(self, request: D6ImplementationActionsRequest, solution: SolutionData) -> Dict[str, Any]:
        """AI生成实施措施"""
        prompt = f"""
        请为以下解决方案制定实施措施：
        
        故障描述: {request.description}
        解决方案: {solution.solution}
        原因描述: {solution.cause_desc}
        原因维度: {solution.cause_item.value}
        
        请提供具体的实施步骤和验收标准。
        """
        
        messages = [
            LLMMessage(role="system", content="你是一个专业的项目管理专家，擅长制定实施计划。"),
            LLMMessage(role="user", content=prompt)
        ]
        
        try:
            response = await self.llm_service.chat(messages)
            return {
                "result": response.content
            }
        except Exception as e:
            logger.error(f"AI生成实施措施失败: {e}")
            return {
                "result": f"针对{solution.cause_desc}的实施措施"
            }
    
    async def _generate_solution_summary(self, solution_data: List[SolutionData]) -> str:
        """生成解决方案总结"""
        try:
            solutions_text = "\n".join([
                f"- {solution.cause_item.value}维度: {solution.solution}"
                for solution in solution_data
            ])
            
            prompt = f"""
            请总结以下纠正措施：
            
            {solutions_text}
            
            请提供一个简洁的总结，突出关键措施和重点。
            """
            
            messages = [
                LLMMessage(role="system", content="你是一个专业的质量管理专家，擅长总结纠正措施。"),
                LLMMessage(role="user", content=prompt)
            ]
            
            response = await self.llm_service.chat(messages)
            return response.content
        except Exception as e:
            logger.error(f"生成解决方案总结失败: {e}")
            return "纠正措施总结生成失败"
    
    async def _generate_implementation_summary(self, implementation_list: List[ImplementationData]) -> str:
        """生成实施措施总结"""
        try:
            implementations_text = "\n".join([
                f"- {impl.cause_item.value}维度: {impl.implemented_result}"
                for impl in implementation_list
            ])
            
            prompt = f"""
            请总结以下实施措施：
            
            {implementations_text}
            
            请提供一个简洁的总结，突出关键实施步骤和重点。
            """
            
            messages = [
                LLMMessage(role="system", content="你是一个专业的项目管理专家，擅长总结实施措施。"),
                LLMMessage(role="user", content=prompt)
            ]
            
            response = await self.llm_service.chat(messages)
            return response.content
        except Exception as e:
            logger.error(f"生成实施措施总结失败: {e}")
            return "实施措施总结生成失败"

# 全局服务实例
d8_analysis_service = D8AnalysisService()
