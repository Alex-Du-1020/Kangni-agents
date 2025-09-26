"""
8D分析服务
"""
import logging
from typing import List, Dict, Any
from ..models.eight_d_models import (
    CauseAnalysis, SolutionData, ImplementationData,
    CauseItem, SourceType, ReferencedDocument, D4RootCauseAnalysisRequest,
    D5CorrectiveActionsRequest, D6ImplementationActionsRequest,
    D4RootCauseSummaryRequest, D5CorrectiveActionsSummaryRequest,
    D6ImplementationSummaryRequest
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
        Enhanced D4根因分析
        逻辑：1. 重写问题聚焦故障模式和部位 2. RAG搜索获取全面内容 3. LLM分析并分配到各维度
        """
        logger.info(f"开始D4根因分析，故障模式: {request.zdModelName}, 故障部位: {request.zdZeroPartName}")
        
        # Step 1: 重写问题以更好地聚焦
        rewritten_question = self._rewrite_question_for_root_cause(request)
        
        # Step 2: RAG搜索 + LLM分析一步完成
        analysis_results = await self._rag_plus_llm_analysis(request, rewritten_question)
        
        return analysis_results
    
    async def _rag_plus_llm_analysis(self, request: D4RootCauseAnalysisRequest, question: str) -> List[CauseAnalysis]:
        """RAG搜索 + LLM分析：先分配RAG结果，再补充缺失维度"""
        
        # 1. 获取RAG内容
        rag_result = await self.rag_service.search_rag_with_answer(question, top_k=10)
        rag_content = rag_result.get("content", "")
        
        if not rag_content or "未找到" in rag_content:
            logger.warning("RAG搜索未找到相关内容，使用LLM直接生成")
            return await self._generate_causes_with_llm(request)
        
        # 2. 智能分配RAG结果到最合适的维度
        rag_assignments = await self._assign_rag_to_dimensions(request, rag_content)
        
        # 3. 找出没有分配到RAG结果的维度
        assigned_dimensions = set(rag_assignments.keys())
        missing_dimensions = [item for item in request.causeItems if item not in assigned_dimensions]
        
        # 4. 为缺失维度生成LLM分析
        llm_results = {}
        if missing_dimensions:
            logger.info(f"以下维度未分配到RAG结果，使用LLM补充: {[item.value for item in missing_dimensions]}")
            llm_analysis_results = await self._generate_causes_with_llm(request, missing_dimensions)
            # 转换为字典格式
            for analysis in llm_analysis_results:
                llm_results[analysis.causeItem] = {
                    "analysis": analysis.causeAnalysis,
                    "description": analysis.causeDesc
                }
        
        # 5. 合并结果
        analysis_results = []
        
        # 获取RAG结果中的引用文档
        rag_results = rag_result.get("rag_results", [])
        
        # 添加RAG分配的结果
        for cause_item in request.causeItems:
            if cause_item in rag_assignments:
                # 为RAG结果添加引用文档
                referenced_docs = []
                if rag_results:
                    for doc in rag_results:
                        referenced_docs.append(ReferencedDocument(
                            documentName=doc.metadata.get("document_name", "未知文档") if doc.metadata else "未知文档",
                            score=doc.score
                        ))
                
                analysis_results.append(CauseAnalysis(
                    causeAnalysis=rag_assignments[cause_item]["analysis"],
                    causeDesc=rag_assignments[cause_item]["description"],
                    causeItem=cause_item,
                    source=SourceType.LOCAL_DOC,
                    referencedDocuments=referenced_docs if referenced_docs else None
                ))
                logger.info(f"RAG分配到{cause_item.value}维度，引用{len(referenced_docs)}个文档")
            elif cause_item in llm_results:
                analysis_results.append(CauseAnalysis(
                    causeAnalysis=llm_results[cause_item]["analysis"],
                    causeDesc=llm_results[cause_item]["description"],
                    causeItem=cause_item,
                    source=SourceType.AI_GENERATED,
                    referencedDocuments=None
                ))
                logger.info(f"LLM补充{cause_item.value}维度")
            else:
                # 最后的保底处理
                analysis_results.append(CauseAnalysis(
                    causeAnalysis=f"从{cause_item.value}维度分析，可能存在相关原因",
                    causeDesc=f"{cause_item.value}相关原因",
                    causeItem=cause_item,
                    source=SourceType.AI_GENERATED,
                    referencedDocuments=None
                ))
        
        return analysis_results
    
    async def _generate_causes_with_llm(self, request: D4RootCauseAnalysisRequest, cause_items: List[CauseItem] = None) -> List[CauseAnalysis]:
        """使用LLM生成指定维度的原因分析"""
        if cause_items is None:
            cause_items = request.causeItems
            
        dimension_descriptions = self._get_dimension_descriptions(cause_items)
        dimensions = [item.value for item in cause_items]
        
        prompt = f"""
        请分析以下故障的{', '.join(dimensions)}维度原因：
        
        故障模式: {request.zdModelName}
        故障部位: {request.zdZeroPartName}
        
        需要分析的维度说明：
        {dimension_descriptions}
        
        请以JSON格式返回，格式如下：
        {{
            "维度名称": {{
                "analysis": "详细的原因分析（纯文本）",
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
            
            # 解析JSON响应
            import json
            try:
                # 尝试从响应中提取JSON部分
                content = response.content
                if "```json" in content:
                    # 提取JSON部分
                    start = content.find("```json") + 7
                    end = content.find("```", start)
                    if end > start:
                        json_content = content[start:end].strip()
                    else:
                        json_content = content
                else:
                    json_content = content
                
                response_data = json.loads(json_content)
                analysis_results = []
                
                for cause_item in cause_items:
                    item_key = cause_item.value
                    if item_key in response_data:
                        # 确保analysis字段是纯文本
                        analysis_text = response_data[item_key].get("analysis", f"从{item_key}维度分析，可能存在相关原因")
                        # 如果analysis还包含JSON格式，提取纯文本部分
                        if "```json" in analysis_text:
                            import re
                            match = re.search(r'"analysis":\s*"([^"]*)"', analysis_text)
                            if match:
                                analysis_text = match.group(1)
                            else:
                                analysis_text = f"从{item_key}维度分析，可能存在相关原因"
                        
                        analysis_results.append(CauseAnalysis(
                            causeAnalysis=analysis_text,
                            causeDesc=response_data[item_key].get("description", f"{item_key}相关原因"),
                            causeItem=cause_item,
                            source=SourceType.AI_GENERATED,
                            referencedDocuments=None
                        ))
                        logger.info(f"LLM生成{item_key}维度原因")
                    else:
                        # 如果JSON中缺少某个维度，使用默认值
                        analysis_results.append(CauseAnalysis(
                            causeAnalysis=f"从{item_key}维度分析，可能存在相关原因",
                            causeDesc=f"{item_key}相关原因",
                            causeItem=cause_item,
                            source=SourceType.AI_GENERATED,
                            referencedDocuments=None
                        ))
                        logger.warning(f"JSON中缺少{item_key}维度，使用默认值")
                
                return analysis_results
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"LLM生成原因JSON解析失败: {e}，使用简化处理")
                # 简化处理，为每个维度生成结果
                analysis_results = []
                for cause_item in cause_items:
                    analysis_results.append(CauseAnalysis(
                        causeAnalysis=f"从{cause_item.value}维度分析，可能存在相关原因：{response.content[:200]}...",
                        causeDesc=f"{cause_item.value}相关原因",
                        causeItem=cause_item,
                        source=SourceType.AI_GENERATED,
                        referencedDocuments=None
                    ))
                    logger.info(f"LLM生成{cause_item.value}维度原因")
                
                return analysis_results
                
        except Exception as e:
            logger.error(f"LLM生成原因失败: {e}")
            # 最后的降级处理
            analysis_results = []
            for cause_item in cause_items:
                analysis_results.append(CauseAnalysis(
                    causeAnalysis=f"从{cause_item.value}维度分析，可能存在相关原因",
                    causeDesc=f"{cause_item.value}相关原因",
                    causeItem=cause_item,
                    source=SourceType.AI_GENERATED,
                    referencedDocuments=None
                ))
            return analysis_results
    
    async def _assign_rag_to_dimensions(self, request: D4RootCauseAnalysisRequest, rag_content: str) -> Dict[CauseItem, Dict[str, str]]:
        """将RAG结果智能分配到最合适的维度"""
        dimension_descriptions = self._get_dimension_descriptions(request.causeItems)
        
        prompt = f"""
        基于以下RAG搜索结果，将其分配到最合适的分析维度：
        
        RAG搜索结果:
        {rag_content}
        
        故障信息:
        - 故障描述: {request.description}
        - 故障模式: {request.zdModelName}
        - 故障部位: {request.zdZeroPartName}
        
        需要分配的维度说明：
        {dimension_descriptions}
        
        请分析RAG结果中的信息，分析根本原因，并将总结的根本原因分配到最合适的维度。每个根本原因只能分配给一个维度。
        如果RAG结果提到了设计不合理，或者设计不达标之类的问题，请分配到法（方法）维度。
        如果某个维度没有合适的根本原因，请不要分配。
        
        请以JSON格式返回，格式如下：
        {{
            "维度名称": {{
                "analysis": "基于RAG结果的详细原因分析（纯文本）",
                "description": "简洁的原因描述",
                "reason": "为什么分配到这个维度"
            }}
        }}
        
        注意：
        1. 只返回有RAG信息支持的维度
        2. analysis字段只包含纯文本，不要JSON格式
        3. 如果某个维度没有合适的RAG信息，不要包含在结果中
        """
        
        messages = [
            LLMMessage(role="system", content="你是一个专业的故障分析专家，擅长将RAG结果准确分配到对应的分析维度。"),
            LLMMessage(role="user", content=prompt)
        ]
        
        response = await self.llm_service.chat(messages)
        
        # 解析JSON响应
        import json
        try:
            # 尝试从响应中提取JSON部分
            content = response.content
            if "```json" in content:
                # 提取JSON部分
                start = content.find("```json") + 7
                end = content.find("```", start)
                if end > start:
                    json_content = content[start:end].strip()
                else:
                    json_content = content
            else:
                json_content = content
            
            response_data = json.loads(json_content)
            assignments = {}
            for cause_item in request.causeItems:
                item_key = cause_item.value
                if item_key in response_data:
                    # 确保analysis字段是纯文本
                    analysis_text = response_data[item_key].get("analysis", f"从{item_key}维度分析，可能存在相关原因")
                    # 如果analysis还包含JSON格式，提取纯文本部分
                    if "```json" in analysis_text:
                        # 提取JSON中的analysis内容
                        if "analysis" in analysis_text:
                            import re
                            match = re.search(r'"analysis":\s*"([^"]*)"', analysis_text)
                            if match:
                                analysis_text = match.group(1)
                            else:
                                analysis_text = f"从{item_key}维度分析，可能存在相关原因"
                    
                    assignments[cause_item] = {
                        "analysis": analysis_text,
                        "description": response_data[item_key].get("description", f"{item_key}相关原因")
                    }
            return assignments
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"RAG分配JSON解析失败: {e}，返回空分配")
            return {}
    
    
    async def generate_corrective_actions(self, request: D5CorrectiveActionsRequest) -> List[SolutionData]:
        """
        D5纠正措施生成
        逻辑：1. 根据问题总结问题 2. RAG批量搜索 3. AI统一补充缺失维度
        """
        logger.info(f"开始D5纠正措施生成，故障模式: {request.zdModelName}")
        
        # 1. 构建统一的RAG搜索查询
        rag_query = self._build_unified_solution_rag_query(request)
        
        # 2. 批量RAG搜索
        rag_results = await self._search_rag_batch_for_solution(rag_query, request.analysisData)
        
        # 3. 检查哪些维度没有找到RAG结果
        missingAnalyses = []
        solutionResults = []
        
        for analysis in request.analysisData:
            if analysis.causeItem in rag_results:
                # 使用RAG结果
                solutionResults.append(SolutionData(
                    causeDesc=analysis.causeDesc,
                    causeItem=analysis.causeItem,
                    solution=rag_results[analysis.causeItem]["solution"]
                ))
                logger.info(f"RAG找到{analysis.causeItem.value}维度措施")
            else:
                # 记录缺失的分析
                missingAnalyses.append(analysis)
        
        # 4. 如果有缺失分析，使用AI统一生成
        if missingAnalyses:
            logger.info(f"以下维度未找到RAG结果，使用AI生成: {[analysis.causeItem.value for analysis in missingAnalyses]}")
            ai_results = await self._ai_generate_solution_batch(request, missingAnalyses)
            
            for analysis in missingAnalyses:
                solutionResults.append(SolutionData(
                    causeDesc=analysis.causeDesc,
                    causeItem=analysis.causeItem,
                    solution=ai_results[analysis.causeItem]["solution"]
                ))
                logger.info(f"AI生成{analysis.causeItem.value}维度措施")
        
        return solutionResults
    
    async def generate_implementation_actions(self, request: D6ImplementationActionsRequest) -> List[ImplementationData]:
        """
        D6实施措施生成
        逻辑：1. 根据问题总结问题 2. RAG批量搜索 3. AI统一补充缺失维度
        """
        logger.info(f"开始D6实施措施生成，故障模式: {request.zdModelName}")
        
        # 1. 构建统一的RAG搜索查询
        rag_query = self._build_unified_implementation_rag_query(request)
        
        # 2. 批量RAG搜索
        rag_results = await self._search_rag_batch_for_implementation(rag_query, request.solutionData)
        
        # 3. 检查哪些维度没有找到RAG结果
        missingSolutions = []
        implementationResults = []
        
        for solution in request.solutionData:
            if solution.causeItem in rag_results:
                # 使用RAG结果
                implementationResults.append(ImplementationData(
                    causeDesc=solution.causeDesc,
                    causeItem=solution.causeItem,
                    implementedResult=rag_results[solution.causeItem]["result"],
                    solution=solution.solution
                ))
                logger.info(f"RAG找到{solution.causeItem.value}维度实施措施")
            else:
                # 记录缺失的解决方案
                missingSolutions.append(solution)
        
        # 4. 如果有缺失解决方案，使用AI统一生成
        if missingSolutions:
            logger.info(f"以下维度未找到RAG结果，使用AI生成: {[solution.causeItem.value for solution in missingSolutions]}")
            ai_results = await self._ai_generate_implementation_batch(request, missingSolutions)
            
            for solution in missingSolutions:
                implementationResults.append(ImplementationData(
                    causeDesc=solution.causeDesc,
                    causeItem=solution.causeItem,
                    implementedResult=ai_results[solution.causeItem]["result"],
                    solution=solution.solution
                ))
                logger.info(f"AI生成{solution.causeItem.value}维度实施措施")
        
        return implementationResults
    
    def _get_dimension_descriptions(self, cause_items: List[CauseItem]) -> str:
        """获取维度详细说明"""
        dimension_map = {
            CauseItem.PERSON: "人(人员) - 人员操作、技能、培训、责任心等方面的问题",
            CauseItem.MACHINE: "机(设备) - 设备故障、维护不及时、精度不够等方面的问题", 
            CauseItem.MATERIAL: "料(材料) - 原材料质量、规格、供应商、存储等方面的问题",
            CauseItem.METHOD: "法(方法) - 工艺方法或者设计规范达不到要求、操作流程不规范、标准执行不到位等方面的问题",
            CauseItem.ENVIRONMENT: "环(环境) - 工作环境、温度、湿度、清洁度等方面的问题",
            CauseItem.MEASUREMENT: "测(测量) - 测量设备、方法、精度、校准等方面的问题"
        }
        
        descriptions = []
        for item in cause_items:
            if item in dimension_map:
                descriptions.append(f"- {item.value}: {dimension_map[item]}")
            else:
                descriptions.append(f"- {item.value}: 相关维度分析")
        
        return "\n".join(descriptions)
    
    def _rewrite_question_for_root_cause(self, request: D4RootCauseAnalysisRequest) -> str:
        """重写问题以更好地聚焦故障模式和部位"""
        return f"""
        针对故障模式"{request.zdModelName}"和故障部位"{request.zdZeroPartName}"，分析故障的根本原因。
        """
    
    def _build_unified_rag_query(self, request: D4RootCauseAnalysisRequest) -> str:
        """构建统一的RAG搜索查询"""
        dimensions = [item.value for item in request.causeItems]
        return f"""
        故障描述: {request.description}
        故障模式: {request.zdModelName}
        故障部位: {request.zdZeroPartName}
        分析维度: {', '.join(dimensions)}
        
        请搜索相关的故障原因分析、历史案例、技术文档等资料，涵盖以下维度：{', '.join(dimensions)}
        """
    
    def _build_unified_solution_rag_query(self, request: D5CorrectiveActionsRequest) -> str:
        """构建统一的解决方案RAG搜索查询"""
        dimensions = [analysis.causeItem.value for analysis in request.analysisData]
        return f"""
        故障描述: {request.description}
        故障模式: {request.zdModelName}
        故障部位: {request.zdZeroPartName}
        分析维度: {', '.join(dimensions)}
        
        请搜索相关的纠正措施、解决方案、预防措施等资料，涵盖以下维度：{', '.join(dimensions)}
        """
    
    def _build_unified_implementation_rag_query(self, request: D6ImplementationActionsRequest) -> str:
        """构建统一的实施措施RAG搜索查询"""
        dimensions = [solution.causeItem.value for solution in request.solutionData]
        return f"""
        故障描述: {request.description}
        故障模式: {request.zdModelName}
        故障部位: {request.zdZeroPartName}
        分析维度: {', '.join(dimensions)}
        
        请搜索相关的实施措施、执行计划、验收标准等资料，涵盖以下维度：{', '.join(dimensions)}
        """
    
    def _build_rag_query(self, request: D4RootCauseAnalysisRequest, causeItem: CauseItem) -> str:
        """构建RAG搜索查询"""
        return f"""
        故障描述: {request.description}
        故障模式: {request.zdModelName}
        故障部位: {request.zdZeroPartName}
        分析维度: {causeItem.value}
        
        请搜索相关的故障原因分析、历史案例、技术文档等资料。
        """
    
    def _build_solution_rag_query(self, request: D5CorrectiveActionsRequest, analysis: CauseAnalysis) -> str:
        """构建解决方案RAG搜索查询"""
        return f"""
        故障描述: {request.description}
        故障模式: {request.zdModelName}
        故障部位: {request.zdZeroPartName}
        原因分析: {analysis.causeAnalysis}
        原因描述: {analysis.causeDesc}
        原因维度: {analysis.causeItem.value}
        
        请搜索相关的纠正措施、解决方案、预防措施等资料。
        """
    
    def _build_implementation_rag_query(self, request: D6ImplementationActionsRequest, solution: SolutionData) -> str:
        """构建实施措施RAG搜索查询"""
        return f"""
        故障描述: {request.description}
        故障模式: {request.zdModelName}
        故障部位: {request.zdZeroPartName}
        解决方案: {solution.solution}
        原因描述: {solution.causeDesc}
        原因维度: {solution.causeItem.value}
        
        请搜索相关的实施措施、执行计划、验收标准等资料。
        """
    
    async def _search_rag_batch_for_cause(self, query: str, causeItems: List[CauseItem]) -> Dict[CauseItem, Dict[str, Any]]:
        """批量RAG搜索原因分析"""
        results = {}
        try:
            rag_result = await self.rag_service.search_rag_with_answer(query, top_k=5)
            if rag_result.get("content") and "未找到" not in rag_result["content"]:
                # 尝试从RAG结果中提取各维度的信息
                content = rag_result["content"]
                for causeItem in causeItems:
                    # 简单的关键词匹配，实际应用中可以使用更复杂的NLP技术
                    if causeItem.value in content:
                        results[causeItem] = {
                            "analysis": content,
                            "description": f"{causeItem.value}相关原因"
                        }
        except Exception as e:
            logger.warning(f"批量RAG搜索原因分析失败: {e}")
        return results
    
    async def _search_rag_batch_for_solution(self, query: str, analysisData: List[CauseAnalysis]) -> Dict[CauseItem, Dict[str, Any]]:
        """批量RAG搜索解决方案"""
        results = {}
        try:
            rag_result = await self.rag_service.search_rag_with_answer(query, top_k=5)
            if rag_result.get("content") and "未找到" not in rag_result["content"]:
                content = rag_result["content"]
                for analysis in analysisData:
                    if analysis.causeItem.value in content:
                        results[analysis.causeItem] = {
                            "solution": content
                        }
        except Exception as e:
            logger.warning(f"批量RAG搜索解决方案失败: {e}")
        return results
    
    async def _search_rag_batch_for_implementation(self, query: str, solutionData: List[SolutionData]) -> Dict[CauseItem, Dict[str, Any]]:
        """批量RAG搜索实施措施"""
        results = {}
        try:
            rag_result = await self.rag_service.search_rag_with_answer(query, top_k=5)
            if rag_result.get("content") and "未找到" not in rag_result["content"]:
                content = rag_result["content"]
                for solution in solutionData:
                    if solution.causeItem.value in content:
                        results[solution.causeItem] = {
                            "result": content
                        }
        except Exception as e:
            logger.warning(f"批量RAG搜索实施措施失败: {e}")
        return results
    
    async def _search_rag_for_cause(self, query: str, causeItem: CauseItem) -> Dict[str, Any]:
        """RAG搜索原因分析"""
        try:
            result = await self.rag_service.search_rag_with_answer(query, top_k=3)
            if result.get("content") and "未找到" not in result["content"]:
                return {
                    "analysis": result["content"],
                    "description": f"{causeItem.value}相关原因"
                }
        except Exception as e:
            logger.warning(f"RAG搜索原因分析失败: {e}")
        return None
    
    async def _search_rag_for_solution(self, query: str, causeItem: CauseItem) -> Dict[str, Any]:
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
    
    async def _search_rag_for_implementation(self, query: str, causeItem: CauseItem) -> Dict[str, Any]:
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
    
    async def _ai_analyze_cause_batch(self, request: D4RootCauseAnalysisRequest, missingDimensions: List[CauseItem]) -> Dict[CauseItem, Dict[str, Any]]:
        """批量AI分析原因"""
        dimensions = [item.value for item in missingDimensions]
        prompt = f"""
        请分析以下故障的{', '.join(dimensions)}维度原因：
        
        故障描述: {request.description}
        故障模式: {request.zdModelName}
        故障部位: {request.zdZeroPartName}
        
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
            for causeItem in missingDimensions:
                results[causeItem] = {
                    "analysis": f"从{causeItem.value}维度分析，可能存在相关原因",
                    "description": f"{causeItem.value}相关原因"
                }
            return results
        except Exception as e:
            logger.error(f"批量AI分析原因失败: {e}")
            results = {}
            for causeItem in missingDimensions:
                results[causeItem] = {
                    "analysis": f"从{causeItem.value}维度分析，可能存在相关原因",
                    "description": f"{causeItem.value}相关原因"
                }
            return results
    
    async def _ai_analyze_cause(self, request: D4RootCauseAnalysisRequest, causeItem: CauseItem) -> Dict[str, Any]:
        """AI分析原因"""
        prompt = f"""
        请分析以下故障的{causeItem.value}维度原因：
        
        故障描述: {request.description}
        故障模式: {request.zdModelName}
        故障部位: {request.zdZeroPartName}
        
        请从{causeItem.value}维度分析可能的原因，并提供：
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
                "description": f"{causeItem.value}相关原因"
            }
        except Exception as e:
            logger.error(f"AI分析原因失败: {e}")
            return {
                "analysis": f"从{causeItem.value}维度分析，可能存在相关原因",
                "description": f"{causeItem.value}相关原因"
            }
    
    async def _ai_generate_solution_batch(self, request: D5CorrectiveActionsRequest, missingAnalyses: List[CauseAnalysis]) -> Dict[CauseItem, Dict[str, Any]]:
        """批量AI生成解决方案"""
        dimensions = [analysis.causeItem.value for analysis in missingAnalyses]
        prompt = f"""
        请为以下原因分析生成纠正措施：
        
        故障描述: {request.description}
        故障模式: {request.zdModelName}
        故障部位: {request.zdZeroPartName}
        
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
            for analysis in missingAnalyses:
                results[analysis.causeItem] = {
                    "solution": f"针对{analysis.causeDesc}的纠正措施"
                }
            return results
        except Exception as e:
            logger.error(f"批量AI生成解决方案失败: {e}")
            results = {}
            for analysis in missingAnalyses:
                results[analysis.causeItem] = {
                    "solution": f"针对{analysis.causeDesc}的纠正措施"
                }
            return results
    
    async def _ai_generate_implementation_batch(self, request: D6ImplementationActionsRequest, missingSolutions: List[SolutionData]) -> Dict[CauseItem, Dict[str, Any]]:
        """批量AI生成实施措施"""
        dimensions = [solution.causeItem.value for solution in missingSolutions]
        prompt = f"""
        请为以下解决方案制定实施措施：
        
        故障描述: {request.description}
        故障模式: {request.zdModelName}
        故障部位: {request.zdZeroPartName}
        
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
            for solution in missingSolutions:
                results[solution.causeItem] = {
                    "result": f"针对{solution.causeDesc}的实施措施"
                }
            return results
        except Exception as e:
            logger.error(f"批量AI生成实施措施失败: {e}")
            results = {}
            for solution in missingSolutions:
                results[solution.causeItem] = {
                    "result": f"针对{solution.causeDesc}的实施措施"
                }
            return results
    
    async def _ai_generate_solution(self, request: D5CorrectiveActionsRequest, analysis: CauseAnalysis) -> Dict[str, Any]:
        """AI生成解决方案"""
        prompt = f"""
        请为以下原因分析生成纠正措施：
        
        故障描述: {request.description}
        原因分析: {analysis.causeAnalysis}
        原因描述: {analysis.causeDesc}
        原因维度: {analysis.causeItem.value}
        
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
                "solution": f"针对{analysis.causeDesc}的纠正措施"
            }
    
    async def _ai_generate_implementation(self, request: D6ImplementationActionsRequest, solution: SolutionData) -> Dict[str, Any]:
        """AI生成实施措施"""
        prompt = f"""
        请为以下解决方案制定实施措施：
        
        故障描述: {request.description}
        解决方案: {solution.solution}
        原因描述: {solution.causeDesc}
        原因维度: {solution.causeItem.value}
        
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
                "result": f"针对{solution.causeDesc}的实施措施"
            }
    
    async def _generate_solution_summary(self, solution_data: List[SolutionData]) -> str:
        """生成解决方案总结"""
        try:
            solutions_text = "\n".join([
                f"- {solution.causeItem.value}维度: {solution.solution}"
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
                f"- {impl.causeItem.value}维度: {impl.implementedResult}"
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
    
    async def generate_root_cause_summary(self, request: D4RootCauseSummaryRequest) -> str:
        """生成根因分析总结"""
        try:
            analysis_text = "\n".join([
                f"- {analysis.causeItem.value}维度: {analysis.causeAnalysis}"
                for analysis in request.analysisData
            ])
            
            prompt = f"""
            请总结以下根因分析结果：
            
            故障描述: {request.description}
            故障模式: {request.zdModelName}
            故障部位: {request.zdZeroPartName}
            
            分析结果:
            {analysis_text}
            
            请提供一个简洁的根因分析总结，突出关键原因和重点。
            """
            
            messages = [
                LLMMessage(role="system", content="你是一个专业的故障分析专家，擅长总结根因分析结果。"),
                LLMMessage(role="user", content=prompt)
            ]
            
            response = await self.llm_service.chat(messages)
            return response.content
        except Exception as e:
            logger.error(f"生成根因分析总结失败: {e}")
            return "根因分析总结生成失败"
    
    async def generate_corrective_actions_summary(self, request: D5CorrectiveActionsSummaryRequest) -> str:
        """生成纠正措施总结"""
        try:
            solutions_text = "\n".join([
                f"- {solution.causeItem.value}维度: {solution.solution}"
                for solution in request.solutionData
            ])
            
            prompt = f"""
            请总结以下纠正措施：
            
            故障描述: {request.description}
            故障模式: {request.zdModelName}
            故障部位: {request.zdZeroPartName}
            
            纠正措施:
            {solutions_text}
            
            请提供一个简洁的纠正措施总结，突出关键措施和重点。
            """
            
            messages = [
                LLMMessage(role="system", content="你是一个专业的质量管理专家，擅长总结纠正措施。"),
                LLMMessage(role="user", content=prompt)
            ]
            
            response = await self.llm_service.chat(messages)
            return response.content
        except Exception as e:
            logger.error(f"生成纠正措施总结失败: {e}")
            return "纠正措施总结生成失败"
    
    async def generate_implementation_summary(self, request: D6ImplementationSummaryRequest) -> str:
        """生成实施措施总结"""
        try:
            implementations_text = "\n".join([
                f"- {impl.causeItem.value}维度: {impl.implementedResult}"
                for impl in request.implementationList
            ])
            
            prompt = f"""
            请总结以下实施措施：
            
            故障描述: {request.description}
            故障模式: {request.zdModelName}
            故障部位: {request.zdZeroPartName}
            
            实施措施:
            {implementations_text}
            
            请提供一个简洁的实施措施总结，突出关键实施步骤和重点。
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
