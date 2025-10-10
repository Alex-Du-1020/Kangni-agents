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
        rag_result = await self.rag_service.search_rag_with_answer(question, top_k=5, need_rerank=True)
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
        D5纠正措施生成（按维度来源单独处理，AI/文档二选一链式流程）
        """
        logger.info(f"开始D5纠正措施生成（分源/链式），故障模式: {request.zdModelName}，故障部位: {request.zdZeroPartName}")
        rewritten_question = self._rewrite_question_for_solutions(request)
        solutions = []
        for analysis in request.analysisData:
            if analysis.source == SourceType.AI_GENERATED:
                # AI生成的维度，直接LLM生成措施
                solution_text = await self._llm_generate_solution(rewritten_question, analysis.causeAnalysis, analysis.causeDesc, analysis.causeItem)
            else:
                # 文档来的，先尝试RAG查措施
                rag_solution = await self._rag_search_solution_for_analysis(rewritten_question, analysis)
                if rag_solution:
                    solution_text = rag_solution
                else:
                    # 没查到再LLM生成
                    solution_text = await self._llm_generate_solution(rewritten_question, analysis.causeAnalysis, analysis.causeDesc, analysis.causeItem)

            solutions.append(SolutionData(
                causeDesc=analysis.causeDesc,
                causeItem=analysis.causeItem,
                solution=solution_text,
                source=analysis.source))
        return solutions

    async def _llm_generate_solution(self, problem_desc, cause_analysis, cause_desc, cause_item):
        """给出问题描述、原因分析（和CauseItem用于上下文），LLM单独生成措施"""
        prompt = f"""
        问题描述: {problem_desc}
        原因分析: {cause_analysis}
        维度描述: {cause_item.value} ({cause_desc})

        请基于上述问题和原因，为该维度生成严谨、有效、具体的纠正措施建议，避免泛泛而谈。
        """
        messages = [
            LLMMessage(role="system", content="你是一个资深质量管理专家，善于根据具体问题/原因制定纠正措施。"),
            LLMMessage(role="user", content=prompt)
        ]
        response = await self.llm_service.chat(messages)
        return response.content.strip() if response and hasattr(response, 'content') else "未能生成有效纠正措施"

    async def _rag_search_solution_for_analysis(self, query: str, analysis: CauseAnalysis) -> str:
        """针对单个维度，基于文档内容RAG检索纠正措施。如无合适内容则返回None"""
        # 构造维度针对性检索query，可包含描述、模式、部位、当前维度的原因文本
        query = f"纠正措施 {analysis.causeItem.value}: {analysis.causeDesc}\n问题描述: {query} \n原因: {analysis.causeAnalysis}"
        try:
            rag_result = await self.rag_service.search_rag_with_answer(query, top_k=3, need_rerank=True)
            answers = rag_result.get("content", "")
            # 判断是否明显有可用答案
            if answers and "未找到" not in answers and len(answers.strip()) > 10:
                return answers.strip()
            return None
        except Exception as e:
            logger.warning(f"RAG检索纠正措施失败: {e}")
            return None

    async def _rag_search_implement_for_analysis(self, query: str, solution: SolutionData) -> str:
        """针对单个维度，基于文档内容RAG检索纠正措施。如无合适内容则返回None"""
        # 构造维度针对性检索query，可包含描述、模式、部位、当前维度的原因文本
        query = f"问题描述: {query}\n纠正措施 {solution.causeItem.value}: {solution.solution}\n 原因: {solution.causeDesc}"
        try:
            rag_result = await self.rag_service.search_rag_with_answer(query, top_k=3, need_rerank=True)
            answers = rag_result.get("content", "")
            # 判断是否明显有可用答案
            if answers and "未找到" not in answers and len(answers.strip()) > 10:
                return answers.strip()
            return None
        except Exception as e:
            logger.warning(f"RAG检索实施措施失败: {e}")
            return None
    
    async def generate_implementation_actions(self, request: D6ImplementationActionsRequest) -> List[ImplementationData]:
        """
        D6实施措施生成（原因描述+纠正措施+LLM生成实施结果）（按维度来源单独处理，AI/文档二选一链式流程）
        """
        logger.info(f"开始D6实施措施生成（分源/链式），故障模式: {request.zdModelName}，故障部位: {request.zdZeroPartName}")
        rewritten_question = self._rewrite_question_for_implementation(request)
        results = []
        for solution in request.solutionData:
            if solution.source == SourceType.AI_GENERATED:
                # AI生成的维度，直接LLM生成措施
                implemented_text = await self._llm_generate_implementation(rewritten_question, solution.causeDesc, solution.solution, solution.causeItem)
            else:
                # 文档来的，先尝试RAG查措施
                rag_implemented = await self._rag_search_implement_for_analysis(rewritten_question, solution)
                if rag_implemented:
                    implemented_text = rag_implemented
                else:
                    # 没查到再LLM生成
                    implemented_text = await self._llm_generate_implementation(rewritten_question, solution.causeDesc, solution.solution, solution.causeItem)

            results.append(ImplementationData(
                causeDesc=solution.causeDesc,
                causeItem=solution.causeItem,
                implementedResult=implemented_text,
                solution=solution.solution,
                causeConfidence=getattr(solution, 'causeConfidence', 0)))
        return results

    async def _llm_generate_implementation(self, problem_desc, cause_desc, solution_text, cause_item):
        """LLM生成实施措施（根据原因描述与纠正措施）"""
        prompt = f"""
        问题描述: {problem_desc}
        原因描述: {cause_desc}
        纠正措施: {solution_text}
        维度: {cause_item.value}
        请为上述内容生成清晰具体、可操作、可验收的实施措施（实施结果）。避免笼统描述。
        """
        messages = [
            LLMMessage(role="system", content="你是一个有丰富项目落地经验的质量改进专家，擅长制定具体可行的实施措施。"),
            LLMMessage(role="user", content=prompt)
        ]
        response = await self.llm_service.chat(messages)
        return response.content.strip() if response and hasattr(response, 'content') else "未能生成有效实施措施"
    
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

    def _rewrite_question_for_solutions(self, request: D5CorrectiveActionsRequest) -> str:
        """重写问题以更好地聚焦故障模式和部位"""
        return f"""
        针对故障模式"{request.zdModelName}"和故障部位"{request.zdZeroPartName}"，并根据故障的原因分析，生成纠正措施。
        """

    def _rewrite_question_for_implementation(self, request: D6ImplementationActionsRequest) -> str:
        """重写问题以更好地聚焦故障模式和部位"""
        return f"""
        针对故障模式"{request.zdModelName}"和故障部位"{request.zdZeroPartName}"，并根据故障的原因分析和纠正措施，生成实施措施。
        """
    
    async def generate_root_cause_summary(self, request: D4RootCauseSummaryRequest) -> str:
        """生成根因分析总结"""
        try:
            analysis_text = "\n".join([
                f"- {analysis.causeItem.value}维度: {analysis.causeAnalysis}"
                for analysis in request.analysisData
            ])
            
            prompt = f"""
            请总结以下根因分析结果：
            
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
        """生成纠正措施总结（区分置信度，主要/次要措施）"""
        try:
            major_solutions = [s for s in request.solutionData if getattr(s, 'causeConfidence', 0) > 50]
            minor_solutions = [s for s in request.solutionData if 25 < getattr(s, 'causeConfidence', 0) <= 50]
            # 忽略0~25的
            major_text = "\n".join([
                f"- {s.causeItem.value}维度: {s.solution}（置信度{s.causeConfidence}）"
                for s in major_solutions
            ])
            minor_text = "\n".join([
                f"- {s.causeItem.value}维度: {s.solution}（置信度{s.causeConfidence}）"
                for s in minor_solutions
            ])
            prompt = f"""
            请只总结权重较高/重要的纠正措施（causeConfidence>50，视为主要；25~50视为次要建议措施），忽略权重较低的。请突出重点。
            
            故障模式: {request.zdModelName}
            故障部位: {request.zdZeroPartName}
            
            主要纠正措施 (>50):
            {major_text}
            
            次要建议措施 (26~50):
            {minor_text}
            
            只需总结上述内容，忽略未出现的维度。
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
            implementation_texts = ""
            for implementation in request.implementationList:
                implementation_texts += f"- {implementation.causeItem.value}维度: {implementation.implementedResult}\n"

            prompt = f"""
            请总结以下实施措施：

            故障描述: {request.description}
            故障模式: {request.zdModelName}
            故障部位: {request.zdZeroPartName}

            实施措施：
            {implementation_texts}

            请提供一个简洁的实施措施总结，突出关键措施和重点。
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
