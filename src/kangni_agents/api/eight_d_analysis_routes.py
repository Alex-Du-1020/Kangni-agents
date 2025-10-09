"""
8D分析API路由
"""
from fastapi import APIRouter, HTTPException
from typing import List
import logging

from ..models.eight_d_models import (
    D4RootCauseAnalysisRequest, D4RootCauseAnalysisResponse,
    D5CorrectiveActionsRequest, D5CorrectiveActionsResponse,
    D6ImplementationActionsRequest, D6ImplementationActionsResponse,
    D4RootCauseSummaryRequest, D4RootCauseSummaryResponse,
    D5CorrectiveActionsSummaryRequest, D5CorrectiveActionsSummaryResponse,
    D6ImplementationSummaryRequest, D6ImplementationSummaryResponse
)
from ..services.eight_d_analysis_service import d8_analysis_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/qomo/v1/8d", tags=["8d-analysis"])

@router.post("/d4-root-cause-analysis", response_model=D4RootCauseAnalysisResponse)
async def d4_root_cause_analysis(request: D4RootCauseAnalysisRequest):
    """
    D4根因分析接口
    
    根据故障描述、故障模式、故障部位和分析维度，分析问题的根本原因。
    优先使用RAG搜索本地文档，如果找不到则使用AI生成分析结果。
    
    Args:
        request: D4根因分析请求，包含：
            - cause_items: 分析维度列表（人、机、料、法、环、测、其他）
            - description: 故障描述
            - zdModelName: 故障模式名称
            - zdZeroPartName: 故障部位名称
    
    Returns:
        D4RootCauseAnalysisResponse: 包含分析结果列表，每个结果包含：
            - cause_analysis: 原因分析内容
            - cause_desc: 原因描述
            - cause_item: 原因项目
            - source: 数据来源（本地文档/AI生成）
            - referenced_documents: 引用的文档列表（仅当source为本地文档时）
    
    Raises:
        HTTPException: 如果分析失败
    """
    try:
        logger.info(f"开始D4根因分析，故障模式: {request.zdModelName}")
        
        # 调用分析服务
        analysisData = await d8_analysis_service.analyze_root_cause(request)
        
        logger.info(f"D4根因分析完成，共分析{len(analysisData)}个维度")
        
        return D4RootCauseAnalysisResponse(analysisData=analysisData)
        
    except Exception as e:
        logger.error(f"D4根因分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"D4根因分析失败: {str(e)}")

@router.post("/d5-corrective-actions", response_model=D5CorrectiveActionsResponse)
async def d5_corrective_actions(request: D5CorrectiveActionsRequest):
    """
    D5纠正措施接口
    
    根据原因分析结果，生成相应的纠正措施和预防措施。
    优先使用RAG搜索本地文档，如果找不到则使用AI生成解决方案。
    
    Args:
        request: D5纠正措施请求，包含：
            - analysisData: 原因分析数据列表
            - description: 故障描述
            - zdModelName: 故障模式名称
            - zdZeroPartName: 故障部位名称
    
    Returns:
        D5CorrectiveActionsResponse: 包含解决方案列表和总结：
            - solutionData: 解决方案列表
    
    Raises:
        HTTPException: 如果生成失败
    """
    try:
        logger.info(f"开始D5纠正措施生成，故障模式: {request.zdModelName}")
        
        # 调用分析服务生成纠正措施
        solutionData = await d8_analysis_service.generate_corrective_actions(request)
        
        logger.info(f"D5纠正措施生成完成，共生成{len(solutionData)}个解决方案")
        
        return D5CorrectiveActionsResponse(
            solutionData=solutionData
        )
        
    except Exception as e:
        logger.error(f"D5纠正措施生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"D5纠正措施生成失败: {str(e)}")

@router.post("/d6-implementation-actions", response_model=D6ImplementationActionsResponse)
async def d6_implementation_actions(request: D6ImplementationActionsRequest):
    """
    D6实施措施接口
    
    根据解决方案，制定具体的实施措施和执行计划。
    优先使用RAG搜索本地文档，如果找不到则使用AI生成实施措施。
    
    Args:
        request: D6实施措施请求，包含：
            - description: 故障描述
            - solutionData: 解决方案数据列表
            - zdModelName: 故障模式名称
            - zdZeroPartName: 故障部位名称
    
    Returns:
        D6ImplementationActionsResponse: 包含实施措施列表和总结：
            - implementationList: 实施措施列表
    
    Raises:
        HTTPException: 如果生成失败
    """
    try:
        logger.info(f"开始D6实施措施生成，故障模式: {request.zdModelName}")
        
        # 调用分析服务生成实施措施
        implementationList = await d8_analysis_service.generate_implementation_actions(request)
        
        logger.info(f"D6实施措施生成完成，共生成{len(implementationList)}个实施措施")
        
        return D6ImplementationActionsResponse(
            implementationList=implementationList
        )
        
    except Exception as e:
        logger.error(f"D6实施措施生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"D6实施措施生成失败: {str(e)}")

@router.post("/d4-root-cause-summary", response_model=D4RootCauseSummaryResponse)
async def d4_root_cause_summary(request: D4RootCauseSummaryRequest):
    """
    D4根因总结接口
    
    根据原因分析结果，生成根因分析总结。
    
    Args:
        request: D4根因总结请求，包含：
            - analysisData: 原因分析数据列表
            - description: 故障描述
            - zdModelName: 故障模式名称
            - zdZeroPartName: 故障部位名称
    
    Returns:
        D4RootCauseSummaryResponse: 包含原因总结
    
    Raises:
        HTTPException: 如果总结生成失败
    """
    try:
        logger.info(f"开始D4根因总结，故障模式: {request.zdModelName}")
        
        # 调用分析服务生成根因总结
        analysisSummary = await d8_analysis_service.generate_root_cause_summary(request)
        
        logger.info(f"D4根因总结生成完成")
        
        return D4RootCauseSummaryResponse(analysisSummary=analysisSummary)
        
    except Exception as e:
        logger.error(f"D4根因总结生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"D4根因总结生成失败: {str(e)}")

@router.post("/d5-corrective-actions-summary", response_model=D5CorrectiveActionsSummaryResponse)
async def d5_corrective_actions_summary(request: D5CorrectiveActionsSummaryRequest):
    """
    D5纠正措施总结接口
    
    根据解决方案数据，生成纠正措施总结。
    
    Args:
        request: D5纠正措施总结请求，包含：
            - solutionData: 解决方案数据列表
            - description: 故障描述
            - zdModelName: 故障模式名称
            - zdZeroPartName: 故障部位名称
    
    Returns:
        D5CorrectiveActionsSummaryResponse: 包含纠正措施总结
    
    Raises:
        HTTPException: 如果总结生成失败
    """
    try:
        logger.info(f"开始D5纠正措施总结，故障模式: {request.zdModelName}")
        
        # 调用分析服务生成纠正措施总结
        solutionSummary = await d8_analysis_service.generate_corrective_actions_summary(request)
        
        logger.info(f"D5纠正措施总结生成完成")
        
        return D5CorrectiveActionsSummaryResponse(solutionSummary=solutionSummary)
        
    except Exception as e:
        logger.error(f"D5纠正措施总结生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"D5纠正措施总结生成失败: {str(e)}")

@router.post("/d6-implementation-summary", response_model=D6ImplementationSummaryResponse)
async def d6_implementation_summary(request: D6ImplementationSummaryRequest):
    """
    D6实施措施总结接口
    
    根据实施措施数据，生成实施措施总结。
    
    Args:
        request: D6实施措施总结请求，包含：
            - implementationList: 实施措施数据列表
            - description: 故障描述
            - zdModelName: 故障模式名称
            - zdZeroPartName: 故障部位名称
    
    Returns:
        D6ImplementationSummaryResponse: 包含实施措施总结
    
    Raises:
        HTTPException: 如果总结生成失败
    """
    try:
        logger.info(f"开始D6实施措施总结，故障模式: {request.zdModelName}")
        
        # 调用分析服务生成实施措施总结
        implementationSummary = await d8_analysis_service.generate_implementation_summary(request)
        
        logger.info(f"D6实施措施总结生成完成")
        
        return D6ImplementationSummaryResponse(implementationSummary=implementationSummary)
        
    except Exception as e:
        logger.error(f"D6实施措施总结生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"D6实施措施总结生成失败: {str(e)}")
