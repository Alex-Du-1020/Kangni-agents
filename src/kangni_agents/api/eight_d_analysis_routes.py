"""
8D分析API路由
"""
from fastapi import APIRouter, HTTPException
from typing import List
import logging

from ..models.eight_d_models import (
    D4RootCauseAnalysisRequest, D4RootCauseAnalysisResponse,
    D5CorrectiveActionsRequest, D5CorrectiveActionsResponse,
    D6ImplementationActionsRequest, D6ImplementationActionsResponse
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
            - zd_model_name: 故障模式名称
            - zd_zero_part_name: 故障部位名称
    
    Returns:
        D4RootCauseAnalysisResponse: 包含分析结果列表，每个结果包含：
            - cause_analysis: 原因分析内容
            - cause_desc: 原因描述
            - cause_item: 原因项目
            - source: 数据来源（本地文档/AI生成）
    
    Raises:
        HTTPException: 如果分析失败
    """
    try:
        logger.info(f"开始D4根因分析，故障模式: {request.zd_model_name}")
        
        # 调用分析服务
        analysis_data = await d8_analysis_service.analyze_root_cause(request)
        
        logger.info(f"D4根因分析完成，共分析{len(analysis_data)}个维度")
        
        return D4RootCauseAnalysisResponse(analysis_data=analysis_data)
        
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
            - analysis_data: 原因分析数据列表
            - description: 故障描述
            - zd_model_name: 故障模式名称
            - zd_zero_part_name: 故障部位名称
    
    Returns:
        D5CorrectiveActionsResponse: 包含解决方案列表和总结：
            - solution_data: 解决方案列表
            - solution_summary: 解决方案总结
    
    Raises:
        HTTPException: 如果生成失败
    """
    try:
        logger.info(f"开始D5纠正措施生成，故障模式: {request.zd_model_name}")
        
        # 调用分析服务生成纠正措施
        solution_data = await d8_analysis_service.generate_corrective_actions(request)
        
        # 生成解决方案总结
        solution_summary = await d8_analysis_service._generate_solution_summary(solution_data)
        
        logger.info(f"D5纠正措施生成完成，共生成{len(solution_data)}个解决方案")
        
        return D5CorrectiveActionsResponse(
            solution_data=solution_data,
            solution_summary=solution_summary
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
            - solution_data: 解决方案数据列表
            - zd_model_name: 故障模式名称
            - zd_zero_part_name: 故障部位名称
    
    Returns:
        D6ImplementationActionsResponse: 包含实施措施列表和总结：
            - implementation_list: 实施措施列表
            - implementation_summary: 实施措施总结
    
    Raises:
        HTTPException: 如果生成失败
    """
    try:
        logger.info(f"开始D6实施措施生成，故障模式: {request.zd_model_name}")
        
        # 调用分析服务生成实施措施
        implementation_list = await d8_analysis_service.generate_implementation_actions(request)
        
        # 生成实施措施总结
        implementation_summary = await d8_analysis_service._generate_implementation_summary(implementation_list)
        
        logger.info(f"D6实施措施生成完成，共生成{len(implementation_list)}个实施措施")
        
        return D6ImplementationActionsResponse(
            implementation_list=implementation_list,
            implementation_summary=implementation_summary
        )
        
    except Exception as e:
        logger.error(f"D6实施措施生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"D6实施措施生成失败: {str(e)}")
