"""
8D分析相关数据模型
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum

class CauseItem(str, Enum):
    """原因项目枚举"""
    PERSON = "人"
    MACHINE = "机"
    MATERIAL = "料"
    METHOD = "法"
    ENVIRONMENT = "环"
    MEASUREMENT = "测"
    OTHER = "其他"

class SourceType(str, Enum):
    """数据来源类型"""
    LOCAL_DOC = "本地文档"
    AI_GENERATED = "AI生成"

class CauseAnalysis(BaseModel):
    """原因分析数据模型"""
    cause_analysis: str = Field(..., description="原因分析内容")
    cause_desc: str = Field(..., description="原因描述")
    cause_item: CauseItem = Field(..., description="原因项目")
    source: SourceType = Field(..., description="数据来源")

class SolutionData(BaseModel):
    """解决方案数据模型"""
    cause_desc: str = Field(..., description="原因描述")
    cause_item: CauseItem = Field(..., description="原因项目")
    solution: str = Field(..., description="解决方案")

class ImplementationData(BaseModel):
    """实施措施数据模型"""
    cause_item: CauseItem = Field(..., description="原因项目")
    implemented_result: str = Field(..., description="实施结果")
    solution: str = Field(..., description="解决方案")

# D4根因分析请求模型
class D4RootCauseAnalysisRequest(BaseModel):
    """D4根因分析请求"""
    cause_items: List[CauseItem] = Field(..., description="分析维度列表")
    description: str = Field(..., description="问题描述")
    zd_model_name: str = Field(..., description="故障模式名称")
    zd_zero_part_name: str = Field(..., description="故障部位名称")

class D4RootCauseAnalysisResponse(BaseModel):
    """D4根因分析响应"""
    analysis_data: List[CauseAnalysis] = Field(..., description="分析结果列表")

# D5纠正措施请求模型
class D5CorrectiveActionsRequest(BaseModel):
    """D5纠正措施请求"""
    analysis_data: List[CauseAnalysis] = Field(..., description="原因分析数据")
    description: str = Field(..., description="问题描述")
    zd_model_name: str = Field(..., description="故障模式名称")
    zd_zero_part_name: str = Field(..., description="故障部位名称")

class D5CorrectiveActionsResponse(BaseModel):
    """D5纠正措施响应"""
    solution_data: List[SolutionData] = Field(..., description="解决方案列表")
    solution_summary: str = Field(..., description="解决方案总结")

# D6实施措施请求模型
class D6ImplementationActionsRequest(BaseModel):
    """D6实施措施请求"""
    description: str = Field(..., description="问题描述")
    solution_data: List[SolutionData] = Field(..., description="解决方案数据")
    zd_model_name: str = Field(..., description="故障模式名称")
    zd_zero_part_name: str = Field(..., description="故障部位名称")

class D6ImplementationActionsResponse(BaseModel):
    """D6实施措施响应"""
    implementation_list: List[ImplementationData] = Field(..., description="实施措施列表")
    implementation_summary: str = Field(..., description="实施措施总结")
