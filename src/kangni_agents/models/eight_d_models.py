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

class ReferencedDocument(BaseModel):
    """引用文档模型"""
    documentName: str = Field(..., description="文档名称", alias="documentName")
    score: float = Field(..., description="相关性评分")

class CauseAnalysis(BaseModel):
    """原因分析数据模型"""
    causeAnalysis: str = Field(..., description="原因分析内容", alias="causeAnalysis")
    causeDesc: str = Field(..., description="原因描述", alias="causeDesc")
    causeItem: CauseItem = Field(..., description="原因项目", alias="causeItem")
    source: SourceType = Field(..., description="数据来源")
    referencedDocuments: Optional[List[ReferencedDocument]] = Field(None, description="引用的文档列表", alias="referencedDocuments")

class SolutionData(BaseModel):
    """解决方案数据模型"""
    causeDesc: str = Field(..., description="原因描述", alias="causeDesc")
    causeItem: CauseItem = Field(..., description="原因项目", alias="causeItem")
    solution: str = Field(..., description="解决方案")
    causeConfidence: int = Field(0, description="原因置信度，0-100", alias="causeConfidence")
    source: SourceType = Field(..., description="数据来源", alias="source")

class ImplementationData(BaseModel):
    """实施措施数据模型"""
    causeDesc: str = Field(..., description="原因描述", alias="causeDesc")
    causeItem: CauseItem = Field(..., description="原因项目", alias="causeItem")
    implementedResult: str = Field(..., description="实施结果", alias="implementedResult")
    solution: str = Field(..., description="解决方案")

# D4根因分析请求模型
class D4RootCauseAnalysisRequest(BaseModel):
    """D4根因分析请求"""
    causeItems: List[CauseItem] = Field(..., description="分析维度列表", alias="causeItems")
    description: str = Field(..., description="问题描述")
    zdModelName: str = Field(..., description="故障模式名称", alias="zdModelName")
    zdZeroPartName: str = Field(..., description="故障部位名称", alias="zdZeroPartName")

class D4RootCauseAnalysisResponse(BaseModel):
    """D4根因分析响应"""
    analysisData: List[CauseAnalysis] = Field(..., description="分析结果列表", alias="analysisData")

# D5纠正措施请求模型
class D5CorrectiveActionsRequest(BaseModel):
    """D5纠正措施请求"""
    analysisData: List[CauseAnalysis] = Field(..., description="原因分析数据", alias="analysisData")
    description: str = Field(..., description="问题描述")
    zdModelName: str = Field(..., description="故障模式名称", alias="zdModelName")
    zdZeroPartName: str = Field(..., description="故障部位名称", alias="zdZeroPartName")

class D5CorrectiveActionsResponse(BaseModel):
    """D5纠正措施响应"""
    solutionData: List[SolutionData] = Field(..., description="解决方案列表", alias="solutionData")

# D6实施措施请求模型
class D6ImplementationActionsRequest(BaseModel):
    """D6实施措施请求"""
    description: str = Field(..., description="问题描述")
    solutionData: List[SolutionData] = Field(..., description="解决方案数据", alias="solutionData")
    zdModelName: str = Field(..., description="故障模式名称", alias="zdModelName")
    zdZeroPartName: str = Field(..., description="故障部位名称", alias="zdZeroPartName")

class D6ImplementationActionsResponse(BaseModel):
    """D6实施措施响应"""
    implementationList: List[ImplementationData] = Field(..., description="实施措施列表", alias="implementationList")

# D4根因总结请求模型
class D4RootCauseSummaryRequest(BaseModel):
    """D4根因总结请求"""
    analysisData: List[CauseAnalysis] = Field(..., description="原因分析数据", alias="analysisData")
    description: str = Field(..., description="问题描述")
    zdModelName: str = Field(..., description="故障模式名称", alias="zdModelName")
    zdZeroPartName: str = Field(..., description="故障部位名称", alias="zdZeroPartName")

class D4RootCauseSummaryResponse(BaseModel):
    """D4根因总结响应"""
    analysisSummary: str = Field(..., description="原因总结", alias="analysisSummary")

# D5纠正措施总结请求模型
class D5CorrectiveActionsSummaryRequest(BaseModel):
    """D5纠正措施总结请求"""
    solutionData: List[SolutionData] = Field(..., description="解决方案数据", alias="solutionData")
    description: str = Field(..., description="问题描述")
    zdModelName: str = Field(..., description="故障模式名称", alias="zdModelName")
    zdZeroPartName: str = Field(..., description="故障部位名称", alias="zdZeroPartName")

class D5CorrectiveActionsSummaryResponse(BaseModel):
    """D5纠正措施总结响应"""
    solutionSummary: str = Field(..., description="纠正措施总结", alias="solutionSummary")

# D6实施措施总结请求模型
class D6ImplementationSummaryRequest(BaseModel):
    """D6实施措施总结请求"""
    implementationList: List[ImplementationData] = Field(..., description="实施措施数据", alias="implementationList")
    description: str = Field(..., description="问题描述")
    zdModelName: str = Field(..., description="故障模式名称", alias="zdModelName")
    zdZeroPartName: str = Field(..., description="故障部位名称", alias="zdZeroPartName")

class D6ImplementationSummaryResponse(BaseModel):
    """D6实施措施总结响应"""
    implementationSummary: str = Field(..., description="措施实施总结", alias="implementationSummary")
