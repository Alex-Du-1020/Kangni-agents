"""
8D分析API测试
"""
import pytest
from fastapi.testclient import TestClient
from kangni_agents.main import app

client = TestClient(app)

def test_d4_root_cause_analysis():
    """测试D4根因分析接口"""
    import asyncio
    from kangni_agents.services.eight_d_analysis_service import d8_analysis_service
    from kangni_agents.models.eight_d_models import D4RootCauseAnalysisRequest, CauseItem, SourceType
    
    # 构建请求
    request = D4RootCauseAnalysisRequest(
        causeItems=[CauseItem.PERSON, CauseItem.MACHINE, CauseItem.MATERIAL, CauseItem.METHOD, CauseItem.ENVIRONMENT, CauseItem.MEASUREMENT],
        description="2025年08月09日,配属南昌车辆段南昌客整所CR200J3-A-5014车组(直体长编),担当D133次（北京丰台-井冈山,17:43—次日8:59),机械师反馈南昌西6:39司机通知501407车4位门关不上，机械师6:43手动关门，6:44开车，晚点5分钟。",
        zdModelName="卡滞",
        zdZeroPartName="踏板电机"
    )
    
    # 直接调用service方法
    async def run_test():
        analysis_results = await d8_analysis_service.analyze_root_cause(request)
        
        # 检查结果数量
        assert len(analysis_results) == 6, f"期望6个分析结果，实际得到{len(analysis_results)}个"
        
        # 检查每个分析结果
        for analysis in analysis_results:

            assert hasattr(analysis, 'causeAnalysis'), "缺少causeAnalysis字段"
            assert hasattr(analysis, 'causeDesc'), "缺少causeDesc字段"
            assert hasattr(analysis, 'causeItem'), "缺少causeItem字段"
            assert hasattr(analysis, 'source'), "缺少source字段"
            assert hasattr(analysis, 'referencedDocuments'), "缺少referencedDocuments字段"

            if analysis.causeItem == CauseItem.METHOD:
                assert analysis.source == SourceType.LOCAL_DOC, "法维度原因分析不正确"
            
            # 检查causeAnalysis是纯文本，不包含JSON格式
            assert isinstance(analysis.causeAnalysis, str), "causeAnalysis应该是字符串"
            assert "```json" not in analysis.causeAnalysis, f"causeAnalysis包含JSON格式: {analysis.causeAnalysis}"
            assert len(analysis.causeAnalysis) > 10, "causeAnalysis内容太短"
            
            # 检查source字段
            assert analysis.source in ["本地文档", "AI生成"], f"无效的source值: {analysis.source}"
            
            # 检查referencedDocuments字段
            if analysis.source == SourceType.LOCAL_DOC:
                assert analysis.referencedDocuments is not None, "本地文档来源应该有引用文档"
                assert len(analysis.referencedDocuments) > 0, "引用文档列表不应为空"
                for doc in analysis.referencedDocuments:
                    assert hasattr(doc, 'documentName'), "引用文档缺少documentName字段"
                    assert hasattr(doc, 'score'), "引用文档缺少score字段"
                    assert isinstance(doc.score, (int, float)), "score应该是数字"
            else:
                assert analysis.referencedDocuments is None, "AI生成来源不应该有引用文档"
            
            print(f"维度: {analysis.causeItem.value}, 来源: {analysis.source}")
            print(f"分析: {analysis.causeAnalysis[:100]}...")
            if analysis.referencedDocuments:
                print(f"引用文档数量: {len(analysis.referencedDocuments)}")
            print("---")
        
        # 检查期望的source分布
        sources = [analysis.source for analysis in analysis_results]
        print(f"所有来源: {sources}")
        
        # 检查是否有法维度
        method_analysis = next((a for a in analysis_results if a.causeItem == CauseItem.METHOD), None)
        if method_analysis:
            print(f"法维度来源: {method_analysis.source}")
            print(f"法维度分析: {method_analysis.causeAnalysis[:200]}...")
        
        # 验证JSON格式清理
        for analysis in analysis_results:
            if "```json" in analysis.causeAnalysis:
                print(f"警告: {analysis.causeItem.value}维度的分析仍包含JSON格式")
            else:
                print(f"✓ {analysis.causeItem.value}维度的分析格式正确")
        
        return analysis_results
    
    # 运行异步测试
    analysis_results = asyncio.run(run_test())
    
    print(f"D4根因分析完成，共{len(analysis_results)}个结果")

def test_d5_corrective_actions():
    """测试D5纠正措施接口"""
    request_data = {
        "analysisData": [
            {
                "causeAnalysis": "作业人员未严格按照操作手册进行操作",
                "causeDesc": "操作不当",
                "causeItem": "人",
                "source": "AI生成"
            },
            {
                "causeAnalysis": "设备运行时间过长导致故障",
                "causeDesc": "设备运行参数异常",
                "causeItem": "机",
                "source": "AI生成"
            }
        ],
        "description": "踏板电机故障导致车门无法正常关闭",
        "zdModelName": "卡滞",
        "zdZeroPartName": "踏板电机"
    }
    
    response = client.post("/qomo/v1/8d/d5-corrective-actions", json=request_data)
    
    # 检查响应状态
    assert response.status_code == 200
    
    # 检查响应结构
    data = response.json()
    assert "solutionData" in data
    assert "solutionSummary" in data
    assert isinstance(data["solutionData"], list)
    
    print(f"D5纠正措施响应: {data}")

def test_d6_implementation_actions():
    """测试D6实施措施接口"""
    request_data = {
        "description": "踏板电机故障导致车门无法正常关闭",
        "solutionData": [
            {
                "causeDesc": "操作不当",
                "causeItem": "人",
                "solution": "加强操作培训，建立操作检查制度"
            },
            {
                "causeDesc": "设备运行参数异常",
                "causeItem": "机",
                "solution": "定期维护保养，建立设备监控系统"
            }
        ],
        "zdModelName": "卡滞",
        "zdZeroPartName": "踏板电机"
    }
    
    response = client.post("/qomo/v1/8d/d6-implementation-actions", json=request_data)
    
    # 检查响应状态
    assert response.status_code == 200
    
    # 检查响应结构
    data = response.json()
    assert "implementationList" in data
    assert "implementationSummary" in data
    assert isinstance(data["implementationList"], list)
    
    print(f"D6实施措施响应: {data}")

def test_d4_root_cause_summary():
    """测试D4根因总结接口"""
    request_data = {
        "analysisData": [
            {
                "causeAnalysis": "作业人员未严格按照操作手册进行操作",
                "causeDesc": "操作不当",
                "causeItem": "人",
                "source": "AI生成"
            },
            {
                "causeAnalysis": "设备运行时间过长导致故障",
                "causeDesc": "设备运行参数异常",
                "causeItem": "机",
                "source": "AI生成"
            },
            {
                "causeAnalysis": "物料已超过最大使用年限未进行更换",
                "causeDesc": "材料老化",
                "causeItem": "料",
                "source": "本地文档"
            }
        ],
        "description": "踏板电机故障导致车门无法正常关闭",
        "zdModelName": "卡滞",
        "zdZeroPartName": "踏板电机"
    }
    
    response = client.post("/qomo/v1/8d/d4-root-cause-summary", json=request_data)
    
    # 检查响应状态
    assert response.status_code == 200
    
    # 检查响应结构
    data = response.json()
    assert "analysisSummary" in data
    assert isinstance(data["analysisSummary"], str)
    assert len(data["analysisSummary"]) > 0
    
    print(f"D4根因总结响应: {data}")

def test_d5_corrective_actions_summary():
    """测试D5纠正措施总结接口"""
    request_data = {
        "solutionData": [
            {
                "causeDesc": "操作不当",
                "causeItem": "人",
                "solution": "加强操作培训，建立操作检查制度"
            },
            {
                "causeDesc": "设备运行参数异常",
                "causeItem": "机",
                "solution": "定期维护保养，建立设备监控系统"
            },
            {
                "causeDesc": "材料老化",
                "causeItem": "料",
                "solution": "定期检查物料是否过期，判断是否有老化风险"
            }
        ],
        "description": "踏板电机故障导致车门无法正常关闭",
        "zdModelName": "卡滞",
        "zdZeroPartName": "踏板电机"
    }
    
    response = client.post("/qomo/v1/8d/d5-corrective-actions-summary", json=request_data)
    
    # 检查响应状态
    assert response.status_code == 200
    
    # 检查响应结构
    data = response.json()
    assert "solutionSummary" in data
    assert isinstance(data["solutionSummary"], str)
    assert len(data["solutionSummary"]) > 0
    
    print(f"D5纠正措施总结响应: {data}")

def test_d6_implementation_summary():
    """测试D6实施措施总结接口"""
    request_data = {
        "implementationList": [
            {
                "causeDesc": "操作不当",
                "causeItem": "人",
                "implementedResult": "已完成操作培训，建立检查制度",
                "solution": "加强操作培训，建立操作检查制度"
            },
            {
                "causeDesc": "设备运行参数异常",
                "causeItem": "机",
                "implementedResult": "已完成设备维护，建立监控系统",
                "solution": "定期维护保养，建立设备监控系统"
            },
            {
                "causeDesc": "材料老化",
                "causeItem": "料",
                "implementedResult": "定期检查物料做好记录",
                "solution": "定期检查物料是否过期，判断是否有老化风险"
            }
        ],
        "description": "踏板电机故障导致车门无法正常关闭",
        "zdModelName": "卡滞",
        "zdZeroPartName": "踏板电机"
    }
    
    response = client.post("/qomo/v1/8d/d6-implementation-summary", json=request_data)
    
    # 检查响应状态
    assert response.status_code == 200
    
    # 检查响应结构
    data = response.json()
    assert "implementationSummary" in data
    assert isinstance(data["implementationSummary"], str)
    assert len(data["implementationSummary"]) > 0
    
    print(f"D6实施措施总结响应: {data}")

if __name__ == "__main__":
    # 运行测试
    test_d4_root_cause_analysis()
    # test_d5_corrective_actions()
    # test_d6_implementation_actions()
    # test_d4_root_cause_summary()
    # test_d5_corrective_actions_summary()
    # test_d6_implementation_summary()
    print("所有8D分析测试通过!")
