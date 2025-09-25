"""
8D分析API测试
"""
import pytest
from fastapi.testclient import TestClient
from src.kangni_agents.main import app

client = TestClient(app)

def test_d4_root_cause_analysis():
    """测试D4根因分析接口"""
    request_data = {
        "cause_items": ["人", "机", "料"],
        "description": "2025年08月09日,配属南昌车辆段南昌客整所CR200J3-A-5014车组(直体长编),担当D133次（北京丰台-井冈山,17:43—次日8:59),机械师反馈南昌西6:39司机通知501407车4位门关不上，机械师6:43手动关门，6:44开车，晚点5分钟。",
        "zd_model_name": "卡滞",
        "zd_zero_part_name": "踏板电机"
    }
    
    response = client.post("/qomo/v1/8d/d4-root-cause-analysis", json=request_data)
    
    # 检查响应状态
    assert response.status_code == 200
    
    # 检查响应结构
    data = response.json()
    assert "analysis_data" in data
    assert isinstance(data["analysis_data"], list)
    
    # 检查分析结果
    if data["analysis_data"]:
        analysis = data["analysis_data"][0]
        assert "cause_analysis" in analysis
        assert "cause_desc" in analysis
        assert "cause_item" in analysis
        assert "source" in analysis
        assert analysis["source"] in ["本地文档", "AI生成"]
    
    print(f"D4根因分析响应: {data}")

def test_d5_corrective_actions():
    """测试D5纠正措施接口"""
    request_data = {
        "analysis_data": [
            {
                "cause_analysis": "作业人员未严格按照操作手册进行操作",
                "cause_desc": "操作不当",
                "cause_item": "人",
                "source": "AI生成"
            },
            {
                "cause_analysis": "设备运行时间过长导致故障",
                "cause_desc": "设备运行参数异常",
                "cause_item": "机",
                "source": "AI生成"
            }
        ],
        "description": "踏板电机故障导致车门无法正常关闭",
        "zd_model_name": "卡滞",
        "zd_zero_part_name": "踏板电机"
    }
    
    response = client.post("/qomo/v1/8d/d5-corrective-actions", json=request_data)
    
    # 检查响应状态
    assert response.status_code == 200
    
    # 检查响应结构
    data = response.json()
    assert "solution_data" in data
    assert "solution_summary" in data
    assert isinstance(data["solution_data"], list)
    
    print(f"D5纠正措施响应: {data}")

def test_d6_implementation_actions():
    """测试D6实施措施接口"""
    request_data = {
        "description": "踏板电机故障导致车门无法正常关闭",
        "solution_data": [
            {
                "cause_desc": "操作不当",
                "cause_item": "人",
                "solution": "加强操作培训，建立操作检查制度"
            },
            {
                "cause_desc": "设备运行参数异常",
                "cause_item": "机",
                "solution": "定期维护保养，建立设备监控系统"
            }
        ],
        "zd_model_name": "卡滞",
        "zd_zero_part_name": "踏板电机"
    }
    
    response = client.post("/qomo/v1/8d/d6-implementation-actions", json=request_data)
    
    # 检查响应状态
    assert response.status_code == 200
    
    # 检查响应结构
    data = response.json()
    assert "implementation_list" in data
    assert "implementation_summary" in data
    assert isinstance(data["implementation_list"], list)
    
    print(f"D6实施措施响应: {data}")

if __name__ == "__main__":
    # 运行测试
    test_d4_root_cause_analysis()
    test_d5_corrective_actions()
    test_d6_implementation_actions()
    print("所有8D分析测试通过!")
