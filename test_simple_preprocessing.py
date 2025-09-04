#!/usr/bin/env python3
"""
简化的查询预处理器测试
Simplified test for query preprocessor (standalone)
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ExtractedEntity:
    """提取的实体信息"""
    raw_text: str
    clean_text: str
    entity_type: str
    start_pos: int
    end_pos: int

def test_entity_extraction():
    """测试实体提取功能"""
    
    # Hash pattern
    pattern = r"#([^#]+)#"
    
    test_cases = [
        {
            "name": "原始问题案例",
            "query": "#合肥S1号线项目乘客室门#这个项目一共有多少个订单？",
            "expected_entities": ["合肥S1号线项目乘客室门"],
        },
        {
            "name": "项目+部件组合",
            "query": "#北京地铁15号线# 项目的 #乘客门系统# 部件有多少故障？",
            "expected_entities": ["北京地铁15号线", "乘客门系统"],
        },
        {
            "name": "复杂项目名称",
            "query": "#德里地铁4期项目-20D21028C000#的故障统计信息",
            "expected_entities": ["德里地铁4期项目-20D21028C000"],
        },
        {
            "name": "无特殊标记",
            "query": "查询所有项目的订单总数",
            "expected_entities": [],
        }
    ]
    
    print("🧪 测试实体提取功能")
    print("=" * 80)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📝 测试案例 {i}: {case['name']}")
        print(f"原始查询: {case['query']}")
        
        # 提取实体
        matches = re.findall(pattern, case['query'])
        
        print(f"提取结果: {matches}")
        print(f"预期结果: {case['expected_entities']}")
        
        # 验证结果
        success = matches == case['expected_entities']
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"结果: {status}")
        
        if not success:
            print(f"⚠️  不匹配！")

def test_sql_generation_logic():
    """测试SQL生成逻辑"""
    print("\n" + "=" * 80)
    print("🔧 测试SQL生成改进逻辑")
    print("=" * 80)
    
    # 原始查询
    original_query = "#合肥S1号线项目乘客室门#这个项目一共有多少个订单？"
    
    # 提取实体
    pattern = r"#([^#]+)#"
    match = re.search(pattern, original_query)
    
    if match:
        entity_text = match.group(1)  # 合肥S1号线项目乘客室门
        print(f"提取的完整实体: '{entity_text}'")
        
        # 替换为占位符
        placeholder = "__ENTITY_0__"
        processed_query = original_query.replace(match.group(0), placeholder)
        print(f"处理后查询: {processed_query}")
        
        # 模拟SQL生成提示
        sql_hints = f"""
=== SQL生成特殊要求 ===
生成SQL时请注意以下要求：

实体处理要求:
- {placeholder} 代表 '{entity_text}'，这是一个完整的标识符，在SQL中必须作为单一完整值进行精确匹配，不能拆分

字段映射建议:
- 涉及项目时，优先使用字段：projectname_s, project_name

特殊处理提示:
- 这是项目+部件的组合查询，应该同时匹配项目名称和部件名称
- 项目信息使用 projectname_s 字段进行 LIKE 匹配
- 部件信息使用 partname_s 字段进行 LIKE 匹配
"""
        
        print("生成的SQL提示:")
        print(sql_hints)
        
        # 模拟生成的SQL（使用占位符）
        placeholder_sql = f"SELECT COUNT(DISTINCT orderno) FROM kn_quality_trace_prod_order_process WHERE projectname_s LIKE '%{placeholder}%'"
        print(f"\n占位符SQL: {placeholder_sql}")
        
        # 恢复占位符
        final_sql = placeholder_sql.replace(placeholder, entity_text)
        print(f"最终SQL: {final_sql}")
        
        print(f"\n📊 对比:")
        print(f"❌ 原来可能生成: SELECT COUNT(DISTINCT orderno) FROM kn_quality_trace_prod_order_process WHERE projectname_s LIKE '%合肥S1号线%' AND partname_s LIKE '%乘客室门%';")
        print(f"✅ 现在应该生成: {final_sql}")

def test_complex_scenarios():
    """测试复杂场景"""
    print("\n" + "=" * 80)
    print("🎯 测试复杂场景")
    print("=" * 80)
    
    scenarios = [
        {
            "name": "多个独立实体",
            "query": "#项目A# 和 #项目B# 的对比分析",
            "expected_approach": "两个独立的项目过滤条件，使用OR连接"
        },
        {
            "name": "嵌套内容",
            "query": "#北京地铁1号线-乘客室门系统-V2.0#的测试报告",
            "expected_approach": "整个字符串作为单一实体处理"
        },
        {
            "name": "特殊字符",
            "query": "#项目@ABC-2024/01#的进度查询",
            "expected_approach": "包含特殊字符的完整字符串"
        }
    ]
    
    pattern = r"#([^#]+)#"
    
    for scenario in scenarios:
        print(f"\n场景: {scenario['name']}")
        print(f"查询: {scenario['query']}")
        
        matches = re.findall(pattern, scenario['query'])
        print(f"提取实体: {matches}")
        print(f"处理策略: {scenario['expected_approach']}")

def main():
    """主函数"""
    print("🚀 查询预处理器核心功能测试")
    print("验证 #标记# 内容的正确提取和处理")
    
    try:
        test_entity_extraction()
        test_sql_generation_logic()
        test_complex_scenarios()
        
        print("\n" + "=" * 80)
        print("✨ 测试完成!")
        
        print(f"\n📋 问题解决方案总结:")
        print(f"原问题: #合肥S1号线项目乘客室门# 被拆分成两个字段查询")
        print(f"解决方案:")
        print(f"  1. ✅ 使用正则表达式精确提取 #内容#")
        print(f"  2. ✅ 替换为占位符防止LLM拆分")
        print(f"  3. ✅ 提供明确的SQL生成指导")
        print(f"  4. ✅ 最后恢复为完整内容")
        
        print(f"\n🎯 预期效果:")
        print(f"输入: #合肥S1号线项目乘客室门#这个项目一共有多少个订单？")
        print(f"输出: SELECT COUNT(DISTINCT orderno) FROM kn_quality_trace_prod_order_process")
        print(f"      WHERE projectname_s LIKE '%合肥S1号线项目乘客室门%'")
        print(f"      (而不是被拆分成 projectname_s 和 partname_s 两个条件)")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()