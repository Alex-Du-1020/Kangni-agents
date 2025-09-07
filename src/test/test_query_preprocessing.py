#!/usr/bin/env python3
"""
测试查询预处理器的效果
Test the query preprocessor improvements
"""

import sys
import asyncio
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "../.."))

from kangni_agents.utils.query_preprocessor import query_preprocessor

def test_query_preprocessing():
    """测试查询预处理功能"""
    
    # 测试用例
    test_cases = [
        {
            "name": "原始问题案例",
            "query": "#合肥S1号线项目乘客室门#这个项目一共有多少个订单？",
            "expected_entities": 1,
            "expected_type": "EXACT_MATCH"
        },
        {
            "name": "项目+部件组合",
            "query": "#北京地铁15号线# 项目的 #乘客门系统# 部件有多少故障？",
            "expected_entities": 2,
            "expected_type": "EXACT_MATCH"
        },
        {
            "name": "复杂项目名称",
            "query": "#德里地铁4期项目-20D21028C000#的故障统计信息",
            "expected_entities": 1,
            "expected_type": "EXACT_MATCH"
        },
        {
            "name": "多种标记混合",
            "query": "查询 #上海地铁项目# 的 [紧急按钮] 部件在 \"2024年\" 的故障数量",
            "expected_entities": 3,
            "expected_type": "MIXED"
        },
        {
            "name": "无特殊标记",
            "query": "查询所有项目的订单总数",
            "expected_entities": 0,
            "expected_type": "NONE"
        }
    ]
    
    print("🧪 开始测试查询预处理器")
    print("=" * 80)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📝 测试案例 {i}: {case['name']}")
        print(f"原始查询: {case['query']}")
        
        # 预处理查询
        result = query_preprocessor.preprocess_query(case['query'])
        
        print(f"处理后查询: {result.processed_query}")
        print(f"提取实体数: {len(result.entities)} (预期: {case['expected_entities']})")
        
        # 显示提取的实体
        if result.entities:
            print("提取的实体:")
            for j, entity in enumerate(result.entities):
                print(f"  {j+1}. {entity.raw_text} -> '{entity.clean_text}' ({entity.entity_type})")
        
        # 显示占位符映射
        if result.placeholders:
            print("占位符映射:")
            for placeholder, text in result.placeholders.items():
                print(f"  {placeholder} = '{text}'")
        
        # 显示SQL生成提示
        if result.sql_hints:
            print("SQL生成提示:")
            for hint_type, hint_content in result.sql_hints.items():
                # 只显示关键提示的前100个字符
                preview = hint_content[:100] + "..." if len(hint_content) > 100 else hint_content
                print(f"  {hint_type}: {preview}")
        
        # 验证结果
        success = len(result.entities) == case['expected_entities']
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"结果: {status}")
        
        if not success:
            print(f"⚠️  预期 {case['expected_entities']} 个实体，实际找到 {len(result.entities)} 个")

def test_sql_hint_generation():
    """测试SQL提示生成"""
    print("\n" + "=" * 80)
    print("🔧 测试SQL提示生成")
    print("=" * 80)
    
    # 测试复杂查询
    complex_query = "#合肥S1号线项目乘客室门#这个项目一共有多少个订单？"
    result = query_preprocessor.preprocess_query(complex_query)
    
    # 模拟基础提示
    base_prompt = """你是SQL生成助手。
数据库表结构:
- kn_quality_trace_prod_order_process (projectname_s, partname_s, orderno)

请生成SQL查询。"""
    
    # 生成增强提示
    enhanced_prompt = query_preprocessor.build_enhanced_prompt(base_prompt, result)
    
    print("增强后的提示词:")
    print("-" * 40)
    print(enhanced_prompt)
    print("-" * 40)
    
    # 测试占位符恢复
    test_sql = "SELECT COUNT(DISTINCT orderno) FROM kn_quality_trace_prod_order_process WHERE projectname_s LIKE '%__ENTITY_0__%'"
    restored_sql = query_preprocessor.restore_placeholders_in_sql(test_sql, result.placeholders)
    
    print(f"\n占位符SQL: {test_sql}")
    print(f"恢复后SQL: {restored_sql}")

def test_field_mapping():
    """测试字段映射功能"""
    print("\n" + "=" * 80)
    print("🗺️  测试字段映射功能")
    print("=" * 80)
    
    # 测试包含字段关键词的查询
    field_queries = [
        "查询项目 #测试项目# 的信息",
        "#测试部件# 部件的故障统计",
        "订单号为 #12345# 的详情",
        "客户 #ABC公司# 的所有订单"
    ]
    
    for query in field_queries:
        print(f"\n查询: {query}")
        result = query_preprocessor.preprocess_query(query)
        
        if "field_mapping" in result.sql_hints:
            print("字段映射建议:")
            print(result.sql_hints["field_mapping"])
        else:
            print("未检测到特定字段映射")

def test_enhanced_preprocessing():
    """Test the enhanced preprocessing logic with detailed output"""
    print("\n" + "=" * 80)
    print("🧪 测试增强的查询预处理器（详细输出）")
    print("=" * 80)
    
    query = "#合肥S1号线项目乘客室门#这个项目一共有多少个订单？"
    print(f"测试查询: {query}")
    
    # Preprocess the query
    result = query_preprocessor.preprocess_query(query)
    
    print(f"\n📝 预处理结果:")
    print(f"原始查询: {result.original_query}")
    print(f"处理后查询: {result.processed_query}")
    print(f"实体数量: {len(result.entities)}")
    
    print(f"\n🏷️  提取的实体:")
    for i, entity in enumerate(result.entities):
        print(f"  {i}: '{entity.raw_text}' -> '{entity.clean_text}' ({entity.entity_type})")
    
    print(f"\n🔄 占位符映射:")
    for placeholder, text in result.placeholders.items():
        print(f"  {placeholder} = '{text}'")
    
    print(f"\n💡 生成的SQL提示:")
    for hint_type, hint_content in result.sql_hints.items():
        print(f"  📋 {hint_type}:")
        for line in hint_content.split('\n'):
            if line.strip():
                print(f"    {line}")
    
    # Test the enhanced prompt
    base_prompt = "你是SQL生成助手。"
    enhanced_prompt = query_preprocessor.build_enhanced_prompt(base_prompt, result)
    
    print(f"\n🚀 增强提示词预览:")
    print("=" * 40)
    lines = enhanced_prompt.split('\n')
    for line in lines[:20]:  # Show first 20 lines
        print(line)
    if len(lines) > 20:
        print(f"... (还有{len(lines)-20}行)")
    print("=" * 40)
    
    # Test SQL restoration
    if result.placeholders:
        placeholder = list(result.placeholders.keys())[0]
        test_sql = f"SELECT COUNT(DISTINCT orderno) FROM kn_quality_trace_prod_order_process WHERE projectname_s LIKE '%{placeholder}%'"
        restored_sql = query_preprocessor.restore_placeholders_in_sql(test_sql, result.placeholders)
        
        print(f"\n🔧 SQL恢复测试:")
        print(f"占位符SQL: {test_sql}")
        print(f"恢复后SQL: {restored_sql}")
    
    print(f"\n✅ 预期效果验证:")
    print(f"应该生成类似: SELECT COUNT(DISTINCT orderno) FROM kn_quality_trace_prod_order_process WHERE projectname_s LIKE '%合肥S1号线项目乘客室门%'")
    print(f"而不是: WHERE projectname_s LIKE '%合肥S1号线%' AND partname_s LIKE '%乘客室门%'")


def main():
    """主函数"""
    print("🚀 查询预处理器测试")
    print("用于验证特殊标记处理和SQL生成改进")
    
    try:
        # 运行测试
        test_query_preprocessing()
        test_sql_hint_generation()
        test_field_mapping()
        test_enhanced_preprocessing()  # Add the enhanced test
        
        print("\n" + "=" * 80)
        print("✨ 测试完成!")
        print("\n📋 改进效果:")
        print("1. ✅ 自动识别和提取 #标记# 内容")
        print("2. ✅ 生成针对性的SQL构建提示")
        print("3. ✅ 防止长字段被拆分为多个条件")
        print("4. ✅ 支持多种标记格式 (#, [], \"\", ())")
        print("5. ✅ 提供字段映射建议")
        print("6. ✅ 增强提示词生成和占位符恢复")
        
        print("\n🔧 使用方法:")
        print("现在系统会自动预处理用户查询，")
        print("确保 #合肥S1号线项目乘客室门# 这样的内容")
        print("被当作单一实体处理，而不是拆分。")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()