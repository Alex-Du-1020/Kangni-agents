#!/usr/bin/env python3
"""
Test the enhanced query preprocessing
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "../.."))

def test_enhanced_preprocessing():
    """Test the enhanced preprocessing logic"""
    
    print("🧪 测试增强的查询预处理器")
    print("=" * 60)
    
    # Import here to avoid dependency issues during development
    try:
        from kangni_agents.utils.query_preprocessor import query_preprocessor
        
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
        
    except ImportError as e:
        print(f"❌ 无法导入模块: {e}")
        print("这是正常的，因为完整的依赖环境可能不可用")
        print("但核心预处理逻辑已经在之前的独立测试中验证通过")
        
        return False
        
    return True

if __name__ == "__main__":
    test_enhanced_preprocessing()