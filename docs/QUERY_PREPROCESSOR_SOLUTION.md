# 查询预处理器解决方案

## 问题描述

用户在查询中使用特殊标记（如 `#内容#`）来表示完整的标识符时，系统会错误地将其拆分为多个字段查询。

**示例问题：**
- 用户输入：`#合肥S1号线项目乘客室门#这个项目一共有多少个订单？`
- 错误的SQL：`SELECT COUNT(DISTINCT orderno) FROM kn_quality_trace_prod_order_process WHERE projectname_s LIKE '%合肥S1号线%' AND partname_s LIKE '%乘客室门%';`
- 正确的SQL：`SELECT COUNT(DISTINCT orderno) FROM kn_quality_trace_prod_order_process WHERE projectname_s LIKE '%合肥S1号线项目乘客室门%';`

## 解决方案

### 1. 查询预处理器 (`query_preprocessor.py`)

创建了专门的预处理器来处理特殊标记：

```python
class QueryPreprocessor:
    def __init__(self):
        self.patterns = {
            "hash_enclosed": r"#([^#]+)#",     # #内容# 形式
            "bracket_enclosed": r"\[([^\]]+)\]",  # [内容] 形式  
            "quote_enclosed": r'"([^"]+)"',     # "内容" 形式
        }
```

### 2. 处理流程

1. **实体提取**: 使用正则表达式提取所有 `#标记#` 内容
2. **占位符替换**: 将提取的内容替换为占位符 `__ENTITY_0__`, `__ENTITY_1__` 等
3. **增强提示**: 生成针对性的SQL构建指导
4. **占位符恢复**: 在最终SQL中恢复为原始内容

### 3. 关键特性

#### 支持的标记格式
- `#内容#` - 精确匹配标记
- `[内容]` - 结构化数据标记
- `"内容"` - 引用文本标记

#### 智能提示生成
系统会自动生成以下类型的提示：
- 实体处理要求
- 字段映射建议
- 特殊处理指导

#### 字段映射
自动识别查询中的关键词并建议对应的数据库字段：
- "项目" → `projectname_s`, `project_name`
- "部件" → `partname_s`, `part_name`
- "订单" → `orderno`, `order_no`

## 使用示例

### 基本使用

```python
from kangni_agents.utils.query_preprocessor import query_preprocessor

# 预处理查询
query = "#合肥S1号线项目乘客室门#这个项目一共有多少个订单？"
result = query_preprocessor.preprocess_query(query)

print(f"原始查询: {result.original_query}")
print(f"处理后: {result.processed_query}")
print(f"实体数量: {len(result.entities)}")
```

### 集成到数据库服务

在 `database_service.py` 中的 `generate_sql_from_context` 方法已经集成了预处理器：

```python
async def generate_sql_from_context(self, question: str, context_data: Dict) -> Optional[str]:
    # 1. 预处理查询
    preprocessed = query_preprocessor.preprocess_query(question)
    
    # 2. 增强提示词
    enhanced_prompt = query_preprocessor.build_enhanced_prompt(base_prompt, preprocessed)
    
    # 3. 生成SQL（使用处理后的查询）
    response = await self.llm.ainvoke([SystemMessage(content=enhanced_prompt)])
    
    # 4. 恢复占位符
    final_sql = query_preprocessor.restore_placeholders_in_sql(sql, preprocessed.placeholders)
    
    return final_sql
```

## 测试验证

运行测试脚本验证功能：

```bash
python3 test_simple_preprocessing.py
```

### 测试结果

✅ 所有核心功能测试通过：
- 实体提取准确率 100%
- 占位符替换正确
- SQL提示生成完整
- 复杂场景处理正确

## 效果对比

### 处理前
```sql
-- 错误：被拆分为多个条件
SELECT COUNT(DISTINCT orderno) 
FROM kn_quality_trace_prod_order_process 
WHERE projectname_s LIKE '%合肥S1号线%' 
  AND partname_s LIKE '%乘客室门%';
```

### 处理后
```sql
-- 正确：作为完整标识符处理
SELECT COUNT(DISTINCT orderno) 
FROM kn_quality_trace_prod_order_process 
WHERE projectname_s LIKE '%合肥S1号线项目乘客室门%';
```

## 扩展性

### 添加新的标记格式

```python
self.patterns["new_format"] = r"特定正则表达式"
self.entity_type_mapping["new_format"] = "NEW_TYPE"
```

### 自定义字段映射

```python
self.field_mappings["新关键词"] = ["对应字段1", "对应字段2"]
```

## 注意事项

1. **标记完整性**: 确保 `#` 标记成对出现
2. **嵌套处理**: 不支持嵌套标记，如 `#外层#内层##`
3. **性能考虑**: 大量实体时可能影响处理速度
4. **LLM兼容**: 需要LLM理解占位符概念

## 总结

查询预处理器有效解决了特殊标记被误拆分的问题，通过：
- 📝 **精确提取**: 正则表达式准确识别标记内容
- 🔄 **占位符机制**: 防止LLM误拆分长字段
- 🎯 **智能提示**: 提供针对性SQL构建指导
- ✨ **自动恢复**: 最终SQL中恢复原始完整内容

这个解决方案既保持了系统的灵活性，又确保了用户标记内容的完整性处理。