"""
查询预处理器
处理用户查询中的特殊标记和结构化数据
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ExtractedEntity:
    """提取的实体信息"""
    raw_text: str  # 原始文本（包含标记符）
    clean_text: str  # 清理后的文本（不含标记符）
    entity_type: str  # 实体类型
    start_pos: int  # 开始位置
    end_pos: int  # 结束位置

@dataclass
class PreprocessedQuery:
    """预处理后的查询"""
    original_query: str  # 原始查询
    processed_query: str  # 处理后的查询
    entities: List[ExtractedEntity]  # 提取的实体
    placeholders: Dict[str, str]  # 占位符映射
    sql_hints: Dict[str, str]  # SQL生成提示

class QueryPreprocessor:
    """查询预处理器"""
    
    def __init__(self):
        # 支持的标记模式
        self.patterns = {
            "hash_enclosed": r"#([^#]+)#",  # #内容# 形式
            "bracket_enclosed": r"\[([^\]]+)\]",  # [内容] 形式  
            "quote_enclosed": r'"([^"]+)"',  # "内容" 形式
            "parenthesis_enclosed": r"\(([^)]+)\)"  # (内容) 形式，但需要更谨慎处理
        }
        
        # 实体类型映射
        self.entity_type_mapping = {
            "hash_enclosed": "EXACT_MATCH",  # 精确匹配
            "bracket_enclosed": "STRUCTURED_DATA", 
            "quote_enclosed": "QUOTED_TEXT",
            "parenthesis_enclosed": "GROUPED_INFO"
        }
        
        # 常见的项目字段映射
        self.field_mappings = {
            "项目": ["projectname_s", "project_name", "projectcode_s"],
            "项目名称": ["projectname_s", "project_name"],
            "项目代码": ["projectcode_s", "project_code"],
            "部件": ["partname_s", "part_name", "component_name"],
            "部件名称": ["partname_s", "part_name"],
            "订单": ["orderno", "order_no", "order_number"],
            "订单号": ["orderno", "order_no"],
            "客户": ["customer_name", "client_name"],
            "供应商": ["supplier_name", "vendor_name"]
        }
        
        # 核心表结构映射
        self.table_mappings = {
            "订单": "kn_quality_trace_prod_order_process",
            "项目订单": "kn_quality_trace_prod_order_process",
            "生产订单": "kn_quality_trace_prod_order_process",
            "故障": "fault_info",
            "故障信息": "fault_info"
        }

    def extract_entities(self, query: str) -> List[ExtractedEntity]:
        """提取查询中的特殊实体"""
        entities = []
        
        for pattern_name, pattern in self.patterns.items():
            matches = re.finditer(pattern, query)
            for match in matches:
                entity = ExtractedEntity(
                    raw_text=match.group(0),
                    clean_text=match.group(1),
                    entity_type=self.entity_type_mapping.get(pattern_name, "UNKNOWN"),
                    start_pos=match.start(),
                    end_pos=match.end()
                )
                entities.append(entity)
                
        # 按位置排序
        entities.sort(key=lambda x: x.start_pos)
        logger.info(f"Extracted {len(entities)} entities from query")
        
        return entities

    def create_placeholders(self, entities: List[ExtractedEntity]) -> Dict[str, str]:
        """为实体创建占位符"""
        placeholders = {}
        
        for i, entity in enumerate(entities):
            placeholder = f"__ENTITY_{i}__"
            placeholders[placeholder] = entity.clean_text
            
        return placeholders

    def replace_with_placeholders(self, query: str, entities: List[ExtractedEntity]) -> str:
        """用占位符替换实体"""
        processed_query = query
        
        # 从后往前替换，避免位置偏移
        for i, entity in enumerate(reversed(entities)):
            placeholder = f"__ENTITY_{len(entities)-1-i}__"
            processed_query = (
                processed_query[:entity.start_pos] + 
                placeholder + 
                processed_query[entity.end_pos:]
            )
            
        return processed_query

    def generate_sql_hints(self, query: str, entities: List[ExtractedEntity], placeholders: Dict[str, str]) -> Dict[str, str]:
        """生成SQL构建提示"""
        hints = {}
        
        # 基本提示
        hints["base_instruction"] = "生成SQL时请注意以下要求："
        
        # 实体处理提示
        if entities:
            entity_instructions = []
            for i, entity in enumerate(entities):
                placeholder = f"__ENTITY_{i}__"
                if entity.entity_type == "EXACT_MATCH":
                    entity_instructions.append(
                        f"- {placeholder} 代表 '{entity.clean_text}'，这是一个完整的标识符，"
                        f"在SQL中必须作为单一完整值进行精确匹配，不能拆分"
                    )
                elif entity.entity_type == "STRUCTURED_DATA":
                    entity_instructions.append(
                        f"- {placeholder} 代表结构化数据 '{entity.clean_text}'，应该精确匹配"
                    )
            
            hints["entity_handling"] = "\n".join(entity_instructions)
        
        # 字段映射提示
        field_hints = []
        for entity in entities:
            for field_keyword, db_fields in self.field_mappings.items():
                if field_keyword in query:
                    field_hints.append(
                        f"- 涉及{field_keyword}时，优先使用字段：{', '.join(db_fields[:2])}"
                    )
        
        # 表映射提示
        table_hints = []
        for table_keyword, table_name in self.table_mappings.items():
            if table_keyword in query:
                table_hints.append(
                    f"- 涉及{table_keyword}查询时，使用表：{table_name}"
                )
        
        # 合并映射提示
        mapping_hints = field_hints + table_hints
        if mapping_hints:
            hints["field_and_table_mapping"] = "\n".join(mapping_hints)
            
        # 特殊处理提示
        special_hints = []
        
        # 检测组合查询
        if len(entities) > 1:
            special_hints.append(
                "- 这是一个包含多个实体的查询，每个实体都应该作为独立的过滤条件"
            )
            special_hints.append(
                "- 多个实体之间通常使用 AND 连接，除非语义上明确表示 OR 关系"
            )
        
        # 检测项目+部件组合 vs 单一项目实体
        project_entities = [entity for entity in entities if "#" in entity.raw_text]
        
        # 检查是否是单个完整项目名称（包含项目和部件的组合名称）
        if len(project_entities) == 1:
            entity = project_entities[0] 
            # 如果实体包含"项目"和其他部件相关词汇，但是作为一个完整名称
            if "项目" in entity.clean_text and any(keyword in entity.clean_text for keyword in ["门", "系统", "部件"]):
                special_hints.append(
                    f"- 检测到完整的项目标识符：'{entity.clean_text}'，这是一个完整的项目名称，应该使用 projectname_s 字段进行完整匹配"
                )
                special_hints.append(
                    f"- 不要拆分这个标识符，它应该作为一个完整的项目名称进行查询"
                )
                special_hints.append(
                    f"- 推荐SQL模式：WHERE projectname_s LIKE '%{entity.clean_text}%'"
                )
            elif "项目" in entity.clean_text:
                special_hints.append(
                    f"- 这是项目查询，使用 projectname_s 字段进行匹配"
                )
        
        # 检测多个独立的项目+部件实体组合
        elif len(project_entities) > 1:
            has_project_only = any("项目" in entity.clean_text and not any(keyword in entity.clean_text for keyword in ["门", "系统", "部件"]) for entity in project_entities)
            has_part_only = any(any(keyword in entity.clean_text for keyword in ["门", "系统", "部件"]) and "项目" not in entity.clean_text for entity in project_entities)
            
            if has_project_only and has_part_only:
                special_hints.append(
                    "- 这是项目+部件的独立组合查询，应该同时匹配项目名称和部件名称"
                )
                special_hints.append(
                    "- 项目信息使用 projectname_s 字段进行 LIKE 匹配"
                )
                special_hints.append(
                    "- 部件信息使用 partname_s 字段进行 LIKE 匹配"
                )
        
        if special_hints:
            hints["special_handling"] = "\n".join(special_hints)
            
        return hints

    def preprocess_query(self, query: str) -> PreprocessedQuery:
        """预处理查询"""
        logger.info(f"Preprocessing query: {query}")
        
        # 1. 提取实体
        entities = self.extract_entities(query)
        
        # 2. 创建占位符
        placeholders = self.create_placeholders(entities)
        
        # 3. 替换为占位符
        processed_query = self.replace_with_placeholders(query, entities)
        
        # 4. 生成SQL提示
        sql_hints = self.generate_sql_hints(query, entities, placeholders)
        
        result = PreprocessedQuery(
            original_query=query,
            processed_query=processed_query,
            entities=entities,
            placeholders=placeholders,
            sql_hints=sql_hints
        )
        
        logger.info(f"Preprocessing completed. Found {len(entities)} entities")
        for entity in entities:
            logger.info(f"Entity: {entity.raw_text} -> {entity.clean_text} ({entity.entity_type})")
            
        return result

    def restore_placeholders_in_sql(self, sql: str, placeholders: Dict[str, str]) -> str:
        """在生成的SQL中恢复占位符"""
        restored_sql = sql
        
        for placeholder, original_text in placeholders.items():
            # 替换占位符为原始文本
            restored_sql = restored_sql.replace(placeholder, original_text)
            
        return restored_sql

    def build_enhanced_prompt(self, base_prompt: str, preprocessed_query: PreprocessedQuery) -> str:
        """构建增强的提示词"""
        enhanced_sections = [base_prompt]
        
        # 添加预处理信息
        if preprocessed_query.entities:
            enhanced_sections.append("\n=== 查询预处理信息 ===")
            enhanced_sections.append(f"原始查询: {preprocessed_query.original_query}")
            enhanced_sections.append(f"处理后查询: {preprocessed_query.processed_query}")
            
            enhanced_sections.append("\n提取的实体:")
            for entity in preprocessed_query.entities:
                enhanced_sections.append(f"- {entity.raw_text} -> '{entity.clean_text}' (类型: {entity.entity_type})")
            
            # 添加占位符映射
            enhanced_sections.append("\n占位符映射:")
            for placeholder, text in preprocessed_query.placeholders.items():
                enhanced_sections.append(f"- {placeholder} = '{text}'")
        
        # 添加SQL生成提示
        if preprocessed_query.sql_hints:
            enhanced_sections.append("\n=== SQL生成特殊要求 ===")
            for hint_type, hint_content in preprocessed_query.sql_hints.items():
                enhanced_sections.append(hint_content)
        
        return "\n".join(enhanced_sections)

# 全局实例
query_preprocessor = QueryPreprocessor()