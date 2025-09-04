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
        
        # 常见的项目字段映射 - 基于description.txt完整数据库结构
        self.field_mappings = {
            # 项目相关字段
            "项目": ["projectname_s", "project_name", "projectcode_s", "project_code"],
            "项目名称": ["projectname_s", "project_name"],
            "项目代码": ["projectcode_s", "project_code", "project_id"],
            "项目编号": ["project_code", "projectcode_s", "project_id"],
            "项目年份": ["project_year", "project_project_year"],
            
            # 订单相关字段
            "订单": ["workordernumber_s", "orderno", "order_no", "order_number"],
            "订单号": ["workordernumber_s", "orderno", "order_no"],
            "工单": ["workordernumber_s", "orderno_s"],
            "工单号": ["workordernumber_s", "orderno_s"],
            "订单状态": ["orderstate_s", "order_state"],
            "订单数量": ["quantity_s", "quantity"],
            
            # 物料部件相关字段
            "物料": ["partname_s", "part_name", "material_name", "partno_s"],
            "物料名称": ["partname_s", "part_name", "material_name"],
            "物料编号": ["partno_s", "part_number", "material_number", "material_code"],
            "物料描述": ["partname_s", "material_description"],
            "部件": ["partname_s", "part_name", "component_name"],
            "部件名称": ["partname_s", "part_name"],
            "零件": ["partname_s", "part_name"],
            "零件名称": ["partname_s", "part_name"],
            "图号": ["figureno_s", "figure_type", "zzthzq", "drawing_no"],
            
            # 设备相关字段
            "设备": ["device_name", "mainrescode_s", "equname_s", "equip_name"],
            "设备名称": ["device_name", "mainrescode_s", "equname_s"],
            "设备编号": ["device_id", "deviceid_s", "equip_name"],
            "设备ID": ["device_id", "deviceid_s"],
            
            # 人员相关字段
            "操作人员": ["creator_s", "username", "userno_name"],
            "操作人工号": ["userno_s", "userno"],
            "检验人员": ["inspector", "checkusername", "creator_s"],
            "质检人员": ["username", "checkusername"],
            "申请人": ["sqr", "submit_user_name", "create_name"],
            "创建人": ["creator", "create_by", "create_name"],
            "负责人": ["duty_user_name", "task_user_name"],
            
            # 时间相关字段
            "时间": ["etl_time", "create_date", "create_time"],
            "创建时间": ["create_date", "create_time", "etl_time"],
            "开始时间": ["planstarttime_t", "actualstarttime_t", "starttime", "playtime_t"],
            "结束时间": ["planendtime_t", "actualendtime_t", "endtime", "endtime_t"],
            "异常时间": ["exception_time", "happen_date"],
            "故障时间": ["happen_date", "occur_time"],
            "检验时间": ["checktime", "inspection_start_date", "inspection_end_date"],
            
            # 客户供应商字段
            "客户": ["customer_name", "client_name"],
            "供应商": ["supplier_name", "sup_company_name", "vendor_name"],
            "供应商编码": ["supplier_code", "sup_company_sap_code"],
            
            # 工艺工序字段
            "工序": ["routeoperaname_s", "operationname_s"],
            "工序名称": ["routeoperaname_s", "operationname_s"],
            "工序编号": ["routeoperanumber_s", "processnumber_s"],
            "工艺路线": ["routeno_s", "route_code"],
            "工位": ["wcname_s", "locationnumber_s"],
            "工位名称": ["wcname_s"],
            "工位编号": ["locationnumber_s"],
            
            # 故障异常字段
            "故障": ["defect_name", "fault_name", "description"],
            "故障模式": ["defect_name", "fault_name"],
            "故障原因": ["reason", "cause_analysis"],
            "异常": ["reason", "exception_time", "description"],
            "异常原因": ["reason", "cause_analysis"],
            "不合格": ["bad_describe", "problem_describe"],
            "缺陷": ["defect_name", "fault_name"],
            "缺陷分类": ["defect_category_name", "fault_type_name"],
            
            # 检验质量字段
            "检验": ["testproname", "testsortname_s", "testdescription_s"],
            "检验批次": ["checkbatch", "inspection_lot"],
            "检验结果": ["testsortupcheck_s", "result", "conclusion"],
            "质量": ["quality", "quantity"],
            "数量": ["quantity_s", "quality", "bad_num"],
            
            # 其他业务字段
            "批次": ["batch", "checkbatch"],
            "序列号": ["serialname_s", "door", "vehicle_door_organization_number"],
            "车辆": ["vehicle_code", "vehicle_no", "car_no"],
            "车门": ["door_no", "vehicle_door_id", "door_number"],
            "平台": ["plat_name", "label"],
            "工厂": ["factorynumber_s", "factory", "plant_code"],
            "状态": ["status", "orderstate_s", "checkstate"],
            "备注": ["remarks", "remark", "comment1_s"]
        }
        
        # 核心表结构映射 - 基于description.txt完整数据库结构
        self.table_mappings = {
            # 设备异常相关表
            "设备异常": "kn_quality_equipment_exception_data",
            "设备故障": "kn_quality_equipment_exception_data",
            "异常设备": "kn_quality_equipment_exception_data",
            
            # 生产订单相关表
            "生产订单": "kn_quality_trace_prod_order",
            "订单": "kn_quality_trace_prod_order",
            "项目订单": "kn_quality_trace_prod_order",
            "工单": "kn_quality_trace_prod_order",
            
            # 生产订单过程记录表
            "订单过程": "kn_quality_trace_prod_order_process",
            "生产过程": "kn_quality_trace_prod_order_process",
            "质量检验": "kn_quality_trace_prod_order_process",
            
            # 生产订单工序检验记录表
            "工序检验": "kn_quality_trace_prod_order_route",
            "工序记录": "kn_quality_trace_prod_order_route",
            
            # 自互检和完工检表
            "自检": "kn_quality_trace_prod_order_self",
            "互检": "kn_quality_trace_prod_order_self",
            "完工检": "kn_quality_trace_prod_order_self",
            
            # 用户表
            "操作人员": "kn_quality_trace_prod_order_user",
            "用户": "kn_quality_trace_prod_order_user",
            "人员": "kn_quality_trace_prod_order_user",
            
            # 变更流程表
            "过程变更": "kn_quality_process_change_flow",
            "变更流程": "kn_quality_process_change_flow",
            "变更申请": "kn_quality_process_change_flow",
            
            # 供方变更审批表
            "供应商变更": "kn_quality_supplier_change_approval_form",
            "供方变更": "kn_quality_supplier_change_approval_form",
            "变更审批": "kn_quality_supplier_change_approval_form",
            
            # BOM相关表
            "BOM": "kn_quality_trace_bom_data",
            "物料清单": "kn_quality_trace_bom_data",
            "BOM故障": "kn_quality_trace_bom_data_fault",
            "供方检验报告": "kn_quality_trace_bom_data_supplier",
            
            # 故障相关表
            "故障装车": "kn_quality_trace_fault_part_load",
            "装车信息": "kn_quality_trace_fault_part_load",
            "纠正预防": "kn_quality_trace_history_fault_corrective",
            "首件鉴定": "kn_quality_trace_history_fault_first",
            "故障信息": "kn_quality_trace_history_fault_info",
            "故障记录": "kn_quality_trace_history_fault_info",
            "故障": "kn_quality_trace_history_fault_info",
            "普查整改": "kn_quality_trace_history_fault_inspection",
            "车辆公里数": "kn_quality_trace_history_fault_mileage_time",
            "运行公里数": "kn_quality_trace_history_fault_mileage_time",
            "NCR": "kn_quality_trace_history_fault_ncr",
            "不合格报告": "kn_quality_trace_history_fault_ncr",
            "试验异常": "kn_quality_trace_history_fault_test",
            "试制异常": "kn_quality_trace_history_fault_test",
            
            # 平台项目表
            "平台管理": "kn_quality_trace_history_plat_project",
            "项目管理": "kn_quality_trace_history_plat_project",
            "平台": "kn_quality_trace_history_plat_project",
            
            # 项目交付表
            "项目交付": "kn_quality_trace_project_delivery",
            "交付时间": "kn_quality_trace_project_delivery",
            
            # 字典表
            "字典": "kn_quality_trace_sys_dict_pub",
            "字典数据": "kn_quality_trace_sys_dict_pub"
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