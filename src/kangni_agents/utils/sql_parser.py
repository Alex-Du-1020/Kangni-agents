"""
SQL Parser Utility
Parses SQL queries to extract tables, fields, and conditions for vector search enhancement.
"""

import re
import sqlparse
from typing import Dict, List, Tuple, Optional, Set
import logging

logger = logging.getLogger(__name__)


class SQLParser:
    """Parse SQL queries to extract metadata for vector search enhancement"""
    
    def __init__(self):
        """Initialize the SQL parser"""
        self.table_patterns = [
            r'FROM\s+(\w+)',
            r'JOIN\s+(\w+)',
            r'INTO\s+(\w+)',
            r'UPDATE\s+(\w+)',
            r'TABLE\s+(\w+)'
        ]
        
        self.where_pattern = r'WHERE\s+(.*?)(?:GROUP|ORDER|LIMIT|;|$)'
        self.field_value_patterns = [
            r'(\w+)\s*=\s*[\'"]([^\'\"]+)[\'"]',  # field = 'value'
            r'(\w+)\s+LIKE\s+[\'"]([^\'\"]+)[\'"]',  # field LIKE 'value'
            r'(\w+)\s+IN\s*\(([^)]+)\)',  # field IN (values)
            r'(\w+)\s*=\s*(\d+)',  # field = number
        ]
    
    def parse_sql(self, sql: str) -> Dict[str, any]:
        """
        Parse SQL query to extract tables, fields, and conditions
        
        Args:
            sql: SQL query string
            
        Returns:
            Dictionary containing:
            - tables: List of table names
            - fields: List of field names
            - conditions: Dict of field -> value mappings
            - query_type: Type of SQL query (SELECT, INSERT, etc.)
        """
        try:
            # Normalize SQL
            sql = ' '.join(sql.split())
            sql_upper = sql.upper()
            
            # Determine query type
            query_type = self._get_query_type(sql_upper)
            
            # Extract tables
            tables = self._extract_tables(sql)
            
            # Extract fields and conditions
            fields, conditions = self._extract_fields_and_conditions(sql)
            
            # Extract fields from SELECT clause if applicable
            if query_type == 'SELECT':
                select_fields = self._extract_select_fields(sql)
                fields.update(select_fields)
            
            result = {
                'query_type': query_type,
                'tables': list(tables),
                'fields': list(fields),
                'conditions': conditions,
                'original_sql': sql
            }
            
            logger.debug(f"Parsed SQL: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error parsing SQL: {e}")
            return {
                'query_type': 'UNKNOWN',
                'tables': [],
                'fields': [],
                'conditions': {},
                'original_sql': sql,
                'error': str(e)
            }
    
    def _get_query_type(self, sql_upper: str) -> str:
        """Determine the type of SQL query"""
        for query_type in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP']:
            if sql_upper.strip().startswith(query_type):
                return query_type
        return 'UNKNOWN'
    
    def _extract_tables(self, sql: str) -> Set[str]:
        """Extract table names from SQL query"""
        tables = set()
        
        for pattern in self.table_patterns:
            matches = re.findall(pattern, sql, re.IGNORECASE)
            for match in matches:
                # Clean table name (remove schema prefix if exists)
                table_name = match.split('.')[-1].strip()
                if table_name and not table_name.upper() in ['SELECT', 'WHERE', 'AND', 'OR']:
                    tables.add(table_name.lower())
        
        return tables
    
    def _extract_fields_and_conditions(self, sql: str) -> Tuple[Set[str], Dict[str, List[str]]]:
        """Extract field names and their conditions from WHERE clause"""
        fields = set()
        conditions = {}
        
        # Find WHERE clause
        where_match = re.search(self.where_pattern, sql, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_clause = where_match.group(1)
            
            # Extract field-value pairs
            for pattern in self.field_value_patterns:
                matches = re.findall(pattern, where_clause, re.IGNORECASE)
                for match in matches:
                    if len(match) >= 2:
                        field = match[0].lower().strip()
                        value = match[1].strip().strip('\'"')
                        
                        fields.add(field)
                        
                        # Handle IN clause with multiple values
                        if ',' in value:
                            values = [v.strip().strip('\'"') for v in value.split(',')]
                            conditions.setdefault(field, []).extend(values)
                        else:
                            # Remove SQL wildcards for vector search
                            clean_value = value.replace('%', '').replace('_', '')
                            if clean_value:
                                conditions.setdefault(field, []).append(clean_value)
        
        return fields, conditions
    
    def _extract_select_fields(self, sql: str) -> Set[str]:
        """Extract field names from SELECT clause"""
        fields = set()
        
        # Pattern for SELECT clause
        select_pattern = r'SELECT\s+(.*?)\s+FROM'
        match = re.search(select_pattern, sql, re.IGNORECASE | re.DOTALL)
        
        if match:
            select_clause = match.group(1)
            
            # Handle SELECT *
            if select_clause.strip() == '*':
                return fields
            
            # Parse field list
            field_parts = select_clause.split(',')
            for part in field_parts:
                # Remove aliases, functions, etc.
                part = part.strip()
                
                # Skip aggregate functions
                if any(func in part.upper() for func in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']):
                    # Try to extract field from function
                    func_field_match = re.search(r'\(([^)]+)\)', part)
                    if func_field_match:
                        field_name = func_field_match.group(1).strip()
                        if field_name != '*':
                            fields.add(field_name.lower())
                else:
                    # Extract field name (before AS or space)
                    field_name = re.split(r'\s+AS\s+|\s+', part, flags=re.IGNORECASE)[0]
                    if field_name and field_name != '*':
                        # Remove table prefix if exists
                        field_name = field_name.split('.')[-1]
                        fields.add(field_name.lower())
        
        return fields
    
    def extract_search_terms(self, sql: str) -> List[str]:
        """
        Extract search terms from SQL that might benefit from vector search
        
        Args:
            sql: SQL query string
            
        Returns:
            List of search terms
        """
        search_terms = []
        parsed = self.parse_sql(sql)
        
        # Get all condition values
        for field, values in parsed.get('conditions', {}).items():
            search_terms.extend(values)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in search_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
        
        return unique_terms
    
    def should_use_vector_search(self, sql: str, result_count: int = 0) -> bool:
        """
        Determine if vector search should be used based on SQL and results
        
        Args:
            sql: SQL query string
            result_count: Number of results from the query
            
        Returns:
            True if vector search should be used
        """
        # Use vector search if query returned no results
        if result_count == 0:
            return True
        
        # Check if query has LIKE conditions (fuzzy search)
        if 'LIKE' in sql.upper():
            return True
        
        # Check if query has Chinese characters in conditions (might need fuzzy matching)
        parsed = self.parse_sql(sql)
        for values in parsed.get('conditions', {}).values():
            for value in values:
                if any('\u4e00' <= char <= '\u9fff' for char in value):
                    return True
        
        return False