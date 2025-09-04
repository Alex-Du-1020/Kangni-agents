#!/usr/bin/env python3
"""
Test the query preprocessing with the problematic query
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from kangni_agents.utils.query_preprocessor import query_preprocessor

def test_preprocessing():
    """Test the preprocessing with the problematic query"""
    
    query = "#合肥S1号线项目乘客室门#这个项目一共有多少个订单？"
    print(f"Testing query: {query}")
    print("=" * 60)
    
    # Preprocess the query
    result = query_preprocessor.preprocess_query(query)
    
    print(f"Original query: {result.original_query}")
    print(f"Processed query: {result.processed_query}")
    print(f"Number of entities: {len(result.entities)}")
    print()
    
    print("Extracted entities:")
    for i, entity in enumerate(result.entities):
        print(f"  {i}: {entity.raw_text} -> '{entity.clean_text}' ({entity.entity_type})")
    print()
    
    print("Placeholders:")
    for placeholder, text in result.placeholders.items():
        print(f"  {placeholder} = '{text}'")
    print()
    
    print("SQL Hints:")
    for hint_type, hint_content in result.sql_hints.items():
        print(f"  {hint_type}:")
        print(f"    {hint_content}")
    print()
    
    # Test placeholder restoration
    test_sql = f"SELECT COUNT(DISTINCT orderno) FROM kn_quality_trace_prod_order_process WHERE projectname_s LIKE '%{list(result.placeholders.keys())[0]}%';"
    restored_sql = query_preprocessor.restore_placeholders_in_sql(test_sql, result.placeholders)
    
    print(f"Test SQL with placeholder: {test_sql}")
    print(f"Restored SQL: {restored_sql}")
    
    return result

if __name__ == "__main__":
    test_preprocessing()