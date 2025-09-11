"""
Dictionary Display Utilities
Provides various methods to display dictionary objects in a readable format.
"""

import json
import pprint
from typing import Any, Dict, Union
from datetime import datetime


def display_dict(
    data: Dict[str, Any], 
    method: str = "json", 
    indent: int = 2, 
    width: int = 80,
    depth: int = None,
    ensure_ascii: bool = False
) -> None:
    """
    Display dictionary in various formats.
    
    Args:
        data: Dictionary to display
        method: Display method ("json", "pprint", "custom", "table")
        indent: Indentation level for JSON
        width: Width for pprint
        depth: Maximum depth for pprint
        ensure_ascii: Whether to ensure ASCII encoding for JSON
    """
    if method == "json":
        print(json.dumps(data, indent=indent, ensure_ascii=ensure_ascii, default=str))
    
    elif method == "pprint":
        pprint.pprint(data, width=width, depth=depth)
    
    elif method == "custom":
        _display_custom(data)
    
    elif method == "table":
        _display_table(data)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def _display_custom(data: Dict[str, Any], indent: int = 0) -> None:
    """Custom formatted display with indentation."""
    for key, value in data.items():
        if isinstance(value, dict):
            print('  ' * indent + f'{key}:')
            _display_custom(value, indent + 1)
        elif isinstance(value, list):
            print('  ' * indent + f'{key}: [')
            for item in value:
                if isinstance(item, dict):
                    _display_custom(item, indent + 1)
                else:
                    print('  ' * (indent + 1) + str(item))
            print('  ' * indent + ']')
        else:
            print('  ' * indent + f'{key}: {value}')


def _display_table(data: Dict[str, Any]) -> None:
    """Display dictionary as a table format."""
    print('=' * 60)
    for key, value in data.items():
        print(f'📋 {key.upper()}:')
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                print(f'   • {sub_key}: {sub_value}')
        else:
            print(f'   • {value}')
        print('-' * 40)


def display_query_result(result: Dict[str, Any]) -> None:
    """
    Display query result in a formatted way.
    
    Args:
        result: Query result dictionary
    """
    print("🔍 Query Result:")
    print("=" * 50)
    
    if result.get('success'):
        print("✅ Status: SUCCESS")
    else:
        print("❌ Status: FAILED")
    
    if result.get('sql_query'):
        print(f"📝 SQL Query: {result['sql_query']}")
    
    if result.get('results'):
        print(f"📊 Results: {len(result['results'])} rows")
        for i, row in enumerate(result['results'][:3]):  # Show first 3 rows
            print(f"   Row {i+1}: {row}")
        if len(result['results']) > 3:
            print(f"   ... and {len(result['results']) - 3} more rows")
    
    if result.get('vector_enhanced'):
        print("🎯 Vector Enhanced: Yes")
        if result.get('suggestions_used'):
            print("💡 Suggestions Used:")
            for field, suggestion in result['suggestions_used'].items():
                print(f"   • {field}: {suggestion.get('values', [])[:3]}")
    
    print("=" * 50)


def display_workflow_state(state: Dict[str, Any]) -> None:
    """
    Display workflow state in a formatted way.
    
    Args:
        state: Workflow state dictionary
    """
    print("🔄 Workflow State:")
    print("=" * 50)
    
    print(f"🎯 Intent: {state.get('intent', 'Unknown')}")
    print(f"🔧 Tool Used: {state.get('tool_used', 'None')}")
    print(f"✅ DB Results Valid: {state.get('db_results_valid', False)}")
    print(f"🔍 Vector Search Triggered: {state.get('needs_vector_search', False)}")
    print(f"🎯 Vector Enhanced: {state.get('vector_enhanced', False)}")
    
    if state.get('messages'):
        print(f"💬 Messages: {len(state['messages'])}")
    
    print("=" * 50)


# Example usage
if __name__ == "__main__":
    # Test data
    test_data = {
        'query_result': {
            'success': True,
            'sql_query': 'SELECT COUNT(*) FROM test_table',
            'results': [{'count': 42}],
            'vector_enhanced': True
        },
        'workflow_state': {
            'intent': 'QueryType.DATABASE',
            'tool_used': 'database_query_tool',
            'db_results_valid': True
        }
    }
    
    print("=== Dictionary Display Examples ===")
    print()
    
    print("1. JSON format:")
    display_dict(test_data, method="json")
    print()
    
    print("2. Custom format:")
    display_dict(test_data, method="custom")
    print()
    
    print("3. Table format:")
    display_dict(test_data, method="table")
    print()
    
    print("4. Query result display:")
    display_query_result(test_data['query_result'])
    print()
    
    print("5. Workflow state display:")
    display_workflow_state(test_data['workflow_state'])
