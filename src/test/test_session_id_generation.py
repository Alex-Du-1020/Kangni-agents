#!/usr/bin/env python3
"""
Test session ID generation in query endpoint
"""
import asyncio
import sys
import os
import uuid
import pytest
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "../.."))

from kangni_agents.models import UserQuery
from kangni_agents.api.routes import process_query

@pytest.mark.asyncio
async def test_session_id_generation_no_session():
    """Test that session ID is generated when not provided"""
    query = UserQuery(
        question="What is the total number of users?",
        user_email="test@example.com"
        # session_id is None by default
    )
    
    response = await process_query(query)
    
    # Verify session_id is generated
    assert response.session_id is not None
    assert len(response.session_id) == 36  # UUID4 length
    assert response.session_id.count('-') == 4  # UUID4 format
    
    # Verify it's a valid UUID
    try:
        uuid.UUID(response.session_id)
    except ValueError:
        assert False, f"Generated session_id is not a valid UUID: {response.session_id}"

@pytest.mark.asyncio
async def test_session_id_generation_empty_session():
    """Test that session ID is generated when empty string provided"""
    query = UserQuery(
        question="How many orders are there?",
        user_email="test@example.com",
        session_id=""  # Empty string
    )
    
    response = await process_query(query)
    
    # Verify session_id is generated
    assert response.session_id is not None
    assert len(response.session_id) == 36  # UUID4 length
    assert response.session_id.count('-') == 4  # UUID4 format

@pytest.mark.asyncio
async def test_session_id_preserved_existing():
    """Test that existing session ID is preserved"""
    existing_session_id = "existing-session-123"
    query = UserQuery(
        question="What are the recent orders?",
        user_email="test@example.com",
        session_id=existing_session_id
    )
    
    response = await process_query(query)
    
    # Verify existing session_id is preserved
    assert response.session_id == existing_session_id

@pytest.mark.asyncio
async def test_session_id_whitespace_only():
    """Test that session ID is generated when only whitespace provided"""
    query = UserQuery(
        question="Show me the data",
        user_email="test@example.com",
        session_id="   "  # Only whitespace
    )
    
    response = await process_query(query)
    
    # Verify session_id is generated (whitespace should be treated as empty)
    assert response.session_id is not None
    assert len(response.session_id) == 36  # UUID4 length
    assert response.session_id.count('-') == 4  # UUID4 format

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
