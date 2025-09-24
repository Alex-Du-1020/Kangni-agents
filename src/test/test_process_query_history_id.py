"""
Test that process_query endpoint returns history_id
"""
import pytest
from fastapi.testclient import TestClient
from src.kangni_agents.main import app

client = TestClient(app)

def test_process_query_returns_history_id():
    """Test that process_query endpoint returns history_id in response"""
    response = client.post(
        "/qomo/v1/query",
        json={
            "question": "What is Python programming?",
            "user_email": "test@example.com"
        }
    )
    
    # The response should include history_id field
    assert response.status_code == 200
    data = response.json()
    
    # Check that history_id is present in the response
    assert "history_id" in data
    # history_id should be either an integer or None
    assert data["history_id"] is None or isinstance(data["history_id"], int)
    
    print(f"Response includes history_id: {data.get('history_id')}")

if __name__ == "__main__":
    test_process_query_returns_history_id()
    print("Test passed!")
