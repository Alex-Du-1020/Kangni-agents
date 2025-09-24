"""
Test the new LLM answer endpoints
"""
import pytest
import asyncio
from fastapi.testclient import TestClient
from src.kangni_agents.main import app

client = TestClient(app)

def test_llm_answer_with_question():
    """Test LLM answer endpoint with direct question"""
    response = client.post(
        "/qomo/v1/llm-answer",
        params={
            "question": "What is the capital of France?",
            "user_email": "test@example.com"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "answer" in data
    assert "question" in data
    assert "processing_time_ms" in data
    assert "query_type" in data
    assert data["query_type"] == "llm_only"
    assert data["question"] == "What is the capital of France?"
    assert data["user_email"] == "test@example.com"
    assert data["success"] is True
    
    # Check that we got a reasonable answer
    assert len(data["answer"]) > 0
    print(f"LLM Answer: {data['answer']}")

def test_llm_answer_missing_question():
    """Test LLM answer endpoint with missing question parameter"""
    response = client.post("/qomo/v1/llm-answer")
    
    assert response.status_code == 422  # Validation error for missing required parameter
    data = response.json()
    assert "question" in str(data["detail"])

def test_get_question_from_history_invalid_id():
    """Test get question from history endpoint with invalid history ID"""
    response = client.get(
        "/qomo/v1/question-from-history",
        params={
            "history_id": 99999  # Non-existent ID
        }
    )
    
    assert response.status_code == 404
    data = response.json()
    assert "History record with ID 99999 not found" in data["detail"]

def test_get_question_from_history_missing_id():
    """Test get question from history endpoint with missing history_id parameter"""
    response = client.get("/qomo/v1/question-from-history")
    
    assert response.status_code == 422  # Validation error for missing required parameter
    data = response.json()
    assert "history_id" in str(data["detail"])

if __name__ == "__main__":
    # Run the tests
    test_llm_answer_with_question()
    test_llm_answer_missing_question()
    test_get_question_from_history_invalid_id()
    test_get_question_from_history_missing_id()
    print("All tests passed!")
