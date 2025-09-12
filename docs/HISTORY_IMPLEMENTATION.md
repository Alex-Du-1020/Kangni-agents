# Query History Implementation

This document describes the history functionality implementation for the Kangni Agents system.

## Overview

The history functionality allows the system to:
1. Save all queries and responses to a database
2. Track SQL queries and RAG sources used
3. Allow users to provide feedback (like/dislike)
4. Allow users to add comments
5. Query history by user email or session
6. Search through historical queries

## Database Schema

### Tables

1. **query_history**
   - Stores question, answer, SQL query, sources
   - Tracks success/failure and error messages
   - Records processing time and LLM provider details
   - Indexed on user_email, session_id, and created_at

2. **user_feedback**
   - Stores like/dislike feedback for queries
   - Links to query_history via foreign key
   - One feedback per user per query

3. **user_comments**
   - Stores user comments on queries
   - Links to query_history via foreign key
   - Multiple comments allowed per query

## API Endpoints

### History Endpoints (`/qomo/v1/history`)

- `GET /user/{user_email}` - Get query history for a user
- `GET /session/{session_id}` - Get query history for a session
- `GET /search?q={term}` - Search through history
- `GET /recent?hours={n}` - Get recent queries
- `POST /feedback` - Add or update feedback
- `POST /comment` - Add a comment
- `GET /feedback/stats/{query_id}` - Get feedback statistics

### Query Endpoint Enhancement

The main `/qomo/v1/query` endpoint now automatically:
- Saves successful queries with answers
- Records failed queries with error messages
- Tracks processing time
- Stores SQL queries and RAG sources

## Database Configuration

### SQLite (Default for Testing)
```python
# Automatically uses SQLite at ./history.db
# No configuration needed
```

## Migration Management

Using Alembic for database migrations:

```bash
# Create a new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback one version
alembic downgrade -1
```

## Usage Examples

### Python API

```python
from kangni_agents.services.history_service import history_service

# Save query history
history = await history_service.save_query_history(
    session_id="session-123",
    user_email="user@example.com",
    question="What is the total count?",
    answer="The total count is 42",
    sql_query="SELECT COUNT(*) FROM table",
    sources=[{"content": "doc1", "score": 0.9}],
    query_type="database",
    success=True,
    processing_time_ms=150
)

# Get user history
history_items = await history_service.get_user_history(
    user_email="user@example.com",
    limit=50
)

# Add feedback
feedback = await history_service.add_feedback(
    query_id=1,
    user_email="user@example.com",
    feedback_type="like"
)

# Search history
results = await history_service.search_history(
    search_term="total count",
    user_email="user@example.com"
)
```

### REST API

```bash
# Get user history
curl "http://localhost:8000/qomo/v1/history/user/test@example.com"

# Add feedback
curl -X POST "http://localhost:8000/qomo/v1/history/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "query_id": 1,
    "user_email": "test@example.com",
    "feedback_type": "like"
  }'

# Search history
curl "http://localhost:8000/qomo/v1/history/search?q=users&user_email=test@example.com"

# Get recent queries
curl "http://localhost:8000/qomo/v1/history/recent?hours=24"
```

## Testing

Run the test suite:
```bash
python src/tests/test_history.py
```

This will test:
- Saving successful and failed queries
- Retrieving user and session history
- Adding feedback and comments
- Searching history
- Getting recent queries
- Feedback statistics

## Installation Requirements

Add to your `pyproject.toml` or `requirements.txt`:
```
sqlalchemy>=2.0.0
alembic>=1.13.0
sqlalchemy-utils>=0.41.0  # Optional, for database creation
```

## Features

- ✅ Automatic history saving on every query
- ✅ SQL query and RAG source tracking
- ✅ User feedback (like/dislike)
- ✅ User comments
- ✅ Search functionality
- ✅ Session-based history
- ✅ Processing time tracking
- ✅ LLM provider tracking
- ✅ Error tracking for failed queries
- ✅ MySQL/PostgreSQL support for production
- ✅ Alembic migrations

## Notes

1. **Database Choice**: SQLite is used by default for testing. For production, configure a proper database URL.

2. **Session Management**: The history service properly manages SQLAlchemy sessions to avoid detached instance errors.

3. **Async Support**: All operations are async-compatible for use with FastAPI.

4. **Error Handling**: History saving failures don't break the main query flow - errors are logged but the query continues.

5. **Performance**: Indexes are created on frequently queried columns (user_email, session_id, created_at).