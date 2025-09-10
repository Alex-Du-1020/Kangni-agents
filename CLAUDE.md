# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CRITICAL RULES

### Virtual Environment Management
**ALWAYS activate the virtual environment before running any Python or pip commands:**
```bash
source .venv/bin/activate
```

### Package Management
**When installing or updating packages:**
1. Always activate the virtual environment first
2. Update pyproject.toml when adding new dependencies
3. Use editable installation for development: `pip install -e .`

Example workflow for adding a new package:
```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Install the package
pip install package-name

# 3. Update pyproject.toml dependencies list
# Edit pyproject.toml to add the new package

# 4. Reinstall with updated dependencies
pip install -e .
```

## Common Development Commands

### Setup and Installation
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (editable mode)
pip install --upgrade pip
pip install -e .
```

### Database Setup and Migration
```bash
# Set database type (sqlite for testing, postgresql for production)
export DB_TYPE=sqlite  # or postgresql

# For PostgreSQL, set connection variables
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=postgres
export POSTGRES_DATABASE=kangni_ai_chatbot

# Initialize Alembic (already done)
alembic init alembic

# Create a new migration after model changes
alembic revision --autogenerate -m "Description of changes"

# Apply migrations to database
alembic upgrade head

# Downgrade to previous version
alembic downgrade -1

# View migration history
alembic history
```

### Running the Application
```bash
# Development mode with hot reload
./dev_server.sh

# Production mode
./prod_server.sh
# or
source .venv/bin/activate && python -m src.kangni_agents.main

# Docker mode
docker build -t kangni-agents .
docker run -p 8000:8000 kangni-agents
```

### Testing

**IMPORTANT: All test files MUST be placed in the `src/test/` directory, NOT in the root folder.**
- Test file naming convention: `test_*.py` or `*_test.py`
- Keep tests organized in `src/test/` for better project structure
- Do not create test files in the root directory

```bash
# ALWAYS activate virtual environment first
source .venv/bin/activate

# Test history endpoints with mocked data
python src/test/test_history.py

# Quick test - verify basic functionality
python run_tests.py quick

# Comprehensive test suite
python src/test/test_comprehensive.py

# Run specific test modules
python src/test/test_llm_connection.py
python src/test/test_query_preprocessing.py
python src/test/test_rag.py
```

### Linting and Code Quality
```bash
# ALWAYS activate virtual environment first
source .venv/bin/activate

# Install dev dependencies first
pip install -e ".[dev]"

# Run linters (if configured)
black src/
flake8 src/
mypy src/
```

## Architecture Overview

### Core Components

**FastAPI Application (`src/kangni_agents/main.py`)**
- Entry point with fail-fast startup checks for all services
- Lifespan management for proper service initialization/cleanup
- CORS middleware and structured logging configuration

**LangGraph Agent System (`src/kangni_agents/agents/react_agent.py`)**
- ReAct (Reasoning + Acting) pattern implementation using StateGraph
- Manages workflow: intent classification → tool selection → execution → answer generation
- Supports fallback strategies when primary tools fail
- Key nodes: classify_intent, reason_and_decide, execute_tool, generate_answer

**Service Layer**
- **RAG Service** (`services/rag_service.py`): Integrates with RAGFlow MCP server for document retrieval
- **Database Service** (`services/database_service.py`): SQL generation and execution with query preprocessing
- **LLM Service** (`models/llm_implementations.py`): Multi-provider LLM support (DeepSeek default, OpenAI, Alibaba)
- **History Service** (`services/history_service.py`): Tracks query history, user feedback, and comments

**Intent Classification (`utils/intent_classifier.py`)**
- Determines query type: RAG search, database query, or mixed mode
- Uses keyword-based heuristics for routing decisions

**Query Preprocessing (`utils/query_preprocessor.py`)**
- Handles Chinese-to-English field mapping for database queries
- Manages table/field name resolution and query standardization

### LLM Provider System

The system supports multiple LLM providers through a flexible architecture:
- **Default**: DeepSeek (configured via `DEEPSEEK_API_KEY` and `LLM_BASE_URL`)
- **Fallback**: OpenAI and Alibaba Qwen models
- Provider selection is automatic based on available API keys
- Each service (database, RAG, agent) can use different providers

### History Tracking System

**Features**
- Automatic query history saving with questions, answers, SQL, and sources
- User email-based history retrieval
- Like/dislike feedback system
- Comment functionality for queries
- Session-based history tracking

**Database Models** (`models/history.py`)
- `QueryHistory`: Main history table with query details
- `UserFeedback`: Tracks likes/dislikes
- `UserComment`: Stores user comments on queries

**History API Endpoints**
- `GET /qomo/v1/history/user/{email}` - Get user's query history
- `GET /qomo/v1/history/session/{session_id}` - Get session history
- `GET /qomo/v1/history/search` - Search history by keyword
- `GET /qomo/v1/history/recent` - Get recent queries
- `POST /qomo/v1/history/feedback` - Add like/dislike
- `POST /qomo/v1/history/comment` - Add comment
- `GET /qomo/v1/history/feedback/stats/{query_id}` - Get feedback statistics

### Environment Configuration

Required `.env` file variables:
- `LLM_API_KEY` and `DEEPSEEK_API_KEY` for DeepSeek
- `MYSQL_*` variables for database connection
- `RAGFLOW_MCP_SERVER_URL` for RAG service
- Dataset IDs for different RAG collections
- `DB_TYPE` - Database type (sqlite or postgresql)
- `POSTGRES_*` variables for PostgreSQL connection

### API Endpoints

**Core Endpoints**
- `POST /qomo/v1/query` - Main query endpoint (requires user_email)
- `GET /qomo/v1/health` - Health check
- `GET /docs` - Swagger UI documentation
- `GET /qomo/v1/config` - Configuration info

**History Endpoints**
- `GET /qomo/v1/history/user/{email}` - User query history
- `GET /qomo/v1/history/session/{session_id}` - Session history
- `GET /qomo/v1/history/search` - Search history
- `GET /qomo/v1/history/recent` - Recent queries
- `POST /qomo/v1/history/feedback` - Add feedback
- `POST /qomo/v1/history/comment` - Add comment
- `GET /qomo/v1/history/feedback/stats/{query_id}` - Feedback stats

## Important Notes

- The application uses fail-fast behavior - it won't start if any required service is unavailable
- DeepSeek is the default and recommended LLM provider
- All database queries go through preprocessing to handle Chinese field names
- The agent uses a multi-step reasoning process with tool calls logged for debugging
- RAG search supports multiple dataset IDs for different document collections
- **user_email is REQUIRED** for all query requests to enable history tracking
- History is automatically saved after each successful query
- Database migrations are managed through Alembic
- Supports both SQLite (testing) and PostgreSQL (production) databases