# Kangni Agents

Industrial-grade FastAPI backend agent for Q&A with RAG and database integration.

## Quick Start

### 1. Setup Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -e .
```

### 3. Start the Server

#### Development Mode (with hot reload)
```bash
./dev_server.sh
```

#### Production Mode
```bash
./start_server.sh
```

#### Direct Command
```bash
source .venv/bin/activate
python -m src.kangni_agents.main
```

### 4. Test the Installation

#### Quick Test (Recommended)
Run a quick test to verify basic functionality:

```bash
# Using test runner (recommended)
python run_tests.py quick

# Or directly
python src/tests/quick_test.py
```

**Expected Output:**
```
🎉 Quick test passed! System is ready to use.
✅ DeepSeek is configured as the default LLM provider
✅ The specific question returns the expected result (5)
✅ Application can start successfully
```

#### Comprehensive Test
For full system verification, run the comprehensive test suite:

```bash
# Using test runner (recommended)
python run_tests.py all

# Or directly
python src/tests/test_all.py
```

**Expected Output:**
```
🎉 All tests passed! System is working correctly.
✅ DeepSeek is configured as the default LLM provider
✅ The specific question returns the expected result (5)
✅ All services are available and working
✅ Application can start successfully
```

#### All Tests
Run all test suites:

```bash
python run_tests.py all-tests
```

### 5. Access the API

- **Server**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Docker Usage

### Build Image
```bash
docker build -t kangni-agents .
```

### Run Container
```bash
docker run -p 8000:8000 kangni-agents
```

## API Endpoints

- `POST /api/v1/query` - Submit a question for processing
- `GET /api/v1/health` - Health check
- `GET /api/v1/config` - Configuration info
- `GET /docs` - Interactive API documentation

## Environment Variables

Create a `.env` file in the project root:

```env
# LLM Configuration (DeepSeek as default)
LLM_API_KEY=your_deepseek_api_key
LLM_BASE_URL=https://api.deepseek.com
LLM_CHAT_MODEL=deepseek-chat
DEEPSEEK_API_KEY=your_deepseek_api_key

# Database Configuration
MYSQL_HOST=localhost
MYSQL_USER=your_username
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=your_database
MYSQL_PORT=3306

# RAG Configuration (optional, has defaults)
RAGFLOW_MCP_SERVER_URL=http://158.58.50.45:9382/mcp
RAGFLOW_DEFAULT_DATASET_ID=f3073258886911f08bc30242c0a82006

# Database RAG Dataset IDs
DB_DDL_DATASET_ID=10123e2487f211f0aeb40242c0a80006
DB_QUERY_SQL_DATASET_ID=ffcc7faa87f311f09d4a0242c0a80006
DB_DESCRIPTION_DATASET_ID=6768e88087f211f0a8b00242c0a80006
```

## Architecture

The system includes:
- **FastAPI** web framework with async support
- **LangGraph** for agent workflow management
- **RAG Service** for document search via MCP
- **Database Service** for SQL query generation and execution
- **Intent Classification** for query routing
- **Fail-Fast System** for ensuring all services are available before startup
- **DeepSeek LLM** as the default language model provider

## Development

### Project Structure
```
src/kangni_agents/
├── main.py              # FastAPI application
├── config.py            # Configuration management
├── models/              # Pydantic models
├── api/                 # API routes
├── agents/              # Agent implementations
├── services/            # Business logic services
└── utils/               # Utility functions
```

### Adding New Features
1. Define models in `models/`
2. Add business logic to `services/`
3. Create API endpoints in `api/routes.py`
4. Update agent workflows in `agents/`

## Testing

This section provides comprehensive testing instructions for the Kangni Agents system.

### Prerequisites

Before running tests, ensure you have:

1. **Environment Configuration**: Create a `.env` file with proper API keys and database credentials
2. **Dependencies**: All required packages installed (`pip install -e .`)
3. **Services Running**: 
   - RAG MCP Server (http://158.58.50.45:9382/mcp)
   - MySQL Database with required tables
   - LLM API access (DeepSeek, OpenAI, or Alibaba)

### Test Categories

#### 1. Service Availability Tests

Test if all required services are available and properly configured:

```bash
# Test service availability check
python -c "
import sys; sys.path.insert(0, 'src')
from kangni_agents.main import check_service_availability
import asyncio
asyncio.run(check_service_availability())
print('✅ All services are available')
"
```

**Expected Output:**
```
INFO:kangni_agents.main:Checking service availability...
INFO:kangni_agents.main:RAG service imported successfully
INFO:kangni_agents.main:RAG service connection test successful
INFO:kangni_agents.main:Database service imported successfully
INFO:kangni_agents.main:Database connection test successful
INFO:kangni_agents.main:LLM service available
INFO:kangni_agents.main:All services are available
✅ All services are available
```

#### 2. LLM Provider Configuration Tests

Verify that DeepSeek is configured as the default LLM provider:

```bash
# Test LLM provider configuration
python -c "
import sys; sys.path.insert(0, 'src')
from kangni_agents.config import settings
from kangni_agents.services.database_service import db_service
from kangni_agents.agents.react_agent import kangni_agent

print(f'Base URL: {settings.openai_base_url}')
print(f'Model: {settings.llm_chat_model}')
print(f'Database Service Provider: {db_service.llm_provider}')
print(f'React Agent Provider: {kangni_agent.llm_provider}')
"
```

**Expected Output:**
```
Base URL: https://api.deepseek.com
Model: deepseek-chat
Database Service Provider: LLMProvider.DEEPSEEK
React Agent Provider: LLMProvider.DEEPSEEK
```

#### 3. Application Startup Tests

Test that the application can start successfully with fail-fast behavior:

```bash
# Test application startup
python -c "
import sys; sys.path.insert(0, 'src')
from kangni_agents.main import app
print('✅ FastAPI app created successfully')
"
```

#### 4. Specific Question Tests

Test the specific question that should return result "5":

```bash
# Test the specific question
python -c "
import sys; sys.path.insert(0, 'src')
import asyncio
from kangni_agents.services.database_service import db_service

async def test_question():
    question = '德里地铁4期项目(20D21028C000)在故障信息查询中共发生多少起故障？'
    result = await db_service.query_database(question)
    print(f'Question: {question}')
    print(f'Success: {result.get(\"success\")}')
    print(f'Answer: {result.get(\"answer\")}')
    print(f'Results: {result.get(\"results\")}')
    
    if result.get('success') and '5' in str(result.get('results', [])):
        print('✅ Test passed: Answer contains expected result (5)')
    else:
        print('❌ Test failed: Answer does not contain expected result (5)')

asyncio.run(test_question())
"
```

**Expected Output:**
```
Question: 德里地铁4期项目(20D21028C000)在故障信息查询中共发生多少起故障？
Success: True
Answer: 德里地铁4期项目(20D21028C000)在故障信息查询中共发生5起故障。
Results: [{'fault_count': 5}]
✅ Test passed: Answer contains expected result (5)
```

#### 5. RAG Functionality Tests

Test RAG document search functionality:

```bash
# Test RAG search
python -c "
import sys; sys.path.insert(0, 'src')
import asyncio
from kangni_agents.services.rag_service import rag_service

async def test_rag():
    question = '内解锁接地线线束短，无法安装到紧固螺钉位置是那个项目发生的？'
    results = await rag_service.search_rag(question, 'f3073258886911f08bc30242c0a82006')
    print(f'Question: {question}')
    print(f'Results count: {len(results)}')
    if results:
        print(f'First result: {results[0].content[:200]}...')
        if '东莞1号线项目' in results[0].content:
            print('✅ Test passed: Answer contains expected project name')
        else:
            print('❌ Test failed: Answer does not contain expected project name')
    else:
        print('❌ Test failed: No results returned')

asyncio.run(test_rag())
"
```

**Expected Output:**
```
Question: 内解锁接地线线束短，无法安装到紧固螺钉位置是那个项目发生的？
Results count: 5
First result: 该问题发生在东莞1号线项目。具体情况：在东莞1号线项目中，发现内解锁接地线线束过短，导致无法正确安装到指定的紧固螺钉位置...
✅ Test passed: Answer contains expected project name
```

#### 6. API Endpoint Tests

Test the API endpoints using curl:

```bash
# Test health check endpoint
curl "http://localhost:8000/health"

# Test query endpoint
curl -X POST "http://localhost:8000/api/v1/query" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "德里地铁4期项目(20D21028C000)在故障信息查询中共发生多少起故障？",
       "context": null,
       "session_id": "test-session"
     }'
```

#### 7. Comprehensive Test Script

Create a comprehensive test script to run all tests:

```bash
# Create test script
cat > test_all.py << 'EOF'
#!/usr/bin/env python3
"""Comprehensive test script for Kangni Agents"""

import asyncio
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_service_availability():
    """Test service availability"""
    print("Testing service availability...")
    try:
        from kangni_agents.main import check_service_availability
        await check_service_availability()
        print("✅ Service availability test passed")
        return True
    except Exception as e:
        print(f"❌ Service availability test failed: {e}")
        return False

async def test_llm_providers():
    """Test LLM provider configuration"""
    print("\nTesting LLM providers...")
    try:
        from kangni_agents.services.database_service import db_service
        from kangni_agents.agents.react_agent import kangni_agent
        
        if (db_service.llm_provider.value == "deepseek" and 
            kangni_agent.llm_provider.value == "deepseek"):
            print("✅ LLM provider test passed (DeepSeek)")
            return True
        else:
            print("❌ LLM provider test failed")
            return False
    except Exception as e:
        print(f"❌ LLM provider test failed: {e}")
        return False

async def test_specific_question():
    """Test the specific question"""
    print("\nTesting specific question...")
    try:
        from kangni_agents.services.database_service import db_service
        
        question = "德里地铁4期项目(20D21028C000)在故障信息查询中共发生多少起故障？"
        result = await db_service.query_database(question)
        
        if result.get('success') and '5' in str(result.get('results', [])):
            print("✅ Specific question test passed (result: 5)")
            return True
        else:
            print("❌ Specific question test failed")
            return False
    except Exception as e:
        print(f"❌ Specific question test failed: {e}")
        return False

async def test_rag_functionality():
    """Test RAG functionality"""
    print("\nTesting RAG functionality...")
    try:
        from kangni_agents.services.rag_service import rag_service
        
        question = "内解锁接地线线束短，无法安装到紧固螺钉位置是那个项目发生的？"
        results = await rag_service.search_rag(question, 'f3073258886911f08bc30242c0a82006')
        
        if results and '东莞1号线项目' in results[0].content:
            print("✅ RAG functionality test passed")
            return True
        else:
            print("❌ RAG functionality test failed")
            return False
    except Exception as e:
        print(f"❌ RAG functionality test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("Running comprehensive tests for Kangni Agents...")
    
    tests = [
        test_service_availability(),
        test_llm_providers(),
        test_specific_question(),
        test_rag_functionality()
    ]
    
    results = await asyncio.gather(*tests, return_exceptions=True)
    
    passed = sum(1 for r in results if r is True)
    total = len(results)
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is working correctly.")
    else:
        print("💥 Some tests failed! Please check the configuration.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
EOF

# Run comprehensive tests
python src/tests/test_all.py
```

### Test Environment Setup

#### Environment Variables

Ensure your `.env` file contains:

```env
# LLM Configuration (DeepSeek as default)
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=your_deepseek_api_key
LLM_CHAT_MODEL=deepseek-chat
DEEPSEEK_API_KEY=your_deepseek_api_key

# Database Configuration
MYSQL_HOST=your_mysql_host
MYSQL_USER=your_mysql_user
MYSQL_PASSWORD=your_mysql_password
MYSQL_DATABASE=your_mysql_database
MYSQL_PORT=3306

# RAG Configuration
RAGFLOW_MCP_SERVER_URL=http://158.58.50.45:9382/mcp
RAGFLOW_DEFAULT_DATASET_ID=f3073258886911f08bc30242c0a82006
```

#### Required Services

1. **RAG MCP Server**: Must be running at `http://158.58.50.45:9382/mcp`
2. **MySQL Database**: Must be accessible with the configured credentials
3. **LLM API**: DeepSeek API key must be valid and accessible

### Troubleshooting Tests

#### Common Issues

1. **Service Unavailable Errors**:
   - Check if RAG MCP server is running
   - Verify database connection credentials
   - Ensure LLM API key is valid

2. **Configuration Errors**:
   - Verify `.env` file exists and has correct values
   - Check environment variable names match the configuration

3. **Import Errors**:
   - Ensure you're running tests from the project root directory
   - Verify all dependencies are installed: `pip install -e .`

#### Debug Mode

Run tests with debug information:

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python src/tests/test_all.py
```

### Continuous Integration

For CI/CD pipelines, use the comprehensive test script:

```yaml
# Example GitHub Actions workflow
- name: Run Tests
  run: |
    python src/tests/test_all.py
  env:
    LLM_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
    MYSQL_HOST: ${{ secrets.MYSQL_HOST }}
    MYSQL_USER: ${{ secrets.MYSQL_USER }}
    MYSQL_PASSWORD: ${{ secrets.MYSQL_PASSWORD }}
    MYSQL_DATABASE: ${{ secrets.MYSQL_DATABASE }}
```

### API使用示例

#### 查询接口

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "用户总数有多少？",
       "context": null,
       "session_id": "test-session"
     }'
```

#### 健康检查

```bash
curl "http://localhost:8000/api/v1/health"
```

## 功能特性

- 🤖 **智能意图识别**: 自动判断用户问题类型（RAG搜索/数据库查询/混合模式）
- 📚 **RAG文档搜索**: 集成RAGFlow MCP服务，支持多数据集文档检索
- 🗃️ **数据库查询**: 自动生成SQL查询，支持统计和分析类问题
- 🔄 **ReAct Agent**: 基于LangGraph的推理行动循环，智能决策工具调用
- 🚀 **高性能API**: FastAPI框架，支持异步处理和自动API文档
- 📊 **完整日志**: 结构化日志记录，便于监控和调试

## 工作流程

1. **意图识别**: 分析用户问题，识别查询类型
2. **Agent推理**: LangGraph Agent决定使用哪些工具
3. **工具调用**: 执行RAG搜索或数据库查询
4. **结果整合**: 生成最终答案返回用户

## 意图分类规则

- **RAG搜索**: 包含"为什么"、"如何"、"总结"等概念性关键词
- **数据库查询**: 包含"统计"、"多少"、"排序"等数据性关键词
- **混合模式**: 意图不明确时同时使用两种方式

## Troubleshooting

### Missing Dependencies
The system has fallback implementations that work with minimal dependencies. If you see warnings about missing packages, you can either:
1. Install the full dependencies: `pip install langchain langgraph mcp`
2. Continue with fallback mode for basic functionality

### Port Already in Use
If port 8000 is busy, modify the port in:
- `config.py` (api_port setting)
- Scripts: `dev_server.sh`, `start_server.sh`
- Docker: `-p` parameter

## 扩展开发

### 添加新工具

1. 在`agents/react_agent.py`中定义新的`@tool`函数
2. 将工具添加到`tools`列表
3. 更新Agent的系统提示

### 添加新数据集

1. 在`config.py`中添加数据集ID配置
2. 在`services/rag_service.py`中添加搜索方法
3. 更新API接口支持新数据集

## Quick Reference

### Essential Commands

```bash
# Install and setup
pip install -e .

# Quick test (recommended)
python run_tests.py quick

# Comprehensive test
python run_tests.py all

# All tests
python run_tests.py all-tests

# Start development server
./dev_server.sh

# Start production server
./start_server.sh

# Check health
curl http://localhost:8000/health
```

### Key Configuration

- **Default LLM**: DeepSeek (`https://api.deepseek.com`)
- **Default Model**: `deepseek-chat`
- **Fail-Fast**: Application won't start if services are unavailable
- **Test Question**: "德里地铁4期项目(20D21028C000)在故障信息查询中共发生多少起故障？" → Result: 5

### Support

For issues or questions:
1. Check the [Testing](#testing) section for troubleshooting
2. Verify your `.env` file configuration
3. Ensure all required services are running
4. Run `python run_tests.py all` to diagnose issues