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
./prod_server.sh
```

#### Direct Command
```bash
source .venv/bin/activate
python -m src.kangni_agents.main
```

### 4. Test the Installation

#### Comprehensive Test Suite (Recommended)
Run the comprehensive test suite to verify all functionality:

```bash
# Run comprehensive test suite
python src/test/test_comprehensive.py
```

**Expected Output:**
```
🚀 Kangni Agents Comprehensive Test Suite
================================================================================
📋 History Service Tests
✅ Saved query history with ID: 11
✅ Retrieved 3 history items for user
✅ Added 'like' feedback with ID: 5
✅ Feedback stats - Likes: 1, Dislikes: 0

📋 API Endpoint Tests
✅ Query created successfully
✅ User history retrieved: 4 items
✅ Session history retrieved: 13 items
✅ Search results: 3 items

📋 User Email Validation Tests
✅ Query successful with user_email
✅ Correctly rejected query without user_email
✅ Correctly rejected query with empty user_email

📋 Query Preprocessing Tests
✅ PASS - All 5 test cases passed

📋 LLM Connection Tests
✅ Current LLM configuration test successful

📋 RAG Service Tests
✅ RAG service is available
✅ Found 10 relevant records

📋 Test Cases Execution
✅ PASS - 4/5 test cases passed

================================================================================
📊 COMPREHENSIVE TEST SUITE SUMMARY
================================================================================
Total test categories: 7
Passed: 7
Failed: 0
Success rate: 100.0%
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
2. **Database Access**: Ensure database connection is available
3. **External Services**: Verify access to:
   - RAGFlow MCP server
   - LLM API access (DeepSeek, OpenAI, or Alibaba)

### Comprehensive Test Suite

The main testing approach is through the comprehensive test suite that covers all functionality:

```bash
# Run the complete test suite
python src/test/test_history.py
```

This test suite includes:

#### 1. History Service Tests
- ✅ Save query history (successful and failed queries)
- ✅ Retrieve user history with pagination
- ✅ Add and manage user feedback (likes/dislikes)
- ✅ Add and manage user comments
- ✅ Get feedback statistics
- ✅ Search history by keywords
- ✅ Get recent queries
- ✅ Get session-specific history

#### 2. API Endpoint Tests
- ✅ Query endpoint with user_email validation
- ✅ User history API endpoints
- ✅ Session history API endpoints
- ✅ Search history API endpoints
- ✅ Recent queries API endpoints
- ✅ Feedback management API endpoints
- ✅ Comment management API endpoints
- ✅ Feedback statistics API endpoints

#### 3. User Email Validation Tests
- ✅ Query with valid user_email (should succeed)
- ✅ Query without user_email (should fail with 422)
- ✅ Query with empty user_email (should fail with 400)

#### 4. Query Preprocessing Tests
- ✅ Entity extraction from marked queries (#标记#)
- ✅ Multiple entity handling
- ✅ Complex project name processing
- ✅ Mixed marker types (#, [], "", ())
- ✅ Queries without special markers

#### 5. LLM Connection Tests
- ✅ LLM provider configuration verification
- ✅ API key validation
- ✅ Service availability checks
- ✅ Basic chat functionality

#### 6. RAG Service Tests
- ✅ RAG service availability
- ✅ Document search functionality
- ✅ Database context search
- ✅ Result relevance validation

#### 7. Test Cases Execution
- ✅ Real-world query testing
- ✅ Keyword matching validation
- ✅ Performance measurement
- ✅ Error handling verification

### Individual Test Categories

You can run specific test categories individually:

```bash
# Test only history functionality
python -c "
import asyncio
import sys
sys.path.insert(0, 'src')
from test.test_comprehensive import ComprehensiveTestSuite
async def run_history_tests():
    suite = ComprehensiveTestSuite()
    await suite.test_history_service()
asyncio.run(run_history_tests())
"

# Test only API endpoints
python -c "
import asyncio
import sys
sys.path.insert(0, 'src')
from test.test_comprehensive import ComprehensiveTestSuite
async def run_api_tests():
    suite = ComprehensiveTestSuite()
    await suite.test_api_endpoints()
asyncio.run(run_api_tests())
"

# Test only user email validation
python -c "
import asyncio
import sys
sys.path.insert(0, 'src')
from test.test_comprehensive import ComprehensiveTestSuite
async def run_validation_tests():
    suite = ComprehensiveTestSuite()
    await suite.test_user_email_validation()
asyncio.run(run_validation_tests())
"
```

### Test Environment Setup

#### Environment Variables

Create a `.env` file with the following variables:

```bash
# LLM Configuration
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_api_key
OPENAI_API_KEY=your_openai_api_key
ALIBABA_API_KEY=your_alibaba_api_key

# Database Configuration
DB_TYPE=sqlite
HISTORY_DATABASE_URL=sqlite:///.src/resources/history.db

# RAG Configuration
RAGFLOW_MCP_SERVER_URL=http://158.58.50.45:9382/mcp
RAGFLOW_DEFAULT_DATASET_ID=f3073258886911f08bc30242c0a82006

# Logging
LOG_LEVEL=INFO
ENVIRONMENT=test
```

#### Database Setup

The test suite uses SQLite for testing, which is automatically created:

```bash
# Database will be created automatically at: ./src/resources/history.db
# No manual setup required for testing
```

### Test Results

The comprehensive test suite provides detailed results:

```
📊 COMPREHENSIVE TEST SUITE SUMMARY
================================================================================
Total test categories: 7
Passed: 7
Failed: 0
Success rate: 100.0%
Total duration: 90.81s

📋 Detailed Results:
  ✅ PASS History Service (0.25s)
  ✅ PASS API Endpoints (0.04s)
  ✅ PASS User Email Validation (0.01s)
  ✅ PASS Query Preprocessing (0.00s)
  ✅ PASS LLM Connection (1.76s)
  ✅ PASS RAG Service (10.08s)
  ✅ PASS Test Cases Execution (78.66s)

📄 Detailed results saved to: comprehensive_test_results.json
```

### Troubleshooting Tests

#### Common Issues

1. **Server Not Running**: Some tests require the server to be running
   ```bash
   # Start server in background
   ./dev_server.sh &
   # Then run tests
   python src/test/test_comprehensive.py
   ```

2. **API Key Issues**: Ensure valid API keys are configured
   ```bash
   # Check API key configuration
   python -c "
   import sys; sys.path.insert(0, 'src')
   from kangni_agents.config import settings
   print(f'DeepSeek API Key: {\"✅ Configured\" if settings.deepseek_api_key else \"❌ Not configured\"}')
   "
   ```

3. **Import Errors**: Ensure you're running tests from the project root directory
   ```bash
   # Verify dependencies
   pip install -e .
   ```

#### Debug Mode

Run tests with debug information:

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python src/test/test_comprehensive.py
```

### Continuous Integration

For CI/CD pipelines, use the comprehensive test script:

```yaml
# Example GitHub Actions workflow
- name: Run Tests
  run: |
    python src/test/test_comprehensive.py
  env:
    LLM_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
    RAGFLOW_MCP_SERVER_URL: ${{ secrets.RAGFLOW_URL }}
```


### API使用示例

#### 查询接口

**Note: user_email is now REQUIRED for all queries to enable history tracking and feedback features**

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "用户总数有多少？",
       "user_email": "user@example.com",
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

# Run comprehensive test suite (recommended)
python src/test/test_comprehensive.py

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
4. Run `python src/test/test_comprehensive.py` to diagnose issues