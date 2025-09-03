# Kangni Agents

工业级FastAPI后台AI Agent系统，集成RAG文档搜索和数据库查询功能。

## 功能特性

- 🤖 **智能意图识别**: 自动判断用户问题类型（RAG搜索/数据库查询/混合模式）
- 📚 **RAG文档搜索**: 集成RAGFlow MCP服务，支持多数据集文档检索
- 🗃️ **数据库查询**: 自动生成SQL查询，支持统计和分析类问题
- 🔄 **ReAct Agent**: 基于LangGraph的推理行动循环，智能决策工具调用
- 🚀 **高性能API**: FastAPI框架，支持异步处理和自动API文档
- 📊 **完整日志**: 结构化日志记录，便于监控和调试

## 架构设计

```
src/kangni_agents/
├── agents/          # LangGraph ReAct Agent
├── api/             # FastAPI路由和API
├── models/          # Pydantic数据模型
├── services/        # RAG和数据库服务
├── utils/           # 意图分类等工具
├── config.py        # 配置管理
└── main.py          # 主应用
```

## 快速开始

### 1. 安装依赖

```bash
# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows

# 安装依赖
pip install -e .
```

### 2. 配置环境变量

```bash
# 复制环境变量模板
cp .env.template .env

# 编辑.env文件，填入你的配置
vim .env
```

### 3. 启动服务

```bash
# 使用启动脚本
python run.py

# 或直接使用uvicorn
uvicorn src.kangni_agents.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. 访问API文档

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API使用示例

### 查询接口

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "用户总数有多少？",
       "context": null,
       "session_id": "test-session"
     }'
```

### 健康检查

```bash
curl "http://localhost:8000/api/v1/health"
```

## 配置说明

### RAG配置
- `RAGFLOW_MCP_SERVER_URL`: RAGFlow MCP服务地址
- `RAGFLOW_DEFAULT_DATASET_ID`: 默认文档数据集ID

### 数据库RAG数据集
- `DB_DDL_DATASET_ID`: 数据库表结构数据集
- `DB_QUERY_SQL_DATASET_ID`: SQL查询示例数据集  
- `DB_DESCRIPTION_DATASET_ID`: 数据库描述数据集

### LLM配置
- `OPENAI_API_KEY`: OpenAI API密钥
- `OPENAI_BASE_URL`: API服务地址

## 工作流程

1. **意图识别**: 分析用户问题，识别查询类型
2. **Agent推理**: LangGraph Agent决定使用哪些工具
3. **工具调用**: 执行RAG搜索或数据库查询
4. **结果整合**: 生成最终答案返回用户

## 意图分类规则

- **RAG搜索**: 包含"为什么"、"如何"、"总结"等概念性关键词
- **数据库查询**: 包含"统计"、"多少"、"排序"等数据性关键词
- **混合模式**: 意图不明确时同时使用两种方式

## 开发

### 运行测试

```bash
pytest tests/
```

### 代码格式化

```bash
black src/ tests/
```

### 类型检查

```bash
mypy src/
```

## 部署

### Docker部署

```bash
# 构建镜像
docker build -t kangni-agents .

# 运行容器
docker run -p 8000:8000 --env-file .env kangni-agents
```

### 生产部署

```bash
# 使用Gunicorn
gunicorn src.kangni_agents.main:app -w 4 -k uvicorn.workers.UnicornWorker
```

## 扩展开发

### 添加新工具

1. 在`agents/react_agent.py`中定义新的`@tool`函数
2. 将工具添加到`tools`列表
3. 更新Agent的系统提示

### 添加新数据集

1. 在`config.py`中添加数据集ID配置
2. 在`services/rag_service.py`中添加搜索方法
3. 更新API接口支持新数据集

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

MIT License