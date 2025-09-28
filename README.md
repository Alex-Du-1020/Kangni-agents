# Kangni Agents

工业级 FastAPI 后端智能代理，集成 RAG 和数据库查询功能。

## 快速开始

### 1. 设置虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. 安装依赖

```bash
pip install --upgrade pip
pip install -e .
```

### 3. 启动服务器

#### 开发模式（热重载）
```bash
./dev_server.sh
```

#### 生产模式
```bash
./prod_server.sh
```

#### 直接命令
```bash
source .venv/bin/activate
python -m src.kangni_agents.main
```

### 4. 测试安装

运行综合测试套件来验证所有功能：

```bash
# 运行综合测试套件
python src/test/test_comprehensive.py
```

详细测试说明请参考：[测试文档](docs/TESTING.md)

### 5. 访问 API

- **服务器**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/qomo/v1/health

## Docker 使用

### 构建镜像
```bash
docker build -t kangni-agents .
```

### 运行容器
```bash
docker run -p 8000:8000 kangni-agents
```

## API 接口

- `POST /qomo/v1/query` - 提交问题进行处理
- `GET /qomo/v1/health` - 健康检查
- `GET /qomo/v1/config` - 配置信息
- `GET /docs` - 交互式 API 文档

## 环境变量

在项目根目录创建 `.env` 文件：

```env
# LLM 配置（默认使用 DeepSeek）
LLM_API_KEY=your_deepseek_api_key
LLM_BASE_URL=https://api.deepseek.com
LLM_CHAT_MODEL=deepseek-chat
DEEPSEEK_API_KEY=your_deepseek_api_key

# 数据库配置
MYSQL_HOST=localhost
MYSQL_USER=your_username
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=your_database
MYSQL_PORT=3306

# RAG 配置（可选，有默认值）
RAGFLOW_MCP_SERVER_URL=http://158.58.50.45:9382/mcp

# 数据库 RAG 数据集 ID
DB_DDL_DATASET_ID=2eeb6f2a9ac911f094c80242c0a85006
DB_QUERY_SQL_DATASET_ID=3387079e9acc11f0b60f0242c0a87006
DB_DESCRIPTION_DATASET_ID=8c443ba09acb11f093460242c0a87006
```

## 系统架构

系统包含以下组件：
- **FastAPI** - 支持异步的 Web 框架
- **LangGraph** - 智能代理工作流管理
- **RAG 服务** - 通过 MCP 进行文档搜索
- **数据库服务** - SQL 查询生成和执行
- **意图分类** - 查询路由
- **快速失败系统** - 确保所有服务在启动前可用
- **DeepSeek LLM** - 默认语言模型提供商

## 开发

### 项目结构
```
src/kangni_agents/
├── main.py              # FastAPI 应用
├── config.py            # 配置管理
├── models/              # Pydantic 模型
├── api/                 # API 路由
├── agents/              # 智能代理实现
├── services/            # 业务逻辑服务
└── utils/               # 工具函数
```

### 添加新功能
1. 在 `models/` 中定义模型
2. 在 `services/` 中添加业务逻辑
3. 在 `api/routes.py` 中创建 API 端点
4. 在 `agents/` 中更新智能代理工作流

## API 使用示例

### 查询接口

**注意：user_email 现在是所有查询的必需参数，用于启用历史跟踪和反馈功能**

```bash
curl -X POST "http://localhost:8000/qomo/v1/query" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "用户总数有多少？",
       "user_email": "user@example.com",
       "context": null,
       "session_id": "test-session"
     }'
```

### 健康检查

```bash
curl "http://localhost:8000/qomo/v1/health"
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

## 故障排除

### 缺少依赖
系统具有适用于最小依赖的回退实现。如果您看到关于缺少包的警告，您可以：
1. 安装完整依赖：`pip install langchain langgraph mcp`
2. 继续使用回退模式以获得基本功能

### 端口已被占用
如果端口 8000 繁忙，请在以下位置修改端口：
- `config.py`（api_port 设置）
- 脚本：`dev_server.sh`、`start_server.sh`
- Docker：`-p` 参数

## 扩展开发

### 添加新工具

1. 在`agents/react_agent.py`中定义新的`@tool`函数
2. 将工具添加到`tools`列表
3. 更新Agent的系统提示

### 添加新数据集

1. 在`config.py`中添加数据集ID配置
2. 在`services/rag_service.py`中添加搜索方法
3. 更新API接口支持新数据集

## 快速参考

### 基本命令

```bash
# 安装和设置
pip install -e .

# 运行综合测试套件（推荐）
python src/test/test_comprehensive.py

# 启动开发服务器
./dev_server.sh

# 启动生产服务器
./prod_server.sh

# 检查健康状态
curl http://localhost:8000/health
```

### 关键配置

- **默认 LLM**: DeepSeek (`https://api.deepseek.com`)
- **默认模型**: `deepseek-chat`
- **快速失败**: 如果服务不可用，应用程序不会启动
- **测试问题**: "德里地铁4期项目(20D21028C000)在故障信息查询中共发生多少起故障？" → 结果: 5

### 支持

如有问题或疑问：
1. 查看[测试文档](docs/TESTING.md)进行故障排除
2. 验证您的 `.env` 文件配置
3. 确保所有必需服务正在运行
4. 运行 `python src/test/test_comprehensive.py` 来诊断问题