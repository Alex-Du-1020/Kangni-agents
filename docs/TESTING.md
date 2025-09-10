# 测试文档

本文档提供了 Kangni Agents 系统的综合测试说明。

## 前置条件

在运行测试之前，请确保：

1. **环境配置**: 创建包含正确 API 密钥和数据库凭据的 `.env` 文件
2. **数据库访问**: 确保数据库连接可用
3. **外部服务**: 验证以下服务的访问权限：
   - RAGFlow MCP 服务器
   - LLM API 访问（DeepSeek、OpenAI 或 Alibaba）

## 综合测试套件

主要的测试方法是通过综合测试套件，涵盖所有功能：

```bash
# 运行完整的测试套件
python src/test/test_comprehensive.py
```

此测试套件包括：

### 1. 历史服务测试
- ✅ 保存查询历史（成功和失败的查询）
- ✅ 分页检索用户历史
- ✅ 添加和管理用户反馈（点赞/点踩）
- ✅ 添加和管理用户评论
- ✅ 获取反馈统计
- ✅ 按关键词搜索历史
- ✅ 获取最近查询
- ✅ 获取会话特定历史

### 2. API 端点测试
- ✅ 带 user_email 验证的查询端点
- ✅ 用户历史 API 端点
- ✅ 会话历史 API 端点
- ✅ 搜索历史 API 端点
- ✅ 最近查询 API 端点
- ✅ 反馈管理 API 端点
- ✅ 评论管理 API 端点
- ✅ 反馈统计 API 端点

### 3. 用户邮箱验证测试
- ✅ 带有效 user_email 的查询（应该成功）
- ✅ 不带 user_email 的查询（应该失败，返回 422）
- ✅ 带空 user_email 的查询（应该失败，返回 400）

### 4. 查询预处理测试
- ✅ 从标记查询中提取实体（#标记#）
- ✅ 多实体处理
- ✅ 复杂项目名称处理
- ✅ 混合标记类型（#、[]、""、()）
- ✅ 无特殊标记的查询

### 5. LLM 连接测试
- ✅ LLM 提供商配置验证
- ✅ API 密钥验证
- ✅ 服务可用性检查
- ✅ 基本聊天功能

### 6. RAG 服务测试
- ✅ RAG 服务可用性
- ✅ 文档搜索功能
- ✅ 数据库上下文搜索
- ✅ 结果相关性验证

### 7. 测试用例执行
- ✅ 真实世界查询测试
- ✅ 关键词匹配验证
- ✅ 性能测量
- ✅ 错误处理验证

## 测试环境设置

### 环境变量

创建包含以下变量的 `.env` 文件：

```bash
# LLM 配置
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_api_key
OPENAI_API_KEY=your_openai_api_key
ALIBABA_API_KEY=your_alibaba_api_key

# 数据库配置
DB_TYPE=sqlite
HISTORY_DATABASE_URL=sqlite:///.src/resources/history.db

# RAG 配置
RAGFLOW_MCP_SERVER_URL=http://158.58.50.45:9382/mcp
RAGFLOW_DEFAULT_DATASET_ID=f3073258886911f08bc30242c0a82006

# 日志
LOG_LEVEL=INFO
ENVIRONMENT=test
```

### 数据库设置

测试套件使用 SQLite 进行测试，会自动创建：

```bash
# 数据库将自动创建在：./src/resources/history.db
# 测试无需手动设置
```

## 测试结果

综合测试套件提供详细结果：

```
📊 综合测试套件摘要
================================================================================
测试类别总数: 7
通过: 7
失败: 0
成功率: 100.0%
总耗时: 90.81s

📋 详细结果:
  ✅ 通过 历史服务 (0.25s)
  ✅ 通过 API 端点 (0.04s)
  ✅ 通过 用户邮箱验证 (0.01s)
  ✅ 通过 查询预处理 (0.00s)
  ✅ 通过 LLM 连接 (1.76s)
  ✅ 通过 RAG 服务 (10.08s)
  ✅ 通过 测试用例执行 (78.66s)

📄 详细结果已保存到: comprehensive_test_results.json
```

## 测试故障排除

### 常见问题

1. **服务器未运行**: 某些测试需要服务器运行
   ```bash
   # 在后台启动服务器
   ./dev_server.sh &
   # 然后运行测试
   python src/test/test_comprehensive.py
   ```

2. **API 密钥问题**: 确保配置了有效的 API 密钥
   ```bash
   # 检查 API 密钥配置
   python -c "
   import sys; sys.path.insert(0, 'src')
   from kangni_agents.config import settings
   print(f'DeepSeek API Key: {\"✅ 已配置\" if settings.deepseek_api_key else \"❌ 未配置\"}')
   "
   ```

3. **导入错误**: 确保从项目根目录运行测试
   ```bash
   # 验证依赖
   pip install -e .
   ```

### 调试模式

使用调试信息运行测试：

```bash
# 启用调试日志
export LOG_LEVEL=DEBUG
python src/test/test_comprehensive.py
```

## 持续集成

对于 CI/CD 流水线，使用综合测试脚本：

```yaml
# GitHub Actions 工作流示例
- name: 运行测试
  run: |
    python src/test/test_comprehensive.py
  env:
    LLM_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
    RAGFLOW_MCP_SERVER_URL: ${{ secrets.RAGFLOW_URL }}
```

## 测试命令参考

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
curl http://localhost:8000/qomo/v1/health
```

### 关键配置

- **默认 LLM**: DeepSeek (`https://api.deepseek.com`)
- **默认模型**: `deepseek-chat`
- **快速失败**: 如果服务不可用，应用程序不会启动
- **测试问题**: "德里地铁4期项目(20D21028C000)在故障信息查询中共发生多少起故障？" → 结果: 5

### 支持

如有问题或疑问：
1. 查看本测试文档的故障排除部分
2. 验证您的 `.env` 文件配置
3. 确保所有必需服务正在运行
4. 运行 `python src/test/test_comprehensive.py` 来诊断问题

