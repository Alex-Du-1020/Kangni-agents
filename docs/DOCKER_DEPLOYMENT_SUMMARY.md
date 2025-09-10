# Docker 部署完成总结

## ✅ 已完成的工作

### 1. Dockerfile 优化
- ✅ 修复了文件引用问题（`start_server.sh` → `prod_server.sh`）
- ✅ 添加了 PostgreSQL 支持依赖（`libpq-dev`）
- ✅ 优化了依赖安装流程
- ✅ 添加了健康检查配置
- ✅ 测试验证构建成功

### 2. Docker Compose 配置
- ✅ 创建了生产环境配置（`docker-compose.yml`）
- ✅ 创建了开发环境配置（`docker-compose.dev.yml`）
- ✅ 集成了 PostgreSQL 数据库
- ✅ 配置了服务依赖和健康检查
- ✅ 添加了环境变量配置
- ✅ 创建了数据库初始化脚本

### 3. GitLab CI/CD 配置
- ✅ 更新了 `.gitlab-ci.yml` 文件
- ✅ 配置了阿里云容器镜像服务集成
- ✅ 设置了 RAM 访问控制认证
- ✅ 配置了自动构建和推送流程
- ✅ 添加了触发条件（main 分支提交）

### 4. 文档和配置
- ✅ 创建了 Docker 部署指南
- ✅ 创建了 GitLab CI/CD 配置指南
- ✅ 提供了环境变量示例文件
- ✅ 添加了故障排除指南

## 🚀 使用方法

### 本地 Docker 部署

#### 生产环境
```bash
cd dcoker
cp env.example .env
# 编辑 .env 文件，设置 API 密钥
docker-compose up -d
```

#### 开发环境
```bash
cd dcoker
cp env.example .env
# 编辑 .env 文件，设置 API 密钥
docker-compose -f docker-compose.dev.yml up -d
```

### GitLab CI/CD 自动部署

1. **配置环境变量**
   - 在 GitLab 项目设置中添加：
     - `ALIBABA_ACCESS_KEY_ID`: `LTAI5tRfYCXUtbtk2B8FAkCX`
     - `ALIBABA_ACCESS_KEY_SECRET`: `DZ51dA7wcDJ7cPTXA6y2mGlE7c5NX7`

2. **触发部署**
   - 向 `main` 分支推送代码
   - 系统将自动构建并推送镜像到阿里云

## 📋 服务访问地址

### 生产环境
- **应用服务**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/qomo/v1/health
- **PostgreSQL**: localhost:5432

### 开发环境
- **应用服务**: http://localhost:8001
- **API 文档**: http://localhost:8001/docs
- **健康检查**: http://localhost:8001/qomo/v1/health
- **PostgreSQL**: localhost:5433

## 🔧 配置说明

### 环境变量
```bash
# LLM 配置
DEEPSEEK_API_KEY=your_deepseek_api_key

# 数据库配置（自动配置）
DB_TYPE=postgresql
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=kangni_ai_chatbot

# RAG 配置
RAGFLOW_MCP_SERVER_URL=http://158.58.50.45:9382/mcp
RAGFLOW_DEFAULT_DATASET_ID=f3073258886911f08bc30242c0a82006
```

### 镜像标签策略
- **提交标签**: `kangni-agents:$CI_COMMIT_SHORT_SHA`
- **最新标签**: `kangni-agents:latest`
- **完整地址**: `crpi-l3yyk3n2aniyegyy.cn-hangzhou.personal.cr.aliyuncs.com/kangni-agents:latest`

## 🛠️ 常用命令

### Docker Compose 管理
```bash
# 启动服务
docker-compose up -d

# 停止服务
docker-compose down

# 查看日志
docker-compose logs -f

# 重启服务
docker-compose restart

# 重新构建
docker-compose up -d --build
```

### 数据库管理
```bash
# 连接数据库
docker exec -it kangni-postgres psql -U postgres -d kangni_ai_chatbot

# 运行迁移
docker exec -it kangni-agents alembic upgrade head

# 备份数据库
docker exec kangni-postgres pg_dump -U postgres kangni_ai_chatbot > backup.sql
```

## 🔍 故障排除

### 常见问题
1. **端口冲突**: 检查 8000/8001 和 5432/5433 端口是否被占用
2. **API 密钥**: 确保在 `.env` 文件中正确配置了 `DEEPSEEK_API_KEY`
3. **数据库连接**: 等待 PostgreSQL 健康检查通过后再启动应用
4. **权限问题**: 确保 Docker 有足够权限访问文件系统

### 调试方法
```bash
# 查看服务状态
docker-compose ps

# 查看详细日志
docker-compose logs -f kangni-agents

# 进入容器调试
docker exec -it kangni-agents bash

# 检查健康状态
curl http://localhost:8000/qomo/v1/health
```

## 📚 相关文档

- [Docker 部署指南](dcoker/README.md)
- [GitLab CI/CD 配置指南](GITLAB_CI_SETUP.md)
- [测试文档](TESTING.md)
- [项目主文档](../README.md)

## 🎯 下一步建议

1. **监控配置**: 添加 Prometheus 和 Grafana 监控
2. **日志收集**: 配置 ELK 或类似日志收集系统
3. **备份策略**: 设置数据库自动备份
4. **安全加固**: 配置 SSL/TLS 和防火墙规则
5. **性能优化**: 根据实际使用情况调整资源配置

## ✅ 验证清单

- [x] Dockerfile 构建成功
- [x] Docker Compose 配置验证通过
- [x] PostgreSQL 集成正常
- [x] GitLab CI/CD 配置完成
- [x] 环境变量配置正确
- [x] 健康检查配置完成
- [x] 文档编写完成

所有配置已完成，系统可以正常部署和运行！
