# Docker 部署指南

本目录包含 Kangni Agents 系统的 Docker 部署配置。

## 文件说明

- `docker-compose.yml` - 生产环境配置
- `docker-compose.dev.yml` - 开发环境配置
- `env.example` - 环境变量示例文件
- `README.md` - 本说明文件

## 快速开始

### 1. 准备环境变量

复制环境变量示例文件并配置：

```bash
cp env.example .env
```

编辑 `.env` 文件，设置您的 API 密钥：

```bash
# 必须配置的 API 密钥
DEEPSEEK_API_KEY=your_actual_deepseek_api_key
```

### 2. 生产环境部署

```bash
# 构建并启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f kangni-agents
```

### 3. 开发环境部署

```bash
# 使用开发配置启动
docker-compose -f docker-compose.dev.yml up -d

# 查看开发环境日志
docker-compose -f docker-compose.dev.yml logs -f kangni-agents
```

## 服务访问

### 生产环境
- **应用服务**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/api/v1/health
- **PostgreSQL**: localhost:5432

### 开发环境
- **应用服务**: http://localhost:8001
- **API 文档**: http://localhost:8001/docs
- **健康检查**: http://localhost:8001/api/v1/health
- **PostgreSQL**: localhost:5433

## 数据库管理

### 连接数据库

```bash
# 生产环境
docker exec -it kangni-postgres psql -U postgres -d kangni_ai_chatbot

# 开发环境
docker exec -it kangni-postgres-dev psql -U postgres -d kangni_ai_chatbot_dev
```

### 数据库迁移

```bash
# 进入应用容器
docker exec -it kangni-agents bash

# 运行数据库迁移
alembic upgrade head
```

## 常用命令

### 服务管理

```bash
# 启动所有服务
docker-compose up -d

# 停止所有服务
docker-compose down

# 重启服务
docker-compose restart

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f [service_name]
```

### 构建和更新

```bash
# 重新构建镜像
docker-compose build

# 强制重新构建
docker-compose build --no-cache

# 更新并重启服务
docker-compose up -d --build
```

### 清理

```bash
# 停止并删除容器
docker-compose down

# 删除容器和卷
docker-compose down -v

# 删除所有相关镜像
docker-compose down --rmi all
```

## 故障排除

### 常见问题

1. **端口冲突**
   - 生产环境使用 8000 和 5432 端口
   - 开发环境使用 8001 和 5433 端口
   - 确保端口未被占用

2. **API 密钥未配置**
   - 检查 `.env` 文件是否存在
   - 确保 `DEEPSEEK_API_KEY` 已正确设置

3. **数据库连接失败**
   - 等待 PostgreSQL 完全启动（健康检查通过）
   - 检查数据库配置是否正确

4. **服务启动失败**
   - 查看详细日志：`docker-compose logs [service_name]`
   - 检查环境变量配置
   - 验证网络连接

### 调试模式

```bash
# 以调试模式启动
docker-compose -f docker-compose.dev.yml up

# 进入容器调试
docker exec -it kangni-agents bash
```

## 监控和日志

### 健康检查

所有服务都配置了健康检查：

```bash
# 检查服务健康状态
docker-compose ps

# 手动健康检查
curl http://localhost:8000/api/v1/health
```

### 日志查看

```bash
# 查看所有服务日志
docker-compose logs

# 实时查看日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f kangni-agents
docker-compose logs -f postgres
```

## 生产环境建议

1. **安全配置**
   - 修改默认数据库密码
   - 使用 Docker secrets 管理敏感信息
   - 配置防火墙规则

2. **性能优化**
   - 调整 PostgreSQL 配置
   - 配置适当的资源限制
   - 使用多副本部署

3. **监控**
   - 配置日志收集
   - 设置监控告警
   - 定期备份数据库
