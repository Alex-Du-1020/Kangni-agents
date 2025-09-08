# GitLab CI/CD 配置指南

本指南将帮助您配置 GitLab CI/CD 管道，实现自动构建和推送 Docker 镜像到阿里云容器镜像服务。

## 前置条件

1. **阿里云容器镜像服务账号**
   - 已开通阿里云容器镜像服务
   - 已创建命名空间和仓库

2. **RAM 访问控制**
   - 已创建 RAM 用户
   - 已配置适当的权限策略

3. **GitLab 项目**
   - 项目已配置 GitLab CI/CD
   - 具有管理员权限

## 配置步骤

### 1. 在 GitLab 中配置环境变量

在 GitLab 项目设置中添加以下环境变量：

#### 路径：Settings → CI/CD → Variables

| 变量名 | 值 | 描述 | 保护 | 掩码 |
|--------|-----|------|------|------|
| `ALIBABA_ACCESS_KEY_ID` | `LTAI5tRfYCXUtbtk2B8FAkCX` | 阿里云 RAM Access Key ID | ✅ | ❌ |
| `ALIBABA_ACCESS_KEY_SECRET` | `DZ51dA7wcDJ7cPTXA6y2mGlE7c5NX7` | 阿里云 RAM Access Key Secret | ✅ | ✅ |

**重要提示：**
- 勾选 "Protected" 选项，确保只在受保护的分支中使用
- 对于 `ALIBABA_ACCESS_KEY_SECRET`，勾选 "Masked" 选项以隐藏敏感信息

### 2. 验证 GitLab CI/CD 配置

确保 `.gitlab-ci.yml` 文件已正确配置：

```yaml
# 主要配置项
variables:
  ALIBABA_REGISTRY: crpi-l3yyk3n2aniyegyy.cn-hangzhou.personal.cr.aliyuncs.com
  DOCKER_IMAGE_NAME: kangni-agents
  DOCKER_IMAGE_TAG: $CI_COMMIT_SHORT_SHA
```

### 3. 触发条件

管道将在以下情况下自动触发：
- 向 `main` 分支推送新的提交
- 创建合并请求到 `main` 分支

## 管道流程

### 阶段 1：构建 (Build)
- 安装必要的工具和依赖
- 构建 Docker 镜像
- 生成构建产物

### 阶段 2：推送 (Push)
- 配置阿里云认证
- 获取临时登录令牌
- 推送镜像到阿里云容器镜像服务

## 镜像标签策略

- **提交标签**: `kangni-agents:$CI_COMMIT_SHORT_SHA`
- **最新标签**: `kangni-agents:latest`

## 故障排除

### 常见问题

1. **认证失败**
   ```
   Error: authentication failed
   ```
   **解决方案：**
   - 检查 RAM Access Key 是否正确
   - 确认 RAM 用户具有容器镜像服务权限
   - 验证环境变量是否在 GitLab 中正确配置

2. **权限不足**
   ```
   Error: access denied
   ```
   **解决方案：**
   - 检查 RAM 用户权限策略
   - 确认具有 `cr:GetAuthorizationToken` 权限
   - 确认具有 `cr:PushRepository` 权限

3. **网络连接问题**
   ```
   Error: connection timeout
   ```
   **解决方案：**
   - 检查 GitLab Runner 网络连接
   - 确认可以访问阿里云服务
   - 检查防火墙设置

### 调试方法

1. **查看管道日志**
   - 进入 GitLab 项目
   - 点击 "CI/CD" → "Pipelines"
   - 查看失败的作业日志

2. **本地测试**
   ```bash
   # 测试 Docker 构建
   docker build -t kangni-agents:test .
   
   # 测试阿里云登录
   aliyun configure set --profile default --mode AK --region cn-hangzhou --access-key-id YOUR_ACCESS_KEY_ID --access-key-secret YOUR_ACCESS_KEY_SECRET
   aliyun cr GetAuthorizationToken --region cn-hangzhou
   ```

3. **验证环境变量**
   ```bash
   # 在 GitLab CI 作业中添加调试信息
   - echo "Access Key ID: $ALIBABA_ACCESS_KEY_ID"
   - echo "Registry: $ALIBABA_REGISTRY"
   ```

## 安全最佳实践

1. **使用受保护的环境变量**
   - 所有敏感信息都应标记为 "Protected"
   - 限制在受保护的分支中使用

2. **最小权限原则**
   - RAM 用户只分配必要的权限
   - 定期轮换访问密钥

3. **监控和审计**
   - 定期检查管道执行日志
   - 监控镜像推送活动

## 高级配置

### 多环境部署

可以扩展配置以支持多环境部署：

```yaml
# 添加环境特定的配置
deploy-staging:
  stage: deploy
  environment: staging
  script:
    - docker pull $ALIBABA_REGISTRY/$DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG
    - docker run -d --name kangni-agents-staging $ALIBABA_REGISTRY/$DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG
  only:
    - develop

deploy-production:
  stage: deploy
  environment: production
  script:
    - docker pull $ALIBABA_REGISTRY/$DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG
    - docker run -d --name kangni-agents-prod $ALIBABA_REGISTRY/$DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG
  only:
    - main
  when: manual
```

### 通知配置

添加 Slack 或邮件通知：

```yaml
notify-success:
  stage: notify
  script:
    - echo "Pipeline succeeded for commit $CI_COMMIT_SHA"
  only:
    - main
  when: on_success

notify-failure:
  stage: notify
  script:
    - echo "Pipeline failed for commit $CI_COMMIT_SHA"
  only:
    - main
  when: on_failure
```

## 监控和维护

1. **定期检查**
   - 监控管道执行时间
   - 检查镜像大小和构建效率

2. **清理策略**
   - 定期清理旧的镜像标签
   - 优化 Dockerfile 以减少构建时间

3. **更新维护**
   - 定期更新基础镜像
   - 更新依赖包版本

## 支持

如果遇到问题，请：
1. 查看 GitLab CI/CD 文档
2. 检查阿里云容器镜像服务文档
3. 联系系统管理员获取支持
