# LLM多提供商系统使用指南

本系统支持多个大模型提供商，具有自动故障转移功能，方便添加新的大模型支持。

## 支持的提供商

### 已实现的提供商
- **OpenAI** - GPT系列模型
- **Ollama** - 本地模型服务
- **DeepSeek** - DeepSeek系列模型

### 预留的提供商接口
- **百度文心一言** (Baidu)
- **阿里云通义千问** (Alibaba)  
- **智谱AI** (ZhipuAI)
- **月之暗面** (Moonshot)
- **Google** (Gemini)
- **HuggingFace**

## 快速开始

### 1. 环境变量配置

创建 `.env` 文件或设置环境变量：

```bash
# OpenAI配置
OPENAI_API_KEY=your_openai_api_key
LLM_CHAT_MODEL=gpt-3.5-turbo
LLM_BASE_URL=https://api.openai.com/v1

# DeepSeek配置
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_MODEL=deepseek-chat

# Moonshot配置
MOONSHOT_API_KEY=your_moonshot_api_key
MOONSHOT_MODEL=moonshot-v1-8k

# 智谱AI配置
ZHIPU_API_KEY=your_zhipu_api_key
ZHIPU_MODEL=glm-4

# Ollama配置（本地服务）
OLLAMA_ENABLED=true
OLLAMA_MODEL=llama2
OLLAMA_BASE_URL=http://localhost:11434
```

### 2. 基本使用

```python
from src.kangni_agents.models.llm_implementations import llm_service

# 简单聊天
answer = await llm_service.chat_with_system_prompt(
    system_prompt="你是一个有用的AI助手",
    human_prompt="你好，请介绍一下自己"
)
print(answer)

# 带系统提示的聊天
answer = await llm_service.chat_with_system_prompt(
    system_prompt="你是一个专业的AI技术专家",
    human_prompt="解释什么是机器学习"
)
print(answer)

# 多轮对话
from src.kangni_agents.models.llm_providers import LLMMessage
messages = [
    LLMMessage(role="user", content="你好"),
    LLMMessage(role="assistant", content="你好！有什么可以帮助您的吗？"),
    LLMMessage(role="user", content="请解释一下人工智能")
]
response = await llm_service.chat(messages)
print(response.content)
```

### 3. 高级使用

```python
from src.kangni_agents.models import (
    LLMProvider, LLMMessage, OpenAIConfig
)

# 使用现有的集中式LLM服务
from src.kangni_agents.models.llm_implementations import llm_service

# 检查服务状态
provider_info = llm_service.get_provider_info()
print("服务状态:", provider_info)

# 发送请求（自动故障转移）
messages = [
    LLMMessage(role="user", content="你好")
]
response = await llm_service.chat(messages)
print(f"使用的提供商: {response.provider}")
print(f"回答: {response.content}")

# 流式输出
stream = await llm_service.chat(messages, stream=True)
async for chunk in stream:
    print(chunk, end="", flush=True)
```

## API接口

### 健康检查接口

```bash
GET /qomo/v1/llm/health
```

返回：
```json
{
  "status": "healthy",
  "total_providers": 3,
  "healthy_providers": 2,
  "providers": {
    "openai": true,
    "ollama": false
  }
}
```

### 服务信息接口

```bash
GET /qomo/v1/llm/info
```

返回：
```json
{
  "status": "enabled",
  "providers": [
    {
      "provider_key": "openai_0",
      "is_primary": true,
      "provider": "openai",
      "model_name": "gpt-3.5-turbo",
      "temperature": 0.7
    }
  ]
}
```

## 添加新的提供商

### 1. 创建配置类

在 `llm_providers.py` 中添加新的配置类：

```python
class NewProviderConfig(LLMConfig):
    """新提供商配置"""
    provider: LLMProvider = LLMProvider.NEW_PROVIDER
    model_name: str = "default-model"
    custom_param: Optional[str] = None
```

### 2. 创建实现类

在 `llm_implementations.py` 中添加实现：

```python
class NewProvider(BaseLLMProvider):
    """新提供商实现"""
    
    def __init__(self, config: NewProviderConfig):
        super().__init__(config)
        # 初始化逻辑
    
    async def chat(self, messages: List[LLMMessage], stream: bool = False, **kwargs):
        # 实现聊天逻辑
        pass
    
    async def is_available(self) -> bool:
        # 实现健康检查
        pass
```

### 3. 注册提供商

在 `CentralizedLLMService` 中注册：

```python
PROVIDER_IMPLEMENTATIONS = {
    # 现有提供商...
    LLMProvider.NEW_PROVIDER: NewProvider,
}
```

## 配置示例

### OpenAI兼容接口

```python
# 使用其他OpenAI兼容的服务
config = {
    "provider": LLMProvider.OPENAI,
    "model_name": "custom-model",
    "api_key": "your_api_key",
    "base_url": "https://your-custom-endpoint.com/v1",
    "temperature": 0.7
}
```

### 本地Ollama服务

```python
# 连接本地Ollama服务
config = {
    "provider": LLMProvider.OLLAMA,
    "model_name": "llama2",
    "base_url": "http://localhost:11434",
    "temperature": 0.8
}
```

### 故障转移配置

```python
# 按优先级配置多个提供商
configs = [
    # 主要提供商
    {"provider": "openai", "model_name": "gpt-4", "api_key": "key1"},
    # 本地备用
    {"provider": "ollama", "model_name": "llama2", "base_url": "http://localhost:11434"}
]
```

## 最佳实践

### 1. 提供商选择策略

- **生产环境**：配置2-3个不同的云服务提供商作为备用
- **开发环境**：可以使用本地Ollama服务节省费用
- **特定任务**：根据模型特长选择合适的提供商

### 2. 错误处理

```python
try:
    response = await llm_service.chat_with_system_prompt(
        system_prompt="你是一个有用的AI助手",
        human_prompt="你好"
    )
    print(response)
except Exception as e:
    logger.error(f"LLM请求失败: {e}")
    # 处理错误或使用降级方案
```

### 3. 性能优化

- 合理设置 `temperature` 和 `max_tokens`
- 使用流式输出改善用户体验
- 实施请求缓存减少API调用

### 4. 监控和日志

```python
# 检查服务健康状态
provider_info = llm_service.get_provider_info()
if not provider_info['available']:
    # 发送告警
    pass

# 记录使用统计
logger.info(f"使用了提供商: {response.provider}, 消耗tokens: {response.usage}")
```

## 故障排除

### 常见问题

1. **API密钥错误**：检查环境变量设置
2. **网络连接问题**：检查base_url和防火墙设置  
3. **模型不存在**：确认模型名称正确
4. **配额超限**：检查API使用情况

### 调试方法

```python
# 启用详细日志
import logging
logging.getLogger("src.kangni_agents.models").setLevel(logging.DEBUG)

# 检查服务状态
provider_info = llm_service.get_provider_info()
print("服务信息:", provider_info)
```

## 扩展开发

这个系统设计为高度可扩展，您可以：

1. **添加新的提供商支持**
2. **实现自定义负载均衡策略**
3. **添加请求缓存和限流**
4. **集成更多的模型参数和功能**
5. **实现模型性能监控和统计**

有关更多技术细节，请参考源码中的注释和示例。