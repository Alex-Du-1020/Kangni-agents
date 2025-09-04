# Ollama Configuration Example
# 在环境变量中设置 Ollama 配置：

OLLAMA_MODEL_NAME=chatgpt-oss-20b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TEMPERATURE=0.7

# 或者在代码中直接配置：

from src.kangni_agents.models.llm_manager import LLMManager, LLMFactory
from src.kangni_agents.models.llm_providers import LLMProvider

# 创建 Ollama 配置
ollama_config = LLMFactory.create_config(
    provider=LLMProvider.OLLAMA,
    model_name="chatgpt-oss-20b",
    base_url="http://localhost:11434",
    temperature=0.7,
    timeout=120  # Ollama 本地模型可能需要更长时间
)

# 创建 LLM 管理器
llm_manager = LLMManager([ollama_config])

# 使用示例
async def test_ollama():
    from src.kangni_agents.models.llm_providers import LLMMessage
    
    messages = [
        LLMMessage(role="system", content="你是一个有用的助手"),
        LLMMessage(role="user", content="你好，请介绍一下你自己")
    ]
    
    # 检查服务状态
    health = await llm_manager.health_check()
    print(f"Ollama 健康状态: {health}")
    
    # 发送消息
    if health.get('ollama', False):
        response = await llm_manager.chat(messages)
        print(f"Ollama 响应: {response.content}")
    else:
        print("Ollama 服务不可用，请确保：")
        print("1. Ollama 服务正在运行 (ollama serve)")
        print("2. 模型已下载 (ollama pull chatgpt-oss-20b)")
        print("3. 服务地址正确 (默认 http://localhost:11434)")