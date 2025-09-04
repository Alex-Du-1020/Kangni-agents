"""
LLM多提供商使用示例

这个文件展示了如何使用新的LLM多提供商系统。
支持OpenAI、Anthropic、Ollama、DeepSeek等多种LLM提供商，
并且具有自动故障转移功能。
"""

import asyncio
import os
from src.kangni_agents.models import (
    LLMManager, LLMFactory, create_simple_manager, create_multi_provider_manager,
    LLMProvider, LLMMessage, OpenAIConfig, AnthropicConfig, OllamaConfig
)

async def example_simple_usage():
    """简单使用示例：单一提供商"""
    print("=== 简单使用示例 ===")
    
    # 创建简单的OpenAI管理器
    manager = create_simple_manager(
        provider=LLMProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7
    )
    
    # 准备消息
    messages = [
        LLMMessage(role="user", content="你好，请介绍一下自己")
    ]
    
    # 发送聊天请求
    try:
        response = await manager.chat(messages)
        print(f"回答: {response.content}")
        print(f"模型: {response.model}")
        print(f"提供商: {response.provider}")
    except Exception as e:
        print(f"请求失败: {e}")

async def example_multi_provider():
    """多提供商使用示例：支持故障转移"""
    print("\n=== 多提供商示例 ===")
    
    # 配置多个提供商（按优先级排序）
    configs = [
        {
            "provider": LLMProvider.OPENAI,
            "model_name": "gpt-3.5-turbo",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": 0.7
        },
        {
            "provider": LLMProvider.ANTHROPIC,
            "model_name": "claude-3-sonnet-20240229",
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "temperature": 0.7
        },
        {
            "provider": LLMProvider.OLLAMA,
            "model_name": "llama2",
            "base_url": "http://localhost:11434"
        }
    ]
    
    manager = create_multi_provider_manager(configs)
    
    # 检查服务健康状态
    health_status = await manager.health_check()
    print("服务状态:")
    for provider, is_healthy in health_status.items():
        status = "✓ 可用" if is_healthy else "✗ 不可用"
        print(f"  {provider}: {status}")
    
    # 发送聊天请求
    messages = [
        LLMMessage(role="user", content="请用中文简单介绍一下人工智能")
    ]
    
    try:
        response = await manager.chat(messages)
        print(f"\n使用的提供商: {response.provider}")
        print(f"回答: {response.content[:200]}...")
    except Exception as e:
        print(f"请求失败: {e}")

async def example_streaming():
    """流式输出示例"""
    print("\n=== 流式输出示例 ===")
    
    manager = create_simple_manager(
        provider=LLMProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    messages = [
        LLMMessage(role="user", content="请讲一个简短的故事")
    ]
    
    try:
        stream = await manager.chat(messages, stream=True)
        print("流式回答:")
        async for chunk in stream:
            print(chunk, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"流式请求失败: {e}")

async def example_advanced_config():
    """高级配置示例"""
    print("\n=== 高级配置示例 ===")
    
    # 手动创建配置
    openai_config = OpenAIConfig(
        model_name="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.3,
        max_tokens=1000,
        extra_params={"presence_penalty": 0.1}
    )
    
    ollama_config = OllamaConfig(
        model_name="llama2",
        base_url="http://localhost:11434",
        temperature=0.8
    )
    
    # 创建管理器
    manager = LLMManager([openai_config, ollama_config])
    
    # 获取提供商信息
    provider_info = manager.get_provider_info()
    print("配置的提供商:")
    for info in provider_info:
        primary = " (主要)" if info["is_primary"] else ""
        print(f"  {info['provider']} - {info['model_name']}{primary}")
    
    # 测试聊天
    messages = [
        LLMMessage(role="system", content="你是一个友好的AI助手"),
        LLMMessage(role="user", content="解释一下什么是机器学习")
    ]
    
    try:
        response = await manager.chat(messages)
        print(f"\n回答摘要: {response.content[:100]}...")
    except Exception as e:
        print(f"请求失败: {e}")

async def example_provider_management():
    """提供商管理示例"""
    print("\n=== 提供商管理示例 ===")
    
    # 创建初始管理器
    manager = create_simple_manager(
        provider=LLMProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    print(f"初始提供商数量: {len(manager.providers)}")
    
    # 动态添加新提供商
    ollama_config = OllamaConfig(
        model_name="llama2",
        base_url="http://localhost:11434"
    )
    
    new_provider_key = manager.add_provider(ollama_config)
    print(f"添加新提供商: {new_provider_key}")
    print(f"当前提供商数量: {len(manager.providers)}")
    
    # 切换主要提供商
    success = manager.switch_primary_provider(new_provider_key)
    if success:
        print(f"已切换主要提供商为: {new_provider_key}")
    
    # 移除提供商
    removed = manager.remove_provider(new_provider_key)
    if removed:
        print(f"已移除提供商: {new_provider_key}")
        print(f"剩余提供商数量: {len(manager.providers)}")

def example_configuration_from_env():
    """从环境变量读取配置的示例"""
    print("\n=== 环境变量配置示例 ===")
    
    # 从环境变量读取配置
    configs = []
    
    # OpenAI配置
    if os.getenv("OPENAI_API_KEY"):
        configs.append({
            "provider": LLMProvider.OPENAI,
            "model_name": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("OPENAI_BASE_URL"),
            "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
        })
    
    # Anthropic配置
    if os.getenv("ANTHROPIC_API_KEY"):
        configs.append({
            "provider": LLMProvider.ANTHROPIC,
            "model_name": os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "temperature": float(os.getenv("ANTHROPIC_TEMPERATURE", "0.7"))
        })
    
    # Ollama配置（本地服务，无需API密钥）
    if os.getenv("OLLAMA_ENABLED", "false").lower() == "true":
        configs.append({
            "provider": LLMProvider.OLLAMA,
            "model_name": os.getenv("OLLAMA_MODEL", "llama2"),
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        })
    
    if configs:
        manager = create_multi_provider_manager(configs)
        print(f"从环境变量创建了包含 {len(configs)} 个提供商的管理器")
        
        # 显示配置信息
        for info in manager.get_provider_info():
            print(f"  - {info['provider']}: {info['model_name']}")
    else:
        print("未找到有效的LLM配置，请设置相应的环境变量")
        print("支持的环境变量:")
        print("  - OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL")
        print("  - ANTHROPIC_API_KEY, ANTHROPIC_MODEL")
        print("  - OLLAMA_ENABLED, OLLAMA_MODEL, OLLAMA_BASE_URL")

async def main():
    """主函数：运行所有示例"""
    print("LLM多提供商系统使用示例")
    print("=" * 50)
    
    # 运行各种示例
    await example_simple_usage()
    await example_multi_provider()
    await example_streaming()
    await example_advanced_config()
    await example_provider_management()
    example_configuration_from_env()
    
    print("\n示例完成！")

if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())