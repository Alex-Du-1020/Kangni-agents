#!/usr/bin/env python3
"""
LLM连接测试脚本
用于诊断和测试不同LLM提供商的连接状态
"""

import asyncio
import pytest
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 手动加载.env文件
from dotenv import load_dotenv

# 查找并加载.env文件
env_path = Path(__file__).parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ 已加载.env文件: {env_path}")
else:
    print(f"⚠️  未找到.env文件: {env_path}")

from kangni_agents.models.llm_implementations import llm_service, CentralizedLLMService
from kangni_agents.models.llm_providers import LLMMessage
from kangni_agents.config import settings
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.mark.asyncio

async def test_llm_provider(provider_name: str):
    """测试指定的LLM提供商"""
    print(f"\n{'='*50}")
    print(f"测试LLM提供商: {provider_name.upper()}")
    print(f"{'='*50}")
    
    try:
        # 创建新的LLM服务实例并切换提供商
        test_service = CentralizedLLMService()
        success = test_service.switch_provider(provider_name)
        
        if not success:
            print(f"❌ 无法初始化 {provider_name} 提供商")
            return False
        
        print(f"✅ {provider_name} 提供商初始化成功")
        
        # 检查服务可用性
        available = await test_service.is_available()
        if not available:
            print(f"❌ {provider_name} 服务不可用")
            return False
        
        print(f"✅ {provider_name} 服务可用性检查通过")
        
        # 测试简单聊天
        test_messages = [
            LLMMessage(role="user", content="请回答: 1+1等于几?")
        ]
        
        print(f"🔄 正在测试 {provider_name} 聊天功能...")
        response = await test_service.chat(test_messages)
        
        print(f"✅ {provider_name} 聊天测试成功")
        print(f"📝 响应内容: {response.content[:100]}{'...' if len(response.content) > 100 else ''}")
        print(f"🏷️  模型: {response.model}")
        print(f"🔧 提供商: {response.provider}")
        
        return True
        
    except Exception as e:
        print(f"❌ {provider_name} 测试失败: {str(e)}")
        logger.exception(f"Error testing {provider_name}")
        return False

@pytest.mark.asyncio

async def test_current_config():
    """测试当前配置的LLM提供商"""
    print(f"\n{'='*50}")
    print("测试当前配置")
    print(f"{'='*50}")
    
    print(f"当前配置的提供商: {settings.llm_provider}")
    print(f"OpenAI API Key: {'✅ 已配置' if settings.openai_api_key else '❌ 未配置'}")
    print(f"DeepSeek API Key: {'✅ 已配置' if settings.deepseek_api_key else '❌ 未配置'}")
    print(f"Alibaba API Key: {'✅ 已配置' if settings.alibaba_api_key else '❌ 未配置'}")
    
    # 显示API key的前几位用于调试（不显示完整key）
    if settings.openai_api_key:
        print(f"OpenAI API Key (前8位): {settings.openai_api_key[:8]}...")
    if settings.deepseek_api_key:
        print(f"DeepSeek API Key (前8位): {settings.deepseek_api_key[:8]}...")
    if settings.alibaba_api_key:
        print(f"Alibaba API Key (前8位): {settings.alibaba_api_key[:8]}...")
    
    # 获取提供商信息
    provider_info = llm_service.get_provider_info()
    print(f"LLM服务状态: {'✅ 可用' if provider_info['available'] else '❌ 不可用'}")
    print(f"当前客户端类型: {provider_info['client_type']}")
    
    if provider_info['available']:
        try:
            # 测试简单对话
            test_messages = [
                LLMMessage(role="user", content="你好，请简单介绍一下你自己")
            ]
            
            print("🔄 正在测试当前配置的LLM...")
            response = await llm_service.chat(test_messages)
            print("✅ 当前LLM配置测试成功")
            print(f"📝 响应: {response.content[:200]}{'...' if len(response.content) > 200 else ''}")
            return True
            
        except Exception as e:
            print(f"❌ 当前LLM配置测试失败: {str(e)}")
            logger.exception("Current config test failed")
            return False
    else:
        print("❌ 当前LLM服务不可用")
        return False

async def main():
    """主测试函数"""
    print("🚀 开始LLM连接测试")
    print(f"Python路径: {sys.executable}")
    print(f"工作目录: {os.getcwd()}")
    print(f"项目根目录: {Path(__file__).parent.parent.parent}")

    # 首先测试当前配置
    current_success = await test_current_config()
    
    # 测试所有可用的提供商
    providers_to_test = []
    
    # 根据API key可用性决定测试哪些提供商
    if settings.deepseek_api_key:
        providers_to_test.append("deepseek")
    if settings.openai_api_key:
        providers_to_test.append("openai") 
    if settings.alibaba_api_key:
        providers_to_test.append("alibaba")
    
    # Ollama不需要API key，总是测试
    providers_to_test.append("ollama")
    
    print(f"\n将要测试的提供商: {providers_to_test}")
    
    results = {}
    for provider in providers_to_test:
        results[provider] = await test_llm_provider(provider)
    
    # 总结测试结果
    print(f"\n{'='*50}")
    print("测试结果总结")
    print(f"{'='*50}")
    
    working_providers = []
    failed_providers = []
    
    for provider, success in results.items():
        if success:
            working_providers.append(provider)
            print(f"✅ {provider.upper()}: 工作正常")
        else:
            failed_providers.append(provider)
            print(f"❌ {provider.upper()}: 连接失败")
    
    print(f"\n📊 可用提供商数量: {len(working_providers)}/{len(providers_to_test)}")
    
    if working_providers:
        print(f"✅ 推荐使用: {working_providers[0].upper()}")
        
        # 如果当前配置不工作但有其他可用选项，建议切换
        if not current_success and working_providers:
            print(f"\n💡 建议: 当前配置({settings.llm_provider})不可用，建议切换到 {working_providers[0]}")
            print(f"可以在.env文件中设置: LLM_PROVIDER={working_providers[0]}")
    else:
        print("❌ 没有可用的LLM提供商，请检查配置和网络连接")
    
    return len(working_providers) > 0

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)