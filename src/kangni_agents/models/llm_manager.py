import asyncio
from typing import Dict, List, Optional, Union, AsyncGenerator, Type
from .llm_providers import (
    BaseLLMProvider, LLMProvider, LLMConfig, LLMMessage, LLMResponse,
    OpenAIConfig, DeepSeekConfig, AlibabaConfig, OllamaConfig, CONFIG_MAPPING
)
from .llm_implementations import (
    OpenAIProvider, DeepSeekProvider, AlibabaProvider, OllamaProvider, FallbackProvider
)
import logging

logger = logging.getLogger(__name__)

class LLMFactory:
    """LLM提供商工厂类"""
    
    # 提供商实现映射
    PROVIDER_IMPLEMENTATIONS: Dict[LLMProvider, Type[BaseLLMProvider]] = {
        LLMProvider.OPENAI: OpenAIProvider,
        LLMProvider.DEEPSEEK: DeepSeekProvider,
        LLMProvider.ALIBABA: AlibabaProvider,
        LLMProvider.OLLAMA: OllamaProvider,
    }
    
    @classmethod
    def create_provider(cls, config: LLMConfig) -> BaseLLMProvider:
        """创建LLM提供商实例"""
        provider_class = cls.PROVIDER_IMPLEMENTATIONS.get(config.provider)
        
        if not provider_class:
            logger.warning(f"Provider {config.provider} not implemented, using fallback")
            return FallbackProvider(config)
        
        try:
            return provider_class(config)
        except Exception as e:
            logger.error(f"Failed to create provider {config.provider}: {e}")
            return FallbackProvider(config)
    
    @classmethod
    def create_config(
        self,
        provider: Union[str, LLMProvider],
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ) -> LLMConfig:
        """创建LLM配置"""
        if isinstance(provider, str):
            provider = LLMProvider(provider)
        
        config_class = CONFIG_MAPPING.get(provider, LLMConfig)
        
        return config_class(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            **kwargs
        )
    
    @classmethod
    def get_supported_providers(cls) -> List[LLMProvider]:
        """获取支持的提供商列表"""
        return list(cls.PROVIDER_IMPLEMENTATIONS.keys())

class LLMManager:
    """LLM管理器：支持多个提供商和故障转移"""
    
    def __init__(self, configs: List[LLMConfig], enable_fallback: bool = True):
        """
        初始化LLM管理器
        
        Args:
            configs: LLM配置列表，按优先级排序
            enable_fallback: 是否启用故障转移
        """
        self.configs = configs
        self.enable_fallback = enable_fallback
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.primary_provider_key: Optional[str] = None
        
        # 初始化提供商
        self._initialize_providers()
    
    def _initialize_providers(self):
        """初始化所有提供商"""
        for i, config in enumerate(self.configs):
            provider_key = f"{config.provider}_{i}"
            provider = LLMFactory.create_provider(config)
            self.providers[provider_key] = provider
            
            if i == 0:
                self.primary_provider_key = provider_key
        
        logger.info(f"Initialized {len(self.providers)} LLM providers")
    
    async def chat(
        self,
        messages: List[LLMMessage],
        stream: bool = False,
        use_fallback: bool = True,
        **kwargs
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """
        聊天接口，支持故障转移
        
        Args:
            messages: 消息列表
            stream: 是否流式输出
            use_fallback: 是否使用故障转移
            **kwargs: 额外参数
        """
        
        # 首先尝试主要提供商
        if self.primary_provider_key:
            primary_provider = self.providers[self.primary_provider_key]
            
            try:
                if await primary_provider.is_available():
                    logger.info(f"Using primary provider: {primary_provider.config.provider}")
                    return await primary_provider.chat(messages, stream=stream, **kwargs)
                else:
                    logger.warning(f"Primary provider {primary_provider.config.provider} is not available")
            except Exception as e:
                logger.error(f"Primary provider {primary_provider.config.provider} failed: {e}")
        
        # 如果启用了故障转移，尝试其他提供商
        if use_fallback and self.enable_fallback:
            for provider_key, provider in self.providers.items():
                if provider_key == self.primary_provider_key:
                    continue
                
                try:
                    if await provider.is_available():
                        logger.info(f"Falling back to provider: {provider.config.provider}")
                        return await provider.chat(messages, stream=stream, **kwargs)
                except Exception as e:
                    logger.error(f"Fallback provider {provider.config.provider} failed: {e}")
        
        # 所有提供商都失败，返回错误响应
        error_message = "所有LLM服务都不可用，请稍后重试。"
        
        if stream:
            async def error_stream():
                for char in error_message:
                    yield char
                    await asyncio.sleep(0.01)
            return error_stream()
        else:
            return LLMResponse(
                content=error_message,
                model="error",
                provider=LLMProvider.OPENAI,  # 默认值
                finish_reason="error"
            )
    
    async def health_check(self) -> Dict[str, bool]:
        """检查所有提供商的健康状态"""
        health_status = {}
        
        tasks = []
        provider_keys = []
        
        for provider_key, provider in self.providers.items():
            tasks.append(provider.is_available())
            provider_keys.append(provider_key)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for provider_key, result in zip(provider_keys, results):
            provider = self.providers[provider_key]
            if isinstance(result, Exception):
                health_status[f"{provider.config.provider}"] = False
                logger.error(f"Health check failed for {provider.config.provider}: {result}")
            else:
                health_status[f"{provider.config.provider}"] = result
        
        return health_status
    
    def get_provider_info(self) -> List[Dict[str, any]]:
        """获取所有提供商信息"""
        info = []
        for provider_key, provider in self.providers.items():
            info.append({
                "provider_key": provider_key,
                "is_primary": provider_key == self.primary_provider_key,
                **provider.get_model_info()
            })
        return info
    
    def switch_primary_provider(self, provider_key: str) -> bool:
        """切换主要提供商"""
        if provider_key in self.providers:
            self.primary_provider_key = provider_key
            logger.info(f"Switched primary provider to: {provider_key}")
            return True
        return False
    
    def add_provider(self, config: LLMConfig) -> str:
        """动态添加提供商"""
        provider_key = f"{config.provider}_{len(self.providers)}"
        provider = LLMFactory.create_provider(config)
        self.providers[provider_key] = provider
        
        if not self.primary_provider_key:
            self.primary_provider_key = provider_key
        
        logger.info(f"Added new provider: {provider_key}")
        return provider_key
    
    def remove_provider(self, provider_key: str) -> bool:
        """移除提供商"""
        if provider_key in self.providers:
            del self.providers[provider_key]
            
            if provider_key == self.primary_provider_key:
                # 选择新的主要提供商
                self.primary_provider_key = next(iter(self.providers.keys()), None)
            
            logger.info(f"Removed provider: {provider_key}")
            return True
        return False

# 便捷函数
def create_simple_manager(
    provider: Union[str, LLMProvider],
    model_name: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs
) -> LLMManager:
    """创建简单的单提供商管理器"""
    config = LLMFactory.create_config(
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        **kwargs
    )
    return LLMManager([config])

def create_multi_provider_manager(configs: List[Dict]) -> LLMManager:
    """从配置字典列表创建多提供商管理器"""
    llm_configs = []
    
    for config_dict in configs:
        provider = config_dict.pop('provider')
        model_name = config_dict.pop('model_name')
        
        config = LLMFactory.create_config(
            provider=provider,
            model_name=model_name,
            **config_dict
        )
        llm_configs.append(config)
    
    return LLMManager(llm_configs)