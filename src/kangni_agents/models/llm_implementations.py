import httpx
import json
import asyncio
from typing import List, Union, AsyncGenerator, Dict, Any, Optional
from .llm_providers import (
    BaseLLMProvider, LLMMessage, LLMResponse, LLMProvider,
    OpenAIConfig, DeepSeekConfig, AlibabaConfig, OllamaConfig
)
from ..config import settings
import logging

logger = logging.getLogger(__name__)

class LLMProviderConfig:
    """LLM提供商配置类"""
    
    # 从settings获取提供商选择
    @staticmethod
    def get_provider():
        return settings.llm_provider.lower()
    
    # 各提供商的配置
    DEEPSEEK_CONFIG = {
        "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-chat",
        "temperature": 0.1,
        "max_tokens": None,
        "timeout": 30
    }
    
    OPENAI_CONFIG = {
        "base_url": "https://api.chatanywhere.tech/v1",
        "model_name": "gpt-4.1-ca",
        "temperature": 0.1,
        "max_tokens": None,
        "timeout": 30,
        "organization": None
    }
    
    ALIBABA_CONFIG = {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_name": "qwen-turbo",
        "temperature": 0.1,
        "max_tokens": None,
        "timeout": 30
    }
    
    OLLAMA_CONFIG = {
        "base_url": "http://localhost:11434",
        "model_name": "gpt-oss:20b",
        "temperature": 0.1,
        "max_tokens": None,
        "timeout": 180  # Increased timeout for local Ollama
    }

class OpenAIProvider(BaseLLMProvider):
    """OpenAI提供商实现"""
    
    def __init__(self, config: OpenAIConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        if config.organization:
            self.headers["OpenAI-Organization"] = config.organization
    
    async def chat(
        self,
        messages: List[LLMMessage],
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """OpenAI聊天接口"""
        
        # 转换消息格式
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        payload = {
            "model": self.config.model_name,
            "messages": openai_messages,
            "temperature": self.config.temperature,
            "stream": stream,
            **self.config.extra_params,
            **kwargs
        }
        
        if self.config.max_tokens:
            payload["max_tokens"] = self.config.max_tokens
        
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            if stream:
                return self._stream_chat(client, payload)
            else:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                choice = result["choices"][0]
                
                return LLMResponse(
                    content=choice["message"]["content"],
                    usage=result.get("usage"),
                    model=result["model"],
                    provider=self.config.provider,
                    finish_reason=choice.get("finish_reason")
                )
    
    async def _stream_chat(self, client: httpx.AsyncClient, payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """流式聊天"""
        async with client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if chunk["choices"][0]["delta"].get("content"):
                            yield chunk["choices"][0]["delta"]["content"]
                    except json.JSONDecodeError:
                        continue
    
    async def is_available(self) -> bool:
        """检查OpenAI服务是否可用"""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers=self.headers
                )
                return response.status_code == 200
        except Exception as e:
            self.logger.warning(f"OpenAI service unavailable: {e}")
            return False

class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek提供商实现"""
    
    def __init__(self, config: DeepSeekConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://api.deepseek.com"
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
    
    async def chat(
        self,
        messages: List[LLMMessage],
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """DeepSeek聊天接口（类似OpenAI格式）"""
        
        deepseek_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        payload = {
            "model": self.config.model_name,
            "messages": deepseek_messages,
            "temperature": self.config.temperature,
            "stream": stream,
            **self.config.extra_params,
            **kwargs
        }
        
        if self.config.max_tokens:
            payload["max_tokens"] = self.config.max_tokens
        
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            
            if stream:
                return self._parse_stream_response(response)
            else:
                result = response.json()
                choice = result["choices"][0]
                
                return LLMResponse(
                    content=choice["message"]["content"],
                    usage=result.get("usage"),
                    model=result["model"],
                    provider=self.config.provider,
                    finish_reason=choice.get("finish_reason")
                )
    
    async def _parse_stream_response(self, response) -> AsyncGenerator[str, None]:
        """解析流式响应"""
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    if chunk["choices"][0]["delta"].get("content"):
                        yield chunk["choices"][0]["delta"]["content"]
                except json.JSONDecodeError:
                    continue
    
    async def is_available(self) -> bool:
        """检查DeepSeek服务是否可用"""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers=self.headers
                )
                return response.status_code == 200
        except Exception as e:
            self.logger.warning(f"DeepSeek service unavailable: {e}")
            return False

class AlibabaProvider(BaseLLMProvider):
    """Alibaba提供商实现"""
    
    def __init__(self, config: AlibabaConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://api.alibaba.com"
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
    
    async def chat(
        self,
        messages: List[LLMMessage],
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """Alibaba聊天接口"""
        
        alibaba_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        payload = {
            "model": self.config.model_name,
            "messages": alibaba_messages,
            "temperature": self.config.temperature,
            "stream": stream,
            **self.config.extra_params,
            **kwargs
        }
        
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            if stream:
                return self._stream_chat(client, payload)
            else:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                choice = result["choices"][0]
                
                return LLMResponse(
                    content=choice["message"]["content"],
                    usage=result.get("usage"),
                    model=result["model"],
                    provider=self.config.provider,
                    finish_reason=choice.get("finish_reason")
                )
    
    async def _stream_chat(self, client: httpx.AsyncClient, payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """流式聊天"""
        async with client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if chunk["choices"][0]["delta"].get("content"):
                            yield chunk["choices"][0]["delta"]["content"]
                    except json.JSONDecodeError:
                        continue
    
    async def is_available(self) -> bool:
        """检查Alibaba服务是否可用"""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                # Use a simple test message to check availability
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json={
                        "model": self.config.model_name,
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 1
                    }
                )
                return response.status_code in [200, 400]  # 400也表示服务可用，只是请求格式问题
        except Exception as e:
            self.logger.warning(f"Alibaba service unavailable: {e}")
            return False

class OllamaProvider(BaseLLMProvider):
    """Ollama本地大模型提供商实现"""
    
    def __init__(self, config: OllamaConfig):
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:11434"
        self.headers = {
            "Content-Type": "application/json"
        }
    
    async def chat(
        self,
        messages: List[LLMMessage],
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """Ollama聊天接口"""
        
        # 转换消息格式
        ollama_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        payload = {
            "model": self.config.model_name,
            "messages": ollama_messages,
            "stream": stream,
            "options": {
                "temperature": self.config.temperature,
                **self.config.extra_params
            },
            **kwargs
        }
        
        if self.config.max_tokens:
            payload["options"]["num_predict"] = self.config.max_tokens
        
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            if stream:
                return self._stream_chat(client, payload)
            else:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                
                return LLMResponse(
                    content=result["message"]["content"],
                    usage={
                        "prompt_tokens": result.get("prompt_eval_count", 0),
                        "completion_tokens": result.get("eval_count", 0),
                        "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                    },
                    model=result.get("model", self.config.model_name),
                    provider=self.config.provider,
                    finish_reason=result.get("done_reason", "stop")
                )
    
    async def _stream_chat(self, client: httpx.AsyncClient, payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """流式聊天"""
        async with client.stream(
            "POST",
            f"{self.base_url}/api/chat",
            headers=self.headers,
            json=payload
        ) as response:
            async for line in response.aiter_lines():
                if line.strip():
                    try:
                        chunk = json.loads(line)
                        if "message" in chunk and "content" in chunk["message"]:
                            content = chunk["message"]["content"]
                            if content:
                                yield content
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
    
    async def is_available(self) -> bool:
        """检查Ollama服务是否可用"""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                # 检查Ollama服务状态
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code != 200:
                    return False
                
                # 检查模型是否存在
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]
                
                # 检查目标模型是否可用
                return any(self.config.model_name in name for name in model_names)
                
        except Exception as e:
            self.logger.warning(f"Ollama service unavailable: {e}")
            return False

class FallbackProvider(BaseLLMProvider):
    """降级提供商：当所有服务都不可用时使用"""
    
    async def chat(
        self,
        messages: List[LLMMessage],
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """降级聊天响应"""
        content = f"抱歉，当前所有LLM服务都不可用。您的问题是：{messages[-1].content if messages else '未知'}。请稍后重试。"
        
        if stream:
            async def stream_fallback():
                for char in content:
                    yield char
                    await asyncio.sleep(0.01)
            return stream_fallback()
        else:
            return LLMResponse(
                content=content,
                model="fallback",
                provider=self.config.provider,
                finish_reason="fallback"
            )
    
    async def is_available(self) -> bool:
        """降级提供商始终可用"""
        return True


class CentralizedLLMService:
    """集中式LLM服务，统一管理所有LLM通信"""
    
    def __init__(self):
        self.llm_provider = None
        self.llm_client = None
        self.llm_available = False
        self._initialize_llm()
    
    def _initialize_llm(self):
        """初始化LLM客户端"""
        try:
            # 根据settings中的配置选择LLM提供商
            provider = LLMProviderConfig.get_provider()
            
            if provider == "deepseek":
                api_key = settings.deepseek_api_key
                if not api_key:
                    logger.warning("DeepSeek API key not available")
                    return
                    
                config = LLMProviderConfig.DEEPSEEK_CONFIG
                llm_config = DeepSeekConfig(
                    model_name=config["model_name"],
                    api_key=api_key,
                    base_url=config["base_url"],
                    temperature=config["temperature"],
                    max_tokens=config["max_tokens"],
                    timeout=config["timeout"]
                )
                self.llm_provider = LLMProvider.DEEPSEEK
                self.llm_client = DeepSeekProvider(llm_config)
                
            elif provider == "openai":
                api_key = settings.openai_api_key
                if not api_key:
                    logger.warning("OpenAI API key not available")
                    return
                    
                config = LLMProviderConfig.OPENAI_CONFIG
                llm_config = OpenAIConfig(
                    model_name=config["model_name"],
                    api_key=api_key,
                    base_url=config["base_url"],
                    temperature=config["temperature"],
                    max_tokens=config["max_tokens"],
                    timeout=config["timeout"],
                    organization=config.get("organization")
                )
                self.llm_provider = LLMProvider.OPENAI
                self.llm_client = OpenAIProvider(llm_config)
                
            elif provider == "alibaba":
                api_key = settings.alibaba_api_key
                if not api_key:
                    logger.warning("Alibaba API key not available")
                    return
                    
                config = LLMProviderConfig.ALIBABA_CONFIG
                llm_config = AlibabaConfig(
                    model_name=config["model_name"],
                    api_key=api_key,
                    base_url=config["base_url"],
                    temperature=config["temperature"],
                    max_tokens=config["max_tokens"],
                    timeout=config["timeout"]
                )
                self.llm_provider = LLMProvider.ALIBABA
                self.llm_client = AlibabaProvider(llm_config)
                
            elif provider == "ollama":
                # Ollama不需要API key，使用本地服务
                config = LLMProviderConfig.OLLAMA_CONFIG
                llm_config = OllamaConfig(
                    model_name=config["model_name"],
                    api_key="",  # Ollama不需要API key
                    base_url=config["base_url"],
                    temperature=config["temperature"],
                    max_tokens=config["max_tokens"],
                    timeout=config["timeout"]
                )
                self.llm_provider = LLMProvider.OLLAMA
                self.llm_client = OllamaProvider(llm_config)
                
            else:
                logger.warning(f"Unknown provider: {provider}, falling back to Ollama")
                # 回退到Ollama
                config = LLMProviderConfig.OLLAMA_CONFIG
                llm_config = OllamaConfig(
                    model_name=config["model_name"],
                    api_key="",
                    base_url=config["base_url"],
                    temperature=config["temperature"],
                    max_tokens=config["max_tokens"],
                    timeout=config["timeout"]
                )
                self.llm_provider = LLMProvider.OLLAMA
                self.llm_client = OllamaProvider(llm_config)
            
            self.llm_available = True
            logger.info(f"Centralized LLM service initialized with provider: {self.llm_provider} (model: {config['model_name']})")
            
        except Exception as e:
            logger.warning(f"Failed to initialize centralized LLM service: {e}")
            self.llm_available = False
            self.llm_client = None
    
    async def chat(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """统一的聊天接口"""
        if not self.llm_available or not self.llm_client:
            # Instead of raising error, try to reinitialize
            logger.warning("LLM service not available, attempting to reinitialize...")
            self._initialize_llm()
            
            if not self.llm_available or not self.llm_client:
                # Still not available, use fallback
                fallback = FallbackProvider(None)
                return await fallback.chat(messages, **kwargs)
        
        try:
            return await self.llm_client.chat(messages, **kwargs)
        except Exception as e:
            logger.error(f"LLM chat error with {self.llm_provider}: {e}")
            # Try to fallback to another provider
            return await self._fallback_chat(messages, **kwargs)
    
    async def _fallback_chat(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """尝试使用其他可用的提供商"""
        current_provider = LLMProviderConfig.get_provider()
        fallback_providers = ["deepseek", "openai", "alibaba"]
        
        # Remove current provider from fallback list
        if current_provider in fallback_providers:
            fallback_providers.remove(current_provider)
        
        for provider in fallback_providers:
            try:
                # Check if we have the necessary API key
                if provider == "deepseek" and not settings.deepseek_api_key:
                    continue
                elif provider == "openai" and not settings.openai_api_key:
                    continue
                elif provider == "alibaba" and not settings.alibaba_api_key:
                    continue
                
                logger.info(f"Trying fallback provider: {provider}")
                temp_service = CentralizedLLMService()
                success = temp_service.switch_provider(provider)
                
                if success and await temp_service.is_available():
                    return await temp_service.chat(messages, **kwargs)
                    
            except Exception as e:
                logger.warning(f"Fallback provider {provider} failed: {e}")
                continue
        
        # All providers failed, use fallback response
        logger.error("All LLM providers failed, using fallback response")
        fallback = FallbackProvider(None)
        return await fallback.chat(messages, **kwargs)
    
    
    async def chat_with_system_prompt(self, system_prompt: str, user_message: str, **kwargs) -> str:
        """便捷的聊天接口，自动构建系统提示"""
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_message)
        ]
        
        response = await self.chat(messages, **kwargs)
        return response.content
    
    async def is_available(self) -> bool:
        """检查LLM服务是否可用"""
        if not self.llm_available or not self.llm_client:
            return False
        
        try:
            return await self.llm_client.is_available()
        except Exception as e:
            logger.warning(f"LLM availability check failed: {e}")
            return False
    
    def get_provider_info(self) -> Dict[str, Any]:
        """获取提供商信息"""
        return {
            "provider": self.llm_provider,
            "available": self.llm_available,
            "client_type": type(self.llm_client).__name__ if self.llm_client else None,
            "current_config": LLMProviderConfig.get_provider()
        }
    
    def switch_provider(self, provider: str) -> bool:
        """动态切换LLM提供商"""
        valid_providers = ["deepseek", "openai", "alibaba", "ollama"]
        if provider.lower() not in valid_providers:
            logger.error(f"Invalid provider: {provider}. Valid options: {valid_providers}")
            return False
        
        # 更新settings中的配置
        settings.llm_provider = provider.lower()
        
        # 重新初始化
        self._initialize_llm()
        
        logger.info(f"Switched to provider: {provider}")
        return self.llm_available
    
    def get_available_providers(self) -> List[str]:
        """获取可用的提供商列表"""
        return ["deepseek", "openai", "alibaba", "ollama"]
    
    def get_provider_config(self, provider: str = None) -> Dict[str, Any]:
        """获取指定提供商的配置"""
        if provider is None:
            provider = LLMProviderConfig.get_provider()
        
        provider = provider.lower()
        if provider == "deepseek":
            return LLMProviderConfig.DEEPSEEK_CONFIG
        elif provider == "openai":
            return LLMProviderConfig.OPENAI_CONFIG
        elif provider == "alibaba":
            return LLMProviderConfig.ALIBABA_CONFIG
        elif provider == "ollama":
            return LLMProviderConfig.OLLAMA_CONFIG
        else:
            return {}


# 全局LLM服务实例
llm_service = CentralizedLLMService()