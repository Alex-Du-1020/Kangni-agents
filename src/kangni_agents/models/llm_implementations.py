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
            "stream": stream,
            "options": {
                "temperature": self.config.temperature,
                **self.config.extra_params
            },
            **kwargs
        }
        
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            if stream:
                return self._stream_chat(client, payload)
            else:
                response = await client.post(
                    f"{self.base_url}/v1/messages",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                
                return LLMResponse(
                    content=result["content"][0]["text"],
                    usage=result.get("usage"),
                    model=result["model"],
                    provider=self.config.provider,
                    finish_reason=result.get("stop_reason")
                )
    
    async def _stream_chat(self, client: httpx.AsyncClient, payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """流式聊天"""
        async with client.stream(
            "POST",
            f"{self.base_url}/v1/messages",
            headers=self.headers,
            json=payload
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    try:
                        chunk = json.loads(data)
                        if chunk["type"] == "content_block_delta":
                            if chunk["delta"].get("text"):
                                yield chunk["delta"]["text"]
                    except json.JSONDecodeError:
                        continue
    
    async def is_available(self) -> bool:
        """检查Alibaba服务是否可用"""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                # Alibaba没有public的health check endpoint，使用一个简单请求测试
                response = await client.post(
                    f"{self.base_url}/v1/messages",
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
        api_key = settings.openai_api_key or settings.deepseek_api_key
        if not api_key:
            logger.warning("No API key available, LLM features will be disabled")
            return
        
        try:
            # 根据配置选择LLM提供商，默认使用DeepSeek
            if "alibaba" in (settings.openai_base_url or "").lower() or "qwen" in (settings.openai_base_url or "").lower():
                llm_config = AlibabaConfig(
                    model_name="qwen-turbo",
                    api_key=api_key,
                    base_url=settings.openai_base_url,
                    temperature=0.1
                )
                self.llm_provider = LLMProvider.ALIBABA
                self.llm_client = AlibabaProvider(llm_config)
            elif "openai" in (settings.openai_base_url or "").lower() or "api.openai.com" in (settings.openai_base_url or ""):
                # 明确指定OpenAI时才使用
                llm_config = OpenAIConfig(
                    model_name="gpt-4.1-ca",
                    api_key=api_key,
                    base_url=settings.openai_base_url,
                    temperature=0.1
                )
                self.llm_provider = LLMProvider.OPENAI
                self.llm_client = OpenAIProvider(llm_config)
            else:
                # 默认使用DeepSeek
                llm_config = DeepSeekConfig(
                    model_name=settings.llm_chat_model or "deepseek-chat",
                    api_key=api_key,
                    base_url=settings.openai_base_url or "https://api.deepseek.com",
                    temperature=0.1
                )
                self.llm_provider = LLMProvider.DEEPSEEK
                self.llm_client = DeepSeekProvider(llm_config)
            
            self.llm_available = True
            logger.info(f"Centralized LLM service initialized with provider: {self.llm_provider}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize centralized LLM service: {e}")
            self.llm_available = False
            self.llm_client = None
    
    async def chat(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """统一的聊天接口"""
        if not self.llm_available or not self.llm_client:
            raise RuntimeError("LLM service not available")
        
        return await self.llm_client.chat(messages, **kwargs)
    
    async def chat_with_system_prompt(self, system_prompt: str, user_message: str, **kwargs) -> str:
        """便捷的聊天接口，自动构建系统提示"""
        if not self.llm_available or not self.llm_client:
            raise RuntimeError("LLM service not available")
        
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_message)
        ]
        
        response = await self.llm_client.chat(messages, **kwargs)
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
            "client_type": type(self.llm_client).__name__ if self.llm_client else None
        }


# 全局LLM服务实例
llm_service = CentralizedLLMService()