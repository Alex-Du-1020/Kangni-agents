import httpx
import json
import asyncio
from typing import List, Union, AsyncGenerator, Dict, Any
from .llm_providers import (
    BaseLLMProvider, LLMMessage, LLMResponse, LLMProvider,
    OpenAIConfig, DeepSeekConfig, AlibabaConfig
)
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