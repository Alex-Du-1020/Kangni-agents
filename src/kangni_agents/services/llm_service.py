"""
LLM多提供商集成服务

这个服务类集成了新的LLM多提供商系统到现有的kangni-agents架构中。
可以通过配置文件或环境变量来配置多个LLM提供商。
"""

import os
from typing import List, Optional, Dict, Any
from ..config import settings
from ..models import (
    LLMManager, create_multi_provider_manager, LLMProvider, LLMMessage, LLMResponse
)
import logging

logger = logging.getLogger(__name__)

class LLMService:
    """LLM服务：集成多提供商支持到现有系统"""
    
    def __init__(self):
        self.manager: Optional[LLMManager] = None
        self._initialize_manager()
    
    def _initialize_manager(self):
        """初始化LLM管理器"""
        try:
            configs = self._load_llm_configs()
            if configs:
                self.manager = create_multi_provider_manager(configs)
                logger.info(f"LLM service initialized with {len(configs)} providers")
            else:
                logger.warning("No LLM configurations found, service will be disabled")
                self.manager = None
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {e}")
            self.manager = None
    
    def _load_llm_configs(self) -> List[Dict[str, Any]]:
        """从设置和环境变量加载LLM配置"""
        configs = []
        
        # OpenAI配置（优先级最高）
        if settings.openai_api_key:
            config = {
                "provider": LLMProvider.OPENAI,
                "model_name": settings.llm_chat_model or "gpt-3.5-turbo",
                "api_key": settings.openai_api_key,
                "temperature": 0.7
            }
            
            if settings.openai_base_url:
                config["base_url"] = settings.openai_base_url
            
            configs.append(config)
            logger.info("Added OpenAI configuration")
        
        # Anthropic配置
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            configs.append({
                "provider": LLMProvider.ANTHROPIC,
                "model_name": os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
                "api_key": anthropic_key,
                "temperature": 0.7
            })
            logger.info("Added Anthropic configuration")
        
        # DeepSeek配置
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        if deepseek_key:
            configs.append({
                "provider": LLMProvider.DEEPSEEK,
                "model_name": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                "api_key": deepseek_key,
                "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                "temperature": 0.7
            })
            logger.info("Added DeepSeek configuration")
        
        # Moonshot配置
        moonshot_key = os.getenv("MOONSHOT_API_KEY")
        if moonshot_key:
            configs.append({
                "provider": LLMProvider.MOONSHOT,
                "model_name": os.getenv("MOONSHOT_MODEL", "moonshot-v1-8k"),
                "api_key": moonshot_key,
                "base_url": os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1"),
                "temperature": 0.7
            })
            logger.info("Added Moonshot configuration")
        
        # 智谱AI配置
        zhipu_key = os.getenv("ZHIPU_API_KEY")
        if zhipu_key:
            configs.append({
                "provider": LLMProvider.ZHIPU,
                "model_name": os.getenv("ZHIPU_MODEL", "glm-4"),
                "api_key": zhipu_key,
                "temperature": 0.7
            })
            logger.info("Added ZhipuAI configuration")
        
        # Ollama配置（本地服务）
        if os.getenv("OLLAMA_ENABLED", "false").lower() == "true":
            configs.append({
                "provider": LLMProvider.OLLAMA,
                "model_name": os.getenv("OLLAMA_MODEL", "llama2"),
                "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                "temperature": 0.8
            })
            logger.info("Added Ollama configuration")
        
        return configs
    
    async def chat(
        self,
        messages: List[str],
        system_prompt: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> LLMResponse:
        """
        聊天接口
        
        Args:
            messages: 消息列表（字符串格式）
            system_prompt: 系统提示词
            stream: 是否流式输出
            **kwargs: 额外参数
        """
        if not self.manager:
            return LLMResponse(
                content="LLM服务未初始化或配置不正确",
                model="error",
                provider=LLMProvider.OPENAI,
                finish_reason="error"
            )
        
        # 转换消息格式
        llm_messages = []
        
        if system_prompt:
            llm_messages.append(LLMMessage(role="system", content=system_prompt))
        
        for i, msg in enumerate(messages):
            role = "user" if i % 2 == 0 else "assistant"
            llm_messages.append(LLMMessage(role=role, content=msg))
        
        # 如果最后一条不是用户消息，添加一条
        if llm_messages and llm_messages[-1].role != "user":
            llm_messages.append(LLMMessage(role="user", content=messages[-1] if messages else ""))
        
        return await self.manager.chat(llm_messages, stream=stream, **kwargs)
    
    async def simple_chat(self, question: str, system_prompt: Optional[str] = None) -> str:
        """简化的聊天接口，返回字符串回答"""
        response = await self.chat([question], system_prompt=system_prompt)
        return response.content
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        if not self.manager:
            return {
                "status": "disabled",
                "message": "LLM service not initialized",
                "providers": {}
            }
        
        provider_health = await self.manager.health_check()
        healthy_count = sum(1 for status in provider_health.values() if status)
        
        return {
            "status": "healthy" if healthy_count > 0 else "unhealthy",
            "total_providers": len(provider_health),
            "healthy_providers": healthy_count,
            "providers": provider_health
        }
    
    def get_service_info(self) -> Dict[str, Any]:
        """获取服务信息"""
        if not self.manager:
            return {
                "status": "disabled",
                "providers": []
            }
        
        return {
            "status": "enabled",
            "providers": self.manager.get_provider_info()
        }
    
    def is_available(self) -> bool:
        """检查服务是否可用"""
        return self.manager is not None
    
    async def reload_config(self) -> bool:
        """重新加载配置"""
        try:
            self._initialize_manager()
            return True
        except Exception as e:
            logger.error(f"Failed to reload LLM config: {e}")
            return False

# 全局LLM服务实例
llm_service = LLMService()