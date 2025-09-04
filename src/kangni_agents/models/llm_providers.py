from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncGenerator, Union
from pydantic import BaseModel, Field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class LLMProvider(str, Enum):
    """支持的LLM提供商"""
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    ALIBABA = "alibaba"

class LLMConfig(BaseModel):
    """LLM配置基类"""
    provider: LLMProvider
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    timeout: int = Field(default=60, gt=0)
    extra_params: Dict[str, Any] = Field(default_factory=dict)

class OpenAIConfig(LLMConfig):
    """OpenAI配置"""
    provider: LLMProvider = LLMProvider.OPENAI
    model_name: str = "gpt-4.1-ca"
    organization: Optional[str] = None

class DeepSeekConfig(LLMConfig):
    """DeepSeek配置"""
    provider: LLMProvider = LLMProvider.DEEPSEEK
    model_name: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"

class AlibabaConfig(LLMConfig):
    """阿里云通义千问配置"""
    provider: LLMProvider = LLMProvider.ALIBABA
    model_name: str = "qwen-turbo"
    access_key_id: Optional[str] = None
    access_key_secret: Optional[str] = None

class LLMMessage(BaseModel):
    """LLM消息"""
    role: str  # system, user, assistant
    content: str

class LLMResponse(BaseModel):
    """LLM响应"""
    content: str
    usage: Optional[Dict[str, int]] = None
    model: str
    provider: LLMProvider
    finish_reason: Optional[str] = None

class BaseLLMProvider(ABC):
    """LLM提供商基类"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.provider}")
    
    @abstractmethod
    async def chat(
        self,
        messages: List[LLMMessage],
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """聊天接口"""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """检查服务是否可用"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "provider": self.config.provider,
            "model_name": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }

# 配置类映射
CONFIG_MAPPING = {
    LLMProvider.OPENAI: OpenAIConfig,
    LLMProvider.DEEPSEEK: DeepSeekConfig,
    LLMProvider.ALIBABA: AlibabaConfig,
}