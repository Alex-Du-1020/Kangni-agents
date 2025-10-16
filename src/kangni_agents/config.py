from pydantic_settings import BaseSettings
from typing import Optional, List
from pydantic import Field

class Settings(BaseSettings):
    # RAG设置
    ragflow_mcp_server_url: str = Field(default="http://158.58.50.45:9382/mcp", alias="RAGFLOW_MCP_SERVER_URL")
    ragflow_dataset_ids: List[str] = Field(
        default=["e1c90b209ad311f09b6d0242c0a87006", "b32951fc9acf11f0a21c0242c0a87006"],
        alias="RAGFLOW_DATASET_IDS"
    )
    
    # 数据库相关RAG数据集IDs
    db_ddl_dataset_id: str = "2eeb6f2a9ac911f094c80242c0a85006"
    db_query_sql_dataset_id: str = "3387079e9acc11f0b60f0242c0a87006"
    db_description_dataset_id: str = "8c443ba09acb11f093460242c0a87006"
    
    # 数据库配置
    mysql_host: Optional[str] = "localhost"
    mysql_user: Optional[str] = None
    mysql_password: Optional[str] = None
    mysql_database: Optional[str] = None
    mysql_port: int = 3306
    
    # API设置
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # LLM设置 - 简化的配置
    llm_provider: str = Field(default="ollama", alias="LLM_PROVIDER")  # 可选: "deepseek", "openai", "alibaba", "ollama"
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: Optional[str] = Field(default=None, alias="OPENAI_MODEL")
    deepseek_api_key: Optional[str] = Field(default=None, alias="DEEPSEEK_API_KEY")
    alibaba_api_key: Optional[str] = Field(default=None, alias="ALIBABA_API_KEY")
    embedding_api_key: Optional[str] = Field(default=None, alias="EMBEDDING_API_KEY")
    
    # Ollama配置
    ollama_base_url: str = Field(default="http://158.158.4.66:4434/v1", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="gpt-oss-20b", alias="OLLAMA_MODEL")
    ollama_api_key: str = Field(default="test_api_key", alias="OLLAMA_API_KEY")
    
    # Minimum similarity score to consider a match
    similarity_threshold: Optional[float] = Field(default=0.3, alias="SIMILARITY_THRESHOLD")
  
    # Maximum number of suggestions per field
    max_suggestions: Optional[int] = Field(default=3, alias="MAX_SUGGESTIONS")
        
    # 日志设置
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    environment: str = Field(default="production", alias="ENVIRONMENT")
    
    class Config:
        env_file = ".env"
        populate_by_name = True
        extra = "ignore"  # Ignore extra fields from .env file
    
    def get_log_level(self) -> str:
        """根据环境获取日志级别"""
        return self.log_level.upper()

settings = Settings()