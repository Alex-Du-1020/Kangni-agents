from pydantic_settings import BaseSettings
from typing import Optional
from pydantic import Field

class Settings(BaseSettings):
    # RAG设置
    ragflow_mcp_server_url: str = "http://158.58.50.45:9382/mcp"
    ragflow_default_dataset_id: str = "f3073258886911f08bc30242c0a82006"
    
    # 数据库相关RAG数据集IDs
    db_ddl_dataset_id: str = "10123e2487f211f0aeb40242c0a80006"
    db_query_sql_dataset_id: str = "ffcc7faa87f311f09d4a0242c0a80006"
    db_description_dataset_id: str = "6768e88087f211f0a8b00242c0a80006"
    
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
    llm_provider: str = Field(default="deepseek", alias="LLM_PROVIDER")  # 可选: "deepseek", "openai", "alibaba", "ollama"
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    deepseek_api_key: Optional[str] = Field(default=None, alias="DEEPSEEK_API_KEY")
    alibaba_api_key: Optional[str] = Field(default=None, alias="ALIBABA_API_KEY")
    
    # 日志设置
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    environment: str = Field(default="production", alias="ENVIRONMENT")
    
    class Config:
        env_file = ".env"
        populate_by_name = True
    
    def get_log_level(self) -> str:
        """根据环境获取日志级别"""
        if self.environment.lower() == "development" or self.environment.lower() == "dev":
            return "DEBUG"
        return self.log_level.upper()

settings = Settings()