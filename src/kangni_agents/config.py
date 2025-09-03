from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # RAG设置
    ragflow_mcp_server_url: str = "http://158.58.50.45:9382/mcp"
    ragflow_default_dataset_id: str = "f3073258886911f08bc30242c0a82006"
    
    # 数据库相关RAG数据集IDs
    db_ddl_dataset_id: str = "10123e2487f211f0aeb40242c0a80006"
    db_query_sql_dataset_id: str = "ffcc7faa87f311f09d4a0242c0a80006"
    db_description_dataset_id: str = "6768e88087f211f0a8b00242c0a80006"
    
    # API设置
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # LLM设置
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    
    # 日志设置
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"

settings = Settings()