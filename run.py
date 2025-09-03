#!/usr/bin/env python3
"""
Kangni Agents启动脚本
"""

import sys
import os
import uvicorn

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from kangni_agents.config import settings

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Starting Kangni Agents Server")
    print(f"📍 Host: {settings.api_host}")
    print(f"🔌 Port: {settings.api_port}")
    print(f"🔍 RAG Server: {settings.ragflow_mcp_server_url}")
    print(f"📊 Log Level: {settings.log_level}")
    print("=" * 50)
    
    uvicorn.run(
        "kangni_agents.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )