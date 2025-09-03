FROM python:3.11-slim

WORKDIR /app

# 复制项目文件
COPY pyproject.toml ./
COPY src/ ./src/
COPY test_rag.py ./

# 安装依赖
RUN pip install -e .

# 设置环境变量
ENV PYTHONPATH=/app/src

# 默认命令
CMD ["python", "test_rag.py"]