FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY start_server.sh ./

# Create virtual environment
RUN python -m venv .venv

# Activate virtual environment and install dependencies
RUN . .venv/bin/activate && \
    pip install --upgrade pip && \
    pip install fastapi 'uvicorn[standard]' pydantic pydantic-settings python-dotenv && \
    pip install -e .

# Make startup script executable
RUN chmod +x start_server.sh

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default command to run the FastAPI server
CMD ["./start_server.sh"]