#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Set development environment
export ENVIRONMENT=development
export LOG_LEVEL=INFO

# Check if .env file exists and load it
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    export $(grep -v '^#' .env | xargs)
fi

# Start the development server with hot reload
echo "Starting Kangni Agents development server..."
echo "Virtual environment: $(which python)"
echo "Python version: $(python --version)"
echo ""
echo "Server will be available at: http://localhost:8000"
echo "API documentation: http://localhost:8000/docs"
echo "Health check: http://localhost:8000/api/v1/health"
echo ""

# Run with uvicorn directly for development with hot reload
exec uvicorn src.kangni_agents.main:app --host 0.0.0.0 --port 8000 --reload