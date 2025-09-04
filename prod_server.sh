#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Set production environment
export ENVIRONMENT=production
export LOG_LEVEL=INFO

# Check if .env file exists and load it
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    export $(grep -v '^#' .env | xargs)
fi

# Start the production server
echo "Starting Kangni Agents production server..."
echo "Virtual environment: $(which python)"
echo "Python version: $(python --version)"
echo ""
echo "Server will be available at: http://localhost:8000"
echo "API documentation: http://localhost:8000/docs"
echo "Health check: http://localhost:8000/api/v1/health"
echo ""

# Run with uvicorn for production
exec uvicorn src.kangni_agents.main:app --host 0.0.0.0 --port 8000 --workers 4
