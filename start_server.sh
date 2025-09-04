#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Start the server
echo "Starting Kangni Agents server..."
echo "Virtual environment: $(which python)"
echo "Python version: $(python --version)"

# Run the FastAPI application
exec python -m src.kangni_agents.main