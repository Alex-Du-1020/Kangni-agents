#!/bin/bash
# Development helper script for Kangni Agents project

# Always activate virtual environment first
echo "Activating virtual environment..."
source .venv/bin/activate

# Check if command is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <command>"
    echo "Available commands:"
    echo "  test          - Run all tests"
    echo "  test-vector   - Run vector embedding tests"
    echo "  test-workflow - Run workflow tests"
    echo "  server        - Start development server"
    echo "  install       - Install/update dependencies"
    echo "  lint          - Run linting"
    exit 1
fi

COMMAND=$1

case $COMMAND in
    "test")
        echo "Running all tests..."
        python -m pytest src/test/ -v
        ;;
    "test-vector")
        echo "Running vector embedding tests..."
        python src/test/test_vector_sync_and_workflow.py
        ;;
    "test-workflow")
        echo "Running workflow tests..."
        python src/test/test_vector_workflow.py
        ;;
    "server")
        echo "Starting development server..."
        python src/kangni_agents/main.py
        ;;
    "install")
        echo "Installing/updating dependencies..."
        pip install -e .
        ;;
    "lint")
        echo "Running linting..."
        python -m flake8 src/
        python -m black --check src/
        ;;
    *)
        echo "Unknown command: $COMMAND"
        exit 1
        ;;
esac
