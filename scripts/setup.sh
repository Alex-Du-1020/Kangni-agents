#!/bin/bash
# Setup script for Kangni Agents project

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install/update dependencies
echo "Installing dependencies..."
pip install -e .

# Run tests
echo "Running tests..."
python -m pytest src/test/ -v

echo "Setup complete!"
