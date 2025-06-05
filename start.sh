#!/bin/bash

# Get the port from environment variable (Render sets this)
PORT=${PORT:-8000}

echo "Starting PSX Financial Assistant on port $PORT..."

# Start the Chainlit application
exec python -m chainlit run Step10MCPClientPsxGPT.py --host 0.0.0.0 --port $PORT --headless 