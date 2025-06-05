# Simple single-stage build for minimal client
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary application files
COPY Step10MCPClientPsxGPT.py .
COPY prompts.py .
COPY tickers.json .
COPY start.sh .
COPY .chainlit/ .chainlit/

# Create necessary directories
RUN mkdir -p enhanced_client_contexts

# Make start script executable
RUN chmod +x start.sh

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port (Render will set the PORT env var)
EXPOSE $PORT

# Start the application
CMD ["./start.sh"] 