# Multi-stage build for smaller production image
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Set working directory
WORKDIR /app

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
ENV PATH=/root/.local/bin:$PATH

# Expose port (Render will set the PORT env var)
EXPOSE $PORT

# Non-root user for security
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Start the application
CMD ["./start.sh"] 