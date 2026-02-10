# MCP Tools Service Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (poppler-utils needed by pdf2image)
RUN apt-get update && apt-get install -y \
    curl \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy shared module first & install its deps
COPY shared /app/shared
RUN pip install --no-cache-dir -r /app/shared/requirements.txt

# Copy requirements and install
COPY mcp-tools/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY mcp-tools/ .

# Make shared importable without sys.path hacks
ENV PYTHONPATH=/app/shared:$PYTHONPATH

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Run the application with FastMCP's built-in HTTP server
CMD ["python", "main.py", "--transport", "sse", "--host", "0.0.0.0", "--port", "8002"]
