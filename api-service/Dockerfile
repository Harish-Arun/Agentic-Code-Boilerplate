# API Service Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy shared module first & install its deps
COPY shared /app/shared
RUN pip install --no-cache-dir -r /app/shared/requirements.txt

# Copy requirements and install
COPY api-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api-service/ .

# Create data directory
RUN mkdir -p /app/data

# Make shared importable without sys.path hacks
ENV PYTHONPATH=/app/shared:$PYTHONPATH

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
