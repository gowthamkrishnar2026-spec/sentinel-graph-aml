# Python 3.11 Slim Multi-purpose AML Forensic Container
FROM python:3.11-slim

# Install system dependencies for C extensions and curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source tree
COPY . .

# Set environment defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    NEO4J_URI=bolt://neo4j:7687 \
    NEO4J_USER=neo4j \
    NEO4J_PASSWORD=password123 \
    PORT=8000

# Expose FastAPI & Dashboard Port
EXPOSE 8000

# Healthcheck to verify the backend is up and responding
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/api/overview || exit 1

# Start the uvicorn web server
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
