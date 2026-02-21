FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (long timeout for slow VPS / large packages like torch)
RUN pip install --no-cache-dir --default-timeout=600 -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY data/ ./data/
COPY public/ ./public/

# Create necessary directories
RUN mkdir -p public/images/upload public/images/preload public/edited public/patterns .cache/huggingface

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONWARNINGS="ignore::FutureWarning"
ENV HF_HOME=/app/.cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
