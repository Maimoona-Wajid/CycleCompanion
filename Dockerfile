# Use Python 3.11 slim image to keep container size minimal
FROM python:3.11-slim

# Install system dependencies required for compilation and PostgreSQL connectivity
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy dependency requirements list and install packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code and saved ML model pickles
COPY . .

# Expose default port (Render sets $PORT dynamically)
EXPOSE 10000

# Run FastAPI with Gunicorn: 2 workers (free tier = 512MB RAM),
# 120s timeout for model loading on cold start
CMD gunicorn -w 2 -k uvicorn.workers.UvicornWorker --timeout 120 --bind 0.0.0.0:${PORT:-10000} main:app


