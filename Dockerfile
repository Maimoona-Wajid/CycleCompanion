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

# Run FastAPI using Gunicorn process manager with 4 concurrent Uvicorn workers
# Uses shell form so $PORT is expanded at runtime (Render injects this)
CMD gunicorn -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-10000} main:app
