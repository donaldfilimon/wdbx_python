FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libffi-dev \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Install additional dependencies for plugins and development
RUN pip install numpy matplotlib scikit-learn pytest mypy flake8 black isort \
    discord.py openai huggingface_hub sentence_transformers requests \
    cachetools

# Copy application code
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/wdbx_data \
             /app/demo_visualizations \
             /app/wdbx_model_cache \
             /app/.wdbx

# Set permissions
RUN chmod +x /app/run_wdbx.py

# Expose port for HTTP server
EXPOSE 8080

# Default command
CMD ["python", "run_wdbx.py", "--interactive"]

# Build with:
# docker build -t wdbx-dev .
#
# Run with:
# docker run -it --rm -v $(pwd):/app -p 8080:8080 wdbx-dev 