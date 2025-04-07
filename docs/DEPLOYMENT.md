# WDBX Deployment Guide

This guide covers different approaches for deploying WDBX in development and production environments.

## Deployment Options

WDBX can be deployed using several approaches, depending on your requirements:

1. **Docker Container**: The simplest way to deploy WDBX in a consistent environment
2. **Python Package**: Deploy WDBX as a package in your existing Python environment
3. **Web Service**: Run WDBX as a web service with REST API endpoints
4. **Interactive Mode**: Run WDBX in interactive mode for development and exploration

## Docker Deployment

### Prerequisites

- Docker installed on your system
- Git for cloning the repository (optional)

### Basic Deployment

```bash
# Clone the repository
git clone https://github.com/example/wdbx_python.git
cd wdbx_python

# Build the Docker image
docker build -t wdbx-dev .

# Run the container with interactive mode
docker run -it --rm -v $(pwd):/app -p 8080:8080 wdbx-dev
```

### Using Docker Compose

For a more complete setup with optional Ollama integration:

```bash
# Start all services
docker-compose up -d

# Start only WDBX without Ollama
docker-compose up -d wdbx

# Run in interactive mode
docker-compose run wdbx python run_wdbx.py --interactive

# Access bash shell
docker-compose exec wdbx bash
```

### Environment Variables

You can configure WDBX using environment variables in your Docker deployment:

```bash
docker run -it --rm \
  -e WDBX_LOG_LEVEL=INFO \
  -e WDBX_VECTOR_DIMENSION=768 \
  -e WDBX_DATA_DIR=/data \
  -v $(pwd)/data:/data \
  -p 8080:8080 \
  wdbx-dev
```

## Python Package Deployment

### Installation

```bash
# Install from source
pip install -e .

# Install with ML acceleration
pip install -e ".[ml]"

# Install with vector search optimization
pip install -e ".[vector]"

# Install everything
pip install -e ".[full]"
```

### Usage in Your Application

```python
from wdbx import WDBX, WDBXConfig, EmbeddingVector
import numpy as np

# Configure WDBX
config = WDBXConfig(
    vector_dimension=1024,
    num_shards=2,
    data_dir="./wdbx_data"
)

# Create instance
db = WDBX(config=config)

# Use WDBX in your application
# ...
```

## Web Service Deployment

### Local Deployment

```bash
# Start the HTTP server
python -m wdbx.cli server --host 0.0.0.0 --port 8080
```

### Production Deployment with Gunicorn

For production deployments, we recommend using Gunicorn:

```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:8080 wdbx.server:app
```

## Streamlit UI Deployment

WDBX includes a Streamlit-based UI for easy interaction:

```bash
# Install Streamlit
pip install streamlit

# Run the Streamlit app
streamlit run wdbx/ui/streamlit_app.py
```

### Deploying to Streamlit Sharing

1. Create a GitHub repository for your WDBX project
2. Create a `requirements.txt` file with necessary dependencies
3. Push your code to GitHub
4. Sign up for Streamlit Sharing (streamlit.io/sharing)
5. Connect your GitHub repository
6. Deploy your app

## Cloud Deployment Options

### AWS Deployment

Deploy WDBX on AWS using:

1. **EC2**: For full control over the deployment
2. **ECS/EKS**: For container-based deployments using Docker
3. **Lambda + API Gateway**: For serverless deployments (suitable for low-volume usage)

### Google Cloud Platform

Deploy WDBX on GCP using:

1. **Compute Engine**: Similar to EC2 for full control
2. **GKE**: Container orchestration with Kubernetes
3. **Cloud Run**: Serverless container deployment

### Azure

Deploy WDBX on Azure using:

1. **Azure VMs**: Full control over the environment
2. **AKS**: Azure Kubernetes Service for container orchestration
3. **Azure Container Instances**: Simple container deployment

## Performance Considerations

### Scaling Vector Store

For large vector collections, consider:

1. **Increasing shards**: Set higher `num_shards` value to distribute vectors
2. **FAISS optimization**: Use the `vector` extra to enable FAISS acceleration
3. **GPU acceleration**: Use GPU-enabled backends for faster processing
4. **Memory optimization**: Configure appropriate cache sizes for your hardware

### Backend Selection

Choose the appropriate ML backend based on your deployment environment:

1. **PyTorch**: Best general-purpose backend for CPU environments
2. **JAX**: Consider for TPU environments or specialized workloads
3. **FAISS**: Highly recommended for large-scale vector search

## Monitoring and Maintenance

### Health Checks

Implement health checks to monitor the status of your WDBX deployment:

```python
from wdbx.health import check_health

status = check_health()
if status["healthy"]:
    print("WDBX is running normally")
else:
    print("WDBX has issues:", status["issues"])
```

### Backup and Recovery

Regularly backup your vector store:

```bash
# Backup WDBX data
python -m wdbx.cli backup --output backup.wdbx

# Restore from backup
python -m wdbx.cli restore --input backup.wdbx
```

## Troubleshooting

### Common Issues

1. **Memory errors**: Increase available memory or reduce vector dimension
2. **Slow search performance**: Enable FAISS, adjust number of shards
3. **Import errors**: Ensure all dependencies are installed correctly

### Getting Help

If you encounter issues with your deployment:

1. Check the documentation
2. Review logs with `--log-level=DEBUG`
3. File an issue on the GitHub repository 