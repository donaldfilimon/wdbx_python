# WDBX Documentation Portal

Welcome to the WDBX documentation portal. This site contains comprehensive documentation for the WDBX vector database and search engine.

## Key Features

- **High-performance Vector Database**: Store and search vector embeddings with optimal performance
- **ML Framework Integration**: Native support for NumPy, PyTorch, and JAX
- **Memory Optimization**: Advanced memory management for handling large embedding datasets
- **Production Ready**: Kubernetes deployment, monitoring, and diagnostics tools
- **Extensible Architecture**: Plugin system and customizable components

## Documentation Highlights

### ML Integration

WDBX provides powerful integration with machine learning frameworks including PyTorch and JAX:

- **Unified ML Backend Interface**: Consistent API across different ML frameworks
- **Optimized Vector Operations**: Up to 4x speedup for batch cosine similarity operations
- **FAISS Integration**: High-performance vector search with specialized backends
- **Framework Selection**: Automatic selection of the best available backend

[Learn more about ML Integration](ML_INTEGRATION.html)

### Diagnostics System

WDBX includes a comprehensive diagnostics system for monitoring and performance analysis:

- **Resource Monitoring**: Track memory, CPU, and disk usage
- **Performance Profiling**: Time critical operations and identify bottlenecks
- **Event Logging**: Track important system events and errors
- **Custom Component Monitoring**: Register and monitor your own components

[Learn more about Diagnostics](diagnostics.html)

### Code Quality

WDBX follows strict code quality guidelines to ensure a reliable and maintainable codebase:

- **Style Guidelines**: PEP 8 compliance with additional conventions
- **Documentation Standards**: Google-style docstrings and type annotations
- **Error Handling**: Robust exception handling and recovery
- **Performance Optimization**: Guidelines for efficient resource usage

[Learn more about Code Quality](CODE_QUALITY.html)

## Getting Started

### Installation

Install WDBX using pip:

```bash
pip install wdbx
```

### Basic Usage

```python
from wdbx.client import WDBXClient
import numpy as np

# Initialize client
client = WDBXClient()

# Create vectors
vector1 = client.create_vector(
    vector_data=np.random.rand(768).astype(np.float32),
    metadata={"description": "Example vector"}
)

# Search for similar vectors
results = client.find_similar_vectors(query_vector=vector1, top_k=5)
```

## Documentation Sections

- [API Reference](api/index.html): Complete API documentation
- [User Guide](guide/index.html): Comprehensive usage guides
- [Plugins](plugins/index.html): Available plugins and extensions
- [Operations](operations/index.html): Deployment and monitoring guides
- [Development](development/index.html): Contributing guidelines and best practices 