# WDBX - Vector Database and Embedding Management System

WDBX is a powerful vector database and embedding management system designed specifically for AI applications. It provides efficient storage, indexing, and retrieval of high-dimensional vector embeddings.

## Features

- Fast similarity search with configurable indexing
- Support for various embedding models
- Storage and retrieval of vector embeddings with metadata
- Customizable vector database with pluggable backends
- Comprehensive UI dashboard for visualization and management
- Visualization tools for vector embeddings
- Plugin system for extensibility

## Installation

```bash
# Basic installation
pip install wdbx

# Install with UI components
pip install wdbx[ui]
```

## Quick Start

```python
from wdbx import VectorDB

# Create a new vector database
db = VectorDB(dimension=1536)

# Add vectors with metadata
db.add_vector([0.1, 0.2, ...], metadata={"text": "Example text"})

# Find similar vectors
results = db.find_similar([0.1, 0.2, ...], top_k=5)
```

## UI Dashboard

WDBX includes a full-featured UI dashboard built with Streamlit that provides visualization and management capabilities for your vector database.

### Launching the UI

You can launch the UI with:

```bash
# Using the CLI (if installed with UI components)
wdbx-ui

# Or directly with Streamlit
streamlit run /path/to/wdbx/ui/streamlit_app.py
```

### UI Features

The UI dashboard provides:

- **Modular Architecture** - Easily extensible with new pages and features
- **Vector Visualization** - Visualize your vectors in 2D or 3D space using dimensionality reduction
- **Similarity Search Interface** - Interactive search for similar vectors
- **Database Statistics** - View and monitor your database metrics
- **Theme Customization** - Multiple themes including light, dark, and specialized options
- **Advanced Settings** - Toggle advanced features for power users

### UI Screenshot

![WDBX Dashboard](https://example.com/dashboard-screenshot.png)

## Deploying to Streamlit Cloud

The WDBX UI can be easily deployed to [Streamlit Cloud](https://streamlit.io/cloud) for sharing with your team or showcasing your vector database.

### Using the Deployment Helper

1. Use the deployment helper script to prepare your files:

```bash
python -m wdbx.ui.deployment.deploy_to_cloud --target ./deploy
```

2. Push the generated files to a GitHub repository

3. Connect the repository to Streamlit Cloud and specify `streamlit_app.py` as the main file

For detailed deployment instructions, see [UI Deployment Guide](./src/wdbx/ui/docs/modular_ui.md).

## Documentation

- [API Reference](https://wdbx.readthedocs.io/)
- [UI Developer Guide](./src/wdbx/ui/docs/ui_guide.md)
- [Modular UI Architecture](./src/wdbx/ui/docs/modular_ui.md)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
