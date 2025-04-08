# WDBX Documentation

Welcome to the WDBX documentation system. This documentation is built using the new unified WDBX documentation builder with enhanced features and styling.

## Overview

WDBX is a high-performance vector database and machine learning integration system that offers:

- Fast vector search and retrieval
- Blockchain-based data validation
- Advanced ML integration capabilities
- Extensible plugin architecture
- Cross-platform compatibility

## Getting Started

To get started with WDBX, follow these simple steps:

1. **Installation**: Install WDBX using pip
   ```
   pip install wdbx
   ```

2. **Initialize**: Create a new WDBX instance
   ```python
   from wdbx import WDBX
   
   # Create a new WDBX instance
   db = WDBX(data_dir="./data")
   ```

3. **Add data**: Store vectors and data
   ```python
   # Create a vector
   vector = db.create_vector([0.1, 0.2, 0.3, 0.4], metadata={"source": "example"})
   
   # Create a block with data and embedding
   block = db.create_block(
       data={"text": "Example data", "category": "sample"},
       embeddings=[vector]
   )
   ```

4. **Search**: Find similar vectors and data
   ```python
   # Find similar vectors
   results = db.find_similar_vectors([0.1, 0.2, 0.3, 0.4], top_k=5)
   
   # Search blocks
   blocks = db.search_blocks("example query", top_k=5)
   ```

## Documentation Builder

This documentation is built using the new unified `wdbx_builder.py` script which combines the best features from various documentation tools including:

- Markdown to HTML conversion with beautiful styling
- Sphinx integration for advanced documentation
- File watching with live reload
- Automated table of contents
- Mobile-friendly responsive design
- Dark mode support

To build the documentation:

```bash
python wdbx_builder.py --mode markdown --clean --open
```

To watch for changes:

```bash
python wdbx_builder.py --mode markdown --watch
```

## Next Steps

- Explore the [API Reference](api_reference.md)
- Learn about the [Plugin System](plugin_system_overview.md)
- Read about [Performance Optimization](performance_profiling.md)
- Understand the [Project Structure](project_structure.md)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to get involved. 