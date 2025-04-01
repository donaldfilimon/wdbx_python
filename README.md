# WDBX: Wide Distributed Block Database

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python Versions](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)]()
[![Status](https://img.shields.io/badge/status-beta-yellow)]()

WDBX is a high-performance, distributed database system specifically designed for multi-personality AI systems. It combines vector similarity search, blockchain-style integrity, multiversion concurrency control (MVCC), and multi-head attention mechanisms to create a robust and flexible foundation for advanced AI applications.

## ✨ Features

- **Embedding Vector Storage:** Store, update, and search high-dimensional embedding vectors using efficient similarity search.
- **Blockchain-Inspired Data Integrity:** Create and validate blocks that link embedding vectors together with cryptographic integrity.
- **MVCC Concurrency:** Handle read/write operations concurrently with conflict resolution.
- **Neural Backtracking:** Trace activation patterns and detect semantic drift in AI responses.
- **Multi-Head Attention:** Apply transformer-like attention for sequence modeling and enhanced context understanding.
- **Multi-Persona Framework:** Manage multiple AI personas and blend responses based on context.
- **Content Filtering & Bias Mitigation:** Ensure responses are safe and balanced.
- **Asynchronous HTTP Server:** Serve API endpoints using aiohttp for high-performance access.
- **Pure Python Implementation:** Works with standard library dependencies for core functionality.
- **Optional High-Performance Backend:** Accelerate operations with FAISS, JAX, and other libraries when available.

## 🚀 Quick Start

### Installation

WDBX can be installed directly from GitHub or PyPI:

```bash
# Install from PyPI with basic dependencies
pip install wdbx

# Install with all dependencies for optimal performance
pip install wdbx[full]

# Install directly from GitHub
pip install git+https://github.com/username/wdbx.git
```

### Basic Usage

Here's a simple example of using WDBX to store and retrieve embedding vectors:

```python
import numpy as np
from wdbx import WDBX
from wdbx.data_structures import EmbeddingVector

# Initialize WDBX with 512-dimensional vectors
wdbx = WDBX(vector_dimension=512, num_shards=4)

# Create and store an embedding vector
vector = np.random.randn(512).astype(np.float32)
embedding = EmbeddingVector(
    vector=vector,
    metadata={"description": "Sample embedding", "source": "example"}
)
vector_id = wdbx.store_embedding(embedding)
print(f"Stored embedding with ID: {vector_id}")

# Search for similar vectors
results = wdbx.search_similar_vectors(vector, top_k=5)
for vector_id, similarity in results:
    print(f"Vector ID: {vector_id}, Similarity: {similarity:.4f}")

# Create a neural trace
trace_id = wdbx.create_neural_trace(vector)
print(f"Created trace with ID: {trace_id}")
```

### Using the Multi-Persona Framework

```python
from wdbx import WDBX
from wdbx.persona import PersonaManager

# Initialize WDBX
wdbx = WDBX(vector_dimension=1024)

# Initialize the persona manager
persona_manager = PersonaManager(wdbx)

# Process a user message
user_input = "Can you explain how vector databases work?"
context = {"chain_id": None, "block_ids": []}
response, block_id = persona_manager.process_user_input(user_input, context)

print(f"User: {user_input}")
print(f"AI: {response}")
print(f"Block ID: {block_id}")
```

### Running the Interactive Mode

WDBX comes with an interactive mode that allows you to explore its functionality from the command line:

```bash
# Start interactive mode
python -m wdbx.cli interactive

# Run the example demonstration
python -m wdbx.cli example

# Start the HTTP server
python -m wdbx.cli server --host 127.0.0.1 --port 8080
```

## 🧩 Architecture

WDBX consists of several key components:

### Core Components

- **Vector Store:** Manages embedding vectors with efficient similarity search.
- **Blockchain Manager:** Handles data integrity using blockchain-inspired techniques.
- **MVCC Manager:** Provides multiversion concurrency control for transactions.
- **Neural Backtracker:** Traces activation patterns through the system.
- **Shard Manager:** Distributes data across multiple shards for scalability.
- **Persona Manager:** Manages multiple AI personas and response generation.

### Database Components

- **HTTP Client:** Connect to WDBX over HTTP.
- **Socket Client:** Connect to WDBX using raw sockets for high-performance access.
- **Filesystem Client:** Use the local filesystem as a database backend.

### System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                          WDBX Client                           │
└───────────────────────────────┬────────────────────────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
┌─────────────▼──────┐ ┌────────▼───────┐ ┌───────▼────────┐
│    HTTP Client     │ │  Socket Client  │ │ Filesystem Client │
└─────────────┬──────┘ └────────┬───────┘ └───────┬────────┘
              │                 │                 │
┌─────────────▼─────────────────▼─────────────────▼────────────┐
│                       WDBX Server                            │
├──────────────┬──────────────┬───────────────┬───────────────┤
│ Vector Store │ Blockchain   │ MVCC Manager  │ Neural        │
│              │ Manager      │               │ Backtracker   │
├──────────────┼──────────────┼───────────────┼───────────────┤
│ Shard        │ Persona      │ Content       │ Multi-Head    │
│ Manager      │ Manager      │ Filter        │ Attention     │
└──────────────┴──────────────┴───────────────┴───────────────┘
```

## 📚 API Reference

### WDBX Class

The main entry point for interacting with WDBX:

```python
wdbx = WDBX(vector_dimension=1024, num_shards=8)
```

- `store_embedding(embedding_vector)`: Store an embedding vector and return its ID.
- `create_conversation_block(data, embeddings, chain_id=None, context_references=None)`: Create a block containing conversation data and embeddings.
- `search_similar_vectors(query_vector, top_k=10)`: Search for vectors similar to the query vector.
- `create_neural_trace(query_vector)`: Create a neural trace for the query vector.
- `get_conversation_context(block_ids)`: Get the conversation context for the given block IDs.
- `get_system_stats()`: Get statistics about system usage.

### EmbeddingVector Class

Represents a high-dimensional embedding vector with metadata:

```python
embedding = EmbeddingVector(vector=vector_data, metadata={"source": "user_query"})
```

- `normalize()`: Return a normalized copy of the vector.
- `serialize()`: Serialize the vector to bytes.
- `deserialize(data)`: Create an EmbeddingVector from serialized bytes.

### Block Class

Represents a data block with blockchain-inspired integrity:

```python
block = Block(id=block_id, timestamp=time.time(), data=data, embeddings=embeddings)
```

- `calculate_hash()`: Calculate the cryptographic hash of the block.
- `mine_block()`: Mine the block by finding a nonce that produces a hash with the required difficulty.
- `validate()`: Validate the block's integrity.
- `serialize()`: Serialize the block to bytes.
- `deserialize(data)`: Create a Block from serialized bytes.

### PersonaManager Class

Manages multi-persona interactions:

```python
persona_manager = PersonaManager(wdbx)
```

- `determine_optimal_persona(user_input, context=None)`: Determine the optimal persona for the user input.
- `generate_response(user_input, persona_id, context=None)`: Generate a response using the specified persona.
- `process_user_input(user_input, context=None)`: Process a user input and generate a response.

## 🧪 Performance

WDBX is designed for high performance and scalability:

- **Vector Operations:** Efficiently handle millions of high-dimensional vectors.
- **Sharding:** Distribute data across multiple shards for horizontal scaling.
- **Concurrency:** Handle concurrent read/write operations with MVCC.
- **Optimization:** Automatically use available accelerators (FAISS, JAX, etc.) when available.

## 📋 Requirements

### Core Requirements
- Python 3.9 or higher

### Optional Dependencies for Enhanced Performance
- NumPy: For efficient vector operations
- FAISS: For high-performance vector similarity search
- aiohttp: For HTTP server functionality
- scikit-learn: For advanced clustering algorithms
- JAX: For JIT acceleration in MLIR

## 🛠️ Development Status

The WDBX project is currently in beta, with a fully functional Python implementation available for immediate use alongside ongoing work on the high-performance backend.

## 🚶 Roadmap

- **v1.0 (Current):** Core Python implementation with all features
- **v1.1:** Enhanced performance with JAX acceleration
- **v1.2:** Distributed deployment support
- **v1.3:** Advanced monitoring and observability
- **v2.0:** Full production hardening with enterprise features

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

```WDBX is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full license text.```

---

`previous/ongoing completion plan`
# WDBX Project Completion Plan

## 1. Fix Module Structure

### Issues to Resolve:
- Fix circular imports between modules
- Resolve duplicate WDBX class definitions
- Fix incorrect imports in `__init__.py`
- Clean up constants vs data structures

### Implementation:
1. Reorganize `__init__.py` to properly expose the API
2. Fix imports in all modules
3. Move WDBX class to a single location
4. Properly separate constants from data structures

## 2. Fix Dependencies

### Issues to Resolve:
- Handle optional dependencies properly
- Implement fallbacks for external libraries
- Update requirements.txt

### Implementation:
1. Add try/except blocks for optional dependencies like faiss
2. Implement fallback mechanisms using pure Python
3. Update requirements with proper version constraints

## 3. Complete Core Features

### Issues to Resolve:
- Implement missing features mentioned in README
- Complete server implementation
- Add full CLI functionality

### Implementation:
1. Complete neural backtracking implementation
2. Finish multi-persona framework
3. Implement content filtering & bias mitigation
4. Enhance multi-head attention implementation
5. Complete HTTP server

## 4. Add Documentation

### Issues to Resolve:
- Improve API documentation
- Add usage examples
- Create clear installation guide

### Implementation:
1. Add docstrings to all classes and methods
2. Create detailed API reference
3. Write step-by-step tutorials
4. Update README with detailed installation and usage

## 5. Improve Testing

### Issues to Resolve:
- Fix existing tests
- Add more test coverage
- Implement integration tests

### Implementation:
1. Fix import issues in tests
2. Add unit tests for all core features
3. Add integration tests for the full system
4. Add performance benchmarks

## 6. Optimize Performance

### Issues to Resolve:
- Optimize vector operations
- Improve blockchain mining
- Enhance concurrency

### Implementation:
1. Optimize vector similarity search
2. Add caching for frequent operations
3. Implement more efficient blockchain mining
4. Enhance MVCC implementation

## 7. Package and Distribution

### Issues to Resolve:
- Create proper package structure
- Add setup tools configuration
- Prepare for PyPI submission

### Implementation:
1. Finalize package structure
2. Update setup.py with complete metadata
3. Create CI/CD pipeline
4. Prepare documentation for PyPI
