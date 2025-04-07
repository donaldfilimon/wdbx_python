# WDBX Python

A high-performance vector database with blockchain-inspired features and multi-version concurrency control.

## Introduction

[Existing introduction content...]

## Project Structure

The WDBX project is organized as follows:

```
wdbx_python/
├── data/                   # Data storage directory
├── docs/                   # Documentation
├── examples/               # Example usage scripts
├── scripts/                # Utility scripts
│   ├── benchmarks/         # Performance testing tools
│   ├── linters/            # Code quality tools
│   └── runners/            # Application runners
├── src/                    # Source code
│   └── wdbx/               # Main package
│       ├── core/           # Core functionality
│       ├── ml/             # Machine learning components
│       ├── storage/        # Data storage modules
│       └── ui/             # User interfaces
├── tests/                  # Tests
│   ├── imports/            # Import validation tests
│   ├── integration/        # Integration tests
│   ├── ml/                 # ML-specific tests
│   └── unit/               # Unit tests
└── wdbx_tool.py            # Unified command launcher
```

Use the unified tool launcher for common operations:

```bash
# Run the application
python scripts/wdbx_tool.py run

# Launch the web UI
python scripts/wdbx_tool.py web

# Run tests
python scripts/wdbx_tool.py test

# Run benchmarks
python scripts/wdbx_tool.py benchmark
```

## Addressing Circular Imports

This project had circular dependency issues that were resolved using the following strategies:

1. **Strategic use of `__init__.py` files**:
   - Created placeholder classes in module `__init__.py` files to break circular dependencies
   - Defined shared types and constants in module-level scope

2. **TYPE_CHECKING conditional imports**:
   - Used `if TYPE_CHECKING:` conditional imports for type checking without runtime dependencies
   - This allows proper type hinting while avoiding circular imports at runtime

3. **Dynamic imports**:
   - Implemented helper functions like `get_mvcc_transaction()` to dynamically import required classes
   - Only imports dependencies when actually needed during execution

4. **Centralized constants**:
   - Created a core constants module at `src/wdbx/core/constants.py`
   - All system-wide constants are defined in one place for better maintainability

5. **Standardized logging**:
   - Implemented a comprehensive logging system in `src/wdbx/utils/logging_utils.py`
   - Provides consistent log formatting, rotating file handlers, and level management across all components

## Development

To run tests for import validation:

```
python test_imports.py
```

## Dependencies

- NumPy: Core numerical operations
- FAISS (optional): For efficient vector similarity search
- LMDB (optional): For persistent storage
- JAX/PyTorch (optional): For accelerated vector operations

## Features

- High-performance vector similarity search
- Distributed query planning across multiple shards
- Blockchain-based storage for data provenance
- MVCC transactional support
- Web UI for visualization and monitoring
- Python API for easy integration
- Customizable CLI with plugin system and theming
- Multiple backend support (JAX, PyTorch, NumPy)

## Installation

```bash
# Install from source
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Install with ML acceleration
pip install -e ".[ml]"

# Install with vector search optimization
pip install -e ".[vector]"

# Install everything
pip install -e ".[full]"
```

## Quick Start

```python
from wdbx import WDBX, WDBXConfig, EmbeddingVector
import numpy as np

# Create a WDBX instance with default configuration
config = WDBXConfig(
    vector_dimension=1024,
    num_shards=2,
    data_dir="./wdbx_data"
)
db = WDBX(config=config)

# Store a vector
vector = np.random.randn(config.vector_dimension).astype(np.float32)
embedding = EmbeddingVector(
    vector=vector,
    metadata={
        "description": "Example vector",
        "timestamp": 1649926800,
        "source": "example"
    }
)
vector_id = db.store_embedding(embedding)
print(f"Stored vector with ID: {vector_id}")

# Search for similar vectors
query_vector = np.random.randn(config.vector_dimension).astype(np.float32)
results = db.search_similar_vectors(query_vector, top_k=5)
for vector_id, similarity in results:
    print(f"Vector ID: {vector_id}, Similarity: {similarity:.4f}")

# Start the Streamlit UI
python scripts/runners/run_streamlit.py
```

## Command Line Interface

WDBX offers a powerful command-line interface with multiple modes and customization options.

### Interactive Mode

```bash
wdbx interactive [options]
```

Options:
- `--data-dir`: Set the data directory (default: ./data)
- `--dimension`: Set vector dimension (default: 128)
- `--ml-backend`: Choose ML backend (numpy, jax, torch)
- `--use-gpu`: Use GPU if available
- `--theme`: Choose UI theme (default, dark, light)
- `--plugins`: Comma-separated list of plugins to load

### Server Mode

```bash
wdbx server [options]
```

Options:
- `--host`: Server host (default: localhost)
- `--port`: Server port (default: 8080)
- `--workers`: Number of worker processes
- `--log-level`: Set logging level

### Web UI Mode

```bash
wdbx web [options]
```

Options:
- `--port`: Web UI port (default: 3000)
- `--server-url`: URL of the WDBX server
- `--theme`: Web UI theme (light, dark, auto)

### Benchmark Mode

```bash
wdbx benchmark [options]
```

Options:
- `--vectors`: Number of vectors to use in benchmarks
- `--dimension`: Vector dimension
- `--test`: Specific test to run
- `--output`: Output file for benchmark results

## Plugin System

The CLI can be extended with plugins that add new commands:

```python
def register_commands(plugin_registry):
    plugin_registry["mycmd"] = my_command_function
    
def my_command_function(db, args):
    print("My custom command!")
```

Available plugins include:
- `visualization`: Adds commands for visualizing vector data

## Web UI

WDBX includes a Streamlit-based UI for visualizing and interacting with vectors:

```bash
# Run the Streamlit app
python scripts/runners/run_streamlit.py

# Or use Streamlit directly
streamlit run src/wdbx/ui/streamlit_app.py
```

The Streamlit UI provides:
- Interactive vector visualization with PCA, TSNE, and UMAP dimensionality reduction
- Similarity search functionality
- Store management
- Metadata exploration

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
python scripts/runners/run_tests.py

# Run with coverage
python scripts/runners/run_tests.py --coverage

# Run vector store benchmarks
python scripts/benchmarks/vector_store_benchmark.py

# Run linters/code quality tools
python scripts/linters/fix_lint.py src/wdbx
```

## Debugging

The wdbx package includes enhanced debugging capabilities:

```python
from wdbx.debugger import set_trace, wdbx_debug, debug_vector

# Use as a decorator
@wdbx_debug
def my_function():
    ...

# Debug specific vectors
debug_vector(vector, "input_vector")
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Memory Optimization

WDBX now includes automatic memory optimization features to prevent out-of-memory errors in production:

- **Automatic memory monitoring**: The system monitors memory usage and triggers optimization when usage exceeds configured thresholds.
- **Configurable thresholds**: Set `WDBX_MAX_MEMORY_PERCENT` (default: 85.0) to control when optimization starts.
- **Periodic checks**: Configure check frequency with `WDBX_MEMORY_CHECK_INTERVAL` (default: 10 refresh cycles).
- **Manual optimization**: Trigger optimization programmatically via `vector_store.optimize_memory()`.

Example configuration:

```bash
# Set memory threshold to 75%
export WDBX_MAX_MEMORY_PERCENT=75.0

# Check memory usage every 5 refresh cycles
export WDBX_MEMORY_CHECK_INTERVAL=5

# Disable automatic memory optimization
export WDBX_MEMORY_OPTIMIZATION_ENABLED=false
```

## System Diagnostics

WDBX includes comprehensive system diagnostics to monitor and optimize performance:

```python
from wdbx.utils.diagnostics import SystemDiagnostics, start_monitoring

# Create a diagnostics instance (or use the global singleton)
diagnostics = SystemDiagnostics()

# Register WDBX components for monitoring
diagnostics.register_component("vector_store", vector_store)
diagnostics.register_component("block_manager", block_manager)
diagnostics.register_component("tx_manager", transaction_manager)

# Start automatic monitoring (uses background thread)
diagnostics.start_monitoring()

# Get current memory usage
memory_info = diagnostics.get_memory_usage()
print(f"Current memory: {memory_info['percent']:.1f}% ({memory_info['used'] / 1024**2:.1f} MB)")

# Get system information
system_info = diagnostics.get_system_info()
print(f"Platform: {system_info['platform']}")
print(f"CPU Count: {system_info.get('cpu_count', 'Unknown')}")

# View performance metrics history
metrics = diagnostics.get_metrics_history()
print(f"Recorded {len(metrics['memory_usage'])} memory data points")

# Log custom events
diagnostics.log_event("operation", {
    "operation": "vector_search",
    "duration_ms": 156,
    "results_count": 10
})

# Optimize memory on demand
diagnostics.optimize_memory()

# Stop monitoring before exiting
diagnostics.stop_monitoring()
```

The diagnostics system automatically:
- Tracks memory usage over time
- Optimizes memory when thresholds are exceeded
- Collects performance metrics for later analysis
- Collects timing information for key operations
- Provides detailed system information for troubleshooting

## Database Management

WDBX includes built-in database management tools:

```bash
# Initialize database
python wdbx_tool.py db init

# Check database status
python wdbx_tool.py db status

# Backup database
python wdbx_tool.py db backup --target backup_file.zip

# Restore database
python wdbx_tool.py db restore --target backup_file.zip

# Clean up and optimize database
python wdbx_tool.py db cleanup
```

## Performance Monitoring

Monitor system and WDBX performance metrics:

```bash
# Collect metrics until interrupted
python wdbx_tool.py metrics

# Collect metrics for a specific duration
python wdbx_tool.py metrics --duration 3600  # 1 hour

# Specify output directory
python wdbx_tool.py metrics --output-dir ./metrics_output

# Connect to running WDBX server
python wdbx_tool.py metrics --server 127.0.0.1:8080
```

Metrics are saved as CSV and JSON files, and visualizations are generated (if matplotlib is installed).

## Environment Configuration

WDBX can be configured using environment variables or a `.env` file:

```bash
# Use a specific .env file
python wdbx_tool.py --env my_custom.env run
```

Key configuration options include:

- `WDBX_DATA_DIR`: Database storage location
- `WDBX_VECTOR_DIMENSION`: Default vector dimension
- `WDBX_MAX_MEMORY_PERCENT`: Memory threshold for optimization
- `WDBX_DEFAULT_ML_BACKEND`: ML backend selection (numpy, torch, jax)

See `.env.template` for all available options.
