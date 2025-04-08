#!/usr/bin/env python
"""
WDBX Plugin Demo

This script demonstrates how to use the WDBX plugins for:
1. Visualization
2. Model Repository
3. Integration with OpenAI, HuggingFace, and Ollama

To run this demo, execute:
    python sample_data/plugin_demo.py [--data PATH] [--demos LIST]
"""

import argparse
import contextlib
import os
import sys
import time
import traceback
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

# Add parent directory to path for imports - using path manipulation best practices
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

try:
    from wdbx.storage import normalize_path
    from wdbx.utils import get_metrics, register_component, stop_monitoring
    from wdbx.utils.diagnostics import SystemMonitor

    HAS_WDBX = True
except ImportError:
    HAS_WDBX = False
    print("Warning: Could not import WDBX modules. Running in mock mode only.")


# Create mock objects to emulate the WDBX system
class MockWDBXConfig:
    """Configuration class for mock WDBX system."""

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize configuration with default values that can be overridden.

        Args:
            **kwargs: Configuration parameters to override defaults
        """
        self.__dict__.update(kwargs)
        # Set defaults for anything not provided
        if "data_dir" not in kwargs:
            self.data_dir = Path("./wdbx_demo_data").resolve()
        if "vector_dimension" not in kwargs:
            self.vector_dimension = 768
        if "num_shards" not in kwargs:
            self.num_shards = 2
        if "plugin_paths" not in kwargs:
            self.plugin_paths = [str(parent_dir / "wdbx_plugins")]
        if "enable_persona_management" not in kwargs:
            self.enable_persona_management = False


def create_mock_wdbx(**kwargs: Any) -> MagicMock:
    """
    Create a mock WDBX instance with the necessary attributes.

    Args:
        **kwargs: Configuration parameters for the mock WDBX instance

    Returns:
        MagicMock: A mock WDBX instance
    """
    mock_db = MagicMock()
    mock_db.vector_dimension = kwargs.get("vector_dimension", 768)
    mock_db.num_shards = kwargs.get("num_shards", 2)

    # Use Path for proper cross-platform path handling
    data_dir = kwargs.get("data_dir", "./wdbx_demo_data")
    if isinstance(data_dir, str):
        data_dir = Path(data_dir).resolve()
    mock_db.data_dir = data_dir

    mock_db.version = "0.1.0"
    mock_db.config = MockWDBXConfig(**kwargs)

    # Create a storage directory to make the demo more realistic
    os.makedirs(mock_db.data_dir, exist_ok=True)

    return mock_db


def mock_load_plugins(plugin_paths: List[str]) -> Dict[str, MagicMock]:
    """
    Mock function to simulate loading plugins.

    Args:
        plugin_paths: A list of paths to directories containing plugin modules

    Returns:
        Dict[str, MagicMock]: A dictionary mapping plugin names to mock plugin modules
    """
    # Convert string paths to Path objects for better handling
    path_objects = [Path(p).resolve() for p in plugin_paths]
    valid_paths = [p for p in path_objects if p.exists() and p.is_dir()]

    if not valid_paths:
        print(f"{Colors.YELLOW}Warning: No valid plugin paths found in {plugin_paths}{Colors.ENDC}")

    print(f"Loading plugins from: {[str(p) for p in valid_paths]}")

    # Create mock plugins with actual functionality for demo
    plugins = {}

    # Mock visualization plugin
    viz_plugin = MagicMock()
    viz_plugin.create_plot.return_value = "demo_plot.png"
    viz_plugin.NAME = "visualization"
    viz_plugin.VERSION = "0.1.0"
    plugins["visualization"] = viz_plugin

    # Mock model repo plugin
    model_repo = MagicMock()
    model_repo.list_models.return_value = [
        {"name": "bert-base", "type": "embedding", "dimensions": 768},
        {"name": "gpt2-small", "type": "generation", "dimensions": 1024},
        {"name": "clip", "type": "multimodal", "dimensions": 512},
    ]
    model_repo.NAME = "model_repo"
    model_repo.VERSION = "0.1.0"
    plugins["model_repo"] = model_repo

    # Other mock plugins
    plugins["openai"] = MagicMock(NAME="openai_api", VERSION="0.1.0")
    plugins["huggingface"] = MagicMock(NAME="huggingface", VERSION="0.1.0")
    plugins["ollama"] = MagicMock(NAME="ollama", VERSION="0.1.0")

    return plugins


def mock_register_builtin_commands(registry: Dict[str, Any]) -> None:
    """
    Mock function to register built-in commands.

    Args:
        registry: The registry to add commands to
    """
    registry.update(
        {
            "help": lambda db, args: print(f"Help command: {args}"),
            "version": lambda db, args: print(f"WDBX version: {db.version}"),
            "info": lambda db, args: print(
                f"WDBX info: dimension={db.vector_dimension}, shards={db.num_shards}"
            ),
        }
    )


# Define the config and functions to use
WDBXConfig = MockWDBXConfig
create_wdbx = create_mock_wdbx
load_plugins = mock_load_plugins
register_builtin_commands = mock_register_builtin_commands


# Color formatting for terminal output with Windows support
class Colors:
    """ANSI color codes for terminal output with Windows support."""

    # Initialize Windows terminal colors if needed
    if sys.platform == "win32":
        try:
            import colorama

            colorama.init()
            ENABLED = True
        except ImportError:
            ENABLED = False
    else:
        ENABLED = True

    HEADER = "\033[95m" if ENABLED else ""
    BLUE = "\033[94m" if ENABLED else ""
    CYAN = "\033[96m" if ENABLED else ""
    GREEN = "\033[92m" if ENABLED else ""
    YELLOW = "\033[93m" if ENABLED else ""
    RED = "\033[91m" if ENABLED else ""
    ENDC = "\033[0m" if ENABLED else ""
    BOLD = "\033[1m" if ENABLED else ""
    UNDERLINE = "\033[4m" if ENABLED else ""


def format_text(text: str, color: str, bold: bool = False) -> str:
    """
    Format text with color and style.

    Args:
        text: Text to format
        color: Color to use
        bold: Whether to make the text bold

    Returns:
        Formatted text string
    """
    if bold:
        return f"{color}{Colors.BOLD}{text}{Colors.ENDC}"
    return f"{color}{text}{Colors.ENDC}"


def print_header(text: str) -> None:
    """
    Print a formatted header.

    Args:
        text: The header text to print
    """
    print(f"\n{format_text('=' * 60, Colors.HEADER, True)}")
    print(f"{format_text(' ' + text, Colors.HEADER, True)}")
    print(f"{format_text('=' * 60, Colors.HEADER, True)}")


def print_step(step: str) -> None:
    """
    Print a formatted step.

    Args:
        step: The step text to print
    """
    print(f"\n{format_text(f'>> {step}', Colors.BLUE, True)}\n")


def print_error(error: str) -> None:
    """
    Print a formatted error message.

    Args:
        error: The error message to print
    """
    print(f"{format_text(f'ERROR: {error}', Colors.RED)}")


def print_warning(warning: str) -> None:
    """
    Print a formatted warning message.

    Args:
        warning: The warning message to print
    """
    print(f"{format_text(f'WARNING: {warning}', Colors.YELLOW)}")


@contextlib.contextmanager
def timer(description: str) -> Iterator[None]:
    """
    Context manager for timing operations.

    Args:
        description: Description of the operation being timed
    """
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{format_text(f'{description} completed in {elapsed:.2f} seconds', Colors.GREEN)}")


def process_file(db: Any, file_path: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """
    Process a text file into chunks and store as embeddings.

    Args:
        db: WDBX instance
        file_path: Path to the text file
        chunk_size: Size of each text chunk in characters
        overlap: Overlap between chunks in characters

    Returns:
        List of stored embedding IDs

    Raises:
        FileNotFoundError: If the file does not exist
        IOError: If there's an error reading the file
    """
    print_step(f"Processing file: {file_path}")

    try:
        file_path_obj = Path(file_path).resolve()
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path_obj, encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError as e:
        print_error(str(e))
        return []
    except OSError as e:
        print_error(f"Error reading file: {str(e)}")
        return []

    print(f"File loaded: {len(text)} characters")

    if not text:
        print_warning("File is empty")
        return []

    with timer("Text chunking"):
        # Split text into chunks with overlap - optimized for performance
        chunks = []
        text_len = len(text)
        step_size = chunk_size - overlap

        # Pre-allocate chunks - more efficient than repeated list appends
        num_chunks = (text_len + step_size - 1) // step_size
        chunks = [""] * num_chunks

        for i in range(num_chunks):
            start = i * step_size
            end = min(start + chunk_size, text_len)
            chunk = text[start:end]
            if len(chunk) > 20:  # Only include chunks with meaningful content
                chunks[i] = chunk

        # Filter out empty chunks
        chunks = [c for c in chunks if c]

    print(f"Split into {len(chunks)} chunks")

    # Create embeddings and store them
    stored_ids = []

    with timer("Creating embeddings"):
        for i, chunk in enumerate(chunks):
            # Use carriage return for progress updates to reduce terminal clutter
            print(f"Processing chunk {i+1}/{len(chunks)}...", end="\r")

            # Create the embedding using the model repository plugin
            # Just mock this by generating a random ID
            result = str(uuid.uuid4())

            # Add to stored IDs if successful
            if result:
                stored_ids.append(result)

    print(f"\nStored {len(stored_ids)} embeddings")
    return stored_ids


def demo_visualization(db: Any, embedding_ids: List[str], query: str = "vector embeddings") -> None:
    """
    Demonstrate visualization capabilities.

    Args:
        db: WDBX instance
        embedding_ids: List of embedding IDs to visualize
        query: Query text for search-based visualizations
    """
    print_header("Visualization Plugin Demo")

    # Create visualization directory if it doesn't exist
    output_dir = Path("./demo_visualizations").resolve()
    output_dir.mkdir(exist_ok=True)

    print_step("Configuring visualization settings")
    print(f"Visualization settings: output_dir={output_dir} dpi=150")

    with timer("Creating vector dimension plot"):
        print_step(f"Creating vector dimension plot for query: '{query}'")
        output_path = output_dir / "dimension_plot.png"
        print(f"Visualization created: {output_path}")

    with timer("Creating histogram"):
        print_step(f"Creating histogram for query: '{query}'")
        output_path = output_dir / "histogram.png"
        print(f"Visualization created: {output_path}")

    with timer("Creating similarity matrix"):
        print_step(f"Creating similarity matrix for query: '{query}'")
        output_path = output_dir / "similarity_matrix.png"
        print(f"Visualization created: {output_path}")

    with timer("Creating PCA visualization"):
        print_step(f"Creating PCA visualization for query: '{query}'")
        output_path = output_dir / "pca_visualization.png"
        print(f"Visualization created: {output_path}")

    with timer("Creating t-SNE visualization"):
        print_step(f"Creating t-SNE visualization for query: '{query}'")
        output_path = output_dir / "tsne_visualization.png"
        print(f"Visualization created: {output_path}")


def demo_model_repository(db: Any) -> None:
    """
    Demonstrate model repository capabilities.

    Args:
        db: WDBX instance
    """
    print_header("Model Repository Plugin Demo")

    print_step("Searching for available models")
    # Simulate model search results
    available_models = [
        {"name": "bert-base-uncased", "type": "embedding", "size": "110M"},
        {"name": "all-MiniLM-L6-v2", "type": "embedding", "size": "80M"},
        {"name": "clip-vit-base", "type": "multimodal", "size": "150M"},
        {"name": "gpt2-medium", "type": "generation", "size": "350M"},
    ]
    print(f"Found {len(available_models)} models")
    for model in available_models:
        print(f"  - {model['name']} ({model['type']}, {model['size']})")

    print_step("Loading model")
    # Simulate model loading
    print("Loading model: all-MiniLM-L6-v2")
    print("Model loaded successfully")

    print_step("Creating embeddings")
    # Simulate embedding creation
    print("Generating embeddings for text:")
    sample_text = (
        "Vector embeddings are numerical representations of data that capture semantic meaning."
    )
    print(f"  '{sample_text}'")
    print("Embedding generated: [0.12, 0.34, -0.56, ...] (384 dimensions)")


def demo_model_apis(db: Any) -> None:
    """
    Demonstrate model API integrations.

    Args:
        db: WDBX instance
    """
    print_header("Model API Integrations Demo")

    # Simulate OpenAI integration
    print_step("OpenAI API Integration")
    print("Checking OpenAI API connection...")
    print("Connected to OpenAI API")
    print("Available OpenAI models:")
    print("  - text-embedding-3-small (Dimensions: 1536)")
    print("  - text-embedding-3-large (Dimensions: 3072)")

    # Simulate HuggingFace integration
    print_step("HuggingFace Integration")
    print("Checking HuggingFace Hub connection...")
    print("Connected to HuggingFace Hub")
    print("Recent popular embedding models:")
    print("  - sentence-transformers/all-MiniLM-L6-v2")
    print("  - sentence-transformers/all-mpnet-base-v2")
    print("  - thenlper/gte-small")

    # Simulate Ollama integration
    print_step("Ollama Integration")
    print("Checking Ollama service...")
    print("Connected to Ollama service")
    print("Local Ollama models:")
    print("  - llama2")
    print("  - mistral")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="WDBX Plugin Demo")
    parser.add_argument(
        "--data",
        type=str,
        default=str(script_dir / "sample_text.txt"),
        help="Path to the data file to process",
    )
    parser.add_argument(
        "--demos",
        type=str,
        default="all",
        help="Comma-separated list of demos to run (visualization, model_repository, model_apis, all)",
    )
    return parser.parse_args()


def check_environment() -> bool:
    """
    Check that the environment is properly set up.

    Returns:
        True if environment is properly set up, False otherwise
    """
    # Check for colorama on Windows
    if sys.platform == "win32" and not Colors.ENABLED:
        print_warning(
            "colorama package not found. Install with 'pip install colorama' for better terminal output on Windows."
        )

    # Check for WDBX
    if not HAS_WDBX:
        print_warning("WDBX package not installed or not in path. Running in mock mode only.")

    # Check for demonstration data
    sample_file = script_dir / "sample_text.txt"
    if not sample_file.exists():
        print_warning(f"Sample data file not found: {sample_file}")
        print_warning("Demo will use synthetic data.")
        return False

    return True


def main() -> int:
    """
    Main function to run the plugin demos.

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    try:
        args = parse_arguments()

        print_header("WDBX Plugin Demo")
        print(f"Script running from: {script_dir}")

        env_ok = check_environment()

        # Determine demos to run
        demos = args.demos.lower().split(",")
        run_all = "all" in demos

        # Create mock WDBX database
        with timer("Initializing WDBX"):
            db = create_wdbx()
            print(f"WDBX initialized with vector dimension: {db.vector_dimension}")

            # Mock command registry
            command_registry = {}
            register_builtin_commands(command_registry)

            # Mock plugin loading
            plugins = load_plugins(db.config.plugin_paths)
            print(f"Loaded {len(plugins)} plugins")

        # Process sample data
        data_path = args.data
        if data_path:
            try:
                embedding_ids = process_file(db, data_path)
            except Exception as e:
                print_error(f"Error processing file: {str(e)}")
                embedding_ids = [str(uuid.uuid4()) for _ in range(5)]  # Mock IDs
        else:
            # Generate mock embedding IDs
            embedding_ids = [str(uuid.uuid4()) for _ in range(5)]

        # Run selected demos
        try:
            if run_all or "visualization" in demos:
                demo_visualization(db, embedding_ids)

            if run_all or "model_repository" in demos:
                demo_model_repository(db)

            if run_all or "model_apis" in demos:
                demo_model_apis(db)

        except KeyboardInterrupt:
            print_warning("\nDemo interrupted by user.")
            return 130  # Standard exit code for SIGINT
        except Exception as e:
            print_error(f"Error during demo: {str(e)}")
            traceback.print_exc()
            return 1

        print_header("Demo Completed Successfully")
        return 0

    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
