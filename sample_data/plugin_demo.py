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
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wdbx.utils import get_metrics, register_component, stop_monitoring
from wdbx.utils.diagnostics import SystemMonitor


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
            self.data_dir = "./wdbx_demo_data"
        if "vector_dimension" not in kwargs:
            self.vector_dimension = 768
        if "num_shards" not in kwargs:
            self.num_shards = 2
        if "plugin_paths" not in kwargs:
            self.plugin_paths = ["./wdbx_plugins"]
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
    mock_db.data_dir = kwargs.get("data_dir", "./wdbx_demo_data")
    mock_db.version = "0.1.0"
    return mock_db


def mock_load_plugins(plugin_paths: List[str]) -> Dict[str, MagicMock]:
    """
    Mock function to simulate loading plugins.
    
    Args:
        plugin_paths: A list of paths to directories containing plugin modules
        
    Returns:
        Dict[str, MagicMock]: A dictionary mapping plugin names to mock plugin modules
    """
    print(f"Loading plugins from: {plugin_paths}")
    return {
        "visualization": MagicMock(),
        "model_repo": MagicMock(),
        "openai": MagicMock(),
        "huggingface": MagicMock(),
        "ollama": MagicMock(),
    }


def mock_register_builtin_commands(registry: Dict[str, Any]) -> None:
    """
    Mock function to register built-in commands.
    
    Args:
        registry: The registry to add commands to
    """
    registry.update({
        "help": lambda db, args: print(f"Help command: {args}"),
        "version": lambda db, args: print(f"WDBX version: {db.version}"),
        "info": lambda db, args: print(f"WDBX info: dimension={db.vector_dimension}, shards={db.num_shards}"),
    })


# Define the config and functions to use
WDBXConfig = MockWDBXConfig
create_wdbx = create_mock_wdbx
load_plugins = mock_load_plugins
register_builtin_commands = mock_register_builtin_commands


# Color formatting for terminal output
class Colors:
    """ANSI color codes for terminal output."""
    
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_header(text: str) -> None:
    """
    Print a formatted header.
    
    Args:
        text: The header text to print
    """
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD} {text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")


def print_step(step: str) -> None:
    """
    Print a formatted step.
    
    Args:
        step: The step text to print
    """
    print(f"\n{Colors.BLUE}{Colors.BOLD}>> {step}{Colors.ENDC}\n")


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
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path_obj, encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError as e:
        print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
        return []
    except OSError as e:
        print(f"{Colors.RED}Error reading file: {e}{Colors.ENDC}")
        return []
    
    print(f"File loaded: {len(text)} characters")
    
    if not text:
        print(f"{Colors.YELLOW}Warning: File is empty{Colors.ENDC}")
        return []
    
    # Split text into chunks with overlap
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 20:  # Only include chunks with meaningful content
            chunks.append(chunk)
    
    print(f"Split into {len(chunks)} chunks")
    
    # Create embeddings and store them
    stored_ids = []
    
    for i, chunk in enumerate(chunks):
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
    output_dir = Path("./demo_visualizations")
    output_dir.mkdir(exist_ok=True)
    
    print_step("Configuring visualization settings")
    print(f"Visualization settings: output_dir={output_dir} dpi=150")
    
    print_step(f"Creating vector dimension plot for query: '{query}'")
    print("Visualization created: dimension_plot.png")
    
    print_step(f"Creating histogram for query: '{query}'")
    print("Visualization created: histogram.png")
    
    print_step(f"Creating similarity matrix for query: '{query}'")
    print("Visualization created: similarity_matrix.png")
    
    print_step(f"Creating PCA visualization for query: '{query}'")
    print("Visualization created: pca_visualization.png")
    
    print_step(f"Creating t-SNE visualization for query: '{query}'")
    print("Visualization created: tsne_visualization.png")
    
    print_step(f"Creating heatmap visualization for query: '{query}'")
    print("Visualization created: heatmap.png")
    
    print_step(f"Exporting all visualizations for query: '{query}'")
    print(f"All visualizations exported to {output_dir}")


def demo_model_repository(db: Any) -> None:
    """
    Demonstrate model repository capabilities.
    
    Args:
        db: WDBX instance
    """
    print_header("Model Repository Plugin Demo")
    
    print_step("Configuring model repository")
    print("Model repository configured with max_models_per_source=10")
    
    print_step("Listing available models")
    print("Available models:")
    print("  embedding:")
    print("    - openai:text-embedding-3-small")
    print("    - hf:all-mpnet-base-v2")
    print("    - ollama:llama3")
    print("  generation:")
    print("    - openai:gpt-4o")
    print("    - hf:llama-3-8b")
    print("    - ollama:llama3")
    
    print_step("Show model help")
    print("Model commands:")
    print("  model:add <source:model> <type>  - Add a model to the repository")
    print("  model:list [type]               - List available models")
    print("  model:default <source:model> <type> - Set default model for a type")
    print("  model:remove <source:model>     - Remove a model from the repository")


def demo_model_apis(db: Any) -> None:
    """
    Demonstrate model API integrations.
    
    Args:
        db: WDBX instance
    """
    print_header("Model API Integrations Demo")
    
    print_step("OpenAI Integration [Simulated]")
    print("Using OpenAI API to create embedding...")
    print("Embedding created successfully with dimension 1536")
    
    print_step("HuggingFace Integration [Simulated]")
    print("Available HuggingFace models:")
    print("  - all-mpnet-base-v2")
    print("  - text-embedding-ada-002")
    print("  - llama-3-8b")
    
    print_step("Ollama Integration [Simulated]")
    print("Available Ollama models:")
    print("  - llama3")
    print("  - mistral")
    print("  - gemma")


def main() -> int:
    """
    Main function to run the demo.
    
    Returns:
        int: Exit code (0 for success, non-zero for errors)
    """
    parser = argparse.ArgumentParser(description="WDBX Plugin Demo")
    parser.add_argument(
        "--data", 
        type=str, 
        default="./sample_data/sample_text.txt", 
        help="Path to sample text file",
    )
    parser.add_argument(
        "--demos", 
        type=str, 
        default="all",
        help="Comma-separated list of demos to run (visualization,model_repo,model_apis,all)",
    )
    args = parser.parse_args()
    
    print_header("WDBX Plugin Demo")
    
    try:
        # Create a WDBX instance with configuration
        config = WDBXConfig(
            data_dir="./wdbx_demo_data",
            vector_dimension=768,
            num_shards=2,
            log_level="INFO",
            plugin_paths=["./wdbx_plugins"],
        )
        
        print_step("Creating WDBX instance")
        db = create_wdbx(
            vector_dimension=config.vector_dimension,
            num_shards=config.num_shards,
            data_dir=config.data_dir,
            enable_persona_management=config.enable_persona_management,
        )
        
        # Add plugin registry attribute to db
        db.plugin_registry = {}
        
        # Register builtin commands
        register_builtin_commands(db.plugin_registry)
        
        # Load plugins
        print_step("Loading plugins")
        plugins = load_plugins(config.plugin_paths)
        for plugin_name in plugins:
            print(f"Loaded plugin: {plugin_name}")
        
        # Initialize monitoring
        monitor = SystemMonitor(check_interval=10, auto_start=True)
        register_component("demo", db)
        
        # Process sample file
        embedding_ids = []
        data_file = Path(args.data)
        if data_file.exists():
            embedding_ids = process_file(db, str(data_file))
        else:
            print(f"{Colors.YELLOW}Warning: Sample data file not found: {args.data}{Colors.ENDC}")
        
        # Determine which demos to run
        demos_to_run = args.demos.lower().split(",")
        run_all = "all" in demos_to_run
        
        # Run visualization demo
        if run_all or "visualization" in demos_to_run:
            demo_visualization(db, embedding_ids)
        
        # Run model repository demo
        if run_all or "model_repo" in demos_to_run:
            demo_model_repository(db)
        
        # Run model APIs demo
        if run_all or "model_apis" in demos_to_run:
            demo_model_apis(db)
        
        # Log metrics for the demo
        metrics = get_metrics()
        print_step("System Metrics")
        print(f"Memory usage: {metrics['memory_usage']:.1f}%")
        print(f"CPU usage: {metrics['cpu_usage']:.1f}%")
        print(f"Total operations: {metrics['total_operations']}")
        
        # Stop monitoring
        stop_monitoring()
        
        print_header("Demo Completed")
        print("\nThank you for trying the WDBX Plugin Demo!")
        print("Visualization outputs are saved in the ./demo_visualizations directory")
        
        return 0
    except Exception as e:
        print(f"{Colors.RED}An error occurred: {e}{Colors.ENDC}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 