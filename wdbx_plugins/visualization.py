"""
WDBX Visualization Plugin.

This plugin adds visualization capabilities to the WDBX CLI, enabling users
to visualize vector embeddings, similarity matrices, and distribution histograms.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Callable, Optional

# Constants for magic numbers
MAX_VECTORS_FULL_LEGEND = 10
MAX_VECTORS_SMALL_LEGEND = 20
MIN_VECTORS_FOR_SIMILARITY = 2
MIN_VECTORS_FOR_PCA = 2
MIN_VECTORS_FOR_TSNE = 2
SHORT_ID_DISPLAY_LENGTH = 8
MAX_ID_DISPLAY_LENGTH_BEFORE_TRUNC = 10
PCA_N_COMPONENTS = 2
TSNE_N_COMPONENTS = 2
TSNE_DEFAULT_ITER = 250
TSNE_DEFAULT_PERPLEXITY_LOW = 2
TSNE_DEFAULT_PERPLEXITY_HIGH = 5  # Used for > 10 vectors
TSNE_PERPLEXITY_THRESHOLD = 10  # Num vectors threshold for perplexity calc
RANDOM_STATE_SEED = 42

logger = logging.getLogger("WDBX.plugins.visualization")

# Set up default configuration
DEFAULT_OUTPUT_DIR = "./visualizations"
DEFAULT_COLOR_SCHEME = "viridis"
DEFAULT_DPI = 150
DEFAULT_MAX_VECTORS = 50
DEFAULT_FIG_WIDTH = 10.0
DEFAULT_FIG_HEIGHT = 8.0
DEFAULT_SHOW_PLOTS = False

# Check for dependencies
HAVE_NUMPY = False
HAVE_MATPLOTLIB = False
HAVE_SKLEARN = False

try:
    import numpy as np

    HAVE_NUMPY = True
    logger.debug("NumPy is available for visualization.")
except ImportError:
    logger.warning("NumPy not available. Install with 'pip install numpy' to enable visualization.")

try:
    import matplotlib.pyplot as plt

    HAVE_MATPLOTLIB = True
    logger.debug("Matplotlib is available for visualization.")
except ImportError:
    logger.warning(
        "Matplotlib not available. Install with 'pip install matplotlib' to enable visualization."
    )

try:
    # Use importlib to check for sklearn availability without importing it
    import importlib.util

    if importlib.util.find_spec("sklearn") is not None:
        HAVE_SKLEARN = True
        logger.debug("scikit-learn is available for advanced visualizations.")
    else:
        HAVE_SKLEARN = False
        logger.warning(
            "scikit-learn not available. Install with 'pip install scikit-learn' to enable advanced visualizations."
        )
except ImportError:
    HAVE_SKLEARN = False
    logger.warning(
        "scikit-learn not available. Install with 'pip install scikit-learn' to enable advanced visualizations."
    )

# Global configuration dictionary
vis_config = {
    "output_dir": DEFAULT_OUTPUT_DIR,
    "color_scheme": DEFAULT_COLOR_SCHEME,
    "dpi": DEFAULT_DPI,
    "show_plots": DEFAULT_SHOW_PLOTS,
    "max_vectors": DEFAULT_MAX_VECTORS,
    "figure_width": DEFAULT_FIG_WIDTH,
    "figure_height": DEFAULT_FIG_HEIGHT,
}


def register_commands(plugin_registry: dict[str, Callable]) -> None:
    """
    Register visualization commands with the CLI.

    Args:
        plugin_registry: Registry to add commands to
    """
    plugin_registry["plot"] = cmd_plot
    plugin_registry["histogram"] = cmd_histogram
    plugin_registry["similarity"] = cmd_similarity_matrix
    plugin_registry["pca"] = cmd_pca_visualization
    plugin_registry["tsne"] = cmd_tsne_visualization
    plugin_registry["heatmap"] = cmd_vector_heatmap
    plugin_registry["vis"] = cmd_visualization_help
    plugin_registry["vis:config"] = cmd_visualization_config
    plugin_registry["vis:export"] = cmd_export_all_visualizations

    # Load config and ensure directory exists
    _load_vis_config()
    _ensure_vis_output_dir()  # Ensure dir exists after loading config

    logger.info(
        "Visualization commands registered: plot, histogram, similarity, pca, tsne, heatmap, vis, vis:config, vis:export"
    )


def _get_vis_config_path() -> str:
    """Get the full path to the visualization config file."""
    return os.path.join(os.path.expanduser("~"), ".wdbx", "visualization_config.json")


def _ensure_vis_output_dir() -> None:
    """Ensure the visualization output directory exists based on config."""
    try:
        dir_path = vis_config["output_dir"]
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Visualization output directory ensured: {dir_path}")
    except Exception as e:
        logger.error(
            f"Error creating visualization output directory '{vis_config['output_dir']}': {e}"
        )


def _load_vis_config() -> None:
    """Load visualization configuration from file."""
    config_path = _get_vis_config_path()
    try:
        if os.path.exists(config_path):
            with open(config_path) as f:
                loaded_config = json.load(f)
                # Update defaults with loaded values
                vis_config.update(loaded_config)
                logger.info("Loaded visualization configuration.")
    except json.JSONDecodeError as e:
        logger.error(f"Error loading visualization config: {e}")
        logger.warning("Using default visualization configuration.")
    except Exception as e:
        logger.error(f"Error loading visualization config: {e}")
        logger.warning("Using default visualization configuration.")

    # Ensure output directory exists
    _ensure_vis_output_dir()


def _save_vis_config() -> None:
    """Save visualization configuration to file."""
    config_path = _get_vis_config_path()
    config_dir = os.path.dirname(config_path)
    try:
        os.makedirs(config_dir, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(vis_config, f, indent=2)
        logger.info(f"Saved visualization configuration to {config_path}")
    except Exception as e:
        logger.error(f"Error saving visualization configuration to {config_path}: {e}")


def _print_vis_config():
    """Print the current visualization configuration."""
    for key, value in vis_config.items():
        print(f"  \033[1m{key}\033[0m = {value}")


def _validate_numeric_value(key: str, value: str) -> tuple[bool, int, str]:
    """Validate and convert a numeric configuration value.

    Args:
        key: The configuration key
        value: The string value to convert

    Returns:
        Tuple of (success, converted_value, error_message)
    """
    try:
        num_value = int(value)
        if num_value <= 0:
            return False, 0, "Value must be a positive number."
        return True, num_value, ""
    except ValueError:
        return False, 0, "Value must be a number."


def _update_vis_config_value(key: str, value: str) -> bool:
    """Update a single visualization config value with type conversion and validation."""
    global vis_config

    try:
        if key in ("figure_width", "figure_height", "dpi", "max_vectors", "hist_bins"):
            # Convert numeric values
            success, num_value, error = _validate_numeric_value(key, value)
            if not success:
                print(f"\033[1;31mError: {error}\033[0m")
                return False
            vis_config[key] = num_value
        elif key == "truncate_labels":
            # Convert boolean values
            vis_config[key] = value.lower() in ("true", "yes", "1", "y")
        elif key == "output_dir":
            # Basic validation: ensure it's a string
            if not isinstance(value, str) or not value:
                print("\033[1;31mError: output_dir must be a non-empty string.\033[0m")
                return False
            vis_config[key] = value
        elif key == "color_scheme":
            # Basic validation: ensure it's a string
            if not isinstance(value, str) or not value:
                print("\033[1;31mError: color_scheme must be a non-empty string.\033[0m")
                return False
            vis_config[key] = value
        else:
            # For any other keys
            vis_config[key] = value

        print(f"\033[1;32mUpdated {key} = {vis_config[key]}.\033[0m")
        return True
    except Exception as e:
        print(f"\033[1;31mError updating configuration: {e}.\033[0m")
        return False


def cmd_visualization_config(db, args: str) -> None:
    """
    Configure visualization settings.

    Args:
        db: WDBX instance
        args: Command arguments (key=value pairs)
    """
    print("\033[1;35mWDBX Visualization Configuration\033[0m")

    if not args:
        # Display current configuration
        print("Current configuration:")
        _print_vis_config()
        print("\nTo change a setting, use: vis:config key=value")
        return

    # Parse key=value pairs
    updated_any = False
    parts = args.split()
    for part in parts:
        if "=" in part:
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key in vis_config:
                if _update_vis_config_value(key, value):
                    updated_any = True
            else:
                print(f"\033[1;31mUnknown configuration key: {key}\033[0m")
        else:
            print(f"\033[1;31mInvalid format: '{part}'. Use key=value format.\033[0m")

    if updated_any:
        _save_vis_config()
        _ensure_vis_output_dir()  # Re-ensure dir in case it was changed


def cmd_visualization_help(db, args: str) -> None:
    """
    Show help for visualization commands.

    Args:
        db: WDBX instance
        args: Command arguments (unused)
    """
    print("\033[1;35mWDBX Visualization Commands\033[0m")
    print("The following visualization commands are available:")
    print("\033[1m  plot <query>\033[0m - Plot vector dimensions for vectors matching query")
    print("\033[1m  histogram <query>\033[0m - Plot histogram of vector values")
    print("\033[1m  similarity <query>\033[0m - Create similarity matrix heatmap")
    print("\033[1m  pca <query>\033[0m - Visualize vectors using PCA dimensionality reduction")
    print("\033[1m  tsne <query>\033[0m - Visualize vectors using t-SNE dimensionality reduction")
    print("\033[1m  heatmap <query>\033[0m - Create a heatmap visualization of vector components")
    print("\033[1m  vis:config [key=value]\033[0m - Configure visualization settings")
    print("\033[1m  vis:export <query>\033[0m - Export all visualizations for a query")

    print("\n\033[1;34mVisualization Settings:\033[0m")
    _print_vis_config()

    missing_deps = []
    if not HAVE_MATPLOTLIB:
        missing_deps.append("matplotlib")
    if not HAVE_SKLEARN:
        missing_deps.append("scikit-learn")
    if not HAVE_NUMPY:
        missing_deps.append("numpy")

    if missing_deps:
        print("\n\033[1;31mWarning: Missing dependencies.\033[0m")
        print("Install the following packages to enable all visualization features:")
        print(f"  pip install {' '.join(missing_deps)}")


def _try_get_vector_by_method(db, vec_id: str, method_name: str) -> Optional[np.ndarray]:
    """
    Try to get a vector using a specific method.

    Args:
        db: WDBX instance
        vec_id: Vector ID
        method_name: Method name to try

    Returns:
        Vector as numpy array or None if not found/error
    """
    try:
        if not hasattr(db, method_name):
            return None

        if method_name == "get_vector_by_id":
            vector_data = db.get_vector_by_id(vec_id)
            if vector_data is not None:
                return np.asarray(vector_data, dtype=np.float32)

        elif method_name == "get_embedding_vector":
            embedding_obj = db.get_embedding_vector(vec_id)
            if embedding_obj is not None and hasattr(embedding_obj, "vector"):
                return embedding_obj.vector  # Already numpy array

        elif method_name == "retrieve_vector":
            vector_data = db.retrieve_vector(vec_id)
            if vector_data is not None:
                return np.asarray(vector_data, dtype=np.float32)

        return None
    except Exception as e:
        logger.error(f"Error using {method_name} to get vector {vec_id}: {e}")
        return None


def get_actual_vectors(db, vector_ids: list[str]) -> list[np.ndarray]:
    """
    Retrieve the actual vector arrays from vector IDs.

    Attempts to handle different database implementations by trying multiple
    methods of retrieving vectors.

    Args:
        db: WDBX instance
        vector_ids: List of vector IDs to retrieve

    Returns:
        List of numpy arrays containing the actual vectors
    """
    if not vector_ids:
        return []

    vectors = []
    methods_to_try = ["get_vector_by_id", "get_embedding_vector", "retrieve_vector"]

    for vec_id in vector_ids:
        vector = None

        # Try each method until we get a vector
        for method in methods_to_try:
            vector = _try_get_vector_by_method(db, vec_id, method)
            if vector is not None:
                vectors.append(vector)
                break

        # If no method worked, log a warning
        if vector is None:
            logger.warning(f"Could not retrieve vector with ID: {vec_id}")

    if not vectors:
        logger.warning("No vectors could be retrieved from the provided IDs.")

    return vectors


def get_vectors_for_query(
    db, query: str, max_vectors: int = None
) -> tuple[list[str], list[np.ndarray]]:
    """
    Get vectors for a search query.

    Args:
        db: Database instance
        query: Query string
        max_vectors: Maximum number of vectors to return (if None, uses vis_config["max_vectors"])

    Returns:
        Tuple of (vector_ids, vectors)
    """
    if not HAVE_NUMPY:
        print("\033[1;31mError: NumPy not available. Cannot get vectors.\033[0m")
        print("Install with: pip install numpy")
        return [], []

    if max_vectors is None:
        max_vectors = vis_config.get("max_vectors", DEFAULT_MAX_VECTORS)

    logger.info(f"Searching for vectors matching query: '{query}' (limit: {max_vectors})")

    try:
        # First try to convert query to embedding (for similarity search)
        query_embedding = db.create_embedding_from_text(query)

        try:
            # Try search_similar_vectors method first
            if hasattr(db, "search_similar_vectors"):
                results = db.search_similar_vectors(query_embedding, top_k=max_vectors)
                vector_ids = [result[0] for result in results]
            # Fall back to search method if available
            elif hasattr(db, "search") or hasattr(db.vector_store, "search"):
                search_method = getattr(db, "search", None) or db.vector_store.search
                results = search_method(query_embedding, k=max_vectors)
                vector_ids = [result[0] for result in results]
            # Fall back to search_with_scores method if available
            elif hasattr(db, "search_with_scores") or hasattr(
                db.vector_store, "search_with_scores"
            ):
                search_method = getattr(db, "search_with_scores",
                                        None) or db.vector_store.search_with_scores
                results = search_method(query_embedding, k=max_vectors)
                vector_ids = [result[0] for result in results]
            else:
                logger.error("Database does not implement search or search_with_scores method")
                print("\033[1;31mError: Database does not support vector search.\033[0m")
                return [], []

        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            print(f"\033[1;31mError during vector search: {e}\033[0m")
            return [], []

    except Exception as e:
        logger.error(f"Error creating embedding from query: {e}")
        print(f"\033[1;31mError creating embedding from query '{query}': {e}\033[0m")
        return [], []

    # Get the actual vector embeddings
    vectors = get_actual_vectors(db, vector_ids)

    if not vectors:
        print(f"No vectors found or error retrieving vectors for query '{query}'.")

    return vector_ids, vectors


def save_figure(fig, filename_prefix: str, title: str) -> str:
    """
    Save a matplotlib figure to disk with proper naming.

    Args:
        fig: The matplotlib figure to save
        filename_prefix: Prefix for the filename (e.g., 'pca', 'tsne')
        title: Title or identifier for the plot (e.g., the query)

    Returns:
        Path to the saved file or None if error
    """
    try:
        output_dir = Path(vis_config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        # Create a more descriptive and safe filename
        safe_title = "".join(c for c in title if c.isalnum() or c in (" ", "_", "-")).rstrip()
        safe_prefix = "".join(c for c in filename_prefix if c.isalnum() or c == "_").rstrip("_")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_prefix}_{safe_title.replace(' ', '_')}_{timestamp}.png"

        filepath = output_dir / filename

        fig.savefig(filepath, dpi=vis_config["dpi"], bbox_inches="tight")
        logger.info(f"Saved visualization '{title}' to: {filepath}")
        plt.close(fig)  # Close the figure to free memory
        return str(filepath)
    except Exception as e:
        logger.error(f"Error saving figure: {e}")
        return None


def cmd_plot(db, args: str) -> None:
    """
    Plot vector components as line plots.

    Args:
        db: WDBX instance
        args: Query to find vectors to plot
    """
    if not HAVE_MATPLOTLIB or not HAVE_NUMPY:
        print("\033[1;31mError: Matplotlib and NumPy are required for plotting.\033[0m")
        return

    if not args:
        print("\033[1;33mUsage: plot <query>\033[0m")
        return

    query = args
    print(f"\033[1;34mGenerating vector plot for query: '{query}'...\033[0m")
    vector_ids, vectors = get_vectors_for_query(
        db, query, vis_config.get("max_vectors", DEFAULT_MAX_VECTORS)
    )

    if not vectors:
        return

    try:
        # Create plot
        fig, ax = plt.subplots(
            figsize=(
                vis_config.get("figure_width", DEFAULT_FIG_WIDTH),
                vis_config.get("figure_height", DEFAULT_FIG_HEIGHT),
            )
        )
        color_map = plt.get_cmap(vis_config.get("color_scheme", DEFAULT_COLOR_SCHEME))

        # Get dimensionality
        vec_dim = len(vectors[0])
        x_axis = np.arange(vec_dim)

        # Plot each vector
        for i, (vector_id, vector) in enumerate(zip(vector_ids, vectors)):
            # Generate color based on position in list
            color = color_map(i / len(vectors))

            # Generate label (truncate vector ID for readability)
            truncate_labels = vis_config.get("truncate_labels", True)
            label = f"Vec {i+1} ({vector_id[:8]}...)" if truncate_labels else f"Vec {i+1}"

            # Plot the vector
            ax.plot(x_axis, vector, label=label, alpha=0.7, linewidth=2, color=color)

        # Add labels and title
        ax.set_title(f"Vector Plot for: '{query}'")
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.5)

        # Add legend with smaller font size for many vectors
        if len(vectors) > MAX_VECTORS_FULL_LEGEND:
            ax.legend(fontsize="small", loc="best")
        else:
            ax.legend(loc="best")

        # Save the figure
        output_path = save_figure(fig, "vector_plot", query)
        if output_path:
            print(f"\033[1;32mVector plot saved to: {output_path}\033[0m")

        # Close the figure to free memory
        plt.close(fig)

    except Exception as e:
        print(f"\033[1;31mError creating plot: {e}\033[0m")


def cmd_histogram(db, args: str) -> None:
    """
    Plot a histogram of all component values from vectors matching a query.

    Args:
        db: WDBX instance
        args: Query string
    """
    if not HAVE_MATPLOTLIB or not HAVE_NUMPY:
        print(
            "\033[1;31mError: Matplotlib and/or NumPy not available. Cannot create histogram.\033[0m"
        )
        print("Install with: pip install matplotlib numpy")
        return

    if not args:
        print("Usage: histogram <query>")
        return

    query = args
    print(f"\033[1;34mGenerating histogram for query: '{query}'...\033[0m")
    vector_ids, vectors = get_vectors_for_query(db, query, vis_config["max_vectors"])

    if not vectors:
        print(f"No vectors found or error retrieving vectors for query '{query}'.")
        return

    fig = _generate_histogram(vectors, query)
    if fig:
        filepath = save_figure(fig, "histogram", f"Histogram for '{query}'")
        if filepath and not vis_config["show_plots"]:
            print(f"Histogram saved to: {filepath}")
        elif vis_config["show_plots"]:
            plt.show()
        else:
            print("\033[1;31mError saving histogram figure.\033[0m")
    else:
        print("\033[1;31mFailed to generate histogram.\033[0m")


def cmd_similarity_matrix(db, args: str) -> None:
    """
    Create a heatmap visualizing the cosine similarity matrix between vectors matching a query.

    Args:
        db: WDBX instance
        args: Query string
    """
    # Requires scikit-learn for cosine_similarity
    if not HAVE_MATPLOTLIB or not HAVE_NUMPY or not HAVE_SKLEARN:
        print(
            "\033[1;31mError: Matplotlib, NumPy, or scikit-learn not available. Cannot create similarity matrix.\033[0m"
        )
        print("Install with: pip install matplotlib numpy scikit-learn")
        return

    if not args:
        print("Usage: similarity <query>")
        return

    query = args
    print(f"\033[1;34mGenerating similarity matrix for query: '{query}'...\033[0m")
    vector_ids, vectors = get_vectors_for_query(db, query, vis_config["max_vectors"])

    if not vectors:
        print(f"No vectors found or error retrieving vectors for query '{query}'.")
        return
    if len(vectors) < MIN_VECTORS_FOR_SIMILARITY:
        print(
            f"Need at least {MIN_VECTORS_FOR_SIMILARITY} vectors to compute similarity matrix, found {len(vectors)}.\033[0m"
        )
        return

    fig = _generate_similarity_matrix(vector_ids, vectors, query)
    if fig:
        filepath = save_figure(fig, "similarity_matrix", f"Similarity Matrix for '{query}'")
        if filepath and not vis_config["show_plots"]:
            print(f"Similarity matrix saved to: {filepath}")
        elif vis_config["show_plots"]:
            plt.show()
        else:
            print("\033[1;31mError saving similarity matrix figure.\033[0m")
    else:
        print("\033[1;31mFailed to generate similarity matrix.\033[0m")


def cmd_pca_visualization(db, args: str) -> None:
    """
    Visualize vectors matching a query using PCA dimensionality reduction (2D scatter plot).

    Args:
        db: WDBX instance
        args: Query string
    """
    if not HAVE_MATPLOTLIB or not HAVE_NUMPY or not HAVE_SKLEARN:
        print(
            "\033[1;31mError: Matplotlib, NumPy, or scikit-learn not available. Cannot create PCA plot.\033[0m"
        )
        print("Install with: pip install matplotlib numpy scikit-learn")
        return

    if not args:
        print("Usage: pca <query>")
        return

    query = args
    print(f"\033[1;34mGenerating PCA visualization for query: '{query}'...\033[0m")
    vector_ids, vectors = get_vectors_for_query(db, query, vis_config["max_vectors"])

    if not vectors:
        print(f"No vectors found or error retrieving vectors for query '{query}'.")
        return
    if len(vectors) < MIN_VECTORS_FOR_PCA:
        print(f"Need at least {MIN_VECTORS_FOR_PCA} vectors for PCA, found {len(vectors)}.\033[0m")
        return

    fig = _generate_pca(vector_ids, vectors, query)
    if fig:
        filepath = save_figure(fig, "pca_plot", f"PCA Plot for '{query}'")
        if filepath and not vis_config["show_plots"]:
            print(f"PCA plot saved to: {filepath}")
        elif vis_config["show_plots"]:
            plt.show()
        else:
            print("\033[1;31mError saving PCA plot figure.\033[0m")
    else:
        print("\033[1;31mFailed to generate PCA plot.\033[0m")


def cmd_tsne_visualization(db, args: str) -> None:
    """
    Visualize vectors matching a query using t-SNE dimensionality reduction (2D scatter plot).

    Args:
        db: WDBX instance
        args: Query string
    """
    if not HAVE_MATPLOTLIB or not HAVE_NUMPY or not HAVE_SKLEARN:
        print(
            "\033[1;31mError: Matplotlib, NumPy, or scikit-learn not available. Cannot create t-SNE plot.\033[0m"
        )
        print("Install with: pip install matplotlib numpy scikit-learn")
        return

    if not args:
        print("Usage: tsne <query>")
        return

    query = args
    print(
        f"\033[1;34mGenerating t-SNE visualization for query: '{query}'... (This may take a moment)\033[0m"
    )
    vector_ids, vectors = get_vectors_for_query(
        db, query, vis_config.get("max_vectors", DEFAULT_MAX_VECTORS)
    )

    if not vectors:
        return

    if len(vectors) < MIN_VECTORS_FOR_TSNE:
        print(
            f"\033[1;33mNeed at least {MIN_VECTORS_FOR_TSNE} vectors for t-SNE visualization. Only got {len(vectors)}.\033[0m"
        )
        return

    try:
        # Import t-SNE here to avoid issues if not installed
        from sklearn.manifold import TSNE

        # Convert vectors to numpy array if needed
        vectors_array = np.array(vectors)

        # Apply t-SNE for dimensionality reduction
        tsne = TSNE(
            n_components=TSNE_N_COMPONENTS,
            perplexity=min(TSNE_DEFAULT_PERPLEXITY_HIGH, len(
                vectors) - 1) if len(vectors) > TSNE_PERPLEXITY_THRESHOLD else TSNE_DEFAULT_PERPLEXITY_LOW,
            n_iter=TSNE_DEFAULT_ITER,
            random_state=RANDOM_STATE_SEED,
        )

        # Perform t-SNE
        tsne_result = tsne.fit_transform(vectors_array)

        # Create plot
        fig, ax = plt.subplots(
            figsize=(
                vis_config.get("figure_width", DEFAULT_FIG_WIDTH),
                vis_config.get("figure_height", DEFAULT_FIG_HEIGHT),
            )
        )

        # Get colormap
        try:
            # Use newer API (matplotlib 3.7+)
            colors = plt.colormaps[vis_config.get("color_scheme", DEFAULT_COLOR_SCHEME)]
        except (KeyError, AttributeError):
            # Fallback for older matplotlib versions
            colors = plt.cm.get_cmap(
                vis_config.get("color_scheme", DEFAULT_COLOR_SCHEME), len(vectors)
            )

        # Plot each vector as a point in 2D space
        for i, (vector_id, point) in enumerate(zip(vector_ids, tsne_result)):
            # Shorten ID for label
            vec_id_str = str(vector_id)
            short_id = (
                f"{vec_id_str[:SHORT_ID_DISPLAY_LENGTH]}..."
                if len(vec_id_str) > MAX_ID_DISPLAY_LENGTH_BEFORE_TRUNC
                else vec_id_str
            )

            # Plot point
            ax.scatter(
                point[0],
                point[1],
                color=colors(i),
                label=f"Vec {i+1} ({short_id})",
                s=100,
                alpha=0.7,
            )

            # Add labels to points if not too many
            if len(vectors) <= MAX_VECTORS_SMALL_LEGEND:
                ax.annotate(
                    f"{i+1}",
                    (point[0], point[1]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                )

        # Customize plot
        ax.set_title(f"t-SNE Visualization for: '{query}'")
        ax.grid(True, linestyle="--", alpha=0.5)

        # Handle legend based on number of vectors
        if len(vectors) <= MAX_VECTORS_FULL_LEGEND:
            ax.legend(loc="best")
        elif len(vectors) <= MAX_VECTORS_SMALL_LEGEND:
            ax.legend(fontsize="small", loc="best")

        # Save the figure
        output_path = save_figure(fig, "tsne", query)
        if output_path:
            print(f"\033[1;32mt-SNE visualization saved to: {output_path}\033[0m")

        # Close the figure to free memory
        plt.close(fig)

    except Exception as e:
        print(f"\033[1;31mError performing t-SNE: {e}\033[0m")
        print("\033[1;31mFailed to generate t-SNE plot.\033[0m")


def cmd_vector_heatmap(db, args: str) -> None:
    """
    Create a heatmap visualizing the component values of vectors matching a query.

    Args:
        db: WDBX instance
        args: Query string
    """
    if not HAVE_MATPLOTLIB or not HAVE_NUMPY:
        print(
            "\033[1;31mError: Matplotlib and/or NumPy not available. Cannot create heatmap.\033[0m"
        )
        print("Install with: pip install matplotlib numpy")
        return

    if not args:
        print("Usage: heatmap <query>")
        return

    query = args
    print(f"\033[1;34mGenerating heatmap for query: '{query}'...\033[0m")
    vector_ids, vectors = get_vectors_for_query(db, query, vis_config["max_vectors"])

    if not vectors:
        print(f"No vectors found or error retrieving vectors for query '{query}'.")
        return

    fig = _generate_heatmap(vector_ids, vectors, query)
    if fig:
        filepath = save_figure(fig, "vector_heatmap", f"Heatmap for '{query}'")
        if filepath and not vis_config["show_plots"]:
            print(f"Heatmap saved to: {filepath}")
        elif vis_config["show_plots"]:
            plt.show()
        else:
            print("\033[1;31mError saving heatmap figure.\033[0m")
    else:
        print("\033[1;31mFailed to generate heatmap.\033[0m")


def _setup_export_directory() -> tuple[str, str, bool]:
    """Set up a directory for exporting multiple visualizations.

    Returns:
        tuple[str, str, bool]: export_dir, unique_id, directory_created
    """
    # Create timestamp for consistent filenames
    timestamp = int(time.time())

    # Save original output dir
    original_dir = vis_config["output_dir"]

    # Create subfolder for this export
    export_dir = os.path.join(original_dir, f"export_{timestamp}")
    try:
        os.makedirs(export_dir, exist_ok=True)
        return export_dir, original_dir, True
    except Exception as e:
        logger.error(f"Error creating export directory: {e}")
        return "", original_dir, False


def _generate_all_standard_visualizations(vector_ids, vectors, query: str) -> None:
    """Generate standard visualizations that don't require scikit-learn.

    Args:
        vector_ids: List of vector IDs
        vectors: List of vector arrays
        query: The search query
    """
    print("\033[1;34mGenerating vector plot...\033[0m")
    _generate_vector_plot(vector_ids, vectors, query)

    print("\033[1;34mGenerating histogram...\033[0m")
    _generate_histogram(vectors, query)

    print("\033[1;34mGenerating similarity matrix...\033[0m")
    _generate_similarity_matrix(vector_ids, vectors, query)

    print("\033[1;34mGenerating vector heatmap...\033[0m")
    _generate_heatmap(vector_ids, vectors, query)


def _generate_all_advanced_visualizations(vector_ids, vectors, query: str) -> None:
    """Generate advanced visualizations that require scikit-learn.

    Args:
        vector_ids: List of vector IDs
        vectors: List of vector arrays
        query: The search query
    """
    if HAVE_SKLEARN and len(vectors) >= MIN_VECTORS_FOR_PCA:
        print("\033[1;34mGenerating PCA visualization...\033[0m")
        _generate_pca(vector_ids, vectors, query)

        print("\033[1;34mGenerating t-SNE visualization...\033[0m")
        _generate_tsne(vector_ids, vectors, query)
    elif not HAVE_SKLEARN:
        print("\033[1;33mSkipping PCA and t-SNE (scikit-learn not available).\033[0m")
    elif len(vectors) < MIN_VECTORS_FOR_PCA:
        print(
            f"\033[1;33mSkipping PCA and t-SNE (need at least {MIN_VECTORS_FOR_PCA} vectors).\033[0m"
        )


def cmd_export_all_visualizations(db, args: str) -> None:
    """
    Generates and saves all available visualizations for vectors matching a query.

    Args:
        db: WDBX instance
        args: Query string to search for vectors
    """
    if not args:
        print("\033[1;33mUsage: vis:export <query>\033[0m")
        return

    if not HAVE_MATPLOTLIB or not HAVE_NUMPY:
        print("\033[1;31mError: Matplotlib and NumPy are required for visualizations.\033[0m")
        return

    query = args
    print(f"\033[1;34mExporting all visualizations for query: '{query}'...\033[0m")

    # Instead of processing vectors directly, call the individual command functions
    # These will handle their own vector retrieval and error conditions

    # Standard visualizations
    cmd_plot(db, query)
    cmd_histogram(db, query)
    cmd_similarity_matrix(db, query)
    cmd_vector_heatmap(db, query)

    # Advanced visualizations that require scikit-learn
    if HAVE_SKLEARN:
        cmd_pca_visualization(db, query)
        cmd_tsne_visualization(db, query)
    else:
        print("\033[1;33mSkipping PCA and t-SNE (scikit-learn not available).\033[0m")

    print(f"\033[1;32mAll available visualizations exported to: {vis_config['output_dir']}\033[0m")


def _generate_vector_plot(vector_ids, vectors, query):
    """Internal helper to generate a vector plot for export_all_visualizations."""
    if not HAVE_MATPLOTLIB or not HAVE_NUMPY:
        logger.error("Cannot generate vector plot: Missing matplotlib or numpy.")
        return None
    if not vectors:
        logger.error("Cannot generate vector plot: No vectors provided.")
        return None

    try:
        fig, ax = plt.subplots(figsize=(vis_config["figure_width"], vis_config["figure_height"]))
        num_vectors = len(vectors)
        if num_vectors == 0:
            return None
        dimension = len(vectors[0])
        indices = np.arange(dimension)

        # Use a colormap
        try:
            colors = plt.cm.get_cmap(vis_config["color_scheme"], num_vectors)
        except ValueError:  # Handle invalid colormap name
            logger.warning(
                f"Invalid color_scheme '{vis_config['color_scheme']}'. Falling back to '{DEFAULT_COLOR_SCHEME}'."
            )
            vis_config["color_scheme"] = DEFAULT_COLOR_SCHEME  # Reset to default
            colors = plt.cm.get_cmap(vis_config["color_scheme"], num_vectors)

        for i, vec in enumerate(vectors):
            # Shorten ID for label, handle potential non-string IDs gracefully
            vec_id_str = str(vector_ids[i])
            label = f"Vec {i+1} ({vec_id_str[:8]}...)"
            ax.plot(
                indices, vec, marker=".", linestyle="-", alpha=0.7, color=colors(i), label=label
            )

        ax.set_title(
            f"Vector Component Values for '{query}' ({num_vectors} vectors x {dimension} dims)"
        )
        ax.set_xlabel("Dimension Index")
        ax.set_ylabel("Component Value")
        ax.grid(True, linestyle=":", alpha=0.6)
        # Improve legend placement for potentially many vectors
        if num_vectors <= MAX_VECTORS_FULL_LEGEND:
            ax.legend(loc="best")
        return fig
    except Exception as e:
        logger.error(f"Error generating vector plot: {e}")
        return None


def _generate_histogram(vectors, query):
    """Internal helper to generate a histogram for export_all_visualizations."""
    if not vectors:
        return None

    # Create plot
    fig = plt.figure(figsize=(vis_config["figure_width"], vis_config["figure_height"]))

    # Combine all values for histogram
    all_values = np.concatenate([v.flatten() for v in vectors])

    plt.hist(all_values, bins=30, alpha=0.7, color=plt.get_cmap(vis_config["color_scheme"])(0.5))
    plt.title(f"Value Distribution for Query: {query}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.5)

    # Add distribution statistics
    mean = np.mean(all_values)
    median = np.median(all_values)
    std_dev = np.std(all_values)

    stats_text = f"Mean: {mean:.4f}\nMedian: {median:.4f}\nStd Dev: {std_dev:.4f}"
    plt.figtext(0.15, 0.8, stats_text, bbox={"facecolor": "white", "alpha": 0.8, "pad": 5})

    # Save figure
    filepath = save_figure(fig, "histogram", query)
    if filepath:
        print(f"\033[1;32mHistogram saved to: {filepath}\033[0m")
    return filepath


def _generate_similarity_matrix(vector_ids, vectors, query):
    """
    Generate a similarity matrix visualization for vector comparison.

    Args:
        vector_ids: List of vector IDs
        vectors: List of vector arrays
        query: The query text for the title

    Returns:
        Matplotlib figure object or None if error
    """
    if not HAVE_MATPLOTLIB or not HAVE_NUMPY:
        logger.error("Cannot generate similarity matrix: Missing matplotlib or numpy.")
        return None
    if not vectors or len(vectors) < MIN_VECTORS_FOR_SIMILARITY:
        logger.error(
            f"Cannot generate similarity matrix: Need at least {MIN_VECTORS_FOR_SIMILARITY} vectors."
        )
        return None

    try:
        # Create similarity matrix
        n = len(vectors)
        similarity_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                v1 = vectors[i]
                v2 = vectors[j]

                # Calculate cosine similarity
                dot_product = np.dot(v1, v2)
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)

                # Avoid division by zero
                if norm1 == 0 or norm2 == 0:
                    similarity = 0
                else:
                    similarity = dot_product / (norm1 * norm2)

                similarity_matrix[i, j] = similarity

        # Create heatmap
        fig, ax = plt.subplots(figsize=(vis_config["figure_width"], vis_config["figure_height"]))
        cax = ax.imshow(similarity_matrix, cmap=vis_config["color_scheme"], interpolation="nearest")
        plt.colorbar(cax, label="Cosine Similarity")

        # Add labels
        short_ids = [f"{vid[:6]}..." for vid in vector_ids]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(short_ids, rotation=45)
        ax.set_yticklabels(short_ids)

        ax.set_title(f"Similarity Matrix for Query: {query}")

        # Add grid
        ax.grid(False)

        # Adjust layout for readability
        plt.tight_layout()

        return fig
    except Exception as e:
        logger.error(f"Error generating similarity matrix: {e}")
        return None


def _generate_heatmap(vector_ids, vectors, query):
    """
    Generate a heatmap visualization of vector components.

    Args:
        vector_ids: List of vector IDs
        vectors: List of vector arrays
        query: The query text for the title

    Returns:
        Matplotlib figure object or None if error
    """
    if not HAVE_MATPLOTLIB or not HAVE_NUMPY:
        logger.error("Cannot generate heatmap: Missing matplotlib or numpy.")
        return None
    if not vectors:
        logger.error("Cannot generate heatmap: No vectors provided.")
        return None

    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(vis_config["figure_width"], vis_config["figure_height"]))

        # Limit dimensions for readability
        max_dims = min(50, len(vectors[0]) if vectors and len(vectors) > 0 else 50)

        # Build the matrix for the heatmap
        matrix = np.zeros((len(vectors), max_dims))
        for i, vec in enumerate(vectors):
            matrix[i, :] = vec[:max_dims]

        # Create heatmap
        cax = ax.imshow(matrix, aspect="auto", cmap=vis_config["color_scheme"])
        plt.colorbar(cax, label="Component Value")

        # Add labels
        short_ids = [f"{vid[:6]}..." for vid in vector_ids]
        ax.set_yticks(range(len(vectors)))
        ax.set_yticklabels(short_ids)
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Vector ID")

        ax.set_title(f"Vector Heatmap for Query: {query}")

        # Adjust layout
        plt.tight_layout()

        return fig
    except Exception as e:
        logger.error(f"Error generating heatmap: {e}")
        return None


def _generate_pca(vector_ids, vectors, query):
    """Internal helper to generate a PCA visualization for export_all_visualizations."""
    if not HAVE_SKLEARN or len(vectors) < MIN_VECTORS_FOR_PCA:
        return None

    try:
        from sklearn.decomposition import PCA

        # Create PCA and transform vectors
        pca = PCA(n_components=PCA_N_COMPONENTS)
        vectors_2d = pca.fit_transform(vectors)

        # Create scatter plot
        fig = plt.figure(figsize=(vis_config["figure_width"], vis_config["figure_height"]))
        plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.7)

        # Add labels for points
        for i, vec_id in enumerate(vector_ids):
            plt.annotate(f"{vec_id[:6]}...", (vectors_2d[i, 0], vectors_2d[i, 1]))

        plt.title(f"PCA Visualization for Query: {query}")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True, linestyle="--", alpha=0.5)

        # Add explained variance
        variance_explained = pca.explained_variance_ratio_
        plt.figtext(
            0.15,
            0.85,
            f"Variance explained:\nComponent 1: {variance_explained[0]:.2%}\nComponent 2: {variance_explained[1]:.2%}",
            bbox={"facecolor": "white", "alpha": 0.8, "pad": 5},
        )

        # Save figure
        filepath = save_figure(fig, "pca", query)
        if filepath:
            print(f"\033[1;32mPCA visualization saved to: {filepath}\033[0m")
        return filepath
    except Exception as e:
        print(f"\033[1;31mError performing PCA: {e}\033[0m")
        return None


def _generate_tsne(vector_ids, vectors, query):
    """Internal helper to generate a t-SNE visualization for export_all_visualizations."""
    if not HAVE_SKLEARN or len(vectors) < MIN_VECTORS_FOR_TSNE:
        return None

    try:
        from sklearn.manifold import TSNE

        # Create t-SNE and transform vectors
        perplexity_val = (
            min(TSNE_DEFAULT_PERPLEXITY_HIGH, len(vectors) - 1)
            if len(vectors) > TSNE_PERPLEXITY_THRESHOLD
            else TSNE_DEFAULT_PERPLEXITY_LOW
        )
        tsne = TSNE(
            n_components=TSNE_N_COMPONENTS,
            perplexity=perplexity_val,
            n_iter=TSNE_DEFAULT_ITER,
            random_state=RANDOM_STATE_SEED,
        )
        vectors_2d = tsne.fit_transform(vectors)

        # Create scatter plot
        fig, ax = plt.subplots(
            figsize=(
                vis_config.get("figure_width", DEFAULT_FIG_WIDTH),
                vis_config.get("figure_height", DEFAULT_FIG_HEIGHT),
            )
        )
        plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.7)

        # Add labels for points
        for i, vec_id in enumerate(vector_ids):
            plt.annotate(f"{vec_id[:6]}...", (vectors_2d[i, 0], vectors_2d[i, 1]))

        plt.title(f"t-SNE Visualization for Query: {query}")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True, linestyle="--", alpha=0.5)

        # Handle legend based on number of vectors
        if len(vectors) <= MAX_VECTORS_FULL_LEGEND:
            ax.legend(loc="best")
        elif len(vectors) <= MAX_VECTORS_SMALL_LEGEND:
            ax.legend(fontsize="small", loc="best")

        # Save figure
        output_path = save_figure(fig, "tsne", query)
        if output_path:
            print(f"\033[1;32mt-SNE visualization saved to: {output_path}\033[0m")

        # Close the figure to free memory
        plt.close(fig)

        return output_path
    except Exception as e:
        print(f"\033[1;31mError performing t-SNE: {e}\033[0m")
        return None
