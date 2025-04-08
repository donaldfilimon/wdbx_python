"""
WDBX HuggingFace Hub Plugin

This plugin provides integration with HuggingFace Hub for downloading models,
generating and storing embeddings from HuggingFace models.
"""

import json
import logging
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("WDBX.plugins.huggingface")

# Check for HuggingFace dependencies
HAVE_HUGGINGFACE = False
HAVE_TORCH = False
HAVE_TRANSFORMERS = False

try:
    import torch

    HAVE_TORCH = True
    logger.debug("PyTorch is available for HuggingFace integration.")
except ImportError:
    logger.warning(
        "PyTorch not available. Install with 'pip install torch' to enable HuggingFace integration."
    )

try:
    from transformers import AutoModel, AutoTokenizer

    HAVE_TRANSFORMERS = True
    logger.debug("Transformers is available for HuggingFace integration.")
except ImportError:
    logger.warning(
        "Transformers not available. Install with 'pip install transformers' to enable HuggingFace integration."
    )

try:
    from huggingface_hub import HfApi, login

    HAVE_HUGGINGFACE = True
    logger.debug("HuggingFace Hub is available.")
except ImportError:
    logger.warning(
        "HuggingFace Hub not available. Install with 'pip install huggingface_hub' to enable full HuggingFace integration."
    )

# Default settings
DEFAULT_CONFIG = {
    "api_token": "",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "text_generation_model": "microsoft/phi-2",
    "cache_dir": "./hf_cache",
    "device": "cpu",  # cpu, cuda, mps (for Mac M-series)
    "use_auth": False,
    "max_length": 512,
    "batch_size": 8,
}

# Global configuration
hf_config = DEFAULT_CONFIG.copy()

# Model cache
embedding_model_cache = {}
tokenizer_cache = {}


def register_commands(plugin_registry: Dict[str, Callable]) -> None:
    """
    Register HuggingFace plugin commands with the WDBX CLI.

    Args:
        plugin_registry: Registry to add commands to
    """
    plugin_registry["hf:search"] = cmd_huggingface_search
    plugin_registry["hf:model"] = cmd_huggingface_models
    plugin_registry["hf:download"] = cmd_huggingface_batch
    plugin_registry["hf:config"] = cmd_huggingface_config
    plugin_registry["hf"] = cmd_huggingface_help

    logger.info("HuggingFace commands registered: hf:search, hf:model, hf:download, " "hf:config")

    # Load config if exists
    _load_huggingface_config()

    # Create cache directory
    _ensure_cache_dir()


def cmd_huggingface_missing(db, args: str) -> None:
    """
    Show message when HuggingFace dependencies are missing.

    Args:
        db: WDBX instance
        args: Command arguments (unused)
    """
    print("\033[1;31mError: HuggingFace dependencies are not installed.\033[0m")
    print("Install the required packages with:")
    print("  pip install torch transformers huggingface_hub")
    print("Then restart the WDBX CLI to enable HuggingFace integration.")


def _get_config_path() -> str:
    """Get the full path to the config file."""
    return os.path.join(os.path.expanduser("~"), ".wdbx", "huggingface_config.json")


def _ensure_cache_dir() -> None:
    """Ensure the HuggingFace cache directory exists."""
    try:
        dir_path = hf_config["cache_dir"]
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"HuggingFace cache directory: {dir_path}")
    except Exception as e:
        logger.error(f"Error creating HuggingFace cache directory: {e}")


def _load_huggingface_config() -> None:
    """Load HuggingFace configuration from file."""
    config_path = _get_config_path()
    try:
        if os.path.exists(config_path):
            with open(config_path) as f:
                loaded_config = json.load(f)
                hf_config.update(loaded_config)
                logger.info("Loaded HuggingFace configuration from file.")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding HuggingFace config file {config_path}: {e}")
    except Exception as e:
        logger.error(f"Error loading HuggingFace configuration from {config_path}: {e}")


def _save_huggingface_config() -> None:
    """Save HuggingFace configuration to file."""
    config_path = _get_config_path()
    config_dir = os.path.dirname(config_path)
    try:
        os.makedirs(config_dir, exist_ok=True)
        # Don't save token to disk for security reasons
        save_config = hf_config.copy()
        save_config["api_token"] = "********" if hf_config["api_token"] else ""
        with open(config_path, "w") as f:
            json.dump(save_config, f, indent=2)
        logger.info(f"Saved HuggingFace configuration to {config_path} (without API token).")
    except Exception as e:
        logger.error(f"Error saving HuggingFace configuration to {config_path}: {e}")


def _print_safe_hf_config():
    """Print the current configuration safely (masking API token)."""
    safe_config = hf_config.copy()
    if safe_config["api_token"]:
        safe_config["api_token"] = "********"
    for key, value in safe_config.items():
        print(f"  \033[1m{key}\033[0m = {value}")


def cmd_huggingface_help(db, args: str) -> None:
    """
    Show help for HuggingFace commands.

    Args:
        db: WDBX instance
        args: Command arguments (unused)
    """
    print("\033[1;35mWDBX HuggingFace Integration\033[0m")
    print("The following HuggingFace commands are available:")
    print(
        "\033[1m  hf:embed <text>\033[0m - Create and store an embedding using a HuggingFace model"
    )
    print("\033[1m  hf:models [type=<type>]\033[0m - List popular HuggingFace models")
    print("\033[1m  hf:search <query>\033[0m - Search for models on HuggingFace Hub")
    print("\033[1m  hf:config [key=value]\033[0m - Configure HuggingFace integration settings")
    print("\033[1m  hf:login\033[0m - Login to HuggingFace Hub")
    print("\033[1m  hf:batch <file_path>\033[0m - Process a batch of texts from a file")

    print("\n\033[1;34mHuggingFace Configuration:\033[0m")
    _print_safe_hf_config()

    if not HAVE_HUGGINGFACE or not HAVE_TORCH or not HAVE_TRANSFORMERS:
        print("\n\033[1;31mWarning: Some HuggingFace dependencies are missing.\033[0m")
        if not HAVE_TORCH:
            print("  - PyTorch not found. Install with: pip install torch")
        if not HAVE_TRANSFORMERS:
            print("  - Transformers not found. Install with: pip install transformers")
        if not HAVE_HUGGINGFACE:
            print("  - HuggingFace Hub not found. Install with: pip install huggingface_hub")


def _update_hf_config_value(key: str, value: str):
    """Update a single HuggingFace config value with type conversion."""
    global hf_config
    try:
        if key in ["max_length", "batch_size"]:
            hf_config[key] = int(value)
        elif key in ["use_auth"]:
            hf_config[key] = value.lower() in ("true", "yes", "1", "y")
        elif key == "device":
            # Validate device
            if value not in ["cpu", "cuda", "mps"]:
                print(
                    f"\033[1;31mWarning: Invalid device '{value}'. Using 'cpu'. Valid options: cpu, cuda, mps.\033[0m"
                )
                hf_config[key] = "cpu"
            elif value == "cuda" and not (HAVE_TORCH and torch.cuda.is_available()):
                print(
                    "\033[1;31mWarning: CUDA device requested but not available/found. Using 'cpu'.\033[0m"
                )
                hf_config[key] = "cpu"
            elif value == "mps" and not (HAVE_TORCH and torch.backends.mps.is_available()):
                print(
                    "\033[1;31mWarning: MPS device requested but not available (macOS M-series only). Using 'cpu'.\033[0m"
                )
                hf_config[key] = "cpu"
            else:
                hf_config[key] = value
        else:
            hf_config[key] = value  # Includes api_token, embedding_model, cache_dir etc.
        print(
            f"\033[1;32mSet {key} = {'********' if key == 'api_token' else hf_config[key]}\033[0m"
        )
        return True
    except ValueError:
        print(f"\033[1;31mError: Invalid value type for {key}. Expected int/bool/string.\033[0m")
        return False


def cmd_huggingface_config(db, args: str) -> None:
    """
    Configure HuggingFace integration settings.

    Args:
        db: WDBX instance
        args: Command arguments (key=value pairs)
    """
    global hf_config

    print("\033[1;35mWDBX HuggingFace Configuration\033[0m")

    if not args:
        # Display current configuration
        print("Current configuration:")
        _print_safe_hf_config()
        print("\nTo change a setting, use: hf:config key=value")
        return

    # Parse key=value pairs
    updated_any = False
    parts = args.split()
    for part in parts:
        if "=" in part:
            key, value = part.split("=", 1)
            if key in hf_config:
                if _update_hf_config_value(key, value):
                    updated_any = True
            else:
                print(f"\033[1;31mUnknown configuration key: {key}\033[0m")
        else:
            print(f"\033[1;31mInvalid format: {part}. Use key=value format.\033[0m")

    if updated_any:
        # Save configuration
        _save_huggingface_config()

    # Create cache directory if needed
    _ensure_cache_dir()


def _check_cache_for_model(
    model_name: str,
) -> Tuple[Optional[AutoModel], Optional[AutoTokenizer], Optional[str]]:
    """
    Check if a model and tokenizer are already in the cache.

    Args:
        model_name: The identifier of the model on HuggingFace Hub.

    Returns:
        Tuple of (model, tokenizer, device_name) or (None, None, None) if not in cache.
    """
    global embedding_model_cache, tokenizer_cache

    # Check if model is already in cache
    if model_name in embedding_model_cache and model_name in tokenizer_cache:
        logger.debug(f"Using cached model and tokenizer: {model_name}")
        model = embedding_model_cache[model_name]
        device = str(next(model.parameters()).device)  # Get device from cached model
        return model, tokenizer_cache[model_name], device

    return None, None, None


def _determine_device() -> str:
    """
    Determine the appropriate device (CUDA, MPS, or CPU) based on config and availability.

    Returns:
        String representing the device to use ('cuda', 'mps', or 'cpu')
    """
    target_device = hf_config["device"]

    # Determine actual device based on config and availability
    if target_device == "auto":
        if torch.cuda.is_available():
            actual_device = "cuda"
        elif (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):  # Check MPS for Apple Silicon
            actual_device = "mps"
        else:
            actual_device = "cpu"
        logger.info(f"Auto-detected device: {actual_device}")
    elif target_device == "cuda" and torch.cuda.is_available():
        actual_device = "cuda"
    elif (
        target_device == "mps"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        actual_device = "mps"
    else:
        if target_device != "cpu":
            logger.warning(
                f"Device '{target_device}' requested but not available. Falling back to CPU."
            )
        actual_device = "cpu"

    return actual_device


def _load_tokenizer(model_name: str, use_auth_token: Optional[str]) -> Optional[AutoTokenizer]:
    """
    Load a tokenizer from HuggingFace Hub.

    Args:
        model_name: The model identifier
        use_auth_token: Authentication token or None

    Returns:
        Loaded tokenizer or None on error
    """
    try:
        logger.debug(f"Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=hf_config["cache_dir"], token=use_auth_token
        )
        logger.debug(f"Tokenizer loaded successfully for {model_name}.")
        return tokenizer
    except Exception as e:
        logger.error(f"Error loading tokenizer for {model_name}: {e}")
        return None


def _load_model_to_device(
    model_name: str, use_auth_token: Optional[str], device: str
) -> Optional[AutoModel]:
    """
    Load a model from HuggingFace Hub to a specific device.

    Args:
        model_name: The model identifier
        use_auth_token: Authentication token or None
        device: The device to load the model to

    Returns:
        Loaded model or None on error
    """
    try:
        logger.debug(f"Loading model {model_name} to device {device}...")
        model = AutoModel.from_pretrained(
            model_name, cache_dir=hf_config["cache_dir"], token=use_auth_token
        ).to(device)
        logger.info(f"Model loaded successfully: {model_name} (device: {device})")
        return model
    except Exception as e:
        logger.error(f"Error loading model {model_name} to device {device}: {e}")
        return None


def _load_model_and_tokenizer(
    model_name: str,
) -> Tuple[Optional[AutoModel], Optional[AutoTokenizer], Optional[str]]:
    """
    Load a model and tokenizer from HuggingFace Hub, handling caching and device selection.

    Args:
        model_name: The identifier of the model on HuggingFace Hub.

    Returns:
        Tuple of (model, tokenizer, device_name) or (None, None, None) on error.
    """
    global embedding_model_cache, tokenizer_cache

    # Check cache first
    model, tokenizer, device = _check_cache_for_model(model_name)
    if model is not None:
        return model, tokenizer, device

    logger.info(f"Loading model and tokenizer: {model_name}")

    # Handle authentication
    use_auth_token = hf_config["api_token"] if hf_config["use_auth"] else None

    # Load tokenizer first
    tokenizer = _load_tokenizer(model_name, use_auth_token)
    if tokenizer is None:
        return None, None, None

    # Determine target device
    actual_device = _determine_device()

    # Load model
    model = _load_model_to_device(model_name, use_auth_token, actual_device)
    if model is None:
        return None, None, None

    # Cache the model and tokenizer for future use
    embedding_model_cache[model_name] = model
    tokenizer_cache[model_name] = tokenizer

    return model, tokenizer, actual_device


def _get_embedding_from_huggingface(text: str, model_name: str = None) -> Optional[List[float]]:
    """
    Get an embedding from a HuggingFace model.

    Args:
        text: Text to embed
        model_name: Name of the model to use (defaults to config setting).

    Returns:
        List of floats representing the embedding, or None on error.
    """
    if not HAVE_TORCH or not HAVE_TRANSFORMERS:
        logger.error(
            "Cannot generate HuggingFace embedding: Missing torch or transformers library."
        )
        return None

    if not model_name:
        model_name = hf_config["embedding_model"]

    try:
        # Load model and tokenizer
        model, tokenizer, device = _load_model_and_tokenizer(model_name)

        if model is None or tokenizer is None:
            logger.error(f"Failed to load model or tokenizer for {model_name}")
            return None

        logger.debug(
            f"Generating embedding for text using {model_name} on device {device}: '{text[:100]}...'"
        )

        # Tokenize input
        # Ensure the tokenizer and model are on the same device if possible
        tokens = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=hf_config["max_length"],
            return_tensors="pt",  # PyTorch tensors
        ).to(device)

        # Generate embeddings
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            outputs = model(**tokens)

        # Get the embedding from the last hidden state or pooler output
        if hasattr(outputs, "pooler_output"):
            # BERT-like models
            logger.debug("Using pooler output for embedding.")
            embedding = outputs.pooler_output.squeeze().cpu().numpy()
        else:
            # Models without pooler
            # Mean pooling - take average of all last hidden states
            logger.debug("Using mean pooling of last hidden state for embedding.")
            last_hidden_state = outputs.last_hidden_state
            attention_mask = tokens["attention_mask"]

            # Perform mean pooling
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            )
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = (sum_embeddings / sum_mask).squeeze().cpu().numpy()

        logger.info(
            f"Embedding generated successfully from {model_name} (shape: {embedding.shape})"
        )
        return embedding.tolist()

    except RuntimeError as e:
        # Catch CUDA out of memory errors, etc.
        logger.error(f"RuntimeError during HuggingFace embedding generation: {e}")
        if "out of memory" in str(e):
            logger.error(
                "Consider reducing batch_size or max_length in config, or using a smaller model or CPU."
            )
    except Exception as e:
        logger.error(f"Unexpected error getting embedding from HuggingFace: {e}", exc_info=True)

    return None


def cmd_huggingface_embed(db, args: str) -> None:
    """
    Create and store an embedding using a HuggingFace model.

    Args:
        db: WDBX instance
        args: Text to embed, optionally specify model with 'model=<model_id>'
    """
    if not args:
        print("Usage: hf:embed <text> [model=<model_id>]")
        return

    # Parse arguments
    model_name = hf_config["embedding_model"]
    text_to_embed = args

    # Simple argument parsing: Check if ' model=' is present
    if " model=" in args:
        try:
            # Find the last occurrence of ' model='
            split_index = args.rindex(" model=")
            text_to_embed = args[:split_index].strip()
            model_part = args[split_index:].strip()
            if model_part.startswith("model="):
                potential_model_name = model_part.split("=", 1)[1].strip()
                if potential_model_name:  # Ensure it's not empty
                    model_name = potential_model_name
                else:
                    print(
                        "\033[1;33mWarning: Empty model name provided after 'model=', using default.\033[0m"
                    )
            else:
                # This case should ideally not happen with rindex + startswith logic
                print("\033[1;33mWarning: Malformed model argument, using default model.\033[0m")

        except ValueError:
            # ' model=' not found, use the whole string as text
            pass  # text_to_embed and model_name are already set correctly

    if not text_to_embed:
        print("\033[1;31mError: No text provided to embed.\033[0m")
        return

    print(
        f"\033[1;34mCreating HuggingFace embedding for text using {model_name}:\033[0m '{text_to_embed[:100]}...'"
    )

    try:
        # Get embedding from HuggingFace
        vector = _get_embedding_from_huggingface(text_to_embed, model_name)

        if vector is None:
            print(
                f"\033[1;31mError: Failed to get embedding using {model_name}. Check logs and model availability.\033[0m"
            )
            return

        print(f"\033[1;32mGot embedding with {len(vector)} dimensions.\033[0m")

        # Create embedding vector
        import numpy as np

        from wdbx.data_structures import EmbeddingVector

        embedding = EmbeddingVector(
            vector=np.array(vector).astype(np.float32),  # Ensure correct dtype
            metadata={
                "source": "huggingface",
                "model": model_name,
                "text": text_to_embed,
                "timestamp": time.time(),
            },
        )

        # Store embedding
        vector_id = db.store_embedding(embedding)

        print(f"\033[1;32mEmbedding stored with ID:\033[0m {vector_id}")

        # Create conversation block
        block_id = db.create_conversation_block(
            data={
                "source": "huggingface",
                "type": "embedding_creation",
                "model": model_name,
                "text": text_to_embed,
                "vector_id": vector_id,
                "timestamp": time.time(),
            },
            # embeddings=[embedding] # Embeddings might be stored implicitly or not needed here
        )

        logger.info(f"Created conversation block {block_id} for HuggingFace embedding {vector_id}")

    except Exception as e:
        # Catch errors during embedding storage or block creation
        print(f"\033[1;31mError storing embedding or creating conversation block: {e}\033[0m")
        logger.error(f"Error in hf:embed storage phase: {e}", exc_info=True)


def _parse_model_list_args(args: str) -> dict:
    """
    Parse arguments for listing models from the HuggingFace Hub.

    Args:
        args: Command line arguments as a string

    Returns:
        Dictionary of parsed arguments
    """
    # Set defaults
    list_args = {"sort": "downloads", "direction": -1, "limit": 15}

    if not args:
        return list_args

    # Parse key=value pairs
    for part in args.split():
        if "=" in part:
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key == "limit":
                try:
                    list_args[key] = int(value)
                except ValueError:
                    print(f"\033[1;33mWarning: Invalid limit '{value}', using default.\033[0m")
            elif key == "type":  # Map 'type' to 'task'
                list_args["task"] = value
            else:
                list_args[key] = value  # Pass other args like sort, direction
        else:
            print(
                f"\033[1;33mWarning: Ignoring invalid argument '{part}'. Use key=value format.\033[0m"
            )

    return list_args


def _fetch_and_display_models(api: HfApi, list_args: dict) -> None:
    """
    Fetch and display models from the HuggingFace Hub.

    Args:
        api: HuggingFace Hub API instance
        list_args: Dictionary of arguments for listing models
    """
    task_filter = list_args.get("task")
    filter_desc = f" ({task_filter} task)" if task_filter else ""
    print(
        f"\033[1;34mFetching top {list_args['limit']} HuggingFace models{filter_desc} sorted by downloads...\033[0m"
    )

    try:
        models_iterator = api.list_models(**list_args)
        models = list(models_iterator)  # Convert iterator to list

        if not models:
            print(f"No models found matching the criteria: {list_args}")
            return

        # Display models
        _display_model_list(models)

    except Exception as e:
        print(f"\033[1;31mError fetching models from HuggingFace Hub: {e}\033[0m")
        logger.error(f"Error listing HuggingFace models: {e}", exc_info=True)


def _display_model_list(models: list) -> None:
    """
    Display a formatted list of HuggingFace models.

    Args:
        models: List of model information objects
    """
    print("\033[1;32mFound Models:\033[0m")
    print(f"{'-' * 100}")

    current_embed_model = hf_config["embedding_model"]
    # current_gen_model = hf_config["text_generation_model"] # Add if relevant

    for model_info in models:
        model_id = model_info.id
        downloads = model_info.downloads
        likes = model_info.likes
        task = model_info.pipeline_tag or "unknown"

        # Highlight current models
        highlight = ""
        if model_id == current_embed_model:
            highlight = " \033[1;32m(current embed)\033[0m"
        # Add similar check for generation model if needed

        print(f"{model_id:<60} {downloads:<15,} {likes:<10,} {task}{highlight}")

    print("\nTo set the default embedding model: hf:config embedding_model=<model_id>")


def cmd_huggingface_models(db, args: str) -> None:
    """
    List popular HuggingFace models from the Hub.

    Args:
        db: WDBX instance
        args: Optional arguments (e.g., type=text-classification, limit=20)
    """
    if not HAVE_HUGGINGFACE:
        print("\033[1;31mError: HuggingFace Hub library not available.\033[0m")
        print("Install with: pip install huggingface_hub")
        return

    # Parse arguments
    list_args = _parse_model_list_args(args)

    # Fetch and display models
    api = HfApi()
    _fetch_and_display_models(api, list_args)


def cmd_huggingface_search(db, args: str) -> None:
    """
    Search for models on HuggingFace Hub.

    Args:
        db: WDBX instance
        args: Search query, optionally specify limit with 'limit=<int>'
    """
    if not HAVE_HUGGINGFACE:
        print("\033[1;31mError: HuggingFace Hub library not available.\033[0m")
        print("Install with: pip install huggingface_hub")
        return

    if not args:
        print("Usage: hf:search <query> [limit=<int>]")
        return

    # Simple argument parsing for limit
    query = args
    limit = 10  # Default limit
    if " limit=" in args:
        try:
            split_index = args.rindex(" limit=")
            query = args[:split_index].strip()
            limit_part = args[split_index:].strip()
            if limit_part.startswith("limit="):
                limit_val = limit_part.split("=", 1)[1].strip()
                limit = int(limit_val)
            else:
                print("\033[1;33mWarning: Malformed limit argument, using default limit.\033[0m")
        except (ValueError, IndexError):
            print(f"\033[1;33mWarning: Invalid limit value, using default {limit}.\033[0m")

    if not query:
        print("\033[1;31mError: No search query provided.\033[0m")
        return

    print(f"\033[1;34mSearching HuggingFace Hub for: '{query}' (limit: {limit})...")

    try:
        api = HfApi()
        models_iterator = api.list_models(search=query, limit=limit, sort="downloads", direction=-1)
        models = list(models_iterator)

        if not models:
            print(f"No models found matching query: '{query}'")
            return

        print(f"\033[1;32mFound {len(models)} models matching '{query}':\033[0m")
        # Print header
        print(f"{'-' * 100}")

        for model_info in models:
            model_id = model_info.id
            downloads = model_info.downloads
            likes = model_info.likes
            task = model_info.pipeline_tag or "unknown"
            print(f"{model_id:<60} {downloads:<15,} {likes:<10,} {task}")

    except Exception as e:
        print(f"\033[1;31mError searching models on HuggingFace Hub: {e}\033[0m")
        logger.error(f"Error searching HuggingFace models: {e}", exc_info=True)


def cmd_huggingface_login(db, args: str) -> None:
    """
    Login to HuggingFace Hub.

    Args:
        db: WDBX instance
        args: Optional token (if provided, attempts non-interactive login)
    """
    if not HAVE_HUGGINGFACE:
        print("\033[1;31mError: HuggingFace Hub library not available.\033[0m")
        print("Install with: pip install huggingface_hub")
        return

    token = args.strip() if args else None

    try:
        if token:
            print("Attempting non-interactive login with provided token...")
            login(token=token)
            # Verify login by getting user info
            api = HfApi()
            user_info = api.whoami()
            print(
                f"\033[1;32mSuccessfully logged in as: {user_info.get('name')} ({user_info.get('email', 'email hidden')})\033[0m"
            )
            # Optionally save the token to config (maybe ask user?)
            save_token = (
                input("Save this token to WDBX configuration for future use? (yes/no): ")
                .lower()
                .strip()
            )
            if save_token == "yes":
                hf_config["api_token"] = token
                hf_config["use_auth"] = True
                _save_huggingface_config()
                print("Token saved to configuration (will be used if use_auth=true).")
            else:
                print("Token not saved to configuration.")
        else:
            print("Opening interactive HuggingFace Hub login...")
            print("Follow the instructions to paste your API token.")
            # Use the interactive login flow from huggingface_hub
            login()
            # Verification happens implicitly within login(), or raises error
            print("\033[1;32mSuccessfully logged in interactively.\033[0m")

        print("Use: hf:config use_auth=true api_token=<your_token>")

    except Exception as e:
        print(f"\033[1;31mError during HuggingFace login: {e}\033[0m")
        logger.error(f"Error during HuggingFace login: {e}", exc_info=True)


def _parse_batch_args(args: str) -> Tuple[str, str]:
    """
    Parse arguments for batch processing.

    Args:
        args: Command line arguments

    Returns:
        Tuple of (file_path, model_name)
    """
    if not args:
        return "", ""

    file_path = args
    model_name = hf_config["embedding_model"]

    if " model=" in args:
        try:
            split_index = args.rindex(" model=")
            file_path = args[:split_index].strip()
            model_part = args[split_index:].strip()
            if model_part.startswith("model="):
                potential_model_name = model_part.split("=", 1)[1].strip()
                if potential_model_name:
                    model_name = potential_model_name
                else:
                    print("\033[1;33mWarning: Empty model name provided, using default.\033[0m")
            else:
                print("\033[1;33mWarning: Malformed model argument, using default.\033[0m")
        except ValueError:
            pass  # Use default model

    return file_path, model_name


def _load_texts_from_file(file_path: str) -> List[str]:
    """
    Load texts from a file, one per line.

    Args:
        file_path: Path to the file

    Returns:
        List of text strings
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines
    except Exception as e:
        print(f"\033[1;31mError reading file: {e}\033[0m")
        return []


def _process_batch_texts(
    db, texts: List[str], model_name: str, model, tokenizer, device: str
) -> Tuple[int, int]:
    """
    Process a batch of texts to generate embeddings.

    Args:
        db: WDBX instance
        texts: List of texts to embed
        model_name: Model name
        model: Loaded model
        tokenizer: Loaded tokenizer
        device: Device for processing

    Returns:
        Tuple of (processed_count, error_count)
    """
    import numpy as np

    from wdbx.data_structures import EmbeddingVector

    processed_count = 0
    error_count = 0
    hf_config["batch_size"]

    start_time = time.time()
    model.eval()  # Set model to evaluation mode

    print(f"Processing {len(texts)} texts... ")

    for i, text in enumerate(texts):
        if i % 50 == 0 and i > 0:  # Print progress periodically
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            print(f"  Processed {i}/{len(texts)} lines ({rate:.1f} lines/sec)... ")

        try:
            # Tokenize single text
            tokens = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=hf_config["max_length"],
                return_tensors="pt",
            ).to(device)

            # Get embedding
            with torch.no_grad():
                outputs = model(**tokens)

            # Process output
            vector_np = _extract_embedding_from_outputs(outputs, tokens)

            if vector_np is not None:
                # Create and store EmbeddingVector
                embedding = EmbeddingVector(
                    vector=vector_np.astype(np.float32),
                    metadata={
                        "source": "huggingface:batch",
                        "text": text,
                        "model": model_name,
                        "timestamp": time.time(),
                    },
                )
                db.store_embedding(embedding)
                processed_count += 1
            else:
                logger.warning(f"Failed to generate embedding for line: {text[:50]}...")
                error_count += 1

        except Exception as e:
            logger.error(f"Error processing line '{text[:50]}...': {e}")
            error_count += 1

    return processed_count, error_count


def _extract_embedding_from_outputs(outputs, tokens):
    """
    Extract embedding vector from model outputs.

    Args:
        outputs: Model outputs
        tokens: Tokenized inputs

    Returns:
        Numpy array with embeddings or None on error
    """
    try:
        if hasattr(outputs, "pooler_output"):
            # BERT-like models
            vector_np = outputs.pooler_output.squeeze().cpu().numpy()
        else:
            # Models without pooler
            # Mean pooling - take average of all last hidden states
            last_hidden_state = outputs.last_hidden_state
            attention_mask = tokens["attention_mask"]

            # Perform mean pooling
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            )
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            vector_np = (sum_embeddings / sum_mask).squeeze().cpu().numpy()

        return vector_np
    except Exception as e:
        logger.error(f"Error extracting embedding: {e}")
        return None


def cmd_huggingface_batch(db, args: str) -> None:
    """
    Process a batch of texts from a file using HuggingFace embeddings.

    Args:
        db: WDBX instance
        args: Path to the input file (one text per line), optionally specify model with 'model=<model_id>'
    """
    if not args:
        print("Usage: hf:batch <file_path> [model=<model_id>]")
        return

    # Parse arguments
    file_path, model_name = _parse_batch_args(args)

    # Validate file exists
    if not os.path.exists(file_path):
        print(f"\033[1;31mError: File not found: {file_path}\033[0m")
        return

    print(
        f"\033[1;34mStarting batch processing for file: {file_path} using model {model_name}\033[0m"
    )

    # Load model
    model, tokenizer, device = _load_model_and_tokenizer(model_name)
    if model is None or tokenizer is None:
        print(f"\033[1;31mError: Failed to load model {model_name} for batch processing.\033[0m")
        return

    # Load texts from file
    texts = _load_texts_from_file(file_path)
    if not texts:
        print("\033[1;33mNo text found in the file.\033[0m")
        return

    print(f"Found {len(texts)} texts to process.")

    # Process texts
    start_time = time.time()
    processed_count, error_count = _process_batch_texts(
        db, texts, model_name, model, tokenizer, device
    )

    # Report results
    end_time = time.time()
    duration = end_time - start_time
    print("\n\033[1;32mBatch processing completed.\033[0m")
    print(f"  Processed: {processed_count} items")
    print(f"  Errors:    {error_count} items")
    print(f"  Duration:  {duration:.2f} seconds")
