"""
WDBX Ollama Plugin.

This plugin provides integration with Ollama for generating and storing
embeddings from local language models.
"""

import json
import logging
import math
import os
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import requests

from wdbx.data_structures import EmbeddingVector

logger = logging.getLogger("WDBX.plugins.ollama")

# Constants for magic numbers
HTTP_STATUS_OK = 200
HTTP_STATUS_BAD_REQUEST = 400
HTTP_STATUS_NOT_FOUND = 404
HTTP_STATUS_REQUEST_TIMEOUT = 408
HTTP_STATUS_INTERNAL_SERVER_ERROR = 500
HTTP_STATUS_GATEWAY_TIMEOUT = 504

# Default settings
DEFAULT_CONFIG = {
    "api_base": "http://localhost:11434",
    "embedding_model": "nomic-embed-text",
    "generation_model": "llama3",
    "timeout": 60,
    "batch_size": 10,
    "max_tokens": 1024,
    "temperature": 0.7,
    "save_responses": True,
    "response_dir": "./wdbx_ollama_responses",
}

# Global configuration
ollama_config = DEFAULT_CONFIG.copy()


def register_commands(plugin_registry: Dict[str, Callable]) -> None:
    """
    Register Ollama commands with the CLI.

    Args:
        plugin_registry: Dictionary mapping command names to functions.
    """
    # Register commands
    plugin_registry["ollama:embed"] = cmd_ollama_embed
    plugin_registry["ollama:generate"] = cmd_ollama_generate
    plugin_registry["ollama:models"] = cmd_ollama_models
    plugin_registry["ollama:config"] = cmd_ollama_config
    plugin_registry["ollama:batch"] = cmd_ollama_batch
    plugin_registry["ollama"] = cmd_ollama_help

    logger.info(
        "Ollama commands registered: ollama:embed, ollama:generate, ollama:models, ollama:config, ollama:batch"
    )

    # Load config if exists
    _load_ollama_config()

    # Create response directory if saving is enabled
    if ollama_config["save_responses"]:
        _ensure_response_dir()


def _ensure_response_dir() -> None:
    """Ensure the response directory exists."""
    try:
        dir_path = ollama_config["response_dir"]
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Ollama response directory: {dir_path}")
    except Exception as e:
        logger.error(f"Error creating Ollama response directory: {e}")


def _get_config_path() -> str:
    """Get the full path to the config file."""
    return os.path.join(os.path.expanduser("~"), ".wdbx", "ollama_config.json")


def _load_ollama_config() -> None:
    """Load Ollama configuration from file."""
    config_path = _get_config_path()
    try:
        if os.path.exists(config_path):
            with open(config_path) as f:
                loaded_config = json.load(f)
                ollama_config.update(loaded_config)
                logger.info("Loaded Ollama configuration from file.")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding Ollama config file {config_path}: {e}")
    except Exception as e:
        logger.error(f"Error loading Ollama configuration from {config_path}: {e}")


def _save_ollama_config() -> None:
    """Save Ollama configuration to file."""
    config_path = _get_config_path()
    config_dir = os.path.dirname(config_path)
    try:
        os.makedirs(config_dir, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(ollama_config, f, indent=2)
        logger.info(f"Saved Ollama configuration to {config_path}")
    except Exception as e:
        logger.error(f"Error saving Ollama configuration to {config_path}: {e}")


def cmd_ollama_help(db, args: str) -> None:
    """
    Show help for Ollama commands.

    Args:
        db: WDBX instance
        args: Command arguments (unused)
    """
    print("\033[1;35mWDBX Ollama Integration\033[0m")
    print("The following Ollama commands are available:")
    print("\033[1m  ollama:embed <text>\033[0m - Create and store an embedding using Ollama")
    print("\033[1m  ollama:generate <text>\033[0m - Generate text using an Ollama model")
    print("\033[1m  ollama:models\033[0m - List available Ollama models")
    print("\033[1m  ollama:config [key=value]\033[0m - Configure Ollama integration settings")
    print("\033[1m  ollama:batch <file_path>\033[0m - Process a batch of texts from a file")

    print("\n\033[1;34mOllama Configuration:\033[0m")
    for key, value in ollama_config.items():
        print(f"  \033[1m{key}\033[0m = {value}")


def _update_ollama_config_value(key: str, value: str):
    """Update a single config value with type conversion."""
    global ollama_config
    try:
        if key in ["timeout", "batch_size", "max_tokens"]:
            ollama_config[key] = int(value)
        elif key in ["temperature"]:
            ollama_config[key] = float(value)
        elif key in ["save_responses"]:
            ollama_config[key] = value.lower() in ("true", "yes", "1", "y")
        else:
            ollama_config[key] = value
        print(f"\033[1;32mSet {key} = {ollama_config[key]}\033[0m")
        return True
    except ValueError:
        print(
            f"\033[1;31mError: Invalid value type for {key}. Expected int/float/bool/string.\033[0m"
        )
        return False


def cmd_ollama_config(db, args: str) -> None:
    """
    Configure Ollama integration settings.

    Args:
        db: WDBX instance
        args: Command arguments (key=value pairs)
    """
    global ollama_config

    print("\033[1;35mWDBX Ollama Configuration\033[0m")

    if not args:
        # Display current configuration
        print("Current configuration:")
        for key, value in ollama_config.items():
            print(f"  \033[1m{key}\033[0m = {value}")
        print("\nTo change a setting, use: ollama:config key=value")
        return

    # Parse key=value pairs
    updated_any = False
    parts = args.split()
    for part in parts:
        if "=" in part:
            key, value = part.split("=", 1)
            if key in ollama_config:
                if _update_ollama_config_value(key, value):
                    updated_any = True
            else:
                print(f"\033[1;31mUnknown configuration key: {key}\033[0m")
        else:
            print(f"\033[1;31mInvalid format: {part}. Use key=value format.\033[0m")

    if updated_any:
        # Save configuration
        _save_ollama_config()

    # Create response directory if saving is enabled
    if ollama_config["save_responses"]:
        _ensure_response_dir()


# --- Helper for API Calls --- #


def _make_api_request(
    url: str, method: str, payload: Dict[str, Any]
) -> Optional[requests.Response]:
    """
    Make an HTTP request to the Ollama API.

    Args:
        url: Full URL to the API endpoint
        method: HTTP method to use (GET or POST)
        payload: Request payload for POST requests

    Returns:
        Response object or None on error
    """
    try:
        if method.upper() == "POST":
            response = requests.post(url, json=payload, timeout=ollama_config["timeout"])
        elif method.upper() == "GET":
            response = requests.get(url, timeout=ollama_config["timeout"])
        else:
            logger.error(f"Unsupported HTTP method: {method}")
            return None

        return response

    except requests.exceptions.ConnectionError:
        logger.error(
            f"Could not connect to Ollama API at {ollama_config['api_base']}. Is Ollama running?"
        )
        return None
    except requests.exceptions.Timeout:
        logger.error(f"Ollama API request timed out ({ollama_config['timeout']}s)")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama API request failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during Ollama API request: {e}")
        return None


def _handle_error_response(
    response: requests.Response, endpoint: str, payload: Dict[str, Any]
) -> str:
    """
    Handle error responses from the Ollama API.

    Args:
        response: The HTTP response
        endpoint: The API endpoint
        payload: The request payload

    Returns:
        Error message string
    """
    error_msg = f"Ollama API Error ({endpoint}): {response.status_code}"

    try:
        error_data = response.json()
        if "error" in error_data:
            error_msg += f" - {error_data['error']}"
        else:
            error_msg += f" - {response.text[:100]}"  # Add raw text if no specific error
    except (json.JSONDecodeError, ValueError):
        error_msg += f" - {response.text[:100]}"

    logger.error(error_msg)

    # Specific suggestions for common errors
    if response.status_code == HTTP_STATUS_NOT_FOUND and endpoint == "/api/embeddings":
        logger.error(f"Model '{payload.get('model')}' not found. Use ollama:models.")
    elif response.status_code == HTTP_STATUS_INTERNAL_SERVER_ERROR:
        logger.error("Ollama server error. Check Ollama service status.")
    elif response.status_code in (
        HTTP_STATUS_REQUEST_TIMEOUT,
        HTTP_STATUS_GATEWAY_TIMEOUT,
    ):
        logger.error("Ollama request timed out. Increase timeout or check server load.")
    elif (
        response.status_code == HTTP_STATUS_BAD_REQUEST
        and "parameter not supported" in error_msg
    ):
        logger.error(
            f"Model '{payload.get('model')}' might not support the requested operation or options."
        )

    return error_msg


def _call_ollama_api(
    endpoint: str, payload: Dict[str, Any], method: str = "POST"
) -> Optional[Dict[str, Any]]:
    """
    Generic helper to call Ollama API endpoints.

    Args:
        endpoint: API endpoint path (e.g., "/api/embeddings")
        payload: Request payload
        method: HTTP method to use (default: POST)

    Returns:
        Response data as dictionary or None on error
    """
    url = f"{ollama_config['api_base']}{endpoint}"

    # Make the API request
    response = _make_api_request(url, method, payload)
    if response is None:
        return None

    # Handle HTTP errors
    if response.status_code != HTTP_STATUS_OK:
        _handle_error_response(response, endpoint, payload)
        return None

    # Parse response
    try:
        resp_data = response.json()

        # Save response if configured
        if ollama_config["save_responses"]:
            _save_ollama_response(endpoint, resp_data)

        return resp_data
    except json.JSONDecodeError:
        logger.error("Invalid JSON response from Ollama API")
        return None


def _save_ollama_response(endpoint: str, data: Dict[str, Any]) -> None:
    """Save the Ollama API JSON response to a file."""
    try:
        filename = f"{endpoint.replace('/', '_')}_{uuid.uuid4()}.json"
        response_file = os.path.join(ollama_config["response_dir"], filename)
        with open(response_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Saved Ollama response ({endpoint}) to {response_file}")
    except Exception as e:
        logger.error(f"Error saving Ollama response: {e}")


# --- Embedding Function --- #


def _get_embedding_from_ollama(text: str, model: str = None) -> Optional[List[float]]:
    """
    Get embedding from Ollama API.

    Args:
        text: Text to embed
        model: Model name (if None, default from config is used)

    Returns:
        Embedding vector as list of floats or None on error
    """
    if not model:
        model = ollama_config["embedding_model"]

    # Call API
    resp_data = _call_ollama_api("/api/embeddings", {"model": model, "prompt": text})

    if resp_data:
        vector = resp_data.get("embedding")
        if isinstance(vector, list) and all(isinstance(x, (int, float)) for x in vector):
            logger.debug(f"Received embedding with {len(vector)} dimensions")
            return vector

        logger.error(f"Invalid embedding format received from Ollama: {type(vector)}")

    logger.error("Invalid or incomplete embedding response from Ollama.")
    return None


# --- Generation Function --- #


def _generate_from_ollama(prompt: str, model: str = None) -> Optional[Dict[str, Any]]:
    """
    Generate text from Ollama model.

    Args:
        prompt: Text prompt
        model: Model name (if None, default from config is used)

    Returns:
        Response data dictionary or None on error
    """
    if not model:
        model = ollama_config["generation_model"]

    # Call API
    resp_data = _call_ollama_api(
        "/api/generate",
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": ollama_config.get("generation_options", {}),
        },
    )

    if resp_data:
        logger.debug(f"Received generation response from Ollama model {model}")
        return resp_data

    logger.error("Invalid or incomplete generation response from Ollama.")
    return None


# --- Model Listing Function --- #


def _get_ollama_models() -> Optional[List[Dict[str, Any]]]:
    """Get list of available models from Ollama."""
    resp_data = _call_ollama_api("/api/tags", {}, method="GET")
    if resp_data:
        models = resp_data.get("models")
        if isinstance(models, list):
            return models

        logger.error("Invalid models list format received from Ollama.")

    logger.error("Failed to get models list or invalid response structure from Ollama.")
    return None


# --- CLI Commands --- #


def cmd_ollama_embed(db: Any, args: str) -> None:
    """
    Create and store an embedding using Ollama.

    Args:
        db: WDBX instance
        args: Text to embed
    """
    if not args:
        print("Usage: ollama:embed <text_to_embed>")
        return

    print(f"\033[1;34mCreating Ollama embedding for:\033[0m {args}")
    vector = _get_embedding_from_ollama(args)

    try:
        if vector:
            print(f"\033[1;32mEmbedding created successfully (dimension: {len(vector)})\033[0m")
            try:
                embedding = EmbeddingVector(
                    vector=np.array(vector).astype(np.float32),  # Ensure float32
                    metadata={
                        "source": "ollama:embed",
                        "text": args,
                        "model": ollama_config["embedding_model"],
                        "timestamp": time.time(),
                    },
                )
                vector_id = db.store_embedding(embedding)
                print(f"\033[1;32mEmbedding stored with ID:\033[0m {vector_id}")

                # Create conversation block (optional)
                block_id = db.create_conversation_block(
                    data={
                        "type": "embedding_creation",
                        "source": "ollama:embed",
                        "text": args,
                        "model": ollama_config["embedding_model"],
                        "vector_id": vector_id,
                    }
                )
                logger.info(f"Created conversation block {block_id} for embedding {vector_id}")
            except Exception as e:
                logger.error(f"Error storing embedding or creating block: {e}")
                print("\033[1;31mError storing embedding or creating conversation block.\033[0m")
        else:
            print("\033[1;31mFailed to create embedding.\033[0m")

    except Exception as e:  # Catch potential errors in the main block
        logger.error(f"Unexpected error in ollama:embed: {e}")
        print(f"\033[1;31mAn unexpected error occurred: {e}\033[0m")


def _display_generation_result(generated_text: str, stats: Dict[str, Any]) -> None:
    """
    Display the generated text and statistics.

    Args:
        generated_text: The text generated by the model
        stats: Dictionary of generation statistics
    """
    if not generated_text:
        print("\033[1;31mOllama returned an empty response.\033[0m")
        return

    print("\n\033[1;32mGenerated Text:\033[0m")
    print(generated_text)

    # Print stats if available
    if stats:
        print("\n\033[1;34mGeneration Stats:\033[0m")
        for k, v in stats.items():
            if "duration" in k:  # Format durations nicely
                print(f"  {k}: {v / 1e9:.3f} s")  # Ollama durations are in nanoseconds
            else:
                print(f"  {k}: {v}")


def _store_generation_result(
    db: Any, prompt: str, generated_text: str, response_data: Dict[str, Any], stats: Dict[str, Any]
) -> None:
    """
    Store the generation result as embeddings in the database.

    Args:
        db: WDBX instance
        prompt: The original prompt text
        generated_text: The text generated by the model
        response_data: The full response data from the API
        stats: Dictionary of generation statistics
    """
    print("\033[1;34mCreating embedding for generated text...\033[0m")

    # Get embedding for the response
    response_vector = _get_embedding_from_ollama(generated_text)
    if not response_vector:
        print("\033[1;31mFailed to create embedding for the response.\033[0m")
        return

    print(f"\033[1;32mEmbedding created (dimension: {len(response_vector)})\033[0m")

    try:
        # Create embedding for prompt
        prompt_vector = _get_embedding_from_ollama(prompt)

        # Create embedding for response
        response_embedding = EmbeddingVector(
            vector=np.array(response_vector).astype(np.float32),
            metadata={
                "source": "ollama:generate:response",
                "text": generated_text,
                "model": ollama_config["embedding_model"],
                "generation_model": response_data.get("model", ollama_config["generation_model"]),
                "prompt": prompt,
                "timestamp": time.time(),
                "generation_stats": stats,
            },
        )
        response_vector_id = db.store_embedding(response_embedding)
        print(f"\033[1;32mResponse embedding stored with ID:\033[0m {response_vector_id}")

        # Prepare data for conversation block
        block_data = {
            "type": "generation",
            "source": "ollama:generate",
            "prompt": prompt,
            "response": generated_text,
            "completion_model": response_data.get("model", ollama_config["generation_model"]),
            "embedding_model": ollama_config["embedding_model"],
            "response_vector_id": response_vector_id,
            "generation_stats": stats,
        }

        # Handle prompt vector if available
        if prompt_vector:
            prompt_embedding = EmbeddingVector(
                vector=np.array(prompt_vector).astype(np.float32),
                metadata={
                    "source": "ollama:generate:prompt",
                    "text": prompt,
                    "model": ollama_config["embedding_model"],
                    "timestamp": time.time(),
                },
            )
            prompt_vector_id = db.store_embedding(prompt_embedding)
            block_data["prompt_vector_id"] = prompt_vector_id
            print(f"\033[1;32mPrompt embedding stored with ID:\033[0m {prompt_vector_id}")

        # Create the conversation block
        block_id = db.create_conversation_block(data=block_data)
        logger.info(f"Created conversation block {block_id} for generation.")

    except Exception as e:
        logger.error(f"Error storing embeddings or creating block: {e}")
        print("\033[1;31mError storing embeddings or creating conversation block.\033[0m")


def cmd_ollama_generate(db: Any, args: str) -> None:
    """
    Generate text using an Ollama model.

    Args:
        db: WDBX instance
        args: Prompt text
    """
    if not args:
        print("Usage: ollama:generate <prompt_text>")
        return

    print(f"\033[1;34mGenerating text from Ollama using prompt:\033[0m {args}")
    response_data = _generate_from_ollama(args)

    if not response_data:
        print("\033[1;31mFailed to generate text.\033[0m")
        return

    # Extract generated text and stats
    generated_text = response_data.get("response", "").strip()
    stats_keys = [
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "eval_count",
        "eval_duration",
    ]
    stats = {k: v for k, v in response_data.items() if k in stats_keys}

    # Display the result
    _display_generation_result(generated_text, stats)

    # Ask if the user wants to store the response
    if generated_text:
        store_choice = (
            input("\n\033[1;34mStore this response as embedding? (y/n): \033[0m").strip().lower()
        )
        if store_choice in ["y", "yes"]:
            _store_generation_result(db, args, generated_text, response_data, stats)


def cmd_ollama_models(db: Any, args: str) -> None:
    """
    List locally available Ollama models.

    Args:
        db: WDBX instance
        args: Command arguments (unused)
    """
    print("\033[1;34mFetching local Ollama models...\033[0m")
    models = _get_ollama_models()

    if models is None:
        print("\033[1;31mFailed to retrieve models list from Ollama.\033[0m")
        return

    if not models:
        print("No local Ollama models found.")
        print("You can download models using: ollama pull <model_name>")
        return

    print("\033[1;32mAvailable Ollama Models:\033[0m")
    # Print header
    print(f"{'NAME':<40} {'SIZE':<15} {'MODIFIED':<25}")
    print("-" * 80)
    for model in models:
        name = model.get("name", "Unknown")
        size = model.get("size", 0)
        modified_at = model.get("modified_at", "Unknown")

        # Format file size
        size_str = _format_size(size)

        # Format modified time (if available and in expected format)
        try:
            # Ollama format: "2024-03-15T10:30:00.123456Z"
            if isinstance(modified_at, str) and "T" in modified_at:
                dt_obj = datetime.fromisoformat(modified_at.replace("Z", "+00:00"))
                modified_str = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
            else:
                modified_str = str(modified_at)
        except ValueError:
            modified_str = str(modified_at)  # Fallback if parsing fails

        print(f"{name:<40} {size_str:<15} {modified_str:<25}")


def _format_size(size_bytes: int) -> str:
    """Format bytes into a human-readable string."""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


def _load_texts_from_file(file_path: str) -> List[str]:
    """
    Load texts from a file, one text per line.

    Args:
        file_path: Path to the text file

    Returns:
        List of non-empty text strings
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    texts = []
    try:
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    texts.append(text)
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        raise

    return texts


def _process_batch(db: Any, texts: List[str], model: str, file_path: str) -> Tuple[int, int]:
    """
    Process a batch of texts, creating and storing embeddings.

    Args:
        db: WDBX instance
        texts: List of text strings to process
        model: Name of the embedding model to use
        file_path: Original file path (for metadata)

    Returns:
        Tuple of (successful_count, error_count)
    """
    processed_count = 0
    error_count = 0

    print(f"Processing batch of {len(texts)} items...")
    for item in texts:
        vector = _get_embedding_from_ollama(item, model)
        if vector:
            try:
                embedding = EmbeddingVector(
                    vector=np.array(vector).astype(np.float32),
                    metadata={
                        "source": "ollama:batch",
                        "text": item,
                        "model": model,
                        "file": file_path,
                        "timestamp": time.time(),
                    },
                )
                # Store embedding
                db.store_embedding(embedding)
                processed_count += 1
            except Exception as e:
                logger.error(f"Error storing embedding for text '{item[:50]}...': {e}")
                error_count += 1
        else:
            error_count += 1

    return processed_count, error_count


def cmd_ollama_batch(db: Any, args: str) -> None:
    """
    Process a batch of texts from a file using Ollama embeddings.

    Args:
        db: WDBX instance
        args: Path to the input file (one text per line)
    """
    if not args:
        print("Usage: ollama:batch <file_path>")
        return

    file_path = args
    try:
        print(f"\033[1;34mStarting batch processing for file: {file_path}\033[0m")
        texts = _load_texts_from_file(file_path)

        if not texts:
            print("\033[1;33mWarning: No texts found in the file.\033[0m")
            return

        model = ollama_config["embedding_model"]
        batch_size = ollama_config["batch_size"]
        total_processed = 0
        total_errors = 0
        start_time = time.time()

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            processed, errors = _process_batch(db, batch, model, file_path)
            total_processed += processed
            total_errors += errors

        end_time = time.time()
        duration = end_time - start_time

        # Display results
        print("\n\033[1;32mBatch processing completed.\033[0m")
        print(f"  Processed: {total_processed} items")
        print(f"  Errors:    {total_errors} items")
        print(f"  Duration:  {duration:.2f} seconds")

    except FileNotFoundError:
        print(f"\033[1;31mError: File not found: {file_path}\033[0m")
    except Exception as e:
        logger.error(f"Error during batch processing: {e}", exc_info=True)
        print(f"\033[1;31mAn error occurred during batch processing: {e}\033[0m")
