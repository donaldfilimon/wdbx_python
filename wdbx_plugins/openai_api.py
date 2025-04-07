"""
WDBX OpenAI API Plugin

This plugin provides integration with OpenAI's APIs for generating and storing
embeddings from OpenAI models.
"""

import json
import logging
import os
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import requests

from wdbx.data_structures import EmbeddingVector

logger = logging.getLogger("WDBX.plugins.openai_api")

# Default settings
DEFAULT_CONFIG = {
    "api_key": "",
    "embedding_model": "text-embedding-3-small",
    "completion_model": "gpt-3.5-turbo",
    "organization_id": "",
    "timeout": 60,
    "max_tokens": 1024,
    "temperature": 0.7,
    "save_responses": True,
    "response_dir": "./wdbx_openai_responses",
}

# Global configuration
openai_config = DEFAULT_CONFIG.copy()


def register_commands(plugin_registry: Dict[str, Callable]) -> None:
    """
    Register OpenAI API commands with the CLI.

    Args:
        plugin_registry: Registry to add commands to
    """
    # Register commands
    plugin_registry["openai:embed"] = cmd_openai_embed
    plugin_registry["openai:generate"] = cmd_openai_generate
    plugin_registry["openai:config"] = cmd_openai_config
    plugin_registry["openai"] = cmd_openai_help

    logger.info("OpenAI API commands registered: openai:embed, openai:generate, openai:config")

    # Load config if exists
    _load_openai_config()

    # Create response directory if saving is enabled
    if openai_config["save_responses"]:
        _ensure_response_dir()


def _ensure_response_dir() -> None:
    """Ensure the response directory exists."""
    try:
        dir_path = openai_config["response_dir"]
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"OpenAI response directory: {dir_path}")
    except Exception as e:
        logger.error(f"Error creating OpenAI response directory: {e}")


def _get_config_path() -> str:
    """Get the full path to the config file."""
    return os.path.join(os.path.expanduser("~"), ".wdbx", "openai_config.json")


def _load_openai_config() -> None:
    """Load OpenAI configuration from file."""
    config_path = _get_config_path()
    try:
        if os.path.exists(config_path):
            with open(config_path) as f:
                loaded_config = json.load(f)
                openai_config.update(loaded_config)
                logger.info("Loaded OpenAI configuration from file.")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding OpenAI config file {config_path}: {e}")
    except Exception as e:
        logger.error(f"Error loading OpenAI configuration from {config_path}: {e}")


def _save_openai_config() -> None:
    """Save OpenAI configuration to file (excluding API key)."""
    config_path = _get_config_path()
    config_dir = os.path.dirname(config_path)
    try:
        os.makedirs(config_dir, exist_ok=True)
        # Save a copy without the API key for security
        save_config = openai_config.copy()
        save_config["api_key"] = "********" if openai_config["api_key"] else ""
        with open(config_path, "w") as f:
            json.dump(save_config, f, indent=2)
        logger.info(f"Saved OpenAI configuration to {config_path} (without API key).")
    except Exception as e:
        logger.error(f"Error saving OpenAI configuration to {config_path}: {e}")


def _print_safe_config():
    """Print the current configuration safely (masking API key)."""
    safe_config = openai_config.copy()
    if safe_config["api_key"]:
        safe_config["api_key"] = "********"
    for key, value in safe_config.items():
        print(f"  \033[1m{key}\033[0m = {value}")


def cmd_openai_help(db: Any, args: str) -> None:
    """
    Show help for OpenAI commands.

    Args:
        db: WDBX instance
        args: Command arguments (unused)
    """
    print("\033[1;35mWDBX OpenAI API Integration\033[0m")
    print("The following OpenAI commands are available:")
    print("\033[1m  openai:embed <text>\033[0m - Create and store an embedding using OpenAI")
    print("\033[1m  openai:generate <text>\033[0m - Generate text using an OpenAI model")
    print("\033[1m  openai:config [key=value]\033[0m - Configure OpenAI integration settings")

    print("\n\033[1;34mOpenAI Configuration:\033[0m")
    _print_safe_config()


def _update_config_value(key: str, value: str):
    """Update a single config value with type conversion."""
    global openai_config
    try:
        if key in ["timeout", "max_tokens"]:
            openai_config[key] = int(value)
        elif key in ["temperature"]:
            openai_config[key] = float(value)
        elif key in ["save_responses"]:
            openai_config[key] = value.lower() in ("true", "yes", "1", "y")
        else:
            openai_config[key] = value
        print(
            f"\033[1;32mSet {key} = {'********' if key == 'api_key' else openai_config[key]}\033[0m"
        )
        return True
    except ValueError:
        print(
            f"\033[1;31mError: Invalid value type for {key}. Expected int/float/bool/string.\033[0m"
        )
        return False


def cmd_openai_config(db: Any, args: str) -> None:
    """
    Configure OpenAI integration settings.

    Args:
        db: WDBX instance
        args: Command arguments (key=value pairs)
    """
    print("\033[1;35mWDBX OpenAI Configuration\033[0m")

    if not args:
        print("Current configuration:")
        _print_safe_config()
        print("\nTo change a setting, use: openai:config key=value")
        print("Note: API key is not written to the config file for security.")
        return

    # Parse key=value pairs
    updated_any = False
    parts = args.split()
    for part in parts:
        if "=" in part:
            key, value = part.split("=", 1)
            if key in openai_config:
                if _update_config_value(key, value):
                    updated_any = True
            else:
                print(f"\033[1;31mUnknown configuration key: {key}\033[0m")
        else:
            print(f"\033[1;31mInvalid format: {part}. Use key=value format.\033[0m")

    if updated_any:
        # Save configuration (without API key)
        _save_openai_config()
        # Ensure response dir exists if needed
        if openai_config["save_responses"]:
            _ensure_response_dir()


def _call_openai_api(endpoint: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Generic helper to call OpenAI API endpoints."""
    if not openai_config["api_key"]:
        logger.error("OpenAI API key not configured")
        return None

    url = f"https://api.openai.com/v1/{endpoint}"
    headers = {
        "Authorization": f"Bearer {openai_config['api_key']}",
        "Content-Type": "application/json",
    }
    if openai_config["organization_id"]:
        headers["OpenAI-Organization"] = openai_config["organization_id"]

    try:
        response = requests.post(
            url, headers=headers, json=payload, timeout=openai_config["timeout"]
        )

        if response.status_code != 200:
            error_msg = f"OpenAI API Error ({endpoint}): {response.status_code}"
            try:
                error_data = response.json()
                if "error" in error_data and isinstance(error_data["error"], dict):
                    error_msg += f" - {error_data['error'].get('message', 'Unknown error')}"
                else:
                    error_msg += f" - {response.text[:100]}"  # Non-standard error format
            except json.JSONDecodeError:
                error_msg += f" - {response.text[:100]}"
            logger.error(error_msg)
            return None

        resp_data = response.json()
        if openai_config["save_responses"]:
            _save_api_response(endpoint, resp_data)
        return resp_data

    except requests.exceptions.Timeout:
        logger.error(f"OpenAI API request timed out ({openai_config['timeout']}s)")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"OpenAI API request failed: {e}")
        return None
    except json.JSONDecodeError:
        logger.error("Invalid JSON response from OpenAI API")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during OpenAI API call: {e}")
        return None


def _save_api_response(endpoint: str, data: Dict[str, Any]) -> None:
    """Save the API JSON response to a file."""
    try:
        filename = f"{endpoint.replace('/', '_')}_{uuid.uuid4()}.json"
        response_file = os.path.join(openai_config["response_dir"], filename)
        with open(response_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Saved OpenAI response ({endpoint}) to {response_file}")
    except Exception as e:
        logger.error(f"Error saving OpenAI response: {e}")


def _get_openai_embedding(text: str, model: Optional[str] = None) -> Optional[List[float]]:
    """
    Get an embedding from OpenAI.

    Args:
        text: Text to embed
        model: Model to use (defaults to config setting)

    Returns:
        List of floating point values representing the embedding, or None on error
    """
    if not text:
        logger.error("Cannot generate embedding for empty text")
        return None

    model = model or openai_config["embedding_model"]
    logger.debug(f"Requesting embedding from OpenAI using model {model}")

    payload = {
        "model": model,
        "input": text,
    }

    resp_data = _call_openai_api("embeddings", payload)

    if resp_data and isinstance(resp_data, dict) and "data" in resp_data:
        if resp_data["data"] and isinstance(resp_data["data"], list):
            embedding_data = resp_data["data"][0]
            if isinstance(embedding_data, dict) and "embedding" in embedding_data:
                vector = embedding_data["embedding"]
                if isinstance(vector, list):
                    return vector
                else:
                    logger.error("Invalid embedding format received from OpenAI.")
            else:
                logger.error("Unexpected embedding data structure from OpenAI.")
        else:
            logger.error("No embedding data found in OpenAI response.")
    else:
        logger.error("Failed to get embedding or invalid response structure from OpenAI.")

    return None


def _generate_openai_completion(prompt: str, model: Optional[str] = None) -> Optional[str]:
    """
    Generate text completion from OpenAI.

    Args:
        prompt: Prompt for text generation
        model: Model to use (defaults to config setting)

    Returns:
        Generated text completion string, or None on error.
    """
    if not prompt:
        logger.error("Cannot generate completion for empty prompt")
        return None

    model = model or openai_config["completion_model"]
    logger.debug(f"Requesting completion from OpenAI using model {model}")

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": openai_config["max_tokens"],
        "temperature": openai_config["temperature"],
    }

    # Handle chat models vs completion models
    if "gpt-4" in model or "gpt-3.5" in model:
        payload.pop("prompt", None)
        payload["messages"] = [{"role": "user", "content": prompt}]
        endpoint = "chat/completions"
    else:
        endpoint = "completions"

    resp_data = _call_openai_api(endpoint, payload)

    if resp_data and isinstance(resp_data, dict) and "choices" in resp_data:
        if resp_data["choices"] and isinstance(resp_data["choices"], list):
            choice = resp_data["choices"][0]
            if isinstance(choice, dict):
                if endpoint == "chat/completions":
                    if (
                        "message" in choice
                        and isinstance(choice["message"], dict)
                        and "content" in choice["message"]
                    ):
                        return choice["message"]["content"].strip()
                elif "text" in choice:
                    return choice["text"].strip()
                else:
                    logger.error("Unexpected completion choice structure from OpenAI.")
            else:
                logger.error("Unexpected choice item type from OpenAI.")
        else:
            logger.error("No choices found in OpenAI completion response.")
    else:
        logger.error("Failed to get completion or invalid response structure from OpenAI.")

    return None


def cmd_openai_embed(db: Any, args: str) -> None:
    """
    Create and store an embedding using OpenAI.

    Args:
        db: WDBX instance
        args: Text to embed
    """
    if not args:
        print("Usage: openai:embed <text_to_embed>")
        return

    print(f"\033[1;34mCreating OpenAI embedding for:\033[0m {args}")
    vector = _get_openai_embedding(args)

    if vector:
        print(f"\033[1;32mEmbedding created successfully (dimension: {len(vector)})\033[0m")
        try:
            # Create embedding vector
            embedding = EmbeddingVector(
                vector=np.array(vector).astype(np.float32),
                metadata={
                    "source": "openai:embed",
                    "text": args,
                    "model": openai_config["embedding_model"],
                    "timestamp": time.time(),
                },
            )

            # Store embedding
            vector_id = db.store_embedding(embedding)
            print(f"\033[1;32mEmbedding stored with ID:\033[0m {vector_id}")

            # Create conversation block (optional)
            block_id = db.create_conversation_block(
                data={
                    "type": "embedding_creation",
                    "source": "openai:embed",
                    "text": args,
                    "model": openai_config["embedding_model"],
                    "vector_id": vector_id,
                }
            )
            logger.info(f"Created conversation block {block_id} for embedding {vector_id}")
        except Exception as e:
            logger.error(f"Error storing embedding or creating block: {e}")
            print("\033[1;31mError storing embedding or creating conversation block.\033[0m")
    else:
        print("\033[1;31mFailed to create embedding.\033[0m")


def cmd_openai_generate(db: Any, args: str) -> None:
    """
    Generate text using an OpenAI model.

    Args:
        db: WDBX instance
        args: Prompt text
    """
    if not args:
        print("Usage: openai:generate <prompt_text>")
        return

    print(f"\033[1;34mGenerating text from OpenAI using prompt:\033[0m {args}")
    generated_text = _generate_openai_completion(args)

    if generated_text:
        print("\n\033[1;32mGenerated Text:\033[0m")
        print(generated_text)

        # Ask if the user wants to store the response
        store_choice = (
            input("\n\033[1;34mStore this response as embedding? (y/n): \033[0m").strip().lower()
        )

        if store_choice in ["y", "yes"]:
            print("\033[1;34mCreating embedding for generated text...\033[0m")
            response_vector = _get_openai_embedding(generated_text)
            if response_vector:
                print(f"\033[1;32mEmbedding created (dimension: {len(response_vector)})\033[0m")
                try:
                    # Create embedding for prompt
                    prompt_vector = _get_openai_embedding(args)

                    # Create embedding for response
                    response_embedding = EmbeddingVector(
                        vector=np.array(response_vector).astype(np.float32),
                        metadata={
                            "source": "openai:generate:response",
                            "text": generated_text,
                            "model": openai_config["embedding_model"],
                            "generation_model": openai_config["completion_model"],
                            "prompt": args,
                            "timestamp": time.time(),
                        },
                    )
                    response_vector_id = db.store_embedding(response_embedding)
                    print(
                        f"\033[1;32mResponse embedding stored with ID:\033[0m {response_vector_id}"
                    )

                    # Create conversation block linking prompt and response
                    block_data = {
                        "type": "generation",
                        "source": "openai:generate",
                        "prompt": args,
                        "response": generated_text,
                        "completion_model": openai_config["completion_model"],
                        "embedding_model": openai_config["embedding_model"],
                        "response_vector_id": response_vector_id,
                    }
                    if prompt_vector:
                        prompt_embedding = EmbeddingVector(
                            vector=np.array(prompt_vector).astype(np.float32),
                            metadata={
                                "source": "openai:generate:prompt",
                                "text": args,
                                "model": openai_config["embedding_model"],
                                "timestamp": time.time(),
                            },
                        )
                        prompt_vector_id = db.store_embedding(prompt_embedding)
                        block_data["prompt_vector_id"] = prompt_vector_id
                        print(
                            f"\033[1;32mPrompt embedding stored with ID:\033[0m {prompt_vector_id}"
                        )

                    block_id = db.create_conversation_block(data=block_data)
                    logger.info(f"Created conversation block {block_id} for generation.")
                except Exception as e:
                    logger.error(f"Error storing embeddings or creating block: {e}")
                    print(
                        "\033[1;31mError storing embeddings or creating conversation block.\033[0m"
                    )
            else:
                print("\033[1;31mFailed to create embedding for the response.\033[0m")
    else:
        print("\033[1;31mFailed to generate text.\033[0m")
