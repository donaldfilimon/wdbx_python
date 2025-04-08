"""
WDBX Model Repository Plugin

This plugin provides a unified interface for managing models from multiple sources,
including OpenAI, HuggingFace, Ollama and local models.
"""

import json
import logging
import os
import time
from typing import Any, Callable, Dict, Optional

# Constants for magic numbers
DEFAULT_MODEL_RETRY_COUNT = 3
DEFAULT_MODEL_RETRY_DELAY = 2
HEALTH_THRESHOLD_GOOD = 80
HEALTH_THRESHOLD_WARN = 50

logger = logging.getLogger("WDBX.plugins.model_repo")

# Default settings
DEFAULT_CONFIG = {
    "default_embedding_model": "openai:text-embedding-3-small",
    "default_generation_model": "ollama:llama3",
    "model_cache_dir": "./wdbx_model_cache",
    "max_models_per_source": 5,
    "auto_discover": True,
}

# Global configuration
repo_config = DEFAULT_CONFIG.copy()

# Model registry
model_registry = {"embedding": {}, "generation": {}}

# Plugin sources
plugin_sources = set()


def register_commands(plugin_registry: Dict[str, Callable]) -> None:
    """
    Register model repository commands with the CLI.

    Args:
        plugin_registry: Registry to add commands to
    """
    # Register commands
    plugin_registry["model:list"] = cmd_model_list
    plugin_registry["model:add"] = cmd_model_add
    plugin_registry["model:remove"] = cmd_model_remove
    plugin_registry["model:default"] = cmd_model_default
    plugin_registry["model:embed"] = cmd_model_embed
    plugin_registry["model:generate"] = cmd_model_generate
    plugin_registry["model:config"] = cmd_model_config
    plugin_registry["model:health"] = cmd_model_health
    plugin_registry["model"] = cmd_model_help

    logger.info(
        "Model repository commands registered: model:list, model:add, model:remove, "
        "model:default, model:embed, model:generate, model:config, model:health"
    )

    # Load config if exists
    _load_model_repo_config()

    # Discover available plugin sources
    _discover_plugin_sources(plugin_registry)

    # Create cache directory
    try:
        os.makedirs(repo_config["model_cache_dir"], exist_ok=True)
        logger.info(f"Model cache directory: {repo_config['model_cache_dir']}")
    except Exception as e:
        logger.error(f"Error creating model cache directory: {e}")

    # Load model registry
    _load_model_registry()


def _discover_plugin_sources(plugin_registry: Dict[str, Callable]) -> None:
    """
    Discover available plugin sources from the registry.

    Args:
        plugin_registry: Plugin command registry
    """
    # Look for known plugin prefixes in the registry
    for command in plugin_registry:
        if command.startswith("openai:"):
            plugin_sources.add("openai")
        elif command.startswith("hf:"):
            plugin_sources.add("huggingface")
        elif command.startswith("ollama:"):
            plugin_sources.add("ollama")

    logger.info(f"Discovered plugin sources: {plugin_sources}")


def _load_model_repo_config() -> None:
    """Load model repository configuration from file."""
    config_path = os.path.join(os.path.expanduser("~"), ".wdbx", "model_repo_config.json")
    try:
        if os.path.exists(config_path):
            with open(config_path) as f:
                loaded_config = json.load(f)
                repo_config.update(loaded_config)
                logger.info("Loaded model repository configuration from file.")
    except Exception as e:
        logger.error(f"Error loading model repository configuration: {e}")


def _save_model_repo_config() -> None:
    """Save model repository configuration to file."""
    config_dir = os.path.join(os.path.expanduser("~"), ".wdbx")
    config_path = os.path.join(config_dir, "model_repo_config.json")
    try:
        os.makedirs(config_dir, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(repo_config, f, indent=2)
            logger.info("Saved model repository configuration to file.")
    except Exception as e:
        logger.error(f"Error saving model repository configuration: {e}")


def _load_model_registry() -> bool:
    """
    Load model registry from file.

    Returns:
        True if loaded successfully, False otherwise
    """
    global model_registry

    # Check for both old and new locations
    config_dir = os.path.join(os.path.expanduser("~"), ".wdbx")
    config_path = os.path.join(config_dir, "model_registry.json")
    old_registry_path = os.path.join(repo_config["model_cache_dir"], "model_registry.json")

    registry_paths = [config_path, old_registry_path]

    for path in registry_paths:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    loaded_registry = json.load(f)

                    # Validate structure
                    if not isinstance(loaded_registry, dict):
                        logger.error(f"Invalid model registry format in {path} (not a dictionary)")
                        continue

                    # Update registry with loaded data
                    model_registry.update(loaded_registry)

                    # Ensure all required types exist
                    for model_type in ["embedding", "generation"]:
                        if model_type not in model_registry:
                            model_registry[model_type] = {}

                    logger.info(f"Loaded model registry from {path}")

                    # If we loaded from old path, save to new location
                    if path == old_registry_path:
                        logger.info("Migrating model registry to new location")
                        _save_model_registry()

                    return True

            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in model registry file: {path}")
            except PermissionError:
                logger.error(f"Permission denied when reading model registry from {path}")
            except Exception as e:
                logger.error(f"Error loading model registry from {path}: {e}")

    # If we reach here, we couldn't load from any location
    logger.info("No existing model registry found, creating new one")

    # Initialize empty registry
    model_registry = {"embedding": {}, "generation": {}}

    return False


def _save_model_registry() -> bool:
    """
    Save model registry to file.

    Returns:
        True if saved successfully, False otherwise
    """
    config_dir = os.path.join(os.path.expanduser("~"), ".wdbx")
    config_path = os.path.join(config_dir, "model_registry.json")

    try:
        # Ensure directory exists
        os.makedirs(config_dir, exist_ok=True)

        # Create a backup of the existing registry first
        if os.path.exists(config_path):
            backup_path = f"{config_path}.bak"
            try:
                import shutil

                shutil.copy2(config_path, backup_path)
                logger.debug(f"Created backup of model registry at {backup_path}")
            except Exception as e:
                logger.warning(f"Failed to create backup of model registry: {e}")

        # Write new registry file
        with open(config_path, "w") as f:
            json.dump(model_registry, f, indent=2)

        logger.info(f"Saved model registry to {config_path}")
        return True

    except PermissionError:
        logger.error(f"Permission denied when saving model registry to {config_path}")
        print("\033[1;31mError: Permission denied when saving model registry\033[0m")
        return False
    except OSError as e:
        logger.error(f"IO error when saving model registry: {e}")
        print(f"\033[1;31mError: Failed to save model registry: {e}\033[0m")
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving model registry: {e}")
        print(f"\033[1;31mError: Failed to save model registry: {e}\033[0m")
        return False


def cmd_model_help(db, args: str) -> None:
    """
    Show help for model repository commands.

    Args:
        db: WDBX instance
        args: Command arguments (unused)
    """
    print("\033[1;35mWDBX Model Repository\033[0m")
    print("The following model repository commands are available:")
    print(
        "\033[1m  model:list [type]\033[0m - List available models (type: embedding or generation)"
    )
    print("\033[1m  model:add <source:model_name> <type>\033[0m - Add a model to the repository")
    print("\033[1m  model:remove <source:model_name>\033[0m - Remove a model from the repository")
    print(
        "\033[1m  model:default <source:model_name> <type>\033[0m - Set a model as default for embedding or generation"
    )
    print(
        "\033[1m  model:embed <text> [model=<source:model_name>]\033[0m - Create embedding using any model"
    )
    print(
        "\033[1m  model:generate <prompt> [model=<source:model_name>]\033[0m - Generate text using any model"
    )
    print("\033[1m  model:config [key=value]\033[0m - Configure model repository settings")
    print("\033[1m  model:health [type]\033[0m - Check the health of registered models")

    print("\n\033[1;34mModel Repository Configuration:\033[0m")
    for key, value in repo_config.items():
        print(f"  \033[1m{key}\033[0m = {value}")

    print("\n\033[1;34mAvailable Plugin Sources:\033[0m")
    if not plugin_sources:
        print("  No plugin sources detected")
    else:
        for source in plugin_sources:
            print(f"  - {source}")

    print("\n\033[1;34mDefault Models:\033[0m")
    print(f"  Embedding: {repo_config['default_embedding_model']}")
    print(f"  Generation: {repo_config['default_generation_model']}")


def cmd_model_config(db, args: str) -> None:
    """
    Configure model repository settings.

    Args:
        db: WDBX instance
        args: Command arguments (key=value pairs)
    """
    global repo_config

    print("\033[1;35mWDBX Model Repository Configuration\033[0m")

    if not args:
        # Display current configuration
        print("Current configuration:")
        for key, value in repo_config.items():
            print(f"  \033[1m{key}\033[0m = {value}")
        print("\nTo change a setting, use: model:config key=value")
        return

    # Parse key=value pairs
    parts = args.split()
    for part in parts:
        if "=" in part:
            key, value = part.split("=", 1)
            if key in repo_config:
                # Convert value to appropriate type
                if key in ["max_models_per_source"]:
                    try:
                        repo_config[key] = int(value)
                    except ValueError:
                        print(f"\033[1;31mError: {key} must be an integer.\033[0m")
                        continue
                elif key in ["auto_discover"]:
                    repo_config[key] = value.lower() in ("true", "yes", "1", "y")
                else:
                    repo_config[key] = value
                print(f"\033[1;32mSet {key} = {repo_config[key]}\033[0m")
            else:
                print(f"\033[1;31mUnknown configuration key: {key}\033[0m")

    # Save configuration
    _save_model_repo_config()


def cmd_model_list(db, args: str) -> None:
    """
    List available models.

    Args:
        db: WDBX instance
        args: Command arguments (model type)
    """
    model_type = args.strip() if args else None

    print("\033[1;35mWDBX Model Repository\033[0m")

    # Validate model type
    valid_types = ["embedding", "generation"]
    if model_type and model_type not in valid_types:
        print(f"\033[1;31mInvalid model type: {model_type}\033[0m")
        print(f"Valid types: {', '.join(valid_types)}")
        return

    # List models
    if model_type:
        _list_models_by_type(model_type)
    else:
        for type_name in valid_types:
            _list_models_by_type(type_name)
            print()


def _list_models_by_type(model_type: str) -> None:
    """
    List models of a specific type.

    Args:
        model_type: Type of models to list ('embedding' or 'generation')
    """
    print(f"\033[1;34mAvailable {model_type} models:\033[0m")

    if model_type in model_registry and model_registry[model_type]:
        # Print table header
        print(f"  {'Source':<12}{'Model Name':<30}{'Status':<10}")
        print(f"  {'-'*12}{'-'*30}{'-'*10}")

        # Print each model
        for source_model, details in model_registry[model_type].items():
            source, model_name = source_model.split(":", 1)
            status = (
                "Default"
                if source_model == repo_config[f"default_{model_type}_model"]
                else "Available"
            )
            status_color = "\033[1;32m" if status == "Default" else "\033[0m"
            print(f"  {source:<12}{model_name:<30}{status_color}{status}\033[0m")
    else:
        print(f"  No {model_type} models registered")


def cmd_model_add(db, args: str) -> None:
    """
    Add a model to the repository.

    Args:
        db: WDBX instance
        args: Command arguments (source:model_name type)
    """
    parts = args.split()
    if len(parts) < 2:
        print("\033[1;31mUsage: model:add <source:model_name> <type>\033[0m")
        print("  <source>: openai, huggingface, ollama")
        print("  <type>: embedding, generation")
        return

    source_model = parts[0]
    model_type = parts[1]

    # Validate source and type
    if ":" not in source_model:
        print(
            "\033[1;31mError: Model name must include source (e.g., openai:text-embedding-3-small)\033[0m"
        )
        return

    source, model_name = source_model.split(":", 1)
    if source not in plugin_sources:
        print(
            f"\033[1;31mError: Unknown source '{source}'. Available sources: {', '.join(plugin_sources)}\033[0m"
        )
        return

    if model_type not in ["embedding", "generation"]:
        print("\033[1;31mError: Type must be 'embedding' or 'generation'\033[0m")
        return

    # Check if model exists
    if model_type in model_registry and source_model in model_registry[model_type]:
        print(
            f"\033[1;33mModel {source_model} is already registered as a {model_type} model\033[0m"
        )
        return

    # Check if we can use this model by calling the plugin
    model_check_cmd = (
        "models" if source == "ollama" else "embed" if model_type == "embedding" else "generate"
    )
    model_check_args = "" if model_check_cmd == "models" else f"Test using {model_name}"

    print(f"\033[1;34mVerifying {source_model} availability...\033[0m")
    check_result = _dispatch_to_plugin(db, model_check_cmd, model_check_args)

    if check_result is None:
        print(
            f"\033[1;31mFailed to verify model {source_model}. Make sure the model exists and the plugin is loaded.\033[0m"
        )
        return

    # Add model to registry
    if model_type not in model_registry:
        model_registry[model_type] = {}

    model_registry[model_type][source_model] = {
        "added_at": time.time(),
        "source": source,
        "model_name": model_name,
    }

    print(f"\033[1;32mAdded {source_model} as a {model_type} model\033[0m")

    # If this is the first model of this type, set it as default
    if len(model_registry[model_type]) == 1:
        repo_config[f"default_{model_type}_model"] = source_model
        print(f"\033[1;32mSet {source_model} as the default {model_type} model\033[0m")

    # Save registry
    _save_model_registry()


def cmd_model_remove(db, args: str) -> None:
    """
    Remove a model from the repository.

    Args:
        db: WDBX instance
        args: Command arguments (source:model_name)
    """
    source_model = args.strip()

    if not source_model:
        print("\033[1;31mUsage: model:remove <source:model_name>\033[0m")
        return

    # Find model in registry
    removed = False
    for model_type in ["embedding", "generation"]:
        if model_type in model_registry and source_model in model_registry[model_type]:
            # Check if it's the default model
            is_default = repo_config[f"default_{model_type}_model"] == source_model

            # Remove from registry
            del model_registry[model_type][source_model]

            print(f"\033[1;32mRemoved {source_model} from {model_type} models\033[0m")
            removed = True

            # If it was the default model, set a new default if possible
            if is_default and model_registry[model_type]:
                new_default = next(iter(model_registry[model_type].keys()))
                repo_config[f"default_{model_type}_model"] = new_default
                print(f"\033[1;32mSet {new_default} as the new default {model_type} model\033[0m")

    if not removed:
        print(f"\033[1;31mModel {source_model} not found in repository\033[0m")
        return

    # Save registry
    _save_model_registry()


def cmd_model_default(db, args: str) -> None:
    """
    Set a model as default for embedding or generation.

    Args:
        db: WDBX instance
        args: Command arguments (source:model_name type)
    """
    parts = args.split()
    if len(parts) < 2:
        print("\033[1;31mUsage: model:default <source:model_name> <type>\033[0m")
        print("  <type>: embedding, generation")
        return

    source_model = parts[0]
    model_type = parts[1]

    # Validate type
    if model_type not in ["embedding", "generation"]:
        print("\033[1;31mError: Type must be 'embedding' or 'generation'\033[0m")
        return

    # Check if model exists in registry
    if model_type not in model_registry or source_model not in model_registry[model_type]:
        print(
            f"\033[1;31mError: Model {source_model} is not registered as a {model_type} model\033[0m"
        )
        print("Use `model:add` to register the model first.")
        return

    # Set as default
    repo_config[f"default_{model_type}_model"] = source_model
    print(f"\033[1;32mSet {source_model} as the default {model_type} model\033[0m")

    # Save configuration
    _save_model_repo_config()


def cmd_model_embed(db, args: str) -> None:
    """
    Create embedding using any model.

    Args:
        db: WDBX instance
        args: Command arguments (text [model=<source:model_name>])
    """
    if not args:
        print("\033[1;33mUsage: model:embed <text> [model=<source:model_name>]\033[0m")
        return

    # Parse arguments to separate text and model
    model_arg = None
    text = args

    # Extract model argument if present
    if " model=" in args:
        parts = args.split(" model=", 1)
        text = parts[0]
        model_arg = parts[1]

    # Use default model if not specified
    source_model = model_arg if model_arg else repo_config["default_embedding_model"]

    # Validate model format
    if ":" not in source_model:
        print(
            f"\033[1;31mError: Invalid model format '{source_model}'. Use format 'source:model_name'\033[0m"
        )
        print(f"Available sources: {', '.join(plugin_sources)}")
        return

    # Check if model exists
    model_type = "embedding"
    if model_type not in model_registry or source_model not in model_registry[model_type]:
        print(
            f"\033[1;31mError: Model {source_model} is not registered as an {model_type} model\033[0m"
        )
        print("Use `model:add` to register the model first.")
        return

    # Extract source and model name
    try:
        source, model_name = source_model.split(":", 1)
    except ValueError:
        print(
            f"\033[1;31mError: Invalid model format '{source_model}'. Use format 'source:model_name'\033[0m"
        )
        return

    # Map source to plugin command prefix
    source_to_prefix = {"openai": "openai", "huggingface": "hf", "ollama": "ollama"}

    if source not in source_to_prefix:
        print(
            f"\033[1;31mError: Unknown source '{source}'. Available sources: {', '.join(source_to_prefix.keys())}\033[0m"
        )
        return

    prefix = source_to_prefix[source]

    # Dispatch to the appropriate plugin
    print(f"\033[1;34mCreating embedding using {source_model}...\033[0m")

    # Format arguments based on the source
    if source == "openai":
        plugin_args = f"{text} model={model_name}"
    elif source == "huggingface":
        plugin_args = f"{text} model_name={model_name}"
    else:  # Ollama
        plugin_args = f"{text} model={model_name}"

    # Construct the full command for dispatch
    command = f"{prefix}:embed"

    # Call the plugin
    result = _dispatch_to_plugin(db, command, plugin_args)

    if result is not None:
        # If the result is a float or list, it's likely the embedding
        if isinstance(result, (list, float)) or (
            hasattr(result, "shape") and hasattr(result, "tolist")
        ):
            print("\033[1;32mSuccessfully created embedding\033[0m")
            # Don't print the actual embedding vector as it would be overwhelming
            if hasattr(result, "shape"):
                print(f"Embedding shape: {result.shape}")
            elif isinstance(result, list):
                print(f"Embedding dimensions: {len(result)}")
        else:
            print(f"\033[1;32mOperation completed: {result}\033[0m")
    else:
        print("\033[1;31mFailed to create embedding\033[0m")


def cmd_model_generate(db, args: str) -> None:
    """
    Generate text using any model.

    Args:
        db: WDBX instance
        args: Command arguments (prompt [model=<source:model_name>])
    """
    if not args:
        print("\033[1;33mUsage: model:generate <prompt> [model=<source:model_name>]\033[0m")
        return

    # Parse arguments to separate prompt and model
    model_arg = None
    prompt = args

    # Extract model argument if present
    if " model=" in args:
        parts = args.split(" model=", 1)
        prompt = parts[0]
        model_arg = parts[1]

    # Use default model if not specified
    source_model = model_arg if model_arg else repo_config["default_generation_model"]

    # Validate model format
    if ":" not in source_model:
        print(
            f"\033[1;31mError: Invalid model format '{source_model}'. Use format 'source:model_name'\033[0m"
        )
        print(f"Available sources: {', '.join(plugin_sources)}")
        return

    # Check if model exists
    model_type = "generation"
    if model_type not in model_registry or source_model not in model_registry[model_type]:
        print(
            f"\033[1;31mError: Model {source_model} is not registered as a generation model\033[0m"
        )
        print("Use `model:add` to register the model first.")
        return

    # Extract source and model name
    try:
        source, model_name = source_model.split(":", 1)
    except ValueError:
        print(
            f"\033[1;31mError: Invalid model format '{source_model}'. Use format 'source:model_name'\033[0m"
        )
        return

    # Map source to plugin command prefix
    source_to_prefix = {"openai": "openai", "huggingface": "hf", "ollama": "ollama"}

    if source not in source_to_prefix:
        print(
            f"\033[1;31mError: Unknown source '{source}'. Available sources: {', '.join(source_to_prefix.keys())}\033[0m"
        )
        return

    prefix = source_to_prefix[source]

    # Dispatch to the appropriate plugin
    print(f"\033[1;34mGenerating text using {source_model}...\033[0m")

    # Format arguments based on the source
    if source == "openai":
        plugin_args = f"{prompt} model={model_name}"
    elif source == "huggingface":
        plugin_args = f"{prompt} model_name={model_name}"
    else:  # Ollama
        plugin_args = f"{prompt} model={model_name}"

    # Construct the full command for dispatch
    command = f"{prefix}:generate"

    # Call the plugin
    result = _dispatch_to_plugin(db, command, plugin_args)

    if result is not None:
        if isinstance(result, str):
            # If it's a string, it's likely the generated text
            print(f"\033[1;32mGenerated text:\033[0m\n{result}")
        else:
            # For other types of results
            print("\033[1;32mOperation completed successfully\033[0m")
    else:
        print("\033[1;31mFailed to generate text\033[0m")


def cmd_model_health(db, args: str) -> None:
    """
    Check the health of registered models.

    Args:
        db: WDBX instance
        args: Command arguments (model type to check)
    """
    model_type = args.strip() if args else None

    print("\033[1;35mModel Health Check\033[0m")

    # Validate model type if provided
    valid_types = ["embedding", "generation"]
    if model_type and model_type not in valid_types:
        print(f"\033[1;31mInvalid model type: {model_type}\033[0m")
        print(f"Valid types: {', '.join(valid_types)}")
        return

    # Determine which types to check
    types_to_check = [model_type] if model_type else valid_types

    # Check each model
    total_models = 0
    healthy_models = 0

    print(f"{'Model':<40} {'Type':<12} {'Status':<10} {'Message'}")
    print(f"{'-'*40} {'-'*12} {'-'*10} {'-'*30}")

    for current_type in types_to_check:
        if current_type not in model_registry or not model_registry[current_type]:
            if model_type:  # Only show this message if a specific type was requested
                print(f"  No {current_type} models registered")
            continue

        for source_model, _details in model_registry[current_type].items():
            total_models += 1
            source, model_name = source_model.split(":", 1)

            # Map source to plugin command prefix
            source_to_prefix = {"openai": "openai", "huggingface": "hf", "ollama": "ollama"}

            if source not in source_to_prefix:
                status = "\033[1;33mWARNING\033[0m"
                message = f"Unknown source '{source}'"
                print(f"{source_model:<40} {current_type:<12} {status:<10} {message}")
                continue

            prefix = source_to_prefix[source]

            # Check if plugin is available
            if not hasattr(db, "plugin_registry") or f"{prefix}:embed" not in db.plugin_registry:
                status = "\033[1;33mWARNING\033[0m"
                message = f"Plugin '{prefix}' not registered"
                print(f"{source_model:<40} {current_type:<12} {status:<10} {message}")
                continue

            # For health check, we'll use different commands based on source and type
            test_text = "Health check test"

            if current_type == "embedding":
                command = f"{prefix}:embed"
                args = (
                    f"{test_text} model={model_name}"
                    if source == "openai"
                    else f"{test_text} model_name={model_name}"
                )
            else:  # generation
                command = f"{prefix}:generate"
                args = f"{test_text} model={model_name}"

            # Attempt to use the model
            try:
                logger.debug(f"Health checking {source_model} with command {command}")
                result = _dispatch_to_plugin(db, command, args)

                if result is not None:
                    status = "\033[1;32mHEALTHY\033[0m"
                    message = "OK"
                    healthy_models += 1
                else:
                    status = "\033[1;31mERROR\033[0m"
                    message = "Failed to use model"
            except Exception as e:
                status = "\033[1;31mERROR\033[0m"
                message = str(e)[:30]

            print(f"{source_model:<40} {current_type:<12} {status:<10} {message}")

    # Print summary
    print("\n\033[1;34mSummary:\033[0m")
    print(f"  Total models: {total_models}")
    print(f"  Healthy models: {healthy_models}")
    if total_models > 0:
        health_percentage = (healthy_models / total_models) * 100
        health_color = (
            "\033[1;32m"
            if health_percentage > HEALTH_THRESHOLD_GOOD
            else "\033[1;33m" if health_percentage > HEALTH_THRESHOLD_WARN else "\033[1;31m"
        )
        print(f"  Overall health: {health_color}{health_percentage:.1f}%\033[0m")

    if healthy_models < total_models:
        print("\n\033[1;33mTip: Use 'model:remove' to remove unhealthy models.\033[0m")


def _dispatch_to_plugin(db, command: str, args: str = "") -> Any:
    """
    Dispatch a command to the appropriate plugin.

    Args:
        db: WDBX instance
        command: Command to dispatch
        args: Command arguments

    Returns:
        Result of the command or None if command not found
    """
    logger.debug(f"Dispatching command: {command} with args: {args}")

    # Check if db has the plugin_registry attribute
    if not hasattr(db, "plugin_registry"):
        logger.error("WDBX instance doesn't have plugin_registry attribute")
        print(
            "\033[1;31mError: Unable to access plugin registry. Make sure WDBX is properly initialized.\033[0m"
        )
        return None

    try:
        # Check if the command exists directly in the plugin registry
        if command in db.plugin_registry:
            logger.debug(f"Found command {command} in plugin registry")
            return db.plugin_registry[command](db, args)

        # If this is a command with a specific format (like "prefix:command")
        # try to match it directly
        if ":" in command:
            # The command already includes the prefix, so try to use it directly
            if command in db.plugin_registry:
                logger.debug(f"Found prefixed command {command} in registry")
                return db.plugin_registry[command](db, args)

        # Log available commands for debugging
        logger.debug(f"Available commands: {list(db.plugin_registry.keys())}")
        logger.warning(f"No plugin found for command: {command}")
        print(f"\033[1;31mError: Command '{command}' not found in plugin registry\033[0m")
        return None

    except Exception as e:
        logger.error(f"Error dispatching command {command}: {e}", exc_info=True)
        print(f"\033[1;31mError executing command: {str(e)}\033[0m")
        return None
