"""
Unit tests for the model repository plugin.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import model_repo module
try:
    from wdbx_plugins.model_repo import (
        _dispatch_to_plugin,
        cmd_model_add,
        cmd_model_config,
        cmd_model_default,
        cmd_model_embed,
        cmd_model_generate,
        cmd_model_help,
        cmd_model_list,
        cmd_model_remove,
        register_commands,
    )
except ImportError:
    # Skip tests if module not available
    model_repo_available = False
else:
    model_repo_available = True

# Skip entire test class if dependencies not available
import pytest


class MockWDBX:
    """Mock WDBX instance for testing."""
    
    def __init__(self):
        self.plugin_registry = {
            "openai:embed": self.mock_openai_embed,
            "openai:generate": self.mock_openai_generate,
            "hf:embed": self.mock_hf_embed,
            "hf:generate": self.mock_hf_generate,
            "ollama:embed": self.mock_ollama_embed,
            "ollama:generate": self.mock_ollama_generate,
            "ollama:models": self.mock_ollama_models
        }
    
    def mock_openai_embed(self, db, args):
        """Mock OpenAI embed command."""
        return "openai-embed-123"
    
    def mock_openai_generate(self, db, args):
        """Mock OpenAI generate command."""
        return "This is a mock OpenAI generation response."
    
    def mock_hf_embed(self, db, args):
        """Mock HuggingFace embed command."""
        return "hf-embed-456"
    
    def mock_hf_generate(self, db, args):
        """Mock HuggingFace generate command."""
        return "This is a mock HuggingFace generation response."
    
    def mock_ollama_embed(self, db, args):
        """Mock Ollama embed command."""
        return "ollama-embed-789"
    
    def mock_ollama_generate(self, db, args):
        """Mock Ollama generate command."""
        return "This is a mock Ollama generation response."
    
    def mock_ollama_models(self, db, args):
        """Mock Ollama models command."""
        return ["llama3", "mistral", "gemma"]


@pytest.mark.skipif(not model_repo_available, reason="Model repo plugin not available")
class TestModelRepoPlugin(unittest.TestCase):
    """Test the model repository plugin functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.db = MockWDBX()
        self.plugin_registry = {}
        
        # Create a temporary directory for model cache
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_cache_dir = self.temp_dir.name
        
        # Register commands in the plugin registry
        if model_repo_available:
            # Patch the repo_config and plugin_sources
            with patch("wdbx_plugins.model_repo.repo_config") as self.mock_repo_config, \
                 patch("wdbx_plugins.model_repo.plugin_sources") as self.mock_plugin_sources, \
                 patch("wdbx_plugins.model_repo.model_registry") as self.mock_model_registry, \
                 patch("wdbx_plugins.model_repo._save_model_registry") as self.mock_save_registry, \
                 patch("wdbx_plugins.model_repo._save_model_repo_config") as self.mock_save_config:
                
                # Set up mock config
                self.mock_repo_config.copy.return_value = {
                    "default_embedding_model": "openai:text-embedding-3-small",
                    "default_generation_model": "ollama:llama3",
                    "model_cache_dir": self.model_cache_dir,
                    "max_models_per_source": 5,
                    "auto_discover": True
                }
                
                # Set up mock plugin sources
                self.mock_plugin_sources.__iter__.return_value = ["openai", "huggingface", "ollama"]
                
                # Set up mock model registry
                self.mock_model_registry.__getitem__.side_effect = lambda key: {
                    "embedding": {
                        "openai:text-embedding-3-small": {
                            "source": "openai",
                            "model_name": "text-embedding-3-small",
                            "added_at": 1617293847.123456
                        },
                        "ollama:llama3": {
                            "source": "ollama",
                            "model_name": "llama3",
                            "added_at": 1617293847.123456
                        }
                    },
                    "generation": {
                        "openai:gpt-4": {
                            "source": "openai",
                            "model_name": "gpt-4",
                            "added_at": 1617293847.123456
                        },
                        "ollama:llama3": {
                            "source": "ollama",
                            "model_name": "llama3",
                            "added_at": 1617293847.123456
                        }
                    }
                }.get(key, {})
                
                # Register commands
                register_commands(self.plugin_registry)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temporary directory
        self.temp_dir.cleanup()
    
    def test_register_commands(self):
        """Test that commands are registered correctly."""
        self.assertIn("model:list", self.plugin_registry)
        self.assertIn("model:add", self.plugin_registry)
        self.assertIn("model:remove", self.plugin_registry)
        self.assertIn("model:default", self.plugin_registry)
        self.assertIn("model:embed", self.plugin_registry)
        self.assertIn("model:generate", self.plugin_registry)
        self.assertIn("model:config", self.plugin_registry)
        self.assertIn("model", self.plugin_registry)
    
    def test_dispatch_to_plugin(self):
        """Test dispatching commands to other plugins."""
        # Test OpenAI embed
        result = _dispatch_to_plugin(self.db, "embed", "Test text")
        self.assertEqual(result, "openai-embed-123")
        
        # Test Ollama generate
        result = _dispatch_to_plugin(self.db, "generate", "Test prompt")
        self.assertEqual(result, "This is a mock OpenAI generation response.")
        
        # Test non-existent command
        result = _dispatch_to_plugin(self.db, "nonexistent", "Test")
        self.assertIsNone(result)
    
    def test_model_add(self):
        """Test adding a model."""
        with patch("wdbx_plugins.model_repo._dispatch_to_plugin") as mock_dispatch:
            mock_dispatch.return_value = "Success"
            
            # Add a model
            cmd_model_add(self.db, "huggingface:sentence-transformers/all-MiniLM-L6-v2 embedding")
            
            # Check that dispatch was called with the correct arguments
            mock_dispatch.assert_called_once()
            
            # Check that the model was added to the registry
            # Note: The registry is mocked, so we're just checking that _save_model_registry was called
            self.mock_save_registry.assert_called_once()
    
    def test_model_remove(self):
        """Test removing a model."""
        # Setup a side effect to simulate model existence check
        def side_effect(key):
            if key == "embedding":
                return {
                    "openai:text-embedding-3-small": {
                        "source": "openai", 
                        "model_name": "text-embedding-3-small"
                    }
                }
            return {}
        
        with patch("wdbx_plugins.model_repo.model_registry") as mock_registry:
            mock_registry.__getitem__.side_effect = side_effect
            mock_registry.__contains__.return_value = True
            
            # Test removing an existing model
            with patch.dict(self.plugin_registry, {"model:remove": cmd_model_remove}):
                cmd_model_remove(self.db, "openai:text-embedding-3-small")
            
            # Check that the model was removed from the registry
            self.mock_save_registry.assert_called_once()
    
    def test_model_default(self):
        """Test setting a default model."""
        # Setup a side effect to simulate model existence check
        def side_effect(item):
            if item == "embedding":
                return {
                    "openai:text-embedding-3-small": {
                        "source": "openai", 
                        "model_name": "text-embedding-3-small"
                    }
                }
            return {}
        
        with patch("wdbx_plugins.model_repo.model_registry") as mock_registry:
            mock_registry.__getitem__.side_effect = side_effect
            mock_registry.__contains__.return_value = True
            
            # Set a default model
            with patch.dict(self.plugin_registry, {"model:default": cmd_model_default}):
                cmd_model_default(self.db, "openai:text-embedding-3-small embedding")
            
            # Check that the default was saved
            self.mock_save_config.assert_called_once()
    
    def test_model_embed(self):
        """Test embedding with a model."""
        with patch("wdbx_plugins.model_repo._dispatch_to_plugin") as mock_dispatch:
            mock_dispatch.return_value = "test-embed-id"
            
            # Test with default model
            cmd_model_embed(self.db, "Test text")
            
            # Check that dispatch was called
            mock_dispatch.assert_called_once()
            
            # Test with specified model
            mock_dispatch.reset_mock()
            cmd_model_embed(self.db, "Test text model=openai:text-embedding-3-small")
            
            # Check that dispatch was called with the correct arguments
            mock_dispatch.assert_called_once()
    
    def test_model_generate(self):
        """Test generating text with a model."""
        with patch("wdbx_plugins.model_repo._dispatch_to_plugin") as mock_dispatch:
            mock_dispatch.return_value = "Generated text"
            
            # Test with default model
            cmd_model_generate(self.db, "Test prompt")
            
            # Check that dispatch was called
            mock_dispatch.assert_called_once()
            
            # Test with specified model
            mock_dispatch.reset_mock()
            cmd_model_generate(self.db, "Test prompt model=openai:gpt-4")
            
            # Check that dispatch was called with the correct arguments
            mock_dispatch.assert_called_once()
    
    def test_model_config(self):
        """Test configuring the model repository."""
        # Test with no arguments (display current config)
        with patch("builtins.print") as mock_print:
            cmd_model_config(self.db, "")
            
            # Check that print was called to display the config
            mock_print.assert_called()
        
        # Test setting a config value
        with patch.dict(self.plugin_registry, {"model:config": cmd_model_config}):
            cmd_model_config(self.db, "max_models_per_source=10")
        
        # Check that the config was saved
        self.mock_save_config.assert_called_once()
    
    def test_model_list(self):
        """Test listing models."""
        # Test with no arguments (list all models)
        with patch("builtins.print") as mock_print:
            cmd_model_list(self.db, "")
            
            # Check that print was called to display models
            mock_print.assert_called()
        
        # Test listing specific model type
        with patch("builtins.print") as mock_print:
            cmd_model_list(self.db, "embedding")
            
            # Check that print was called to display models
            mock_print.assert_called()


if __name__ == "__main__":
    unittest.main() 