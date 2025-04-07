#!/usr/bin/env python3
"""
WDBX Lint Configuration

This module defines configuration settings for linting tools used in the WDBX project.
It provides options for integrating external linting tools like ruff, black, and isort
with our custom linting framework.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
PLUGINS_DIR = PROJECT_ROOT / "wdbx_plugins"

# Linting configurations
RUFF_CONFIG = {
    "enabled": True,
    "fix": True,
    "select": [
        "E",  # pycodestyle errors
        "F",  # pyflakes
        "I",  # isort
        "W",  # pycodestyle warnings
        "N",  # pep8-naming
        "UP",  # pyupgrade
        "B",  # flake8-bugbear
        "C4",  # flake8-comprehensions
        "PLR",  # pylint refactor suggestions
    ],
    "ignore": [
        "E501",  # line length - handle separately
    ],
    "target-version": "py39",
}

BLACK_CONFIG = {
    "enabled": True,
    "line_length": 100,
    "target_version": ["py39"],
    "include": r"\.pyi?$",
    "exclude": r"/(\.git|\.hg|\.mypy_cache|\.nox|\.tox|\.venv|_build|buck-out|build|dist)/"
}

ISORT_CONFIG = {
    "enabled": True,
    "profile": "black",
    "line_length": 100,
}

AUTOFLAKE_CONFIG = {
    "enabled": True,
    "remove_unused_variables": True,
    "remove_all_unused_imports": True,
    "remove_duplicate_keys": True,
}

# Custom linter configurations
WDBX_LINTER_CONFIG = {
    "enabled": True,
    "fixers": [
        "indentation",
        "strings",
        "imports",
        "try_except",
        "docstrings",
        "discord_bot",
    ],
}

# Special config for discord bot fixer
DISCORD_BOT_FIXER_CONFIG = {
    "enabled": True,
    "auto_fix": True,
    "source_path": "wdbx_plugins/discord_bot.py",  # Default path to the Discord bot file
}

# Files and directories to exclude from linting
EXCLUDE_PATTERNS = [
    ".git",
    ".github",
    "__pycache__",
    "*.pyc",
    "venv",
    ".venv",
    "build",
    "dist",
    "*.egg-info",
]

# Mapping of tools to their install commands if not found
TOOL_INSTALL_COMMANDS = {
    "ruff": "pip install ruff",
    "black": "pip install black",
    "isort": "pip install isort",
    "autoflake": "pip install autoflake",
} 