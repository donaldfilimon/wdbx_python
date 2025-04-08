#!/usr/bin/env python
"""WDBX - Vector Database for AI Applications.

This package provides a vector database optimized for AI applications with
plugin support for various integrations.
"""

import os

from setuptools import find_packages, setup

# Get long description from README.md
with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

# Get version from version.py
version = {}
with open(os.path.join("src", "wdbx", "version.py"), encoding="utf-8") as f:
    exec(f.read(), version)

# Core dependencies
install_requires = [
    "numpy>=1.19.0",
    "requests>=2.25.0",
    "typing-extensions>=3.7.4",
    "pydantic>=1.8.0",
    "aiohttp>=3.7.4",
]

# Development dependencies
dev_requires = [
    "black",
    "ruff",
    "mypy",
    "pytest",
    "pytest-cov",
    "pre-commit",
    "tox",
]

# Plugin-specific dependencies
plugin_requires = {
    "discord": ["discord.py>=2.0.0", "matplotlib", "plotly"],
    "ollama": ["httpx>=0.18.0"],
    "lmstudio": ["httpx>=0.18.0"],
    "twitch": ["twitchio>=2.0.0"],
    "youtube": [
        "google-api-python-client",
        "google-auth",
        "google-auth-oauthlib",
        "google-auth-httplib2",
    ],
    "webscraper": ["beautifulsoup4", "lxml", "playwright", "requests"],
    "all": [],  # Will be filled in below
}

# UI dependencies (for Streamlit app)
ui_requires = [
    "streamlit>=1.10.0",
    "pandas>=1.3.0",
    "matplotlib>=3.4.0",
    "scikit-learn>=1.0.0",
    "plotly>=5.0.0",
]

# Add all plugin dependencies to the 'all' extra
plugin_requires["all"] = list(set(dep for deps in plugin_requires.values() for dep in deps))

# Define all extras_require
extras_require = {
    "dev": dev_requires,
    "ui": ui_requires,
    "docs": [
        "mkdocs",
        "mkdocs-material",
        "mdx-include",
    ],
    **plugin_requires,
}

# Add an 'all' extras_require that includes everything
extras_require["all"] = list(set(
    dep for deps in extras_require.values() 
    for dep in (deps if isinstance(deps, list) else [deps])
))

setup(
    name="wdbx",
    version=version.get("__version__", "0.1.0"),
    description="Vector database and embedding management system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="WDBX Team",
    author_email="info@wdbx.ai",
    url="https://github.com/wdbx/wdbx-python",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "wdbx=wdbx.cli:main",
            "wdbx-ui=wdbx.ui.launcher:main",
        ],
    },
    include_package_data=True,
)
