"""
Setuptools configuration for WDBX.

This file is provided for backward compatibility with older tooling.
The primary configuration is in pyproject.toml.
"""

import os
import re

from setuptools import find_packages, setup

# Read version from __init__.py
with open(os.path.join("src", "wdbx", "__init__.py")) as f:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if version_match:
        version = version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string")

# Read long description from README.md
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Core dependencies
install_requires = [
    "numpy>=1.20.0",
    "aiohttp>=3.8.0",
    "scikit-learn>=1.0.0",
]

# Optional dependencies
extras_require = {
    "vector": ["faiss-cpu>=1.7.0"],
    "ml": ["torch>=2.0.0", "sentence-transformers>=2.2.0"],
    "llm": ["openai>=1.0.0", "llama-cpp-python>=0.2.0"],
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "flake8>=5.0.0",
        "black>=22.0.0",
        "isort>=5.0.0",
        "mypy>=0.900",
    ],
}

# Full installation includes all optional dependencies
extras_require["full"] = sorted(set(pkg for group in extras_require.values() for pkg in group))

setup(
    name="wdbx",
    version=version,
    description="Wide Distributed Block Exchange for multi-persona AI systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="WDBX Team",
    author_email="info@example.com",
    url="https://github.com/example/wdbx_python",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Database",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "wdbx=wdbx.cli:main",
            "wdbx-web=wdbx.ui.web:web_main",
        ]
    },
    keywords="database, vector, embedding, blockchain, ai, multi-persona, similarity-search",
    project_urls={
        "Documentation": "https://github.com/example/wdbx_python/docs",
        "Source": "https://github.com/example/wdbx_python",
        "Tracker": "https://github.com/example/wdbx_python/issues",
    },
)

if __name__ == "__main__":
    try:
        setup()
    except Exception:
        print(
            "\n\nAn error occurred during installation!\n\n"
            "Please file an issue at https://github.com/example/wdbx-python\n"
        )
        raise
