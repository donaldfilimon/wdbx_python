from setuptools import setup, find_packages
import os
import re

# Read version from __init__.py
with open(os.path.join('wdbx', '__init__.py'), 'r') as f:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if version_match:
        version = version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string")

# Read long description from README.md
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Core dependencies
install_requires = [
    'numpy>=1.20.0',
]

# Optional dependencies
extras_require = {
    'vector': ['faiss-cpu>=1.7.0'],
    'server': ['aiohttp>=3.8.0'],
    'ml': ['scikit-learn>=1.0.0'],
    'jax': ['jax>=0.3.0', 'jaxlib>=0.3.0'],
    'dev': [
        'pytest>=7.0.0',
        'pytest-cov>=4.0.0',
        'flake8>=5.0.0',
        'black>=22.0.0',
        'isort>=5.0.0',
        'mypy>=0.900',
    ],
}

# Full installation includes all optional dependencies
extras_require['full'] = [pkg for group in extras_require.values() for pkg in group]

setup(
    name="wdbx",
    version="0.0.1",
    description="Wide Distributed Block Exchange for multi-persona AI systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Donald Filimon",
    author_email="donald.filimon@example.com",
    url="https://github.com/donaldfilimon/wdbx_python",
    packages=find_packages(),
    install_requires="requirements.txt",
    extras_require="python3, jax, jaxlib, ",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Database",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "wdbx=wdbx.cli:main"
        ]
    },
    keywords="database, vector, embedding, blockchain, ai, multi-persona, similarity-search",
    project_urls={
        "Documentation": "https://github.com/donaldfilimon/wdbx_python/docs",
        "Source": "https://github.com/donaldfilimon/wdbx_python",
        "Tracker": "https://github.com/donaldfilimon/wdbx_python/issues",
    },
)
