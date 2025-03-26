from setuptools import setup, find_packages

setup(
    name="wdbx",
    version="1.0.0",
    description="Wide Distributed Block Exchange for multi-persona AI systems",
    author="Donald Filimon",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "faiss",
        "aiohttp",
        "scikit-learn"
    ],
    entry_points={
        "console_scripts": [
            "wdbx= wdbx.cli:main"
        ]
    },
)
