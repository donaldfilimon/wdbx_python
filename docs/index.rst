WDBX Documentation
=================

.. toctree::
   :maxdepth: 1
   :caption: Overview
   :hidden:

   landing
   features
   installation
   quick-start

.. toctree::
   :maxdepth: 2
   :caption: User Guides
   :hidden:

   user_guide
   usage/vector_operations
   usage/search_config
   usage/metadata_management
   usage/db_management
   usage/error_handling
   ML_INTEGRATION
   diagnostics
   performance_profiling

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/core
   api/client
   api/storage
   api/search
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Plugins
   :hidden:

   plugins/index
   plugins/development
   discord_bot

.. toctree::
   :maxdepth: 1
   :caption: Operations
   :hidden:

   DEPLOYMENT
   monitoring
   backup_recovery
   scaling

.. toctree::
   :maxdepth: 1
   :caption: Diagnostics & Debugging
   :hidden:

   linting
   diagnostics_visualization
   debugging_tools

.. toctree::
   :maxdepth: 1
   :caption: Development
   :hidden:

   CONTRIBUTING
   CODE_QUALITY
   testing_strategy
   design_principles

.. include:: landing.md

===============================
WDBX Documentation
===============================

.. image:: _static/wdbx_banner.png
   :alt: WDBX Banner
   :align: center

WDBX is a high-performance vector database and search engine optimized for machine learning embeddings.

Features
--------

- **Multiple ML Backend Support**: Integration with various machine learning backends like PyTorch, TensorFlow, and FAISS
- **Memory Optimization**: Advanced memory management for handling large embedding datasets
- **Vector Operations**: Efficient similarity search, dimensionality reduction, and vector manipulation
- **Flexible API**: HTTP API with OpenAPI documentation and both synchronous and asynchronous client libraries
- **Kubernetes Ready**: Production-grade deployment configurations for cloud environments
- **Benchmarking Tools**: Compare performance against other vector databases
- **OpenAPI Documentation**: Comprehensive API documentation with Swagger UI

Getting Started
--------------

Installation
~~~~~~~~~~~

Install WDBX using pip:

.. code-block:: bash

   pip install wdbx

Quick Start
~~~~~~~~~~

.. code-block:: python

   from wdbx.client import WDBXClient
   import numpy as np

   # Initialize WDBX client
   client = WDBXClient(vector_dimension=1536)

   # Create vectors
   vector1 = client.create_vector(
       vector_data=np.random.rand(1536).astype(np.float32),
       metadata={"description": "Example vector 1"}
   )
   
   vector2 = client.create_vector(
       vector_data=np.random.rand(1536).astype(np.float32),
       metadata={"description": "Example vector 2"}
   )

   # Calculate similarity
   similarity = client.core.cosine_similarity(vector1, vector2)
   print(f"Similarity: {similarity:.4f}")

   # Search for similar vectors
   results = client.find_similar_vectors(query=vector1, top_k=5)
   for vector_id, score in results:
       print(f"Vector ID: {vector_id}, Score: {score:.4f}")

   # Shut down
   client.disconnect()

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   quickstart
   cli
   
.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   guide/configuration
   guide/vectors
   guide/blocks
   guide/searching
   guide/persistence
   guide/memory
   guide/performance
   
.. toctree::
   :maxdepth: 2
   :caption: API Reference:
   
   api/client
   api/core
   api/ml_backend
   api/http

.. toctree::
   :maxdepth: 2
   :caption: Plugins:
   
   plugins/discord_bot

.. toctree::
   :maxdepth: 2
   :caption: Operations:
   
   monitoring
   operations/deployment
   operations/scaling
   operations/security
   operations/backup

.. toctree::
   :maxdepth: 1
   :caption: Documentation:
   
   diagnostics
   diagnostics_visualization
   performance_profiling
   optimized_vector_operations
   linting

.. toctree::
   :maxdepth: 1
   :caption: Development & Planning:

   CONTRIBUTING
   CODE_QUALITY
   ML_INTEGRATION
   DEPLOYMENT
   NEXT_STEPS
   RESTRUCTURING
   improvements_summary

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 