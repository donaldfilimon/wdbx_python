============================
WDBX Client Library
============================

The WDBX Client Library provides a simple, high-level interface for interacting with WDBX. It abstracts away the underlying complexity and provides convenient methods for common operations.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

The client library is designed to make it easy to integrate WDBX into applications. It provides both synchronous and asynchronous interfaces, handles resource management, and includes utilities for common operations.

Key features:

* Simple, intuitive API for creating and working with vectors and blocks
* Automatic resource management (including temporary directories)
* Context manager support for clean resource handling
* Batch operations for improved performance
* Memory optimization utilities
* Import/export functionality
* Profiling and error handling

Installation
-----------

The WDBX client is included with the main WDBX package. Install it using pip:

.. code-block:: bash

   pip install wdbx

Basic Usage
----------

Creating a Client
~~~~~~~~~~~~~~~~

.. code-block:: python

   from wdbx.client import WDBXClient

   # Create a client with a specific data directory
   client = WDBXClient(
       data_dir="/path/to/data",
       vector_dimension=1536,
       enable_memory_optimization=True
   )

   # Connect to WDBX (also happens automatically if auto_connect=True)
   client.connect()

   # Use the client...

   # Disconnect when done
   client.disconnect()

Using the Context Manager
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from wdbx.client import wdbx_session

   # Using a context manager (automatically connects and disconnects)
   with wdbx_session(vector_dimension=1536) as client:
       # Use the client...
       pass  # Resources automatically cleaned up

Working with Vectors
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from wdbx.client import WDBXClient

   client = WDBXClient()

   # Create a vector
   vector_data = np.random.rand(1536).astype(np.float32)
   metadata = {"description": "Example vector", "tags": ["example", "test"]}
   vector = client.create_vector(vector_data=vector_data, metadata=metadata)

   # Save the vector
   client.save_vector(vector)

   # Retrieve a vector by ID
   retrieved_vector = client.get_vector(vector.id)

   # Find similar vectors
   similar_vectors = client.find_similar_vectors(query=vector, top_k=10)
   for vector_id, similarity in similar_vectors:
       print(f"Vector ID: {vector_id}, Similarity: {similarity:.4f}")

Working with Blocks
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create a block containing vectors
   block = client.create_block(
       data={"name": "Example block", "description": "Contains example vectors"},
       embeddings=[vector1, vector2, vector3]
   )

   # Save the block
   client.save_block(block)

   # Retrieve a block by ID
   retrieved_block = client.get_block(block.id)

   # Search for relevant blocks
   similar_blocks = client.search_blocks(query=vector, top_k=5)
   for block, similarity in similar_blocks:
       print(f"Block ID: {block.id}, Similarity: {similarity:.4f}")

Batch Operations
~~~~~~~~~~~~~~~

.. code-block:: python

   # Create multiple vectors in batch
   vector_data_list = [np.random.rand(1536).astype(np.float32) for _ in range(10)]
   metadata_list = [{"index": i} for i in range(10)]
   
   vectors = client.batch_create_vectors(
       vector_data_list=vector_data_list,
       metadata_list=metadata_list
   )

   # Find similar vectors for multiple queries
   results = client.batch_find_similar_vectors(
       queries=[vectors[0], vectors[1]],
       top_k=5
   )

Memory Management
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Optimize memory usage
   client.optimize_memory()

   # Clear all in-memory data
   client.clear()

Import/Export
~~~~~~~~~~~~

.. code-block:: python

   # Export data to a directory
   client.export_data(output_dir="/path/to/export", format="json")

   # Import data from a directory
   client.import_data(input_dir="/path/to/export", format="json")

Using the Async Client
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import asyncio

   async def async_operations():
       with wdbx_session() as client:
           # Get the async client
           async_client = client.get_async_client()
           
           # Create a vector asynchronously
           vector = await async_client.create_vector(
               vector_data=vector_data,
               metadata=metadata
           )
           
           # Find similar vectors asynchronously
           similar_vectors = await async_client.find_similar_vectors(
               query_vector=vector,
               top_k=5
           )

   # Run the async function
   asyncio.run(async_operations())

API Reference
------------

WDBXClient
~~~~~~~~~

.. py:class:: WDBXClient

   High-level client for interacting with WDBX.

   .. py:method:: __init__(data_dir=None, vector_dimension=1536, enable_memory_optimization=True, auto_connect=True, config_path=None, **kwargs)

      Initialize the WDBX client.

      :param data_dir: Directory for storing vector data (if None, will use a temporary directory)
      :param vector_dimension: Dimension of vectors
      :param enable_memory_optimization: Whether to enable memory optimization
      :param auto_connect: Whether to connect automatically
      :param config_path: Path to configuration file (if provided, other parameters are ignored)
      :param kwargs: Additional keyword arguments for WDBXCore

   .. py:method:: connect()

      Connect to WDBX.

      :return: Self for chaining

   .. py:method:: disconnect()

      Disconnect from WDBX and release resources.

   .. py:method:: core

      Get the WDBX core instance.

      :return: WDBX core instance
      :raises: WDBXError if not connected to WDBX

   .. py:method:: get_async_client()

      Get the async WDBX client.

      :return: Async WDBX client
      :raises: WDBXError if not connected to WDBX

   .. py:method:: create_vector(vector_data, metadata=None, vector_id=None)

      Create a new embedding vector.

      :param vector_data: Vector data as list or numpy array
      :param metadata: Additional metadata for the vector
      :param vector_id: Optional ID for the vector (generated if not provided)
      :return: Created embedding vector
      :raises: WDBXError if the vector creation fails

   .. py:method:: create_block(data, embeddings=None, block_id=None, references=None)

      Create a new data block.

      :param data: Block data
      :param embeddings: Embedding vectors associated with this block
      :param block_id: Optional ID for the block (generated if not provided)
      :param references: Optional references to other blocks
      :return: Created block
      :raises: WDBXError if the block creation fails

   .. py:method:: find_similar_vectors(query, top_k=10, threshold=0.0)

      Find vectors similar to the query.

      :param query: Query vector or vector ID (if string)
      :param top_k: Number of results to return
      :param threshold: Minimum similarity threshold
      :return: List of (vector_id, similarity) tuples sorted by similarity
      :raises: WDBXError if the search fails

   .. py:method:: search_blocks(query, top_k=10, threshold=0.0)

      Search for blocks relevant to the query.

      :param query: Query as text, dict, embedding vector, or numpy array
      :param top_k: Number of results to return
      :param threshold: Minimum similarity threshold
      :return: List of (block, similarity) tuples sorted by similarity
      :raises: WDBXError if the search fails

   .. py:method:: get_vector(vector_id)

      Get a vector by ID.

      :param vector_id: ID of the vector
      :return: Vector instance or None if not found

   .. py:method:: get_block(block_id)

      Get a block by ID.

      :param block_id: ID of the block
      :return: Block instance or None if not found

   .. py:method:: save_vector(vector, overwrite=False)

      Save a vector to disk.

      :param vector: Vector to save
      :param overwrite: Whether to overwrite if the file exists
      :return: True if successful

   .. py:method:: save_block(block, overwrite=False)

      Save a block to disk.

      :param block: Block to save
      :param overwrite: Whether to overwrite if the file exists
      :return: True if successful

   .. py:method:: batch_create_vectors(vector_data_list, metadata_list=None, vector_ids=None)

      Create multiple vectors in batch.

      :param vector_data_list: List of vector data
      :param metadata_list: List of metadata for each vector
      :param vector_ids: List of IDs for each vector
      :return: List of created vectors

   .. py:method:: batch_find_similar_vectors(queries, top_k=10, threshold=0.0)

      Find vectors similar to multiple queries in batch.

      :param queries: List of query vectors or vector IDs
      :param top_k: Number of results to return for each query
      :param threshold: Minimum similarity threshold
      :return: List of result lists, each containing (vector_id, similarity) tuples

   .. py:method:: optimize_memory()

      Optimize memory usage.

   .. py:method:: clear()

      Clear all in-memory data.

   .. py:method:: get_stats()

      Get processing statistics.

      :return: Dictionary of statistics

   .. py:method:: export_data(output_dir, format="json")

      Export all data to a directory.

      :param output_dir: Output directory
      :param format: Export format (json or binary)
      :return: True if successful

   .. py:method:: import_data(input_dir, format="json")

      Import data from a directory.

      :param input_dir: Input directory
      :param format: Import format (json or binary)
      :return: True if successful

Utility Functions
~~~~~~~~~~~~~~~~

.. py:function:: wdbx_session(data_dir=None, vector_dimension=1536, enable_memory_optimization=True, **kwargs)

   Context manager for a WDBX session.

   :param data_dir: Directory for storing vector data (if None, will use a temporary directory)
   :param vector_dimension: Dimension of vectors
   :param enable_memory_optimization: Whether to enable memory optimization
   :param kwargs: Additional keyword arguments for WDBXCore
   :return: WDBX client

.. py:function:: create_vector(vector_data, metadata=None, vector_id=None, data_dir=None, **kwargs)

   Create a vector without explicitly creating a client.

   :param vector_data: Vector data as list or numpy array
   :param metadata: Additional metadata for the vector
   :param vector_id: Optional ID for the vector (generated if not provided)
   :param data_dir: Directory for storing vector data (if None, will use a temporary directory)
   :param kwargs: Additional keyword arguments for WDBXCore
   :return: Created embedding vector

.. py:function:: find_similar_vectors(query, top_k=10, threshold=0.0, data_dir=None, **kwargs)

   Find similar vectors without explicitly creating a client.

   :param query: Query vector or vector ID (if string)
   :param top_k: Number of results to return
   :param threshold: Minimum similarity threshold
   :param data_dir: Directory for storing vector data (if None, will use a temporary directory)
   :param kwargs: Additional keyword arguments for WDBXCore
   :return: List of (vector_id, similarity) tuples sorted by similarity

Examples
--------

For complete examples, see the `client_usage.py` example file in the WDBX distribution:

.. code-block:: python

   from wdbx.client import WDBXClient, wdbx_session
   
   # Example 1: Basic client usage
   client = WDBXClient(data_dir="/path/to/data")
   # ... (perform operations)
   client.disconnect()
   
   # Example 2: Using a context manager
   with wdbx_session() as client:
       # ... (perform operations)
       # automatically disconnects when done
   
   # Example 3: Memory optimization
   with wdbx_session() as client:
       # ... (create vectors)
       client.optimize_memory()
       # ... (continue working with optimized memory)
   
   # Example 4: Using the async client
   import asyncio
   
   async def async_operations():
       with wdbx_session() as client:
           async_client = client.get_async_client()
           # ... (perform async operations)
   
   asyncio.run(async_operations()) 