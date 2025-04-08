Python Client API
================

.. module:: wdbx.client
   :synopsis: Python client for WDBX vector database

The WDBX Python client provides a high-level interface for interacting with the WDBX server.

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
===========

.. code-block:: bash

   pip install wdbx-python

Basic Usage
==========

Creating a Client
----------------

.. code-block:: python

   from wdbx import WDBXClient

   client = WDBXClient(host="localhost", port=8000)

Using the Context Manager
------------------------

.. code-block:: python

   with WDBXClient(host="localhost", port=8000) as client:
       # Use client here
       pass

Working with Vectors
-------------------

.. code-block:: python

   # Add a vector
   client.add_vector(vector=[1.0, 2.0, 3.0], metadata={"id": "vec1"})

   # Search for similar vectors
   results = client.search(vector=[1.0, 2.0, 3.0], k=5)

Working with Blocks
------------------

.. code-block:: python

   # Create a block
   block = client.create_block(name="my_block")

   # Add vectors to block
   block.add_vectors(vectors=[[1.0, 2.0], [3.0, 4.0]])

Batch Operations
---------------

.. code-block:: python

   # Batch add vectors
   vectors = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
   metadata = [{"id": f"vec{i}"} for i in range(len(vectors))]
   client.batch_add_vectors(vectors=vectors, metadata=metadata)

Memory Management
----------------

.. code-block:: python

   # Clear all vectors
   client.clear()

   # Delete specific vectors
   client.delete_vectors(ids=["vec1", "vec2"])

Import/Export
------------

.. code-block:: python

   # Export vectors
   client.export_vectors("vectors.json")

   # Import vectors
   client.import_vectors("vectors.json")

Using the Async Client
---------------------

.. code-block:: python

   from wdbx import AsyncWDBXClient

   async with AsyncWDBXClient(host="localhost", port=8000) as client:
       # Use async client here
       await client.add_vector(vector=[1.0, 2.0, 3.0])

API Reference
============

WDBXClient
----------

.. autoclass:: wdbx.client.WDBXClient
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
----------------

.. automodule:: wdbx.client.utils
   :members:
   :undoc-members:
   :show-inheritance:

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