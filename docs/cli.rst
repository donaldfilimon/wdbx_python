=======================
Command Line Interface
=======================

WDBX provides a comprehensive command-line interface (CLI) for interacting with the system directly from your terminal. This interface allows you to initialize databases, create vectors and blocks, perform searches, and manage your data without writing any code.

Installation
-----------

The WDBX CLI is automatically installed when you install the WDBX package:

.. code-block:: bash

   pip install wdbx

Basic Usage
-----------

Once installed, you can use the ``wdbx`` command to interact with WDBX:

.. code-block:: bash

   wdbx --help

Global Options
-------------

The following options apply to all commands:

- ``--version``: Show version information
- ``--data-dir PATH``: Data directory location
- ``--vector-dimension N``: Vector dimension (default: 1536)
- ``--log-level LEVEL``: Logging level (debug, info, warning, error, critical)
- ``--config PATH``: Path to configuration file

Commands
--------

The CLI provides the following commands:

Initialization and Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Initialize a new WDBX database
   wdbx init --data-dir ./mydata

   # Initialize with force overwrite
   wdbx init --data-dir ./mydata --force

Vector Operations
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create a vector
   wdbx create-vector --data "[0.1, 0.2, 0.3, 0.4]" --metadata '{"description": "Test vector"}'

   # Create and save a vector with specific ID
   wdbx create-vector --data "[0.1, 0.2, 0.3, 0.4]" --id "my-vector-1" --save

   # Search for similar vectors
   wdbx search --query-id "my-vector-1" --top-k 5

   # Search using vector data
   wdbx search --query-data "[0.1, 0.2, 0.3, 0.4]" --top-k 10 --threshold 0.5

Block Operations
~~~~~~~~~~~~~~

.. code-block:: bash

   # Create a block
   wdbx create-block --data '{"name": "Test Block", "description": "This is a test"}' --vectors '["vector-id-1", "vector-id-2"]'

   # Search for relevant blocks
   wdbx search-blocks --query-id "my-vector-1" --top-k 5

   # Search blocks using text
   wdbx search-blocks --query-text "test query" --top-k 5

Data Management
~~~~~~~~~~~~~~

.. code-block:: bash

   # Get a vector by ID
   wdbx get --id "my-vector-1" --type vector

   # Get a block by ID
   wdbx get --id "my-block-1" --type block

   # Export data
   wdbx export --output-dir ./export-data

   # Import data
   wdbx import --input-dir ./export-data

   # Show statistics
   wdbx stats

   # Clear in-memory data
   wdbx clear

   # Optimize memory usage
   wdbx optimize

Server Operations
~~~~~~~~~~~~~~~

.. code-block:: bash

   # Start the WDBX server
   wdbx server --host 127.0.0.1 --port 8000 --workers 4

Command Details
--------------

init
~~~~

Initialize a new WDBX database.

.. code-block:: bash

   wdbx init --data-dir PATH [--force]

Options:
  - ``--data-dir PATH``: Data directory location (required)
  - ``--force``: Force initialization even if data directory exists

create-vector
~~~~~~~~~~~~

Create a new embedding vector.

.. code-block:: bash

   wdbx create-vector --data JSON_ARRAY [--metadata JSON_OBJECT] [--id ID] [--save]

Options:
  - ``--data JSON_ARRAY``: Vector data as JSON array (required)
  - ``--metadata JSON_OBJECT``: Vector metadata as JSON object
  - ``--id ID``: Vector ID (generated if not provided)
  - ``--save``: Save the vector to disk

create-block
~~~~~~~~~~~

Create a new data block.

.. code-block:: bash

   wdbx create-block --data JSON_OBJECT [--vectors JSON_ARRAY] [--id ID] [--save]

Options:
  - ``--data JSON_OBJECT``: Block data as JSON object (required)
  - ``--vectors JSON_ARRAY``: List of vector IDs to include in the block
  - ``--id ID``: Block ID (generated if not provided)
  - ``--save``: Save the block to disk

search
~~~~~

Search for similar vectors.

.. code-block:: bash

   wdbx search (--query-id ID | --query-data JSON_ARRAY) [--top-k N] [--threshold FLOAT] [--output-format FORMAT]

Options:
  - ``--query-id ID``: ID of the query vector
  - ``--query-data JSON_ARRAY``: Query vector data as JSON array
  - ``--top-k N``: Number of results to return (default: 10)
  - ``--threshold FLOAT``: Minimum similarity threshold (default: 0.0)
  - ``--output-format FORMAT``: Output format (text or json, default: text)

search-blocks
~~~~~~~~~~~

Search for relevant blocks.

.. code-block:: bash

   wdbx search-blocks (--query-id ID | --query-data JSON_ARRAY | --query-text TEXT) [--top-k N] [--threshold FLOAT] [--output-format FORMAT]

Options:
  - ``--query-id ID``: ID of the query vector
  - ``--query-data JSON_ARRAY``: Query vector data as JSON array
  - ``--query-text TEXT``: Query text
  - ``--top-k N``: Number of results to return (default: 10)
  - ``--threshold FLOAT``: Minimum similarity threshold (default: 0.0)
  - ``--output-format FORMAT``: Output format (text or json, default: text)

get
~~~

Get a vector or block by ID.

.. code-block:: bash

   wdbx get --id ID --type TYPE [--output-format FORMAT]

Options:
  - ``--id ID``: ID of the vector or block (required)
  - ``--type TYPE``: Type of object to get (vector or block, required)
  - ``--output-format FORMAT``: Output format (text or json, default: text)

export
~~~~~

Export data to a directory.

.. code-block:: bash

   wdbx export --output-dir PATH [--format FORMAT]

Options:
  - ``--output-dir PATH``: Output directory (required)
  - ``--format FORMAT``: Export format (json or binary, default: json)

import
~~~~~

Import data from a directory.

.. code-block:: bash

   wdbx import --input-dir PATH [--format FORMAT]

Options:
  - ``--input-dir PATH``: Input directory (required)
  - ``--format FORMAT``: Import format (json or binary, default: json)

stats
~~~~

Show statistics.

.. code-block:: bash

   wdbx stats [--output-format FORMAT]

Options:
  - ``--output-format FORMAT``: Output format (text or json, default: text)

clear
~~~~

Clear all in-memory data.

.. code-block:: bash

   wdbx clear [--confirm]

Options:
  - ``--confirm``: Confirm clearing data without prompting

optimize
~~~~~~~

Optimize memory usage.

.. code-block:: bash

   wdbx optimize

server
~~~~~

Start the WDBX server.

.. code-block:: bash

   wdbx server [--host HOST] [--port PORT] [--workers N]

Options:
  - ``--host HOST``: Server host (default: 127.0.0.1)
  - ``--port PORT``: Server port (default: 8000)
  - ``--workers N``: Number of worker processes (default: 1)

Examples
--------

Creating and Searching Vectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create a vector
   wdbx create-vector --data "[0.1, 0.2, 0.3, 0.4]" --metadata '{"description": "Test vector"}' --save

   # Create another vector
   wdbx create-vector --data "[0.15, 0.25, 0.35, 0.45]" --metadata '{"description": "Similar vector"}' --save

   # Search for similar vectors
   wdbx search --query-data "[0.1, 0.2, 0.3, 0.4]" --top-k 5

Working with Blocks
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create vectors
   wdbx create-vector --data "[0.1, 0.2, 0.3, 0.4]" --id "vec1" --save
   wdbx create-vector --data "[0.5, 0.6, 0.7, 0.8]" --id "vec2" --save

   # Create a block with these vectors
   wdbx create-block --data '{"name": "Test Block", "content": "This is a test block"}' --vectors '["vec1", "vec2"]' --save

   # Search for blocks
   wdbx search-blocks --query-text "test block" --top-k 5

Exporting and Importing Data
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create some data
   wdbx create-vector --data "[0.1, 0.2, 0.3, 0.4]" --id "vec1" --save
   wdbx create-vector --data "[0.5, 0.6, 0.7, 0.8]" --id "vec2" --save
   wdbx create-block --data '{"name": "Test Block"}' --vectors '["vec1", "vec2"]' --save

   # Export data
   wdbx export --output-dir ./backup

   # Clear data
   wdbx clear --confirm

   # Import data
   wdbx import --input-dir ./backup

Running the Server
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Start the server
   wdbx server --host 0.0.0.0 --port 8080 --workers 2

Integrating with Shell Scripts
----------------------------

The CLI can be easily integrated into shell scripts for automation:

.. code-block:: bash

   #!/bin/bash
   
   # Initialize database
   wdbx init --data-dir ./mydata --force
   
   # Create vectors
   for i in {1..10}; do
     # Generate random vector data
     vector_data="["
     for j in {1..4}; do
       vector_data+="$(echo "scale=2; $RANDOM/32767" | bc),"
     done
     vector_data="${vector_data%,}]"
     
     # Create vector
     wdbx create-vector --data "$vector_data" --metadata "{\"index\": $i}" --save
   done
   
   # Show statistics
   wdbx stats
   
   echo "Completed vector creation"

Exit Codes
---------

The CLI returns the following exit codes:

- ``0``: Success
- ``1``: Error (with error message) 