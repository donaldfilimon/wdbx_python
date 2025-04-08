Command Line Interface
====================

The WDBX CLI provides a command-line interface for managing the WDBX server and performing operations.

Basic Commands
-------------

Start the server:

.. code-block:: bash

    wdbx server start

Create a vector:

.. code-block:: bash

    wdbx vector create --id my_vector --data "[1.0, 2.0, 3.0]"

Search for vectors:

.. code-block:: bash

    wdbx vector search --query "[1.0, 2.0, 3.0]" --top-k 5

Server Management
----------------

Manage the WDBX server:

.. code-block:: bash

    # Start server
    wdbx server start

    # Stop server
    wdbx server stop

    # Check status
    wdbx server status

    # View logs
    wdbx server logs

Vector Operations
----------------

Work with vectors:

.. code-block:: bash

    # Create vector
    wdbx vector create --id my_vector --data "[1.0, 2.0, 3.0]"

    # Get vector
    wdbx vector get --id my_vector

    # Update vector
    wdbx vector update --id my_vector --data "[4.0, 5.0, 6.0]"

    # Delete vector
    wdbx vector delete --id my_vector

    # Search vectors
    wdbx vector search --query "[1.0, 2.0, 3.0]" --top-k 5

Block Operations
---------------

Work with blocks:

.. code-block:: bash

    # Create block
    wdbx block create --id my_block --vectors "[1.0, 2.0, 3.0]" "[4.0, 5.0, 6.0]"

    # Get block
    wdbx block get --id my_block

    # Update block
    wdbx block update --id my_block --vectors "[7.0, 8.0, 9.0]" "[10.0, 11.0, 12.0]"

    # Delete block
    wdbx block delete --id my_block

    # Search blocks
    wdbx block search --query "[1.0, 2.0, 3.0]" --top-k 5

Configuration
------------

Configure the CLI:

.. code-block:: bash

    # Set default host
    wdbx config set host localhost

    # Set default port
    wdbx config set port 8000

    # View current config
    wdbx config show

    # Reset config
    wdbx config reset

Installation
-----------

Install the WDBX CLI using pip:

.. code-block:: bash

    pip install wdbx-cli

Global Options
-------------

Common options available for all commands:

.. code-block:: bash

    --host TEXT     Server hostname (default: localhost)
    --port INTEGER  Server port (default: 8080)
    --verbose      Enable verbose output
    --quiet       Suppress all output except errors
    --config FILE  Path to config file

Initialization and Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Commands for setting up and configuring WDBX:

.. code-block:: bash

    wdbx init             Initialize a new WDBX instance
    wdbx config set       Set configuration options
    wdbx config show      Display current configuration

Data Management
~~~~~~~~~~~~~

Commands for managing data:

.. code-block:: bash

    wdbx export           Export data to file
    wdbx import           Import data from file
    wdbx backup           Create a backup
    wdbx restore          Restore from backup

Command Details
-------------

Detailed documentation for each command.

create-vector
~~~~~~~~~~~~

Create a new vector in the database:

.. code-block:: bash

    wdbx vector create --dimension 3 --values "1.0,2.0,3.0"

create-block
~~~~~~~~~~~

Create a new block from existing vectors:

.. code-block:: bash

    wdbx block create --vectors "id1,id2,id3"

search
~~~~~~

Search for similar vectors:

.. code-block:: bash

    wdbx vector search --query "1.0,2.0,3.0" --k 10

search-blocks
~~~~~~~~~~~~

Search within specific blocks:

.. code-block:: bash

    wdbx block search --block-id "block1" --query "1.0,2.0,3.0"

export
~~~~~~

Export data to a file:

.. code-block:: bash

    wdbx export --output "backup.wdbx"

import
~~~~~~

Import data from a file:

.. code-block:: bash

    wdbx import --input "backup.wdbx"

stats
~~~~~

Display database statistics:

.. code-block:: bash

    wdbx stats --detailed

clear
~~~~~

Clear database or cache:

.. code-block:: bash

    wdbx clear --cache
    wdbx clear --all

optimize
~~~~~~~~

Optimize database performance:

.. code-block:: bash

    wdbx optimize --full

server
~~~~~~

Manage the WDBX server:

.. code-block:: bash

    wdbx server start --host localhost --port 8080
    wdbx server stop
    wdbx server status

Examples
--------

Creating and Searching Vectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Create a vector
    wdbx vector create --dimension 3 --values "1.0,2.0,3.0"
    
    # Search for similar vectors
    wdbx vector search --query "1.0,2.0,3.0" --k 10

Working with Blocks
~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Create a block
    wdbx block create --vectors "id1,id2,id3"
    
    # Search within the block
    wdbx block search --block-id "block1" --query "1.0,2.0,3.0"

Exporting and Importing Data
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Export data
    wdbx export --output "backup.wdbx"
    
    # Import data
    wdbx import --input "backup.wdbx"

Running the Server
~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Start the server
    wdbx server start --host localhost --port 8080

Integrating with Shell Scripts
----------------------------

Example shell script using WDBX CLI:

.. code-block:: bash

    #!/bin/bash
    
    # Start the server
    wdbx server start
    
    # Create vectors
    wdbx vector create --dimension 3 --values "1.0,2.0,3.0"
    wdbx vector create --dimension 3 --values "4.0,5.0,6.0"
    
    # Search for vectors
    wdbx vector search --query "1.0,2.0,3.0" --k 5
    
    # Stop the server
    wdbx server stop

Exit Codes
---------

The CLI uses the following exit codes:

* 0: Success
* 1: General error
* 2: Configuration error
* 3: Connection error
* 4: Invalid input
* 5: Server error 