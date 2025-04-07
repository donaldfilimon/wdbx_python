Discord Bot Plugin
=================

The Discord Bot Plugin allows users to interact with and monitor the WDBX vector database through Discord.

Features
--------

- Basic commands for viewing WDBX status and health
- Vector search functionality
- Vector visualization with PCA
- Administrative commands for system management
- Batch import/export operations
- Real-time health monitoring and alerts

Installation
-----------

The Discord Bot Plugin requires additional dependencies:

.. code-block:: bash

    pip install discord.py matplotlib numpy scikit-learn

Configuration
------------

Create a configuration file (e.g., ``discord_bot.json``) with the following structure:

.. code-block:: json

    {
      "token": "YOUR_DISCORD_BOT_TOKEN",
      "prefix": "!",
      "status_channel_id": 123456789012345678,
      "admin_role_id": 123456789012345678,
      "wdbx_host": "127.0.0.1",
      "wdbx_port": 8080,
      "monitoring_interval": 300,
      "log_dir": "logs",
      "max_vectors_display": 10,
      "allow_vector_deletion": false
    }

Configuration Parameters:

- ``token``: Your Discord bot token (required)
- ``prefix``: Command prefix (default: ``!``)
- ``status_channel_id``: Channel ID for status updates and alerts
- ``admin_role_id``: Role ID that has access to administrative commands
- ``wdbx_host``: Host of the WDBX server (default: ``127.0.0.1``)
- ``wdbx_port``: Port of the WDBX server (default: ``8080``)
- ``monitoring_interval``: Health check interval in seconds (default: ``300``)
- ``log_dir``: Directory for log files (optional)
- ``max_vectors_display``: Maximum number of vectors to display in search results (default: ``10``)
- ``allow_vector_deletion``: Whether to allow vector deletion commands (default: ``false``)

The configuration file should be placed in one of the following locations:

- ``config/discord_bot.json``
- ``discord_bot.json``
- ``~/.config/wdbx/discord_bot.json``

Usage
-----

The Discord Bot Plugin can be used in two modes:

1. As a plugin within the WDBX ecosystem
2. As a standalone application

As a Plugin
~~~~~~~~~~

The plugin will be automatically loaded by WDBX if it's installed correctly:

.. code-block:: python

    from wdbx.client import WDBXClient
    from wdbx.plugins import load_plugin_module, get_plugin

    # Connect to WDBX
    client = WDBXClient()
    client.connect(host="127.0.0.1", port=8080)

    # Load the Discord bot plugin module
    load_plugin_module("wdbx.plugins.discord_bot")

    # Get the plugin
    discord_plugin = get_plugin("discord_bot")
    plugin_instance = discord_plugin()

    # Initialize the plugin with the WDBX client
    plugin_instance.initialize(client)

    # Shutdown when done
    plugin_instance.shutdown()

As a Standalone Application
~~~~~~~~~~~~~~~~~~~~~~~~~

You can run the Discord bot as a standalone application:

.. code-block:: python

    from wdbx.plugins.discord_bot import run_standalone

    run_standalone("path/to/discord_bot.json")

Or directly from the command line:

.. code-block:: bash

    python -m wdbx.plugins.discord_bot path/to/discord_bot.json

Available Commands
----------------

Basic Commands
~~~~~~~~~~~~~

- ``!status`` - Show WDBX status
- ``!health`` - Show detailed health information
- ``!search <query> [top_k]`` - Search for vectors
- ``!metrics`` - Show system metrics
- ``!visualize [query] [n_vectors]`` - Visualize vectors in 2D space
- ``!stats`` - Show vector statistics
- ``!help_wdbx`` - Show all available commands

Administrative Commands
~~~~~~~~~~~~~~~~~~~~~

The following commands require the Admin role:

- ``!admin status`` - Show detailed system status
- ``!admin clear`` - Clear all vectors and blocks (requires confirmation)
- ``!admin optimize`` - Optimize memory usage

Batch Operations
~~~~~~~~~~~~~~

The following commands require the Admin role:

- ``!batch import`` - Import vectors/blocks from a JSON file (attachment required)
- ``!batch export [all|vectors|blocks]`` - Export vectors and/or blocks to a JSON file

Real-time Health Monitoring
-------------------------

If a ``status_channel_id`` is configured, the bot will periodically check the health of the WDBX system and send alerts to the specified channel when issues are detected.

The monitoring interval can be configured using the ``monitoring_interval`` parameter (in seconds).

Vector Visualization
------------------

The ``!visualize`` command allows you to visualize vectors in 2D space using Principal Component Analysis (PCA).

This feature requires additional libraries:

- matplotlib
- numpy
- scikit-learn

Example:

- ``!visualize`` - Visualize a random selection of vectors
- ``!visualize query 50`` - Visualize 50 vectors similar to "query"

API Reference
-----------

.. py:class:: wdbx.plugins.discord_bot.DiscordBotPlugin

   Discord Bot Plugin for WDBX.

   .. py:method:: initialize(api)

      Initialize the plugin with the WDBX API instance.

      :param api: The WDBX API instance (WDBXClient)
      :return: True if initialization was successful, False otherwise

   .. py:method:: shutdown()

      Perform cleanup when the plugin is shutting down.

      :return: True if shutdown was successful, False otherwise

   .. py:property:: name
      :type: str

      Get the plugin's unique name.

   .. py:property:: description
      :type: str

      Get the plugin's description.

   .. py:property:: version
      :type: str

      Get the plugin's version.

Example
------

Here's a complete example of how to use the Discord Bot Plugin:

.. code-block:: python

    import json
    import asyncio
    from wdbx.client import WDBXClient
    from wdbx.plugins import load_plugin_module, get_plugin

    async def main():
        # Create configuration
        config = {
            "token": "YOUR_DISCORD_BOT_TOKEN",
            "prefix": "!",
            "status_channel_id": 123456789012345678,
            "admin_role_id": 123456789012345678,
            "wdbx_host": "127.0.0.1",
            "wdbx_port": 8080
        }

        # Save configuration
        with open("discord_bot.json", "w") as f:
            json.dump(config, f, indent=2)

        # Connect to WDBX
        client = WDBXClient()
        client.connect(host="127.0.0.1", port=8080)

        # Load the Discord bot plugin
        load_plugin_module("wdbx.plugins.discord_bot")
        discord_plugin = get_plugin("discord_bot")
        plugin = discord_plugin()

        # Initialize the plugin
        success = plugin.initialize(client)
        if not success:
            print("Failed to initialize Discord bot plugin")
            return

        # Keep the bot running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            # Shutdown the plugin
            plugin.shutdown()

    if __name__ == "__main__":
        asyncio.run(main())

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~

1. **Bot doesn't start**: Check that your Discord bot token is correct and the bot has the necessary permissions in Discord.

2. **Commands not working**: Ensure the bot has the necessary permissions in the Discord server and channel.

3. **Can't connect to WDBX**: Verify the host and port settings and ensure the WDBX server is running.

4. **Missing dependencies**: Ensure all required packages are installed, especially for visualization features.

5. **Health monitoring not working**: Check that the `status_channel_id` is correctly configured and the bot has permission to send messages in that channel.

Logging
~~~~~~

The Discord Bot Plugin uses structured logging to help with troubleshooting. Logs are written to:

- Standard output
- A log file in the directory specified by `log_dir` (if configured)

Set the log level to DEBUG for more detailed log messages:

.. code-block:: python

    import logging
    logging.getLogger("wdbx.discord").setLevel(logging.DEBUG) 