"""
Discord Bot Plugin Example

This example demonstrates how to configure and use the Discord bot plugin
for WDBX, both as a plugin within the WDBX ecosystem and as a standalone application.
"""

import os
import json
import logging
import asyncio
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("wdbx.examples.discord_bot")

# Try importing required components
try:
    from wdbx.client import WDBXClient
    from wdbx.plugins import load_plugin_module, get_plugin
except ImportError:
    logger.error("Failed to import WDBX modules. Make sure WDBX is installed.")
    raise


def create_config_file(output_path="discord_bot.json"):
    """
    Create a template configuration file for the Discord bot.
    
    Args:
        output_path: Path to save the configuration file.
    
    Returns:
        bool: True if the file was created successfully, False otherwise.
    """
    config = {
        "token": "YOUR_DISCORD_BOT_TOKEN",
        "prefix": "!",
        "status_channel_id": 123456789012345678,
        "admin_role_id": 123456789012345678,
        "wdbx_host": "127.0.0.1",
        "wdbx_port": 8080,
        "monitoring_interval": 300,
        "log_dir": "logs",
        "max_vectors_display": 10,
        "allow_vector_deletion": False
    }
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Write config file
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Created template configuration file at {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create configuration file: {e}")
        return False


async def run_as_plugin():
    """Run the Discord bot as a plugin within WDBX."""
    logger.info("Running Discord bot as a plugin")
    
    # Connect to WDBX
    client = WDBXClient()
    client.connect(host="127.0.0.1", port=8080)
    logger.info("Connected to WDBX")
    
    # Load the Discord bot plugin module
    load_plugin_module("wdbx.plugins.discord_bot")
    
    # Get the plugin
    discord_plugin = get_plugin("discord_bot")
    if not discord_plugin:
        logger.error("Discord bot plugin not found")
        return
    
    logger.info(f"Found Discord bot plugin: {discord_plugin.name} v{discord_plugin.version}")
    logger.info(f"Description: {discord_plugin.description}")
    
    # Create an instance
    plugin_instance = discord_plugin()
    
    # Initialize the plugin with the WDBX client
    success = plugin_instance.initialize(client)
    if not success:
        logger.error("Failed to initialize Discord bot plugin")
        return
    
    logger.info("Discord bot plugin initialized successfully")
    
    try:
        # Keep the bot running for demonstration
        logger.info("Discord bot is running. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping Discord bot plugin...")
    finally:
        # Shutdown the plugin
        plugin_instance.shutdown()
        logger.info("Discord bot plugin shutdown")


def run_standalone():
    """Run the Discord bot as a standalone application."""
    logger.info("Running Discord bot in standalone mode")
    
    # Ensure configuration file exists
    config_path = "discord_bot.json"
    if not os.path.exists(config_path):
        logger.info("Configuration file not found. Creating template...")
        create_config_file(config_path)
        logger.info(f"Please edit {config_path} with your Discord bot token and other settings.")
        logger.info("Then run this example again.")
        return
    
    # Import the standalone runner
    try:
        from wdbx.plugins.discord_bot import run_standalone
        
        logger.info(f"Starting Discord bot with configuration from {config_path}")
        run_standalone(config_path)
    except ImportError:
        logger.error("Failed to import Discord bot module. Make sure it's installed.")
        return
    except Exception as e:
        logger.error(f"Error running Discord bot: {e}")
        return


def main():
    """Main function to demonstrate both plugin and standalone modes."""
    print("WDBX Discord Bot Plugin Example")
    print("===============================")
    print("1. Run as a plugin within WDBX")
    print("2. Run as a standalone application")
    print("3. Create template configuration file")
    print("4. Exit")
    
    choice = input("Enter your choice (1-4): ")
    
    if choice == "1":
        # Run as a plugin
        asyncio.run(run_as_plugin())
    elif choice == "2":
        # Run as a standalone application
        run_standalone()
    elif choice == "3":
        # Create template configuration
        output_path = input("Enter path for configuration file (default: discord_bot.json): ")
        if not output_path:
            output_path = "discord_bot.json"
        create_config_file(output_path)
        print(f"Template configuration file created at {output_path}")
        print("Edit this file with your Discord bot token and other settings.")
    elif choice == "4":
        print("Exiting...")
    else:
        print("Invalid choice. Exiting...")


if __name__ == "__main__":
    main() 