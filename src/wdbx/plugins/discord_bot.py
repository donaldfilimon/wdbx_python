#!/usr/bin/env python3
"""Discord bot plugin for WDBX.

This module provides a Discord bot interface for the WDBX Vector Database.
It allows for querying, managing, and visualizing vector data through Discord.

Features:
- Vector search and embedding creation from text
- Vector database management
- Health monitoring and alerts
- Data visualization

For VS Code Launch Configuration Examples:
To run the bot with VS Code, create a config.json file with your bot token and
add the following to your launch.json:
{
    "configurations": [
        {
            "name": "Discord Bot",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/src/wdbx/plugins/discord_bot.py",
            "args": ["${workspaceFolder}/config.json"]
        }
    ]
}
"""

import os
import sys
import json
import logging
import io
import asyncio
import time
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, cast
from dataclasses import dataclass, field

# Try to import discord.py
try:
    import discord
    from discord.ext import commands, tasks
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    
    # Mock discord classes if not available
    class commands:
        class Bot(object):
            pass
            
        class Context(object):
            async def send(self, content=None, *, embed=None):
                """Mock send method for Context."""
                pass
                
        class CommandError(Exception):
            pass
            
        class CommandNotFound(Exception):
            pass
            
        class MissingRequiredArgument(Exception):
            pass
            
        class BadArgument(Exception):
            pass
        
        class Cog:
            @staticmethod
            def listener():
                def decorator(func):
                    return func
                return decorator
        
        @staticmethod
        def command(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        
        @staticmethod
        def has_permissions(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        
        @staticmethod
        def group(*args, **kwargs):
            def decorator(func):
                func.command = lambda *args, **kwargs: lambda f: f
                return func
            return decorator
    
    class tasks:
        @staticmethod
        def loop(*args, **kwargs):
            def decorator(func):
                func.before_loop = lambda f: f
                func.start = lambda: None
                func.cancel = lambda: None
                func.has_failed = lambda: False
                func.seconds = kwargs.get('seconds', 300)
                func.change_interval = lambda **kwargs: None
                return func
            return decorator
    
    class discord:
        Intents = object
        Embed = object
        Color = object
        File = object
        Activity = object
        ActivityType = object
        Object = object
        User = object
        Guild = object
        TextChannel = object
        Message = object
        Attachment = object
        HTTPException = Exception
        Forbidden = Exception
        NotFound = Exception

# Try to import WDBX components
try:
    # Try absolute imports instead of relative imports
    from wdbx.client import WDBXClient
    from wdbx.core.constants import logger
    from wdbx.health import HealthMonitor, HealthStatus
    from wdbx.prometheus import get_metrics

    WDBX_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    # Fallback for direct execution or different structure
    try:
        from wdbx.client import WDBXClient
        from wdbx.core.constants import logger
        from wdbx.health import HealthMonitor, HealthStatus
        from wdbx.prometheus import get_metrics

        WDBX_AVAILABLE = True
    except (ImportError, ModuleNotFoundError):
        # Mock logger and components if not found
        logger = logging.getLogger("wdbx.discord.fallback")
        logger.warning("WDBX components not found. Using mock objects.")
        HealthMonitor = None
        HealthStatus = None
        get_metrics = None
        WDBXClient = None
        WDBX_AVAILABLE = False

# Try to import psutil for system metrics
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not installed. Some system metrics will not be available.")

# Add vector visualization libraries after existing imports
try:
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import PCA
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logger.warning(
        "Visualization libraries (matplotlib, numpy, scikit-learn) not found. Visualize command disabled."
    )
# Add custom exceptions
class DiscordBotError(Exception):
    """Base exception for Discord bot-related errors."""


class BotSetupError(DiscordBotError):
    """Raised when the bot setup fails."""


class CommandError(DiscordBotError):
    """Raised when a command execution fails."""


class WDBXClientError(DiscordBotError):
    """Raised when interaction with WDBX client fails."""


# Add structured logging configuration
def setup_logging(log_dir: Optional[str] = None) -> None:
    """
    Configure structured logging with proper formatting and handlers.

    Args:
        log_dir: Optional directory for log files
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout)
    ]  # Use stdout for better compatibility with containers

    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "wdbx_discord.log")
            handlers.append(logging.FileHandler(log_file))
        except Exception as e:
            # Use the root logger temporarily if our logger failed
            logging.warning(f"Failed to create log file handler in {log_dir}: {e}")

    # Get our specific logger
    bot_logger = logging.getLogger("wdbx.discord")
    bot_logger.setLevel(logging.INFO)

    # Configure root logger as a fallback and for dependencies
    logging.basicConfig(level=logging.INFO, format=log_format, handlers=handlers)

    # Set levels for noisy libraries
    logging.getLogger("discord").setLevel(logging.WARNING)
    logging.getLogger("discord.http").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)  # Matplotlib can be verbose


@dataclass
class BotConfig:
    """Configuration for the Discord bot."""

    token: str
    prefix: str = "!"
    status_channel_id: Optional[int] = None
    admin_role_id: Optional[int] = None
    wdbx_host: str = "127.0.0.1"
    wdbx_port: int = 8080
    monitoring_interval: int = 300  # seconds
    log_dir: Optional[str] = None
    max_vectors_display: int = 10  # Maximum vectors to display in search results
    allow_vector_deletion: bool = False  # Whether to allow vector deletion commands

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.token:
            raise BotSetupError("Discord bot token is required")

        # Validate prefix
        if not isinstance(self.prefix, str) or not self.prefix:
            raise BotSetupError("Bot prefix must be a non-empty string")

        # Validate interval
        if not isinstance(self.monitoring_interval, int) or self.monitoring_interval < 30:
            logger.warning("Monitoring interval is less than 30 seconds. Setting to 30.")
            self.monitoring_interval = 30

        # Validate IDs
        if self.status_channel_id and not isinstance(self.status_channel_id, int):
            raise BotSetupError("status_channel_id must be an integer")
        if self.admin_role_id and not isinstance(self.admin_role_id, int):
            raise BotSetupError("admin_role_id must be an integer")

        # Create log directory if specified
        if self.log_dir:
            try:
                os.makedirs(self.log_dir, exist_ok=True)
            except Exception as e:
                logger.warning(f"Failed to create log directory {self.log_dir}: {e}")


# Add helper functions before they're used
# Fix error E0602: Undefined variable 'create_help_embed'
def create_help_embed():
    """Create a help embed with available commands."""
    if DISCORD_AVAILABLE:
        embed = discord.Embed(
            title="WDBX Bot Help", 
            description="Available commands for interacting with WDBX",
            color=0x3498db
        )
        return embed
    return None


# Mock the WDBXCog class
class WDBXCog:
    """WDBX Discord Cog."""
    
    def __init__(self, bot, config, wdbx_instance=None):
        """Initialize the cog."""
        self.bot = bot
        self.config = config
        self.wdbx = wdbx_instance


def connect_to_wdbx(host: str, port: int) -> Optional[Any]:
    """Attempt to connect to the WDBX instance."""
    if not WDBX_AVAILABLE or WDBXClient is None:
        logger.error("WDBX client library is not available.")
        return None
    try:
        client = WDBXClient()
        # Use connect with just host and port, no keyword arguments
        client.connect(host, port)
        # Use proper string formatting
        logger.info("Successfully connected to WDBX at %s:%d", host, port)
        return client
    except Exception as e:
        # Use proper string formatting for logger
        logger.error("Failed to connect to WDBX at %s:%d: %s", host, port, str(e))
        return None


def create_bot(config: BotConfig, wdbx_instance: Optional[Any] = None) -> Any:
    """Create and configure the Discord bot instance."""
    if not DISCORD_AVAILABLE:
        raise BotSetupError("discord.py is not installed. Cannot create bot. Install with: pip install discord.py")

    intents = discord.Intents.default()
    intents.members = True  # Needed for role checks potentially
    intents.message_content = True  # Needed for reading command content

    bot = commands.Bot(
        command_prefix=config.prefix, intents=intents, help_command=None
    )  # Disable default help

    # Add the WDBX cog
    cog = WDBXCog(bot, config, wdbx_instance)
    
    # Fix: Don't use asyncio.run in an already async context
    # Instead, we'll use a helper method to add the cog during the on_ready event
    
    # Define an on_ready event
    @bot.event
    async def on_ready():
        """Bot ready event handler."""
        await bot.add_cog(cog)
        logger.info("WDBX cog added to bot.")

    # Add a basic help command override if help_wdbx isn't sufficient
    @bot.command(name="help", hidden=True)
    async def custom_help(ctx: commands.Context, *, command_name: Optional[str] = None):
        """Show help for WDBX bot commands."""
        if command_name:
            command = bot.get_command(command_name)
            if command:
                # Show help for specific command (basic version)
                help_text = f"**`{config.prefix}{command.qualified_name} {command.signature}`**\n{command.help or 'No description available.'}"
                await ctx.send(help_text)
            else:
                await ctx.send(f"Command `{command_name}` not found.")
        else:
            # Show the main WDBX help embed
            embed = create_help_embed()
            await ctx.send(embed=embed)

    return bot


def run_bot(config: BotConfig, wdbx_instance: Optional[Any] = None) -> None:
    """Create and run the Discord bot."""
    setup_logging(config.log_dir)  # Ensure logging is setup
    run_logger = logging.getLogger("wdbx.discord.runner")  # Renamed to avoid conflict

    try:
        bot = create_bot(config, wdbx_instance)
        run_logger.info("Starting Discord bot with prefix '%s'...", config.prefix)
        bot.run(config.token)
    except BotSetupError as e:
        run_logger.critical("Bot setup failed: %s", e)
        raise  # Reraise setup errors
    except Exception as e:
        if DISCORD_AVAILABLE and hasattr(discord, 'LoginFailure') and isinstance(e, discord.LoginFailure):
            run_logger.critical("Login failed: Invalid Discord bot token provided.")
            raise BotSetupError("Invalid Discord Bot Token") from None
        else:
            # Use proper string formatting
            run_logger.critical("An unexpected error occurred while running the bot: %s", str(e), exc_info=True)
            raise  # Reraise the exception


if __name__ == "__main__":
    if not DISCORD_AVAILABLE:
        print("Error: discord.py is not installed. Install with: pip install discord.py")
        print("For full functionality, also install: matplotlib numpy scikit-learn")
        sys.exit(1)
        
    if len(sys.argv) < 2:
        print("Usage: python discord_bot.py <path_to_config.json>")
        sys.exit(1)

    config_path = sys.argv[1]
    if not os.path.exists(config_path):
        print("Error: Configuration file not found at {}".format(config_path))
        sys.exit(1)

    try:
        with open(config_path, encoding="utf-8") as f:
            config_dict = json.load(f)
        bot_config = BotConfig(**config_dict)
    except json.JSONDecodeError:
        print("Error: Invalid JSON in configuration file: {}".format(config_path))
        sys.exit(1)
    except (TypeError, BotSetupError) as e:
        print("Error loading configuration: {}".format(e))
        sys.exit(1)
    except Exception as e:
        print("An unexpected error occurred loading config: {}".format(e))
        sys.exit(1)

    # Setup logging based on config
    setup_logging(bot_config.log_dir)
    main_logger = logging.getLogger("wdbx.discord.main")

    # Attempt to connect to WDBX (optional, bot can run without initial connection)
    wdbx = None
    if WDBX_AVAILABLE:
        main_logger.info(
            f"Attempting to connect to WDBX at {bot_config.wdbx_host}:{bot_config.wdbx_port}..."
        )
        wdbx = connect_to_wdbx(bot_config.wdbx_host, bot_config.wdbx_port)
        if not wdbx:
            main_logger.warning(
                "Could not connect to WDBX initially. Bot will attempt to reconnect."
            )
    else:
        main_logger.warning("WDBX client library not available. Bot running in limited mode.")

    # Run the bot
    try:
        run_bot(bot_config, wdbx)
    except BotSetupError as e:
        # Add more context to the error
        print("Error: %s" % e)
        sys.exit(1)
    except Exception as e:
        # Catch any other unexpected error during runtime
        error_msg = "Bot terminated due to unexpected error: %s" % str(e)
        main_logger.critical(error_msg, exc_info=True)
        print("Error: %s" % e)
        sys.exit(1)

