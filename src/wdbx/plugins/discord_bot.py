""
Discord Bot Plugin for WDBX.

This module provides a Discord bot that allows users to interact with
and monitor the WDBX vector database through Discord.
""

import asyncio
import datetime
import io
import json
import logging
import os
import platform
import sys
import time
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

try:




except Exception as e:


    logger.error(f"Error: {e}", exc_info=True)
    except Exception as e:


except Exception as e:


    logger.error(f"Error: {e}", exc_info=True)


    await ctx.send(f"❌ An error occurred: {str(e)}")


    logger.error(f"Error: {e}", exc_info=True)

except Exception as e:

    logger.error(f"Error: {e}", exc_info=True)

    await ctx.send(f"❌ An error occurred: {str(e)}")


except Exception as e:

except Exception as e:

    logger.error(f"Error: {e}", exc_info=True)

    await ctx.send(f"❌ An error occurred: {str(e)}")


    logger.error(f"Error: {e}", exc_info=True)

except Exception as e:

    logger.error(f"Error: {e}", exc_info=True)
    import discord
    from discord.ext import commands, tasks
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    # Mock discord classes for validation
    class commands...:
        ""Class commands.""
        Bot = object
        Cog = object
        Context = object
        CommandError = Exception
        CommandNotFound = Exception
        MissingRequiredArgument = Exception
        BadArgument = Exception
        def command(...):
            ""Function command.""
            def decorator(func): return func
            return decorator
        def has_permissions(...):
            ""Function has_permissions.""
            def decorator(func): return func
            return decorator
    class tasks...:
        ""Class tasks.""
        def loop(...):
            ""Function loop.""
            def decorator(func): return func
            return decorator
    class discord...:
        ""Class discord.""
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

from wdbx.core.constants import logger
from wdbx.health import HealthMonitor, HealthStatus
from wdbx.prometheus import get_metrics
from wdbx.client import WDBXClient
from wdbx.plugins import WDBXPlugin, register_plugin

# Add vector visualization libraries after existing imports
try...:

    except Exception as e:

    logger.error(f"Error: {e}", exc_info=True)
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import PCA
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logger.warning("Visualization libraries (matplotlib, numpy, scikit-learn) not found. Visualize command disabled.")

# Add custom exceptions
class DiscordBotError(Exception):
    ""Base exception for Discord bot-related errors."
    pass

class BotSetupError(DiscordBotError):
    ""Raised when the bot setup fails."
    pass

class CommandError(DiscordBotError):
    ""Raised when a command execution fails."
    pass

class WDBXClientError(DiscordBotError):
    ""Raised when interaction with WDBX client fails."
    pass

# Add structured logging configuration
def setup_logging(log_dir: Optional[str] = None) -> None:
    ""
    Configure structured logging with proper formatting and handlers.

    Args:
        log_dir: Optional directory for log files
    ""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]  # Use stdout for better compatibility with containers

    if log_dir:
        try:


        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)
            except Exception as e:

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")

            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "wdbx_discord.log")
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            handlers.append(logging.FileHandler(log_file))
        except Exception as e:
            # Use the root logger temporarily if our logger failed
            logging.warning(f"Failed to create log file handler in {log_dir}: {e}")

    # Get our specific logger
    bot_logger = logging.getLogger("wdbx.discord")
    bot_logger.setLevel(logging.INFO)

    # Configure root logger as a fallback and for dependencies
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers
    )

    # Set levels for noisy libraries
    logging.getLogger("discord").setLevel(logging.WARNING)
    logging.getLogger("discord.http").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)  # Matplotlib can be verbose


@dataclass
class BotConfig:
    ""Configuration for the Discord bot."
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
        ""Validate configuration after initialization."
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


            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)
                except Exception as e:

            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)

                await ctx.send(f"❌ An error occurred: {str(e)}")

            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)

                await ctx.send(f"❌ An error occurred: {str(e)}")

            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)

                await ctx.send(f"❌ An error occurred: {str(e)}")

            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)

                await ctx.send(f"❌ An error occurred: {str(e)}")

                logger.error(f"Error: {e}", exc_info=True)

            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)

            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)

            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)

            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)
                os.makedirs(self.log_dir, exist_ok=True)
            except Exception as e:
                logger.warning(f"Failed to create log directory {self.log_dir}: {e}")
           except Exception as e:
               logger.error(f"Error: {e}", exc_info=True)


def connect_to_wdbx(host: str, port: int) -> Optional[WDBXClient]:
    ""Attempt to connect to the WDBX instance."
    try:


    except Exception as e:

        logger.error(f"Error: {e}", exc_info=True)
        except Exception as e:

    except Exception as e:

        logger.error(f"Error: {e}", exc_info=True)

        await ctx.send(f"❌ An error occurred: {str(e)}")

        logger.error(f"Error: {e}", exc_info=True)

    except Exception as e:

        logger.error(f"Error: {e}", exc_info=True)
        client = WDBXClient()
        client.connect(host=host, port=port)
        logger.info(f"Successfully connected to WDBX at {host}:{port}")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return client
    except Exception as e:
        logger.error(f"Failed to connect to WDBX at {host}:{port}: {e}", exc_info=True)
        return None


def format_health_status(status: Dict[str, Dict]) -> str:
    ""Format the health status dictionary into a readable string."
    lines = ["**WDBX Health Status:**"]
    overall_status = status.get("overall_status", {}).get("status", "UNKNOWN")
    lines.append(f"Overall Status: {overall_status}")
    lines.append("\n**Component Status:**")
    for check_name, check_info in status.items():
        if check_name != "overall_status":
            check_status = check_info.get("status", "UNKNOWN")
            message = check_info.get("message", "No details available.")
            lines.append(f"- `{check_name}`: {check_status} ({message})")
    return "\n".join(lines)


def create_help_embed() -> discord.Embed:
    ""Create an embed message for the help command."
    embed = discord.Embed(
        title="WDBX Discord Bot Help",
        description="Available commands for interacting with WDBX:",
        color=discord.Color.blue()
    )
    embed.add_field(name="`!status`", value="Show WDBX status information.", inline=False)
    embed.add_field(name="`!health`", value="Show detailed WDBX health information.", inline=False)
    embed.add_field(name="`!metrics`", value="Show current system metrics (Prometheus format).", inline=False)
    embed.add_field(name="`!search <query> [top_k=5]`", value="Search for similar vectors.", inline=False)
    embed.add_field(name="`!visualize [query] [n_vectors=20]`", value="Visualize vectors in 2D space (requires libraries).", inline=False)
    embed.add_field(name="`!stats`", value="Show vector database statistics.", inline=False)
    embed.add_field(name="`!admin <action> [args]`", value="Perform administrative actions (requires Admin role).\nActions: `status`, `clear`, `optimize`", inline=False)
    embed.add_field(name="`!batch <operation> [args]`", value="Perform batch operations (import/export, requires Admin role).\nOperations: `import`, `export`", inline=False)
    embed.add_field(name="`!config`", value="Generate a template configuration file.", inline=False)
    embed.add_field(name="`!help_wdbx`", value="Show this help message.", inline=False)
    embed.set_footer(text="WDBX Bot | Use the prefix defined in config (default: !)")
    return embed


class WDBXCog(commands.Cog):
    ""Cog containing commands for interacting with WDBX."

    def __init__(...):
        ""Function __init__.""
        if not DISCORD_AVAILABLE:
            raise BotSetupError("discord.py is not installed. Cannot initialize WDBXCog.")

        self.bot = bot
        self.config = config
        self.wdbx = wdbx  # This might be None initially
        self.health_monitor = None  # Initialize later if WDBX available
        self.logger = logging.getLogger("wdbx.discord.cog")

        # Try to initialize health monitor if WDBX is available
        if self.wdbx:
            try:


            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)
                except Exception as e:

            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)

                await ctx.send(f"❌ An error occurred: {str(e)}")

            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)

                await ctx.send(f"❌ An error occurred: {str(e)}")

            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)

                await ctx.send(f"❌ An error occurred: {str(e)}")

            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)

                await ctx.send(f"❌ An error occurred: {str(e)}")

                logger.error(f"Error: {e}", exc_info=True)

            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)

            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)

            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)

            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)
                self.health_monitor = HealthMonitor(self.wdbx)  # Assuming HealthMonitor takes client
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                self.logger.info("Health monitor initialized.")
            except Exception as e:
                self.logger.error(f"Failed to initialize Health Monitor: {e}", exc_info=True)
                self.health_monitor = None  # Ensure it's None if init fails

        # Start background tasks
        if self.config.status_channel_id:
            self.health_check_loop.start()
            self.logger.info("Health check loop started.")
        else:
            self.logger.warning("status_channel_id not configured. Health check loop not started.")

        self.last_health_status = HealthStatus.UNKNOWN if HealthStatus else "UNKNOWN"

    def cog_unload(self):
        ""Clean up tasks when the cog is unloaded."
        self.health_check_loop.cancel()
        self.logger.info("Health check loop cancelled.")
        if self.wdbx:
            try:


            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)
                except Exception as e:

            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)

                await ctx.send(f"❌ An error occurred: {str(e)}")

            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)

                await ctx.send(f"❌ An error occurred: {str(e)}")

            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)

                await ctx.send(f"❌ An error occurred: {str(e)}")

            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)

                await ctx.send(f"❌ An error occurred: {str(e)}")

                logger.error(f"Error: {e}", exc_info=True)

            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)

            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)

            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)

            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)
                self.wdbx.disconnect()
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                self.logger.info("Disconnected from WDBX.")
            except Exception as e:
                self.logger.error(f"Error disconnecting from WDBX: {e}", exc_info=True)

    async def _ensure_wdbx_connection(self, ctx: commands.Context) -> bool:
        ""Check if connected to WDBX and attempt reconnect if not."
        if self.wdbx and hasattr(self.wdbx, "is_connected") and self.wdbx.is_connected():
            return True

        await ctx.send("⏳ Attempting to reconnect to WDBX...")
        self.wdbx = connect_to_wdbx(self.config.wdbx_host, self.config.wdbx_port)

        if self.wdbx:
            # Re-initialize health monitor if connection is re-established
            if not self.health_monitor:
                try:


                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)
                    except Exception as e:

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                    await ctx.send(f"❌ An error occurred: {str(e)}")

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                    await ctx.send(f"❌ An error occurred: {str(e)}")

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                    await ctx.send(f"❌ An error occurred: {str(e)}")

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                    await ctx.send(f"❌ An error occurred: {str(e)}")

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                    await ctx.send(f"❌ An error occurred: {str(e)}")

                    logger.error(f"Error: {e}", exc_info=True)

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)
                    self.health_monitor = HealthMonitor(self.wdbx)
                except Exception as e:
                    logger.error(f"Error: {e}", exc_info=True)
                    self.logger.info("Health monitor re-initialized after reconnect.")
                except Exception as e:
                    self.logger.error(f"Failed to re-initialize Health Monitor: {e}", exc_info=True)
            await ctx.send("✅ Reconnected to WDBX successfully.")
            return True
        else:
            await ctx.send("❌ Failed to connect to WDBX. Please check the WDBX server and configuration.")
            return False

    async def _create_embed(self, ctx: commands.Context, title: str, description: str = "", color: discord.Color = discord.Color.blue()) -> discord.Embed:
        ""Helper function to create a standard embed."
        embed = discord.Embed(title=title, description=description, color=color)
        if hasattr(ctx.author, "display_name"):
            embed.set_footer(text=f"Requested by {ctx.author.display_name}", 
                            icon_url=ctx.author.avatar_url if hasattr(ctx.author, "avatar_url") and ctx.author.avatar_url else None)
        embed.timestamp = datetime.datetime.utcnow()
        return embed

    @commands.command(name="status", help="Show WDBX status information.help=help=")
    async def status(self, ctx: commands.Context):
        "Show WDBX status information."
        if not await self._ensure_wdbx_connection(ctx):
            return

        try:




        except Exception as e:


            logger.error(f"Error: {e}", exc_info=True)
            except Exception as e:


        except Exception as e:


            logger.error(f"Error: {e}", exc_info=True)


            await ctx.send(f"❌ An error occurred: {str(e)}")


        except Exception as e:


            logger.error(f"Error: {e}", exc_info=True)


            await ctx.send(f"❌ An error occurred: {str(e)}")


        except Exception as e:


            logger.error(f"Error: {e}", exc_info=True)


            await ctx.send(f"❌ An error occurred: {str(e)}")


            logger.error(f"Error: {e}", exc_info=True)


        except Exception as e:


            logger.error(f"Error: {e}", exc_info=True)


        except Exception as e:


            logger.error(f"Error: {e}", exc_info=True)


        except Exception as e:


            logger.error(f"Error: {e}", exc_info=True)
            stats = await self.wdbx.get_stats()  # Assuming async get_stats
            embed = await self._create_embed(ctx, "WDBX Status", color=discord.Color.green())
            embed.add_field(name="Connection", value=f"Connected to {self.config.wdbx_host}:{self.config.wdbx_port}", inline=False)
            embed.add_field(name="Total Vectors", value=str(stats.get('total_vectors', 'N/A')), inline=True)
            embed.add_field(name="Total Blocks", value=str(stats.get('total_blocks', 'N/A')), inline=True)
            embed.add_field(name="Memory Usage", value=stats.get('memory_usage_mb', 'N/A'), inline=True)
            # Add more stats as needed
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            await ctx.send(f"❌ An error occurred: {str(e)}")
            await ctx.send(embed=embed)
        except Exception as e:
            self.logger.error(f"Error getting WDBX status: {e}", exc_info=True)
            await ctx.send(f"❌ Error fetching status: {str(e)}")

    @commands.command(name="health", help="Show detailed WDBX health information.help=help=")
    async def health(self, ctx: commands.Context):
        "Show detailed health information."
        if not await self._ensure_wdbx_connection(ctx):
            return
        if not self.health_monitor:
            await ctx.send("❌ Health monitoring is not available or not initialized.")
            return

        try:




        except Exception as e:


            logger.error(f"Error: {e}", exc_info=True)
            except Exception as e:


        except Exception as e:


            logger.error(f"Error: {e}", exc_info=True)


            await ctx.send(f"❌ An error occurred: {str(e)}")


        except Exception as e:


            logger.error(f"Error: {e}", exc_info=True)


            await ctx.send(f"❌ An error occurred: {str(e)}")


        except Exception as e:


            logger.error(f"Error: {e}", exc_info=True)


            await ctx.send(f"❌ An error occurred: {str(e)}")


            logger.error(f"Error: {e}", exc_info=True)


        except Exception as e:


            logger.error(f"Error: {e}", exc_info=True)


        except Exception as e:


            logger.error(f"Error: {e}", exc_info=True)


        except Exception as e:


            logger.error(f"Error: {e}", exc_info=True)
            health_status = await self.health_monitor.check_health()
            # Format the health status into a readable message
            health_message = format_health_status(health_status)
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            await ctx.send(health_message)
        except Exception as e:
            self.logger.error(f"Error checking health: {e}", exc_info=True)
            await ctx.send(f"❌ Error checking health: {str(e)}")

    @commands.command(name="search", help="Search for similar vectors.help=help=")
    async def search(self, ctx: commands.Context, *, query: str = None, top_k: int = 5):
        "Search for similar vectors."
        if not await self._ensure_wdbx_connection(ctx):
            return
        if not query:
            await ctx.send("❌ Please provide a search query.")
            return

        try:




        except Exception as e:


            logger.error(f"Error: {e}", exc_info=True)
            except Exception as e:


        except Exception as e:


            logger.error(f"Error: {e}", exc_info=True)


            await ctx.send(f"❌ An error occurred: {str(e)}")


        except Exception as e:


            logger.error(f"Error: {e}", exc_info=True)


            await ctx.send(f"❌ An error occurred: {str(e)}")


        except Exception as e:


            logger.error(f"Error: {e}", exc_info=True)


            await ctx.send(f"❌ An error occurred: {str(e)}")


            logger.error(f"Error: {e}", exc_info=True)


        except Exception as e:


            logger.error(f"Error: {e}", exc_info=True)


        except Exception as e:


            logger.error(f"Error: {e}", exc_info=True)


        except Exception as e:


            logger.error(f"Error: {e}", exc_info=True)
            # Limit top_k to reasonable values
            top_k = min(max(1, top_k), self.config.max_vectors_display)
            
            await ctx.send(f"🔍 Searching for vectors similar to: `{query}`...")
            results = await self.wdbx.find_similar_vectors(query=query, top_k=top_k)
            
            if not results or len(results) == 0:
                await ctx.send(f"⚠️ No results found for query: `{query}`")
                return
                
            # Format results
            embed = await self._create_embed(ctx, f"Search Results for '{query}'")
            for i, result in enumerate(results):
                vector_id = result.get("id", f"unknown_{i}")
                similarity = result.get("similarity", 0)
                metadata = result.get("metadata", {})
                
                # Format metadata for display
                metadata_str = ", ".join([f"{k}: {v}" for k, v in metadata.items()])
                if not metadata_str:
                    metadata_str = "No metadata"
                    
                embed.add_field(
                    name=f"{i+1}. Vector {vector_id} (Similarity: {similarity:.4f})",
                    value=metadata_str,
                    inline=False
                )
                
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            await ctx.send(f"❌ An error occurred: {str(e)}")
            await ctx.send(embed=embed)
        except Exception as e:
            self.logger.error(f"Error searching vectors: {e}", exc_info=True)
            await ctx.send(f"❌ Error searching vectors: {str(e)}")

    @commands.command(name="metrics", help="Show current system metrics (Prometheus format).help=help=")
    async def metrics(self, ctx: commands.Context):
        "Show current metrics in Prometheus format."
        if not await self._ensure_wdbx_connection(ctx):
            return
        try:


        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)
            except Exception as e:

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")

            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)
            metrics = get_metrics()
            if not metrics:
                await ctx.send("⚠️ No metrics available.")
                return
                
            # Format metrics for display
            # If too large, save to file and send as attachment
            if len(metrics) > 1900:  # Discord message limit approach
                with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as f:
                    f.write(metrics)
                    metrics_file = f.name
                
                await ctx.send("📊 Metrics are too large to display. Attached as file:", 
                              file=discord.File(metrics_file, filename="wdbx_metrics.txt"))
                
                # Clean up temp file
                try:


                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)
                    except Exception as e:

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                    await ctx.send(f"❌ An error occurred: {str(e)}")

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                    await ctx.send(f"❌ An error occurred: {str(e)}")

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                    await ctx.send(f"❌ An error occurred: {str(e)}")

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                    await ctx.send(f"❌ An error occurred: {str(e)}")

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                    await ctx.send(f"❌ An error occurred: {str(e)}")

                    logger.error(f"Error: {e}", exc_info=True)

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)
                    os.unlink(metrics_file)
                except Exception:
                    pass
            else:
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            await ctx.send(f"❌ An error occurred: {str(e)}")
                await ctx.send(f"```\n{metrics}\n```")
        except Exception as e:
            self.logger.error(f"Error getting metrics: {e}", exc_info=True)
            await ctx.send(f"❌ Error getting metrics: {str(e)}")

    @commands.command(name="visualize", help="Visualize vectors in 2D space.help=help=")
    async def visualize(self, ctx: commands.Context, query: str = None, n_vectors: int = 20):
        "Visualize vectors in 2D space using PCA."
        if not await self._ensure_wdbx_connection(ctx):
            return
        if not VISUALIZATION_AVAILABLE:
            await ctx.send("❌ Visualization libraries (matplotlib, numpy, scikit-learn) are not available.")
            return
            
        try:

            

            
        except Exception as e:

            
            logger.error(f"Error: {e}", exc_info=True)
            except Exception as e:

            
        except Exception as e:

            
            logger.error(f"Error: {e}", exc_info=True)

            
            await ctx.send(f"❌ An error occurred: {str(e)}")

            
        except Exception as e:

            
            logger.error(f"Error: {e}", exc_info=True)

            
            await ctx.send(f"❌ An error occurred: {str(e)}")

            
        except Exception as e:

            
            logger.error(f"Error: {e}", exc_info=True)

            
            await ctx.send(f"❌ An error occurred: {str(e)}")

            
            logger.error(f"Error: {e}", exc_info=True)

            
        except Exception as e:

            
            logger.error(f"Error: {e}", exc_info=True)

            
        except Exception as e:

            
            logger.error(f"Error: {e}", exc_info=True)

            
        except Exception as e:

            

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")

            await ctx.send(f"❌ An error occurred: {str(e)}")

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")
            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)
            # Limit number of vectors
            n_vectors = min(max(5, n_vectors), 100)  # At least 5, at most 100
            
            await ctx.send(f"🔍 Fetching vectors{' for query: `'+query+'`' if query else '}...")
            
            # If query provided, get similar vectors
            if query:
                vectors = await self.wdbx.find_similar_vectors(query=query, top_k=n_vectors)
            else:
                # Otherwise get random vectors
                stats = await self.wdbx.get_stats()
                total_vectors = stats.get('total_vectors', 0)
                if total_vectors == 0:
                    await ctx.send("⚠️ No vectors available to visualize.")
                    return
                    
                # Get vectors (implementation would depend on client API)
                vectors = await self.wdbx.get_random_vectors(n=n_vectors)
                
            if not vectors or len(vectors) < 2:
                await ctx.send("⚠️ Not enough vectors available for visualization (need at least 2).")
                return
                
            # Extract embeddings and metadata
            embeddings = [v.get("embedding", []) for v in vectors]
            labels = [v.get("id", f"Vector {i}") for i, v in enumerate(vectors)]
            
            # Apply PCA to reduce to 2D
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(embeddings)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c='blue', alpha=0.7)
            
            # Add labels
            for i, label in enumerate(labels):
                plt.annotate(label, (reduced_data[i, 0], reduced_data[i, 1]), 
                            fontsize=9, alpha=0.8)
                            
            # Add title and save
            title = f"Vector Visualization - {len(vectors)} vectors"
            if query:
                title += f" for query '{query}'"
            plt.title(title)
            plt.tight_layout()
            
            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            
            # Send image
            await ctx.send(file=discord.File(buf, filename="vector_visualization.png"))
            
        except Exception as e:
            self.logger.error(f"Error visualizing vectors: {e}", exc_info=True)
            await ctx.send(f"❌ Error visualizing vectors: {str(e)}")
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            await ctx.send(f"❌ An error occurred: {str(e)}")

    @commands.command(name="stats", help="Show vector database statistics.help=help=")
    async def stats(self, ctx: commands.Context):
        "Show statistics about the vector database."
        if not await self._ensure_wdbx_connection(ctx):
            return
            
        try:

            

            
        except Exception as e:

            
            logger.error(f"Error: {e}", exc_info=True)
            except Exception as e:

            
        except Exception as e:

            
            logger.error(f"Error: {e}", exc_info=True)

            
            await ctx.send(f"❌ An error occurred: {str(e)}")

            
        except Exception as e:

            
            logger.error(f"Error: {e}", exc_info=True)

            
            await ctx.send(f"❌ An error occurred: {str(e)}")

            
        except Exception as e:

            
            logger.error(f"Error: {e}", exc_info=True)

            
            await ctx.send(f"❌ An error occurred: {str(e)}")

            
            logger.error(f"Error: {e}", exc_info=True)

            
        except Exception as e:

            
            logger.error(f"Error: {e}", exc_info=True)

            
        except Exception as e:

            
            logger.error(f"Error: {e}", exc_info=True)

            
        except Exception as e:

            

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")

            await ctx.send(f"❌ An error occurred: {str(e)}")

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")
            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)
            stats = await self.wdbx.get_stats()
            embed = await self._create_embed(ctx, "WDBX Statistics", color=discord.Color.gold())
            
            # Add stats to embed
            for key, value in stats.items():
                # Format nicely
                if isinstance(value, (int, float)) and 'time' in key.lower():
                    # Format timestamps
                    dt = datetime.datetime.fromtimestamp(value)
                    value = dt.strftime("%Y-%m-%d %H:%M:%S")
                elif isinstance(value, float) and ('size' in key.lower() or 'memory' in key.lower()):
                    # Format sizes in MB/GB
                    if value > 1024:
                        value = f"{value/1024:.2f} GB"
                    else:
                        value = f"{value:.2f} MB"
                        
                embed.add_field(name=key.replace('_', ' ').title(), value=str(value), inline=True)
                
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            await ctx.send(f"❌ An error occurred: {str(e)}")
            await ctx.send(embed=embed)
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}", exc_info=True)
            await ctx.send(f"❌ Error getting stats: {str(e)}")

    @commands.command(name="admin", help="Perform administrative actions.help=help=")
    @commands.has_permissions(administrator=True)
    async def admin(self, ctx: commands.Context, action: str = None, *args):
        "Perform administrative actions (requires Admin role)."
        if not await self._ensure_wdbx_connection(ctx):
            return
            
        if not action:
            await ctx.send("❌ Please specify an action. Available actions: `status`, `clear`, `optimize`")
            return
            
        action = action.lower()
        
        try:

        

        
        except Exception as e:

        
            logger.error(f"Error: {e}", exc_info=True)
            except Exception as e:

        
        except Exception as e:

        
            logger.error(f"Error: {e}", exc_info=True)

        
            await ctx.send(f"❌ An error occurred: {str(e)}")

        
        except Exception as e:

        
            logger.error(f"Error: {e}", exc_info=True)

        
            await ctx.send(f"❌ An error occurred: {str(e)}")

        
        except Exception as e:

        
            logger.error(f"Error: {e}", exc_info=True)

        
            await ctx.send(f"❌ An error occurred: {str(e)}")

        
            logger.error(f"Error: {e}", exc_info=True)

        
        except Exception as e:

        
            logger.error(f"Error: {e}", exc_info=True)

        
        except Exception as e:

        
            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")

        
        except Exception as e:

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")

        
            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)
            if action == "status":
                # Get detailed system status
                stats = await self.wdbx.get_stats()
                health = await self.health_monitor.check_health() if self.health_monitor else {"overall_status": {"status": "UNKNOWN", "message": "Health monitor not available"}}
                
                embed = await self._create_embed(ctx, "WDBX Admin Status", color=discord.Color.dark_gold())
                
                # Add system info
                system_info = f"Platform: {platform.system()} {platform.release()}\n"
                system_info += f"Python: {sys.version.split()[0]}\n"
                system_info += f"WDBX Version: {self.wdbx.version if hasattr(self.wdbx, 'version') else 'Unknown'}\n"
                system_info += f"Discord.py: {discord.__version__ if hasattr(discord, '__version__') else 'Unknown'}"
                
                embed.add_field(name="System Info", value=system_info, inline=False)
                
                # Add health status
                health_status = health.get("overall_status", {}).get("status", "UNKNOWN")
                health_message = health.get("overall_status", {}).get("message", "No details available")
                embed.add_field(name="Health Status", value=f"{health_status}: {health_message}", inline=False)
                
                # Add key stats
                embed.add_field(name="Total Vectors", value=str(stats.get('total_vectors', 'N/A')), inline=True)
                embed.add_field(name="Total Blocks", value=str(stats.get('total_blocks', 'N/A')), inline=True)
                embed.add_field(name="Memory Usage", value=stats.get('memory_usage_mb', 'N/A'), inline=True)
                
                if 'uptime' in stats:
                    uptime_seconds = stats['uptime']
                    days, remainder = divmod(uptime_seconds, 86400)
                    hours, remainder = divmod(remainder, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    uptime_str = f"{int(days)}d {int(hours)}h {int(minutes)}m {int(seconds)}s"
                    embed.add_field(name="Uptime", value=uptime_str, inline=True)
                
                await ctx.send(embed=embed)
                
            elif action == "clear":
                # Ask for confirmation
                confirm_msg = await ctx.send("⚠️ **WARNING**: This will clear all vectors and blocks from the database. Type `confirm` to proceed or wait 20 seconds to cancel.")
                
                def check(...):
                    ""Function check.""
                    return msg.author == ctx.author and msg.content.lower() == "confirm" and msg.channel == ctx.channel
                
                try:

                

                
                except Exception as e:

                
                    logger.error(f"Error: {e}", exc_info=True)
                    except Exception as e:

                
                except Exception as e:

                
                    logger.error(f"Error: {e}", exc_info=True)

                
                    await ctx.send(f"❌ An error occurred: {str(e)}")

                
                    logger.error(f"Error: {e}", exc_info=True)

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                    await ctx.send(f"❌ An error occurred: {str(e)}")

                
                except Exception as e:

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                    await ctx.send(f"❌ An error occurred: {str(e)}")

                
                    logger.error(f"Error: {e}", exc_info=True)

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)
                    await self.bot.wait_for('message', check=check, timeout=20.0)
                    # Proceed with clearing
                    await ctx.send("🔄 Clearing database...")
                    await self.wdbx.clear()
                    await ctx.send("✅ Database cleared successfully.")
                except asyncio.TimeoutError:
                    await ctx.send("❌ Operation cancelled: confirmation timeout.")
                    
            elif action == "optimize":
                await ctx.send("🔄 Optimizing memory usage...")
                result = await self.wdbx.optimize_memory()
                if result:
                    await ctx.send("✅ Memory optimization completed successfully.")
                else:
                    await ctx.send("⚠️ Memory optimization completed with warnings.")
                    
            else:
                await ctx.send(f"❌ Unknown action: `{action}`. Available actions: `status`, `clear`, `optimize`")
                
        except Exception as e:
            self.logger.error(f"Error in admin command ({action}): {e}", exc_info=True)
            await ctx.send(f"❌ Error executing admin command: {str(e)}")
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            await ctx.send(f"❌ An error occurred: {str(e)}")
    
    @admin.error
    async def admin_error(self, ctx, error):
        ""Handle errors in the admin command."
        if isinstance(error, commands.MissingPermissions):
            await ctx.send("❌ You do not have permission to use administrative commands.")
        else:
            self.logger.error(f"Error in admin command: {error}", exc_info=True)
            await ctx.send(f"❌ An error occurred: {str(error)}")

    @commands.command(name="batch", help="Perform batch operations.help=help=")
    @commands.has_permissions(administrator=True)
    async def batch(self, ctx: commands.Context, operation: str = None, *args):
        "Perform batch operations (requires Admin role)."
        if not await self._ensure_wdbx_connection(ctx):
            return
            
        if not operation:
            await ctx.send("❌ Please specify an operation. Available operations: `import`, `export`")
            return
            
        operation = operation.lower()
        
        try:

        

        
        except Exception as e:

        
            logger.error(f"Error: {e}", exc_info=True)
            except Exception as e:

        
        except Exception as e:

        
            logger.error(f"Error: {e}", exc_info=True)

        
            await ctx.send(f"❌ An error occurred: {str(e)}")

        
        except Exception as e:

        
            logger.error(f"Error: {e}", exc_info=True)

        
            await ctx.send(f"❌ An error occurred: {str(e)}")

        
        except Exception as e:

        
            logger.error(f"Error: {e}", exc_info=True)

        
            await ctx.send(f"❌ An error occurred: {str(e)}")

        
            logger.error(f"Error: {e}", exc_info=True)

        
        except Exception as e:

        
            logger.error(f"Error: {e}", exc_info=True)

        
        except Exception as e:

        
            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")

        
        except Exception as e:

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")

        
            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)
            if operation == "import":
                # Check if file is attached
                if not ctx.message.attachments:
                    await ctx.send("❌ Please attach a JSON file containing vectors/blocks to import.")
                    return
                    
                attachment = ctx.message.attachments[0]
                if not attachment.filename.endswith('.json'):
                    await ctx.send("❌ Only JSON files are supported for import.")
                    return
                    
                # Download attachment
                await ctx.send(f"⏳ Downloading and processing {attachment.filename}...")
                file_content = await attachment.read()
                
                # Parse JSON
                try:


                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)
                    except Exception as e:

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                    await ctx.send(f"❌ An error occurred: {str(e)}")

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                    await ctx.send(f"❌ An error occurred: {str(e)}")

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                    await ctx.send(f"❌ An error occurred: {str(e)}")

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                    await ctx.send(f"❌ An error occurred: {str(e)}")

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                    await ctx.send(f"❌ An error occurred: {str(e)}")

                    logger.error(f"Error: {e}", exc_info=True)

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)
                    data = json.loads(file_content.decode('utf-8'))
                except json.JSONDecodeError:
                    await ctx.send("❌ Invalid JSON file. Could not parse.")
                    return
                    
                # Check data structure
                if not isinstance(data, dict) or not ('vectors' in data or 'blocks' in data):
                    await ctx.send("❌ Invalid data format. Expected a JSON object with 'vectors' or 'blocks' array.")
                    return
                    
                # Import data
                vectors_count = 0
                blocks_count = 0
                
                if 'vectors' in data and isinstance(data['vectors'], list):
                    await ctx.send(f"⏳ Importing {len(data['vectors'])} vectors...")
                    vectors_count = await self.wdbx.batch_create_vectors(data['vectors'])
                    
                if 'blocks' in data and isinstance(data['blocks'], list):
                    await ctx.send(f"⏳ Importing {len(data['blocks'])} blocks...")
                    blocks_count = len(data['blocks'])
                    for block in data['blocks']:
                        await self.wdbx.create_block(block)
                        
                await ctx.send(f"✅ Import complete. Imported {vectors_count} vectors and {blocks_count} blocks.")
                
            elif operation == "export":
                # Determine what to export
                export_type = args[0].lower() if args else "all"
                if export_type not in ["all", "vectors", "blocks"]:
                    await ctx.send("❌ Invalid export type. Use: all, vectors, or blocks")
                    return
                    
                await ctx.send(f"⏳ Exporting {export_type}...")
                
                # Get data to export
                data = {}
                if export_type in ["all", "vectors"]:
                    # Get all vectors (implementation depends on client API)
                    stats = await self.wdbx.get_stats()
                    total_vectors = stats.get('total_vectors', 0)
                    
                    if total_vectors > 0:
                        vectors = await self.wdbx.export_data(data_type="vectors")
                        data['vectors'] = vectors
                        
                if export_type in ["all", "blocks"]:
                    # Get all blocks
                    stats = await self.wdbx.get_stats()
                    total_blocks = stats.get('total_blocks', 0)
                    
                    if total_blocks > 0:
                        blocks = await self.wdbx.export_data(data_type="blocks")
                        data['blocks'] = blocks
                        
                # Check if we got any data
                if not data:
                    await ctx.send("⚠️ No data to export.")
                    return
                    
                # Save to file
                json_data = json.dumps(data, indent=2)
                file_name = f"wdbx_export_{int(time.time())}.json"
                
                with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
                    f.write(json_data)
                    export_file = f.name
                    
                # Send file
                vectors_count = len(data.get('vectors', []))
                blocks_count = len(data.get('blocks', []))
                
                await ctx.send(f"✅ Export complete. {vectors_count} vectors and {blocks_count} blocks exported:", 
                              file=discord.File(export_file, filename=file_name))
                              
                # Clean up
                try:


                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)
                    except Exception as e:

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                    await ctx.send(f"❌ An error occurred: {str(e)}")

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                    await ctx.send(f"❌ An error occurred: {str(e)}")

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                    await ctx.send(f"❌ An error occurred: {str(e)}")

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                    await ctx.send(f"❌ An error occurred: {str(e)}")

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                    await ctx.send(f"❌ An error occurred: {str(e)}")

                    logger.error(f"Error: {e}", exc_info=True)

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)
                    os.unlink(export_file)
                except Exception:
                    pass
                    
            else:
                await ctx.send(f"❌ Unknown operation: `{operation}`. Available operations: `import`, `export`")
                
        except Exception as e:
            self.logger.error(f"Error in batch command ({operation}): {e}", exc_info=True)
            await ctx.send(f"❌ Error executing batch command: {str(e)}")
            
    @batch.error
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            await ctx.send(f"❌ An error occurred: {str(e)}")
    async def batch_error(self, ctx, error):
        ""Handle errors in the batch command."
        if isinstance(error, commands.MissingPermissions):
            await ctx.send("❌ You do not have permission to use batch commands.")
        else:
            self.logger.error(f"Error in batch command: {error}", exc_info=True)
            await ctx.send(f"❌ An error occurred: {str(error)}")

    @commands.command(name="config", help="Generate a template configuration file.help=help=")
    async def config(self, ctx: commands.Context):
        "Generate a template configuration file."
        template = {
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
        
        json_template = json.dumps(template, indent=2)
        await ctx.send("Use this template for your configuration file:\n```json\n" + json_template + "\n```")

    @commands.command(name="help_wdbx", help="Show help for WDBX bot commands.help=help=")
    async def help_wdbx(self, ctx: commands.Context):
        "Show help for WDBX-specific commands."
        embed = create_help_embed()
        await ctx.send(embed=embed)

    @tasks.loop(seconds=30)
    async def health_check_loop(self):
        ""Periodically check health and send alerts if needed."
        if not self.wdbx or not self.health_monitor:
            return
            
        try:

            

            
        except Exception as e:

            
            logger.error(f"Error: {e}", exc_info=True)
            except Exception as e:

            
        except Exception as e:

            
            logger.error(f"Error: {e}", exc_info=True)

            
            await ctx.send(f"❌ An error occurred: {str(e)}")

            
        except Exception as e:

            
            logger.error(f"Error: {e}", exc_info=True)

            
            await ctx.send(f"❌ An error occurred: {str(e)}")

            
        except Exception as e:

            
            logger.error(f"Error: {e}", exc_info=True)

            
            await ctx.send(f"❌ An error occurred: {str(e)}")

            
            logger.error(f"Error: {e}", exc_info=True)

            
        except Exception as e:

            
            logger.error(f"Error: {e}", exc_info=True)

            
        except Exception as e:

            
            logger.error(f"Error: {e}", exc_info=True)

            
        except Exception as e:

            

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")

            await ctx.send(f"❌ An error occurred: {str(e)}")

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")
            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)
            # Get status channel
            channel = self.bot.get_channel(self.config.status_channel_id)
            if not channel:
                self.logger.warning(f"Status channel {self.config.status_channel_id} not found.")
                return
                
            # Check health
            health = await self.health_monitor.check_health()
            overall_status = health.get("overall_status", {}).get("status", "UNKNOWN")
            
            # If status changes to WARNING or ERROR, send alert
            if overall_status in [HealthStatus.WARNING, HealthStatus.ERROR] and self.last_health_status != overall_status:
                # Format message
                alert_message = f"⚠️ **WDBX Health Alert**\n\n"
                alert_message += format_health_status(health)
                
                # Send to status channel
                await channel.send(alert_message)
                
            # Update last status
            self.last_health_status = overall_status
            
        except Exception as e:
            self.logger.error(f"Error in health check loop: {e}", exc_info=True)
            
    @health_check_loop.before_loop
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
    async def before_health_check(self):
        ""Wait for bot to be ready before starting loop."
        await self.bot.wait_until_ready()
        
        # Also set the interval from config
        self.health_check_loop.change_interval(seconds=self.config.monitoring_interval)


class DiscordBotPlugin(WDBXPlugin):
    ""Discord Bot Plugin for WDBX."
    
    @property
    def name(self) -> str:
        ""Get the plugin's unique name."
        return "discord_bot"
    
    @property
    def description(self) -> str:
        ""Get the plugin's description."
        return "Provides a Discord bot interface for interacting with WDBX."
    
    @property
    def version(self) -> str:
        ""Get the plugin's version."
        return "0.2.0"
    
    def __init__(self):
        ""Initialize the Discord Bot Plugin."
        self.bot = None
        self.config = None
        self.wdbx_client = None
        self.cog = None
        self.logger = logging.getLogger("wdbx.plugins.discord_bot")
    
    def initialize(self, api: Any) -> bool:
        ""
        Initialize the plugin with the WDBX API instance.
        
        Args:
            api: The WDBX API instance.
            
        Returns:
            bool: True if initialization was successful, False otherwise.
        ""
        if not DISCORD_AVAILABLE:
            self.logger.error("Discord.py is not installed. Cannot initialize Discord Bot Plugin.")
            return False
        
        try:

        

        
        except Exception as e:

        
            logger.error(f"Error: {e}", exc_info=True)
            except Exception as e:

        
        except Exception as e:

        
            logger.error(f"Error: {e}", exc_info=True)

        
            await ctx.send(f"❌ An error occurred: {str(e)}")

        
        except Exception as e:

        
            logger.error(f"Error: {e}", exc_info=True)

        
            await ctx.send(f"❌ An error occurred: {str(e)}")

        
        except Exception as e:

        
            logger.error(f"Error: {e}", exc_info=True)

        
            await ctx.send(f"❌ An error occurred: {str(e)}")

        
            logger.error(f"Error: {e}", exc_info=True)

        
        except Exception as e:

        
            logger.error(f"Error: {e}", exc_info=True)

        
        except Exception as e:

        
            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")

        
        except Exception as e:

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")

        
            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)
            # Set up logging
            setup_logging()
            self.logger.info("Discord Bot Plugin initializing...")
            
            # Look for config file in standard locations
            config_locations = [
                "config/discord_bot.json",
                "discord_bot.json",
                os.path.expanduser("~/.config/wdbx/discord_bot.json")
            ]
            
            config_data = None
            for location in config_locations:
                if os.path.exists(location):
                    try:


                    except Exception as e:

                        logger.error(f"Error: {e}", exc_info=True)
                        except Exception as e:

                    except Exception as e:

                        logger.error(f"Error: {e}", exc_info=True)

                        await ctx.send(f"❌ An error occurred: {str(e)}")

                        logger.error(f"Error: {e}", exc_info=True)

                    except Exception as e:

                        logger.error(f"Error: {e}", exc_info=True)
                        with open(location, 'r') as f:
                            config_data = json.load(f)
                            self.logger.info(f"Loaded configuration from {location}")
                            break
                    except Exception as e:
                        self.logger.warning(f"Failed to load config from {location}: {e}")
            
            if not config_data:
                self.logger.error("No configuration file found. Please create one using the template.")
                return False
            
            # Create bot config
            try:


            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)
                except Exception as e:

            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)

                await ctx.send(f"❌ An error occurred: {str(e)}")

            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)

                await ctx.send(f"❌ An error occurred: {str(e)}")

            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)

                await ctx.send(f"❌ An error occurred: {str(e)}")

            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)

                await ctx.send(f"❌ An error occurred: {str(e)}")

                logger.error(f"Error: {e}", exc_info=True)

            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)

            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)

            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)

            except Exception as e:

                logger.error(f"Error: {e}", exc_info=True)
                self.config = BotConfig(**config_data)
            except Exception as e:
                self.logger.error(f"Invalid configuration: {e}")
                return False
            
            # Connect to WDBX
            if api is not None:
                # Use the API provided through initialization
                self.wdbx_client = api
                self.logger.info("Using provided WDBX API instance.")
            else:
                # Fall back to connecting directly
                self.wdbx_client = connect_to_wdbx(self.config.wdbx_host, self.config.wdbx_port)
                if not self.wdbx_client:
                    self.logger.error("Failed to connect to WDBX.")
                    return False
            
            # Create bot instance
            intents = discord.Intents.default()
            intents.message_content = True
            
            self.bot = commands.Bot(command_prefix=self.config.prefix, intents=intents)
            
            # Add event handlers
            @self.bot.event
            async def on_ready():
                self.logger.info(f"Bot logged in as {self.bot.user.name}")
                # Set activity status
                activity = discord.Activity(
                    type=discord.ActivityType.watching,
                    name="WDBX vectors"
                )
                await self.bot.change_presence(activity=activity)
            
            @self.bot.event
            async def on_command_error(ctx, error):
                if isinstance(error, commands.CommandNotFound):
                    return  # Ignore command not found errors
                elif isinstance(error, commands.MissingRequiredArgument):
                    await ctx.send(f"❌ Missing required argument: `{error.param.name}`")
                elif isinstance(error, commands.BadArgument):
                    await ctx.send(f"❌ Bad argument: {str(error)}")
                else:
                    self.logger.error(f"Command error: {error}", exc_info=True)
                    await ctx.send(f"❌ An error occurred: {str(error)}")
            
            # Add the cog to the bot
            self.cog = WDBXCog(self.bot, self.config, self.wdbx_client)
            self.bot.add_cog(self.cog)
            
            # Start the bot in a separate thread
            self.logger.info("Starting Discord bot...")
            self._start_bot_async()
            
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            await ctx.send(f"❌ An error occurred: {str(e)}")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing Discord Bot Plugin: {e}", exc_info=True)
            return False
    
    def _start_bot_async(self):
        ""Start the bot in an async task."
        asyncio.create_task(self._run_bot())
    
    async def _run_bot(self):
        ""Run the bot."
        try:


        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)
            except Exception as e:

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")

            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)
            await self.bot.start(self.config.token)
        except Exception as e:
            self.logger.error(f"Error running Discord bot: {e}", exc_info=True)
       except Exception as e:
           logger.error(f"Error: {e}", exc_info=True)
    
    def shutdown(self) -> bool:
        ""
        Perform cleanup when the plugin is shutting down.
        
        Returns:
            bool: True if shutdown was successful, False otherwise.
        ""
        try:


        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)
            except Exception as e:

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

            await ctx.send(f"❌ An error occurred: {str(e)}")

            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)

        except Exception as e:

            logger.error(f"Error: {e}", exc_info=True)
            self.logger.info("Shutting down Discord Bot Plugin...")
            
            # Close the bot
            if self.bot:
                asyncio.create_task(self.bot.close())
            
            # Clean up resources
            if self.wdbx_client:
                try:


                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)
                    except Exception as e:

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                    await ctx.send(f"❌ An error occurred: {str(e)}")

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                    await ctx.send(f"❌ An error occurred: {str(e)}")

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                    await ctx.send(f"❌ An error occurred: {str(e)}")

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                    await ctx.send(f"❌ An error occurred: {str(e)}")

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                    await ctx.send(f"❌ An error occurred: {str(e)}")

                    logger.error(f"Error: {e}", exc_info=True)

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)

                except Exception as e:

                    logger.error(f"Error: {e}", exc_info=True)
                    self.wdbx_client.disconnect()
                except Exception as e:
                    self.logger.error(f"Error disconnecting from WDBX: {e}")
            
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            return True
        except Exception as e:
            self.logger.error(f"Error during Discord Bot Plugin shutdown: {e}", exc_info=True)
            return False


# Register the plugin
register_plugin(DiscordBotPlugin)


def run_standalone(config_path: str = None):
    ""Run the Discord bot as a standalone application."
    if not DISCORD_AVAILABLE:
        print("Error: Discord.py is not installed. Please install it with 'pip install discord.py'.")
        return
    
    setup_logging()
    logger.info("Starting Discord bot in standalone mode...")
    
    # Load config
    if not config_path:
        config_path = "discord_bot.json"
    
    try:

    

    
    except Exception as e:

    
        logger.error(f"Error: {e}", exc_info=True)
        except Exception as e:

    
    except Exception as e:

    
        logger.error(f"Error: {e}", exc_info=True)

    
        await ctx.send(f"❌ An error occurred: {str(e)}")

    
    except Exception as e:

    
        logger.error(f"Error: {e}", exc_info=True)

    
        await ctx.send(f"❌ An error occurred: {str(e)}")

    
        logger.error(f"Error: {e}", exc_info=True)

    
    except Exception as e:

    
        logger.error(f"Error: {e}", exc_info=True)

    except Exception as e:

        logger.error(f"Error: {e}", exc_info=True)

        await ctx.send(f"❌ An error occurred: {str(e)}")

    
    except Exception as e:

    except Exception as e:

        logger.error(f"Error: {e}", exc_info=True)

        await ctx.send(f"❌ An error occurred: {str(e)}")

    except Exception as e:

        logger.error(f"Error: {e}", exc_info=True)

        await ctx.send(f"❌ An error occurred: {str(e)}")

    
        logger.error(f"Error: {e}", exc_info=True)

    except Exception as e:

        logger.error(f"Error: {e}", exc_info=True)

    except Exception as e:

        logger.error(f"Error: {e}", exc_info=True)
        with...:
        except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
            config_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        print(f"Error: Failed to load config from {config_path}: {e}")
        return
    
    try:

    

    
    except Exception as e:

    
        logger.error(f"Error: {e}", exc_info=True)
        except Exception as e:

    
    except Exception as e:

    
        logger.error(f"Error: {e}", exc_info=True)

    
        await ctx.send(f"❌ An error occurred: {str(e)}")

    
    except Exception as e:

    
        logger.error(f"Error: {e}", exc_info=True)

    
        await ctx.send(f"❌ An error occurred: {str(e)}")

    
        logger.error(f"Error: {e}", exc_info=True)

    
    except Exception as e:

    
        logger.error(f"Error: {e}", exc_info=True)

    except Exception as e:

        logger.error(f"Error: {e}", exc_info=True)

        await ctx.send(f"❌ An error occurred: {str(e)}")

    
    except Exception as e:

    except Exception as e:

        logger.error(f"Error: {e}", exc_info=True)

        await ctx.send(f"❌ An error occurred: {str(e)}")

    except Exception as e:

        logger.error(f"Error: {e}", exc_info=True)

        await ctx.send(f"❌ An error occurred: {str(e)}")

    
        logger.error(f"Error: {e}", exc_info=True)

    except Exception as e:

        logger.error(f"Error: {e}", exc_info=True)

    except Exception as e:

        logger.error(f"Error: {e}", exc_info=True)
        # Create config
        config = BotConfig(**config_data)
        
        # Connect to WDBX
        wdbx_client = connect_to_wdbx(config.wdbx_host, config.wdbx_port)
        if not wdbx_client:
            logger.error("Failed to connect to WDBX.")
            print("Error: Failed to connect to WDBX.")
            return
        
        # Create and run bot
        intents = discord.Intents.default()
        intents.message_content = True
        
        bot = commands.Bot(command_prefix=config.prefix, intents=intents)
        
        # Add event handlers
        @bot.event
        async def on_ready():
            logger.info(f"Bot logged in as {bot.user.name}")
            # Set activity status
            activity = discord.Activity(
                type=discord.ActivityType.watching,
                name="WDBX vectors"
            )
            await bot.change_presence(activity=activity)
        
        # Add the cog to the bot
        bot.add_cog(WDBXCog(bot, config, wdbx_client))
        
        # Run the bot
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        bot.run(config.token)
    except Exception as e:
        logger.error(f"Error running Discord bot: {e}", exc_info=True)
        print(f"Error: {e}")


if __name__ == "__main__":
    # Run in standalone mode if executed directly
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_standalone(config_path) 