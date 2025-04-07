"""
Discord Bot Plugin for WDBX.

This module provides a Discord bot that allows users to interact with
and monitor the WDBX vector database through Discord.

## Features

- Basic commands for viewing WDBX status and health
- Vector search functionality
- Vector visualization with PCA
- Administrative commands for system management
- Batch import/export operations
- Real-time health monitoring and alerts

## Setup

1. Install the required dependencies:
   ```
   pip install discord.py matplotlib numpy scikit-learn
   ```

2. Create a configuration file (e.g., `config.json`):
   ```json
   {
     "token": "YOUR_DISCORD_BOT_TOKEN",
     "prefix": "!",
     "status_channel_id": 123456789012345678,
     "admin_role_id": 123456789012345678,
     "wdbx_host": "127.0.0.1",
     "wdbx_port": 8080,
     "monitoring_interval": 300,
     "log_dir": "logs"
   }
   ```

3. Run the bot:
   ```
   python discord_bot.py config.json
   ```

## Available Commands

- `!status` - Show WDBX status
- `!health` - Show detailed health information
- `!search <query> [top_k]` - Search for vectors
- `!metrics` - Show system metrics
- `!visualize [query] [n_vectors]` - Visualize vectors in 2D space
- `!admin <action> [args]` - Administrative commands (status, clear, optimize)
- `!batch <operation> [args]` - Batch operations (import, export)
- `!help_wdbx` - Show all available WDBX-specific commands
- `!stats` - Show vector statistics
- `!config` - Generate a template configuration file

## Integration

This bot can be integrated with any WDBX instance by passing the instance
to the bot constructor. See the example in the `if __name__ == "__main__"` section.
"""

import asyncio
import datetime
import io
import json
import logging
import os
import platform
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional, Any, List, Tuple, Union

try:
    import discord
    from discord.ext import commands, tasks

    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False

    # Mock discord classes if not available to allow basic structure loading
    class commands:
        Bot = object
        Cog = object
        Context = object
        CommandError = Exception
        CommandNotFound = Exception
        MissingRequiredArgument = Exception
        BadArgument = Exception

        def command(*args, **kwargs):
            def decorator(func):
                return func

            return decorator

        def has_permissions(*args, **kwargs):
            def decorator(func):
                return func

            return decorator

    class tasks:
        def loop(*args, **kwargs):
            def decorator(func):
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


def connect_to_wdbx(host: str, port: int) -> Optional[Any]:
    """Attempt to connect to the WDBX instance."""
    if not WDBX_AVAILABLE or WDBXClient is None:
        logger.error("WDBX client library is not available.")
        return None
    try:
        client = WDBXClient()
        # Use asyncio.run for synchronous context if necessary, or ensure client has async connect
        # Assuming client.connect is synchronous or handled appropriately elsewhere
        client.connect(host=host, port=port)
        logger.info(f"Successfully connected to WDBX at {host}:{port}")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to WDBX at {host}:{port}: {e}", exc_info=True)
        return None


def format_health_status(status: Dict[str, Dict[str, Any]]) -> str:
    """Format the health status dictionary into a readable string."""
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
    """Create an embed message for the help command."""
    embed = discord.Embed(
        title="WDBX Discord Bot Help",
        description="Available commands for interacting with WDBX:",
        color=discord.Color.blue(),
    )
    embed.add_field(name="`!status`", value="Show WDBX status information.", inline=False)
    embed.add_field(name="`!health`", value="Show detailed WDBX health information.", inline=False)
    embed.add_field(
        name="`!metrics`", value="Show current system metrics (Prometheus format).", inline=False
    )
    embed.add_field(
        name="`!search <query> [top_k=5]`", value="Search for similar vectors.", inline=False
    )
    embed.add_field(
        name="`!visualize [query] [n_vectors=20]`",
        value="Visualize vectors in 2D space (requires libraries).",
        inline=False,
    )
    embed.add_field(name="`!stats`", value="Show vector database statistics.", inline=False)
    embed.add_field(
        name="`!admin <action> [args]`",
        value="Perform administrative actions (requires Admin role).\nActions: `status`, `clear`, `optimize`",
        inline=False,
    )
    embed.add_field(
        name="`!batch <operation> [args]`",
        value="Perform batch operations (import/export, requires Admin role).\nOperations: `import`, `export`",
        inline=False,
    )
    embed.add_field(name="`!config`", value="Generate a template configuration file.", inline=False)
    embed.add_field(name="`!help_wdbx`", value="Show this help message.", inline=False)
    embed.set_footer(text="WDBX Bot | Use the prefix defined in config (default: !)")
    return embed


class WDBXCog(commands.Cog):
    """Cog containing commands for interacting with WDBX."""

    def __init__(self, bot: commands.Bot, config: BotConfig, wdbx_instance: Optional[Any] = None):
        if not DISCORD_AVAILABLE:
            raise BotSetupError("discord.py is not installed. Cannot initialize WDBXCog.")

        self.bot = bot
        self.config = config
        self.wdbx = wdbx_instance  # This might be None initially
        self.health_monitor = None  # Initialize later if WDBX available
        self.logger = logging.getLogger("wdbx.discord.cog")

        # Try to initialize health monitor if WDBX is available
        if self.wdbx and WDBX_AVAILABLE and HealthMonitor:
            try:
                self.health_monitor = HealthMonitor(self.wdbx)  # Assuming HealthMonitor takes client
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
        """Clean up tasks when the cog is unloaded."""
        self.health_check_loop.cancel()
        self.logger.info("Health check loop cancelled.")
        if self.wdbx:
            try:
                # Assuming client has a disconnect or close method
                self.wdbx.disconnect()
                self.logger.info("Disconnected from WDBX.")
            except Exception as e:
                self.logger.error(f"Error disconnecting from WDBX: {e}", exc_info=True)

    async def _ensure_wdbx_connection(self, ctx: commands.Context) -> bool:
        """Check if connected to WDBX and attempt reconnect if not."""
        if self.wdbx and self.wdbx.is_connected():  # Assuming an is_connected method
            return True

        await ctx.send("⏳ Attempting to reconnect to WDBX...")
        self.wdbx = connect_to_wdbx(self.config.wdbx_host, self.config.wdbx_port)

        if self.wdbx:
            # Re-initialize health monitor if connection is re-established
            if WDBX_AVAILABLE and HealthMonitor and not self.health_monitor:
                try:
                    self.health_monitor = HealthMonitor(self.wdbx)
                    self.logger.info("Health monitor re-initialized after reconnect.")
                except Exception as e:
                    self.logger.error(f"Failed to re-initialize Health Monitor: {e}", exc_info=True)
            await ctx.send("✅ Reconnected to WDBX successfully.")
            return True
        else:
            await ctx.send(
                "❌ Failed to connect to WDBX. Please check the WDBX server and configuration."
            )
            return False

    async def _create_embed(
        self,
        ctx: commands.Context,
        title: str,
        description: str = "",
        color: Any = None,
    ) -> discord.Embed:
        """Helper function to create a standard embed."""
        if color is None:
            color = discord.Color.blue()
        embed = discord.Embed(title=title, description=description, color=color)
        embed.set_footer(
            text=f"Requested by {ctx.author.display_name}",
            icon_url=ctx.author.avatar.url if ctx.author.avatar else None,
        )
        embed.timestamp = datetime.datetime.utcnow()
        return embed

    @commands.command(name="status", help="Show WDBX status information.")
    async def status(self, ctx: commands.Context):
        """Show WDBX status information."""
        if not await self._ensure_wdbx_connection(ctx):
            return

        try:
            stats = await self.wdbx.get_stats()  # Assuming async get_stats
            embed = await self._create_embed(ctx, "WDBX Status", color=discord.Color.green())
            embed.add_field(
                name="Connection",
                value=f"Connected to {self.config.wdbx_host}:{self.config.wdbx_port}",
                inline=False,
            )
            embed.add_field(
                name="Total Vectors", value=str(stats.get("total_vectors", "N/A")), inline=True
            )
            embed.add_field(
                name="Total Blocks", value=str(stats.get("total_blocks", "N/A")), inline=True
            )
            embed.add_field(
                name="Memory Usage", value=stats.get("memory_usage_mb", "N/A"), inline=True
            )  # Assuming memory usage is available
            # Add more stats as needed
            embed.add_field(name="Server Version", value=stats.get("version", "N/A"), inline=True)
            embed.add_field(name="Uptime", value=stats.get("uptime", "N/A"), inline=True)

            await ctx.send(embed=embed)
        except WDBXClientError as e:
            self.logger.error(f"WDBX client error in status command: {e}", exc_info=True)
            await ctx.send(f"❌ Error getting WDBX status: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error in status command: {e}", exc_info=True)
            await ctx.send(f"❌ An unexpected error occurred: {str(e)}")

    @commands.command(name="health", help="Show detailed WDBX health information.")
    async def health(self, ctx: commands.Context):
        """Show detailed WDBX health information."""
        if not await self._ensure_wdbx_connection(ctx):
            return

        if not self.health_monitor:
            await ctx.send("❌ Health monitoring is not available.")
            return

        try:
            health_status = await self.health_monitor.check_health()
            formatted_status = format_health_status(health_status)

            # Determine color based on status
            overall_status = health_status.get("overall_status", {}).get("status", "UNKNOWN")
            color = (
                discord.Color.green()
                if overall_status == "OK"
                else discord.Color.orange() if overall_status == "WARNING" else discord.Color.red()
            )

            embed = await self._create_embed(ctx, "WDBX Health Check", color=color)

            # Split into fields if too long
            if len(formatted_status) > 1024:
                parts = []
                current_part = ""
                for line in formatted_status.split("\n"):
                    if len(current_part) + len(line) + 1 > 1024:
                        parts.append(current_part)
                        current_part = line
                    elif current_part:
                        current_part += "\n" + line
                    else:
                        current_part = line
                if current_part:
                    parts.append(current_part)

                for i, part in enumerate(parts):
                    embed.add_field(
                        name=f"Health Details {i+1}/{len(parts)}", value=part, inline=False
                    )
            else:
                embed.add_field(name="Health Details", value=formatted_status, inline=False)

            await ctx.send(embed=embed)
        except Exception as e:
            self.logger.error(f"Error in health command: {e}", exc_info=True)
            await ctx.send(f"❌ Error checking health: {str(e)}")

    @commands.command(name="metrics", help="Show current system metrics (Prometheus format).")
    async def metrics(self, ctx: commands.Context):
        """Show current system metrics in Prometheus format."""
        if not await self._ensure_wdbx_connection(ctx):
            return

        if not WDBX_AVAILABLE or get_metrics is None:
            await ctx.send("❌ Metrics functionality is not available.")
            return

        try:
            metrics_data = await get_metrics(self.wdbx)  # Assuming async get_metrics

            if not metrics_data:
                await ctx.send("❌ No metrics data available.")
                return

            # Format metrics for display
            metrics_text = (
                metrics_data
                if isinstance(metrics_data, str)
                else "\n".join([f"{k} {v}" for k, v in metrics_data.items()])
            )

            # If metrics are too long, send as file
            if len(metrics_text) > 1900:  # Discord message limit is 2000
                file = discord.File(io.StringIO(metrics_text), filename="wdbx_metrics.txt")
                await ctx.send("📊 WDBX Metrics (see attached file):", file=file)
            else:
                await ctx.send(f"📊 **WDBX Metrics:**\n```\n{metrics_text}\n```")

        except Exception as e:
            self.logger.error(f"Error in metrics command: {e}", exc_info=True)
            await ctx.send(f"❌ Error retrieving metrics: {str(e)}")

    @commands.command(name="search", help="Search for similar vectors.")
    async def search(self, ctx: commands.Context, query: str, top_k: int = 5):
        """Search for vectors similar to the query."""
        if not await self._ensure_wdbx_connection(ctx):
            return

        if top_k < 1:
            await ctx.send("❌ top_k must be at least 1.")
            return

        if top_k > 100:  # Reasonable limit
            await ctx.send("⚠️ Limiting search results to 100 for performance reasons.")
            top_k = 100

        await ctx.send(f"🔍 Searching for vectors similar to: `{query}` (top {top_k})...")

        try:
            # Assuming client has a search method
            results = await self.wdbx.search(query=query, top_k=top_k)

            if not results:
                await ctx.send("❓ No similar vectors found.")
                return

            # Format results
            embed = await self._create_embed(
                ctx, f"Search Results for: {query}", color=discord.Color.blue()
            )

            # Limit display count
            display_count = min(len(results), self.config.max_vectors_display)
            if display_count < len(results):
                embed.description = f"Showing top {display_count} of {len(results)} results."

            for i, result in enumerate(results[:display_count]):
                vector_id = result.get("id", "Unknown ID")
                score = result.get("score", 0.0)
                metadata = result.get("metadata", {})

                # Format metadata if present
                metadata_str = ""
                if metadata:
                    metadata_str = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
                    metadata_str = f"\nMetadata:\n```\n{metadata_str}\n```"

                embed.add_field(
                    name=f"{i+1}. Vector {vector_id}",
                    value=f"Score: {score:.4f}{metadata_str}",
                    inline=False,
                )

            await ctx.send(embed=embed)

        except Exception as e:
            self.logger.error(f"Error in search command: {e}", exc_info=True)
            await ctx.send(f"❌ Error during search: {str(e)}")

    @commands.command(name="visualize", help="Visualize vectors in 2D space.")
    async def visualize(
        self, ctx: commands.Context, query: Optional[str] = None, n_vectors: int = 20
    ):
        """Visualize vectors in 2D space using PCA."""
        if not await self._ensure_wdbx_connection(ctx):
            return

        if not VISUALIZATION_AVAILABLE:
            await ctx.send(
                "❌ Visualization libraries (matplotlib, numpy, scikit-learn) are not available."
            )
            return

        if n_vectors < 2:
            await ctx.send("❌ n_vectors must be at least 2 for visualization.")
            return

        if n_vectors > 100:  # Reasonable limit
            await ctx.send("⚠️ Limiting visualization to 100 vectors for performance reasons.")
            n_vectors = 100

        await ctx.send(
            f"🎨 Generating vector visualization{' for query: `' + query + '`' if query else ''}..."
        )

        try:
            # Get vectors to visualize
            vectors = []
            if query:
                # Search for vectors similar to query
                results = await self.wdbx.search(query=query, top_k=n_vectors)
                if not results:
                    await ctx.send("❓ No vectors found for the query.")
                    return

                # Extract vector embeddings
                for result in results:
                    vector_id = result.get("id", "Unknown")
                    embedding = result.get("embedding", result.get("vector", None))
                    if embedding:
                        vectors.append((vector_id, embedding))
            else:
                # Get random sample of vectors
                sample = await self.wdbx.get_random_vectors(n=n_vectors)
                if not sample:
                    await ctx.send("❓ No vectors available for visualization.")
                    return

                # Extract vector embeddings
                for vector in sample:
                    vector_id = vector.get("id", "Unknown")
                    embedding = vector.get("embedding", vector.get("vector", None))
                    if embedding:
                        vectors.append((vector_id, embedding))

            if len(vectors) < 2:
                await ctx.send("❌ Need at least 2 vectors for visualization.")
                return

            # Perform PCA to reduce to 2D
            embeddings = np.array([v[1] for v in vectors])
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(embeddings)

            # Create plot
            plt.figure(figsize=(10, 8))
            plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)

            # Add labels for points
            for i, (vector_id, _) in enumerate(vectors):
                plt.annotate(str(vector_id), (reduced[i, 0], reduced[i, 1]), fontsize=8)

            plt.title(f"Vector Visualization{' for query: ' + query if query else ''}")
            plt.xlabel(f"PCA Component 1 (Variance: {pca.explained_variance_ratio_[0]:.2%})")
            plt.ylabel(f"PCA Component 2 (Variance: {pca.explained_variance_ratio_[1]:.2%})")
            plt.grid(True, linestyle="--", alpha=0.7)

            # Save plot to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            buf.seek(0)

            # Send image
            file = discord.File(buf, filename="vector_visualization.png")
            embed = await self._create_embed(
                ctx,
                f"Vector Visualization{' for: ' + query if query else ''}",
                description=f"Showing {len(vectors)} vectors reduced to 2D using PCA.",
                color=discord.Color.purple(),
            )
            embed.set_image(url="attachment://vector_visualization.png")
            await ctx.send(embed=embed, file=file)

            # Clean up
            plt.close()

        except Exception as e:
            self.logger.error(f"Error in visualize command: {e}", exc_info=True)
            await ctx.send(f"❌ Error during visualization: {str(e)}")

    @commands.group(
        name="admin",
        help="Administrative commands (requires Admin role).",
        invoke_without_command=True,
    )
    @commands.has_permissions(administrator=True)
    async def admin(self, ctx: commands.Context):
        """Base command for administrative actions."""
        if ctx.invoked_subcommand is None:
            await ctx.send_help(ctx.command)

    @admin.command(name="status", help="Show detailed system status.")
    @commands.has_permissions(administrator=True)
    async def admin_status(self, ctx: commands.Context):
        """Show detailed system status for administrators."""
        if not await self._ensure_wdbx_connection(ctx):
            return

        try:
            # Get system info
            system_info = {
                "Platform": platform.platform(),
                "Python Version": platform.python_version(),
                "Process ID": os.getpid(),
                "Process Start Time": datetime.datetime.fromtimestamp(
                    time.time() - (time.process_time() if hasattr(time, "process_time") else 0)
                ).strftime("%Y-%m-%d %H:%M:%S"),
                "Memory Usage": (
                    f"{psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024):.2f} MB"
                    if PSUTIL_AVAILABLE
                    else "N/A"
                ),
            }

            # Get WDBX stats
            wdbx_stats = await self.wdbx.get_stats()

            # Create embed
            embed = await self._create_embed(ctx, "WDBX Admin Status", color=discord.Color.gold())

            # System info section
            system_info_str = "\n".join([f"**{k}:** {v}" for k, v in system_info.items()])
            embed.add_field(name="System Information", value=system_info_str, inline=False)

            # WDBX stats section
            wdbx_stats_str = "\n".join([f"**{k}:** {v}" for k, v in wdbx_stats.items()])
            embed.add_field(name="WDBX Statistics", value=wdbx_stats_str, inline=False)

            # Add connection info
            embed.add_field(
                name="Connection Details",
                value=f"**Host:** {self.config.wdbx_host}\n**Port:** {self.config.wdbx_port}",
                inline=False,
            )

            await ctx.send(embed=embed)

        except Exception as e:
            self.logger.error(f"Error in admin status command: {e}", exc_info=True)
            await ctx.send(f"❌ Error retrieving admin status: {str(e)}")

    @admin.command(name="clear", help="Clear all vectors from the database.")
    @commands.has_permissions(administrator=True)
    async def admin_clear(self, ctx: commands.Context):
        """Clear all vectors from the database."""
        if not await self._ensure_wdbx_connection(ctx):
            return

        # Confirmation message
        confirmation_msg = await ctx.send(
            "⚠️ **WARNING**: This will delete ALL vectors from the database. Are you sure? Reply with 'yes' to confirm."
        )

        def check(m):
            return m.author == ctx.author and m.content.lower() == "yes"

        try:
            await self.bot.wait_for("message", timeout=30.0, check=check)
        except asyncio.TimeoutError:
            await confirmation_msg.edit(
                content="❌ Clear operation cancelled (timeout).", delete_after=10
            )
            return
        except Exception as e:  # Catch potential discord errors
            self.logger.error(f"Error waiting for confirmation: {e}", exc_info=True)
            await confirmation_msg.edit(content=f"❌ Error during confirmation: {str(e)}")
            return

        # Proceed with clearing
        await ctx.send("⏳ Clearing all vectors... Please wait.")
        try:
            # Assuming client has a clear method
            success = await self.wdbx.clear()  # Example method
            if success:
                await ctx.send("✅ All vectors cleared successfully.")
            else:
                # If client method returns bool success status
                await ctx.send("❌ Clear operation failed on the server side. Check WDBX logs.")
        except WDBXClientError as e:
            self.logger.error(f"WDBX client error during clear: {e}", exc_info=True)
            await ctx.send(f"❌ WDBX client error: {str(e)}")
        except NotImplementedError:  # Example specific error
            await ctx.send("❌ Clear operation is not supported by this WDBX instance.")
        except Exception as e:
            self.logger.error(f"Error clearing vectors: {e}", exc_info=True)
            await ctx.send(f"❌ An unexpected error occurred during clear: {str(e)}")

    @admin.command(name="optimize", help="Trigger memory optimization in WDBX.")
    @commands.has_permissions(administrator=True)
    async def admin_optimize(self, ctx: commands.Context):
        """Trigger memory optimization in WDBX."""
        if not await self._ensure_wdbx_connection(ctx):
            return

        await ctx.send("⏳ Triggering memory optimization... This might take a moment.")
        try:
            # Assuming client has optimize_memory method
            success = await self.wdbx.optimize_memory()  # Example method
            if success:
                await ctx.send(
                    "✅ Memory optimization process started successfully. Check WDBX logs for details."
                )
            else:
                await ctx.send("❌ Failed to start memory optimization. Check WDBX logs.")
        except WDBXClientError as e:
            self.logger.error(f"WDBX client error during optimize: {e}", exc_info=True)
            await ctx.send(f"❌ WDBX client error: {str(e)}")
        except NotImplementedError:
            await ctx.send("❌ Optimize memory operation is not supported by this WDBX instance.")
        except Exception as e:
            self.logger.error(f"Error optimizing memory: {e}", exc_info=True)
            await ctx.send(f"❌ An unexpected error occurred during optimization: {str(e)}")

    @commands.group(
        name="batch",
        help="Batch operations (import/export, requires Admin role).",
        invoke_without_command=True,
    )
    @commands.has_permissions(administrator=True)
    async def batch(self, ctx: commands.Context):
        """Base command for batch operations."""
        if ctx.invoked_subcommand is None:
            await ctx.send_help(ctx.command)

    @batch.command(name="import", help="Import vectors from an attached JSON file.")
    @commands.has_permissions(administrator=True)
    async def batch_import(self, ctx: commands.Context):
        """Import vectors from an attached JSON file."""
        if not await self._ensure_wdbx_connection(ctx):
            return

        if not ctx.message.attachments:
            await ctx.send("❌ Please attach a JSON file containing the vectors to import.")
            return

        attachment = ctx.message.attachments[0]
        if not attachment.filename.lower().endswith(".json"):
            await ctx.send("❌ Please attach a valid JSON file (`.json`).")
            return

        await ctx.send(f"⏳ Downloading and processing `{attachment.filename}` for import...")

        try:
            # Read attachment content
            json_data = await attachment.read()
            vectors_to_import = json.loads(json_data.decode("utf-8"))

            if not isinstance(vectors_to_import, list):
                raise ValueError("JSON file must contain a list of vector objects.")

            if not vectors_to_import:
                await ctx.send("❌ The JSON file is empty. No vectors to import.")
                return

            # Validate structure (basic check on first item)
            first_item = vectors_to_import[0]
            if (
                not isinstance(first_item, dict)
                or ("vector_id" not in first_item and "id" not in first_item)
                or ("embedding" not in first_item and "vector" not in first_item)
            ):
                raise ValueError(
                    "Vector objects in JSON must be dictionaries with at least 'vector_id' (or 'id') and 'embedding' (or 'vector')."
                )

            await ctx.send(f"⏳ Importing {len(vectors_to_import)} vectors... This may take time.")

            # Assuming client has batch_create_vectors method
            results = await self.wdbx.batch_create_vectors(
                vectors=vectors_to_import
            )  # Example method

            # Process results (assuming results indicate success/failure count or details)
            success_count = results.get(
                "success_count", len(vectors_to_import)
            )  # Adapt based on actual result format
            failure_count = results.get("failure_count", 0)
            errors = results.get("errors", [])

            embed = await self._create_embed(
                ctx,
                "Batch Import Report",
                color=discord.Color.green() if failure_count == 0 else discord.Color.orange(),
            )
            embed.add_field(name="File", value=attachment.filename, inline=False)
            embed.add_field(
                name="Total Vectors in File", value=str(len(vectors_to_import)), inline=True
            )
            embed.add_field(name="Successfully Imported", value=str(success_count), inline=True)
            embed.add_field(name="Failed Imports", value=str(failure_count), inline=True)

            if errors:
                error_str = "\n".join(errors[:5])  # Show first few errors
                if len(errors) > 5:
                    error_str += f"\n... and {len(errors) - 5} more errors."
                embed.add_field(
                    name="Import Errors (Partial List)",
                    value=f"```\n{error_str}\n```",
                    inline=False,
                )

            await ctx.send(embed=embed)

        except json.JSONDecodeError:
            await ctx.send("❌ Invalid JSON file. Please check the file format.")
        except ValueError as e:  # Catch our validation errors
            await ctx.send(f"❌ Invalid data format in JSON file: {str(e)}")
        except discord.HTTPException as e:
            self.logger.error(f"Discord error downloading attachment: {e}", exc_info=True)
            await ctx.send("❌ Error downloading the attached file.")
        except WDBXClientError as e:
            self.logger.error(f"WDBX client error during batch import: {e}", exc_info=True)
            await ctx.send(f"❌ WDBX client error during import: {str(e)}")
        except NotImplementedError:
            await ctx.send("❌ Batch import operation is not supported by this WDBX instance.")
        except Exception as e:
            self.logger.error(f"Error during batch import: {e}", exc_info=True)
            await ctx.send(f"❌ An unexpected error occurred during import: {str(e)}")

    @batch.command(name="export", help="Export all vectors to a JSON file.")
    @commands.has_permissions(administrator=True)
    async def batch_export(self, ctx: commands.Context):
        """Export all vectors to a JSON file."""
        if not await self._ensure_wdbx_connection(ctx):
            return

        await ctx.send(
            "⏳ Exporting all vectors... This might take some time depending on the database size."
        )

        try:
            # Assuming client has an export_data or get_all_vectors method
            # This might need pagination or streaming for large datasets
            all_vectors = (
                await self.wdbx.export_data()
            )  # Example method, assuming it returns list of vector dicts

            if not all_vectors:
                await ctx.send("❌ No vectors found to export.")
                return

            # Serialize to JSON
            export_data = json.dumps(all_vectors, indent=2)
            export_bytes = export_data.encode("utf-8")

            # Send as file
            filename = f"wdbx_export_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            file = discord.File(io.BytesIO(export_bytes), filename=filename)

            await ctx.send(f"✅ Exported {len(all_vectors)} vectors.", file=file)

        except WDBXClientError as e:
            self.logger.error(f"WDBX client error during batch export: {e}", exc_info=True)
            await ctx.send(f"❌ WDBX client error during export: {str(e)}")
        except NotImplementedError:
            await ctx.send("❌ Batch export operation is not supported by this WDBX instance.")
        except Exception as e:
            self.logger.error(f"Error during batch export: {e}", exc_info=True)
            await ctx.send(f"❌ An unexpected error occurred during export: {str(e)}")

    @commands.command(name="stats", help="Show vector database statistics.")
    async def stats(self, ctx: commands.Context):
        """Show vector database statistics."""
        # This might be redundant with 'status', decide if it needs different info
        # For now, reuse the status command logic
        await self.status(ctx)

    @commands.command(name="config", help="Generate a template configuration file.")
    async def config(self, ctx: commands.Context):
        """Generate a template configuration file for the bot."""
        template_config = {
            "token": "YOUR_DISCORD_BOT_TOKEN",
            "prefix": "!",
            "status_channel_id": 123456789012345678,  # Optional: Channel ID for health alerts
            "admin_role_id": 123456789012345678,  # Optional: Role ID for admin commands
            "wdbx_host": "127.0.0.1",
            "wdbx_port": 8080,
            "monitoring_interval": 300,  # In seconds
            "log_dir": "logs",  # Optional: Directory for logs
            "max_vectors_display": 10,
            "allow_vector_deletion": False,
        }
        try:
            config_json = json.dumps(template_config, indent=2)
            await ctx.send(
                "Here is a template `config.json` file. Replace placeholder values.",
                file=discord.File(io.StringIO(config_json), filename="config_template.json"),
            )
        except Exception as e:
            self.logger.error(f"Error generating config template: {e}", exc_info=True)
            await ctx.send(f"❌ Error generating config template: {str(e)}")

    @commands.command(name="help_wdbx", help="Show available WDBX commands.")
    async def help_wdbx(self, ctx: commands.Context):
        """Show the custom help message for WDBX commands."""
        embed = create_help_embed()
        await ctx.send(embed=embed)

    # --- Background Tasks ---

    @tasks.loop(seconds=300)  # Default, configurable via config.monitoring_interval
    async def health_check_loop(self):
        """Periodically check WDBX health and report changes."""
        await self.bot.wait_until_ready()  # Ensure bot is ready before starting

        if not self.wdbx or not self.health_monitor or not self.config.status_channel_id:
            if not self.health_check_loop.has_failed():  # Log only once
                self.logger.warning(
                    "Health check loop cannot run: WDBX not connected, monitor not available, or status channel not set."
                )
            return  # Don't run if prerequisites aren't met

        try:
            health_status_dict = await self.health_monitor.check_health()
            current_status = health_status_dict.get("overall_status", {}).get(
                "status", HealthStatus.UNKNOWN if HealthStatus else "UNKNOWN"
            )

            if current_status != self.last_health_status:
                self.logger.info(
                    f"Health status changed from {self.last_health_status} to {current_status}"
                )
                channel = self.bot.get_channel(self.config.status_channel_id)
                if channel:
                    status_emoji = {"OK": "✅", "WARNING": "⚠️", "ERROR": "❌", "UNKNOWN": "❓"}.get(
                        current_status, "❓"
                    )

                    embed = discord.Embed(
                        title=f"{status_emoji} WDBX Health Status Change",
                        description=f"Overall status changed from **{self.last_health_status}** to **{current_status}**.",
                        color=(
                            discord.Color.orange()
                            if current_status == "WARNING"
                            else (
                                discord.Color.red()
                                if current_status == "ERROR"
                                else discord.Color.green()
                            )
                        ),
                    )
                    formatted_details = format_health_status(health_status_dict)
                    # Limit description length
                    if len(formatted_details) > 1800:
                        formatted_details = formatted_details[:1800] + "... (details truncated)"
                    embed.add_field(name="Details", value=formatted_details, inline=False)
                    embed.timestamp = datetime.datetime.utcnow()
                    try:
                        await channel.send(embed=embed)
                    except discord.Forbidden:
                        self.logger.error(
                            f"Bot lacks permissions to send messages in status channel {self.config.status_channel_id}"
                        )
                    except discord.HTTPException as e:
                        self.logger.error(
                            f"Failed to send health status update to channel {self.config.status_channel_id}: {e}"
                        )
                else:
                    self.logger.warning(
                        f"Status channel with ID {self.config.status_channel_id} not found."
                    )

                self.last_health_status = current_status  # Update last known status

        except Exception as e:
            self.logger.error(f"Error in health check loop: {e}", exc_info=True)
            # Optionally report error to status channel if it persists
            # Consider adding a backoff or disabling loop if errors are frequent

    @health_check_loop.before_loop
    async def before_health_check_loop(self):
        """Wait until the bot is ready before starting the loop."""
        await self.bot.wait_until_ready()
        # Update loop interval from config if needed
        if self.config.monitoring_interval != self.health_check_loop.seconds:
            self.health_check_loop.change_interval(seconds=self.config.monitoring_interval)
            self.logger.info(
                f"Health check interval updated to {self.config.monitoring_interval} seconds."
            )

    # --- Event Handlers ---

    @commands.Cog.listener()
    async def on_ready(self):
        """Called when the bot is ready and connected to Discord."""
        self.logger.info(f"Logged in as {self.bot.user.name} ({self.bot.user.id})")
        await self.bot.change_presence(
            activity=discord.Activity(type=discord.ActivityType.watching, name="WDBX | !help_wdbx")
        )

        # Attempt initial connection if not already connected
        if not self.wdbx:
            self.logger.info("Attempting initial connection to WDBX on ready...")
            self.wdbx = connect_to_wdbx(self.config.wdbx_host, self.config.wdbx_port)
            if self.wdbx and WDBX_AVAILABLE and HealthMonitor and not self.health_monitor:
                try:
                    self.health_monitor = HealthMonitor(self.wdbx)
                    self.logger.info("Health monitor initialized on ready.")
                except Exception as e:
                    self.logger.error(
                        f"Failed to initialize Health Monitor on ready: {e}", exc_info=True
                    )

    @commands.Cog.listener()
    async def on_command_error(self, ctx: commands.Context, error: commands.CommandError):
        """Handle errors that occur during command invocation."""
        if hasattr(ctx.command, "on_error"):
            return  # Don't interfere with custom error handlers

        error = getattr(error, "original", error)  # Get original error if wrapped

        if isinstance(error, commands.CommandNotFound):
            # Optionally suggest similar commands or ignore
            # await ctx.send(f"❓ Command not found. Try `!help_wdbx`.")
            return  # Avoid spamming for typos
        elif isinstance(error, commands.MissingRequiredArgument):
            await ctx.send(
                f"❌ Missing required argument: `{error.param.name}`. Use `!help {ctx.command.qualified_name}` for details."
            )
        elif isinstance(error, commands.BadArgument):
            await ctx.send(
                f"❌ Invalid argument provided. Use `!help {ctx.command.qualified_name}` for details."
            )
        elif isinstance(error, commands.CheckFailure):  # Catches has_permissions/has_role failures
            await ctx.send("🚫 You do not have permission to use this command.")
        elif isinstance(error, WDBXClientError):
            self.logger.error(
                f"WDBX Client Error in command '{ctx.command.qualified_name}': {error}",
                exc_info=True,
            )
            await ctx.send(f"❌ Error communicating with WDBX: {str(error)}")
        elif isinstance(error, DiscordBotError):  # Catch our custom bot errors
            self.logger.error(
                f"Discord Bot Error in command '{ctx.command.qualified_name}': {error}",
                exc_info=True,
            )
            await ctx.send(f"❌ Bot error: {str(error)}")
        else:
            # Generic error handler
            self.logger.error(
                f"Unhandled error in command '{ctx.command.qualified_name}': {error}", exc_info=True
            )
            await ctx.send(
                "🔧 An unexpected error occurred. Please check the logs or contact an administrator."
            )


def create_bot(config: BotConfig, wdbx_instance: Optional[Any] = None) -> commands.Bot:
    """Create and configure the Discord bot instance."""
    if not DISCORD_AVAILABLE:
        raise BotSetupError("discord.py is not installed. Cannot create bot.")

    intents = discord.Intents.default()
    intents.members = True  # Needed for role checks potentially
    intents.message_content = True  # Needed for reading command content

    bot = commands.Bot(
        command_prefix=config.prefix, intents=intents, help_command=None
    )  # Disable default help

    # Add the WDBX cog
    cog = WDBXCog(bot, config, wdbx_instance)
    asyncio.run(bot.add_cog(cog))  # Use asyncio.run if in sync context, else await
    # await bot.add_cog(cog) # If called from async context

    # Add a basic help command override if help_wdbx isn't sufficient
    @bot.command(name="help", hidden=True)
    async def custom_help(ctx: commands.Context, *, command_name: Optional[str] = None):
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


def run_bot(config: BotConfig, wdbx_instance: Optional[Any] = None):
    """Create and run the Discord bot."""
    setup_logging(config.log_dir)  # Ensure logging is setup
    logger = logging.getLogger("wdbx.discord.runner")  # Use runner specific logger

    try:
        bot = create_bot(config, wdbx_instance)
        logger.info(f"Starting Discord bot with prefix '{config.prefix}'...")
        bot.run(config.token)
    except discord.LoginFailure:
        logger.critical("Login failed: Invalid Discord bot token provided.")
        raise BotSetupError("Invalid Discord Bot Token") from None
    except BotSetupError as e:
        logger.critical(f"Bot setup failed: {e}")
        raise  # Reraise setup errors
    except Exception as e:
        logger.critical(f"An unexpected error occurred while running the bot: {e}", exc_info=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python discord_bot.py <path_to_config.json>")
        sys.exit(1)

    config_path = sys.argv[1]
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)

    try:
        with open(config_path, encoding="utf-8") as f:
            config_dict = json.load(f)
        bot_config = BotConfig(**config_dict)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in configuration file: {config_path}")
        sys.exit(1)
    except (TypeError, BotSetupError) as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred loading config: {e}")
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
    except BotSetupError:
        # Error already logged by run_bot
        sys.exit(1)
    except Exception as e:
        # Catch any other unexpected error during runtime
        main_logger.critical(f"Bot terminated due to unexpected error: {e}", exc_info=True)
        sys.exit(1)
