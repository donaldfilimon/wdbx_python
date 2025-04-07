# WDBX Discord Bot Plugin

This plugin provides a Discord bot interface for WDBX vector database, allowing users to interact with and monitor the database through Discord.

## Features

- **Real-time Monitoring**: Monitor database health with automatic alerts when issues are detected
- **Vector Search**: Search for similar vectors directly from Discord
- **Vector Visualization**: Visualize vectors in 2D space using PCA
- **Administrative Commands**: Perform administrative tasks like memory optimization
- **Batch Operations**: Import and export vectors through Discord
- **Statistics & Metrics**: View detailed system statistics and metrics

## Requirements

- Python 3.8+
- discord.py
- matplotlib, numpy, scikit-learn (for visualization)
- WDBX client library

## Installation

1. Install the required dependencies:

```bash
pip install discord.py matplotlib numpy scikit-learn
```

2. Place the `discord_bot.py` file in your WDBX plugins directory.

## Configuration

Create a configuration file named `config.json` with the following structure:

```json
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
```

Parameters:
- `token`: Your Discord bot token (required)
- `prefix`: Command prefix for the bot (default: `!`)
- `status_channel_id`: Channel ID for status updates and alerts
- `admin_role_id`: Role ID for administrative commands
- `wdbx_host`: Host address of the WDBX server (default: `127.0.0.1`)
- `wdbx_port`: Port of the WDBX server (default: `8080`)
- `monitoring_interval`: Health check interval in seconds (default: `300`)
- `log_dir`: Directory for log files (optional)
- `max_vectors_display`: Maximum vectors to show in search results (default: `10`)
- `allow_vector_deletion`: Whether to allow vector deletion commands (default: `false`)

## Usage

### Starting the Bot

```bash
python discord_bot.py config.json
```

### Available Commands

- `!status` - Show WDBX status
- `!health` - Show detailed health information
- `!search <query> [top_k]` - Search for vectors
- `!metrics` - Show system metrics
- `!visualize [query] [n_vectors]` - Visualize vectors in 2D space
- `!admin <action> [args]` - Administrative commands
  - `!admin status` - Show advanced status
  - `!admin clear` - Clear all vectors (requires confirmation)
  - `!admin optimize` - Optimize memory usage
- `!batch <operation> [args]` - Batch operations
  - `!batch import` - Import vectors from JSON file
  - `!batch export` - Export vectors to JSON file
- `!stats` - Show vector statistics
- `!config` - Generate a template configuration file
- `!help_wdbx` - Show all available WDBX-specific commands

## Integration with Existing WDBX Systems

The Discord bot can be integrated with any existing WDBX system by using the WDBX client library. You can create a custom script that connects to your WDBX instance and starts the Discord bot:

```python
from wdbx.client import WDBXClient
from wdbx_plugins.discord_bot import create_bot, BotConfig

# Connect to WDBX
client = WDBXClient()
client.connect(host="127.0.0.1", port=8080)

# Configure bot
config = BotConfig(
    token="YOUR_DISCORD_BOT_TOKEN",
    prefix="!",
    wdbx_host="127.0.0.1",
    wdbx_port=8080
)

# Create and run bot
bot = create_bot(config, client)
bot.run_bot()
```

## Recent Updates

- Fixed indentation issues and improved error handling throughout the code
- Updated metrics display to use actual data from Prometheus
- Enhanced file operations with proper error handling
- Fixed async/await patterns for better performance
- Improved help command to show all available commands
- Added proper rate limiting for command execution
- Optimized memory usage when handling large vector collections

## Support

For issues and feature requests, please open an issue in the WDBX repository or contact the maintainers.

## License

This plugin is distributed under the same license as the main WDBX project. 