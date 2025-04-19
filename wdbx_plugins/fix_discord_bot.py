#!/usr/bin/env python
"""Fix linter issues in discord_bot.py and structure code properly."""
import os
import re
import sys
import logging
import argparse
from pathlib import Path  # Remove if not needed after fixes

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('fix_discord_bot')


def fix_discord_bot_issues(discord_bot_path):
    """
    Fix common linter issues in discord_bot.py.

    Args:
        discord_bot_path: Path to the discord_bot.py file.
    """
    try:
        if not os.path.exists(discord_bot_path):
            logger.error("File not found: %s", discord_bot_path)  # Use lazy formatting
            return False

        logger.info("Fixing linter issues in %s", discord_bot_path)  # Use lazy formatting

        with open(discord_bot_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Fix 1: Add proper indentation to mock commands class
        mock_class_pattern = r'(class commands:.*?)(class tasks:)'
        mock_class_content = re.search(mock_class_pattern, content, re.DOTALL)

        if mock_class_content:
            old_commands_class = mock_class_content.group(1)

            # Add static methods and fix indentation
            new_commands_class = re.sub(
                r'(\s+)def (command|group|has_permissions)\(([^)]*)\):',
                r'\1@staticmethod\n\1def \2(\3):',
                old_commands_class
            )

            # Fix the Cog subclass with listener method
            new_commands_class = re.sub(
                r'(\s+)(class Cog:)',
                r'\1\2\n\1    @staticmethod\n\1    def listener():',
                new_commands_class
            )

            content = content.replace(old_commands_class, new_commands_class)

        # Fix 2: Add static method to tasks.loop
        tasks_class_pattern = r'(class tasks:.*?)(\s+class|def|\s*$)'
        tasks_class_content = re.search(tasks_class_pattern, content, re.DOTALL)

        if tasks_class_content:
            old_tasks_class = tasks_class_content.group(1)

            # Add static methods to tasks.loop
            new_tasks_class = re.sub(
                r'(\s+)def (loop)\(([^)]*)\):',
                r'\1@staticmethod\n\1def \2(\3):',
                old_tasks_class
            )

            content = content.replace(old_tasks_class, new_tasks_class)

        # Fix 3: Fix the connect_to_wdbx method to remove keyword arguments
        connect_pattern = r'(def connect_to_wdbx.*?)client\.connect\(host=host, port=port\)(.*?\n)'
        content = re.sub(connect_pattern, r'\1client.connect(host, port)\2', content)

        # Fix 4: Fix exception handling order in batch import
        batch_import_pattern = r'(except NotImplementedError:.*?)(except WDBXClientError as e:.*?)(except Exception as e:)'
        content = re.sub(batch_import_pattern, r'\2\1\3', content, flags=re.DOTALL)

        # Fix 5: Fix the redefined logger in run_bot function
        run_bot_pattern = r'(def run_bot.*?\n\s+setup_logging.*?\n\s+)logger = (.*?)\n'
        content = re.sub(run_bot_pattern, r'\1run_logger = \2\n', content)

        # Update all logger references in run_bot function
        run_bot_logger_pattern = r'(def run_bot.*?)logger\.(.*?\n.*?)(def|\s*$)'
        content = re.sub(run_bot_logger_pattern, r'\1run_logger.\2\3', content, flags=re.DOTALL)

        # Fix 6: Add health monitor null checks
        health_check_pattern = r'(health_status = await self\.health_monitor\.check_health\(\))'
        health_check_replacement = r'''if hasattr(self.health_monitor, 'check_health'):
            health_status = await self.health_monitor.check_health()
        else:
            # Mock response if check_health doesn't exist
            health_status = {
                "overall_status": {"status": "UNKNOWN", "message": "Health check not available"},
                "mock_check": {"status": "UNKNOWN", "message": "Mock health status"}
            }'''
        content = re.sub(health_check_pattern, health_check_replacement, content)

        # Fix 7: Convert all logger.info calls using multiple arguments to f-strings
        content = re.sub(
            r'logger\.info\("([^"]+)",\s*([^)]+)\)',
            r'logger.info(f"\1".format(\2))',
            content
        )

        # Fix 8: Convert all logger.critical calls using multiple arguments to f-strings
        content = re.sub(
            r'logger\.critical\("([^"]+)",\s*([^)]+)\)',
            r'logger.critical(f"\1".format(\2))',
            content
        )

        # Fix 9: Convert format strings to f-strings in the main section
        content = re.sub(
            r'print\("Error: \{\}".format\(([^)]+)\)\)',
            r'print(f"Error: {\1}")',
            content
        )

        # Fix 10: Fix the connect_to_wdbx logging
        connect_to_wdbx_logging = r'(def connect_to_wdbx.*?)logger\.info\("Successfully connected to WDBX at %s:%s", host, port\)(.*?\n)'
        content = re.sub(connect_to_wdbx_logging,
                         r'\1logger.info(f"Successfully connected to WDBX at {host}:{port}")\2', content, flags=re.DOTALL)

        # Fix 11: Fix error handling in connect_to_wdbx
        error_connect_pattern = r'(def connect_to_wdbx.*?except.*?)logger\.error\("Failed to connect to WDBX at %s:%s: %s", host, port, str\(e\)\)(.*?\n)'
        content = re.sub(error_connect_pattern,
                         r'\1error_msg = f"Failed to connect to WDBX at {host}:{port}: {str(e)}"\n        logger.error(error_msg)\2', content, flags=re.DOTALL)

        # Fix 12: Fix run_bot logging for critical errors with multiple args
        run_bot_error_pattern = r'(def run_bot.*?except.*?else:.*?)logger\.critical\("An unexpected error occurred while running the bot: %s", e\)(.*?\n)'
        content = re.sub(run_bot_error_pattern, r'\1error_msg = "An unexpected error occurred while running the bot: " + str(e)\n            logger.critical(error_msg, exc_info=True)\2', content, flags=re.DOTALL)

        # Write the updated content back to the file
        with open(discord_bot_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info("Successfully fixed linter issues in %s",
                    discord_bot_path)  # Use lazy formatting
        return True
    except Exception as e:
        # Add specific exception types when possible
        logger.error("Error fixing linter issues: %s", str(e), exc_info=True)  # Use lazy formatting
        return False


def main():
    """Run the fixer script with command line arguments."""
    parser = argparse.ArgumentParser(description='Fix linter issues in discord_bot.py')
    parser.add_argument('--path', type=str, default='src/wdbx/plugins/discord_bot.py',
                        help='Path to discord_bot.py file (default: src/wdbx/plugins/discord_bot.py)')

    args = parser.parse_args()

    if fix_discord_bot_issues(args.path):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
