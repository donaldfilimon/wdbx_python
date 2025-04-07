#!/usr/bin/env python3
"""
Script to fix indentation and try/except issues in the Discord bot implementation.
"""

import re
import sys


def fix_discord_bot():
    """Fix linter errors in the Discord bot implementation."""
    try:
        # Read the file - use the correct relative path
        with open("discord_bot.py", encoding="utf-8") as file:
            content = file.read()

        # Fix the first set of linter errors: Unindent amount and return outside function
        # This is in the command decorators section
        content = re.sub(
            r"@commands\.has_permissions\(.*\)\n(\s+)def decorator\(func\):\n(\s+)return func\n(\s+)return decorator",
            r"@commands.has_permissions(\\1)\n\\1def decorator(func):\n\\1    return func\n\\1return decorator",
            content,
        )

        # Fix try statements without except clauses
        # Status command
        content = re.sub(
            r'@self\.command\(name="status", help="Show WDBX status"\)\n\s+async def status\(ctx\):\n\s+"""Show WDBX status information\."""\n\s+if not self\.wdbx:\n\s+await ctx\.send\("❌ Bot is not connected to WDBX instance\."\)\n\s+return\n\n\s+try:',
            r'@self.command(name="status", help="Show WDBX status")\n        async def status(ctx):\n            """Show WDBX status information."""\n            if not self.wdbx:\n                await ctx.send("❌ Bot is not connected to WDBX instance.")\n                return\n\n            try:',
            content,
        )

        # Health command
        content = re.sub(
            r'@self\.command\(name="health", help="Show detailed health information"\)\n\s+async def health\(ctx\):\n\s+"""Show detailed health information\."""\n\s+if not self\.wdbx or not self\.health_monitor:\n\s+await ctx\.send\("❌ Health monitoring is not available\."\)\n\s+return\n\n\s+try:',
            r'@self.command(name="health", help="Show detailed health information")\n        async def health(ctx):\n            """Show detailed health information."""\n            if not self.wdbx or not self.health_monitor:\n                await ctx.send("❌ Health monitoring is not available.")\n                return\n\n            try:',
            content,
        )

        # Search command
        content = re.sub(
            r'@self\.command\(name="search", help="Search for vectors"\)\n\s+async def search\(ctx, query: str, top_k: int = 5\):\n\s+"""\n\s+Search for vectors in WDBX\.\n\s+\n\s+Args:\n\s+query: Search query\n\s+top_k: Number of results to return\n\s+"""\n\s+if not self\.wdbx:\n\s+await ctx\.send\("❌ Bot is not connected to WDBX instance\."\)\n\s+return\n\s+\n\s+try:',
            r'@self.command(name="search", help="Search for vectors")\n        async def search(ctx, query: str, top_k: int = 5):\n            """\n            Search for vectors in WDBX.\n            \n            Args:\n                query: Search query\n                top_k: Number of results to return\n            """\n            if not self.wdbx:\n                await ctx.send("❌ Bot is not connected to WDBX instance.")\n                return\n            \n            try:',
            content,
        )

        # Fix the expected expression and unexpected indentation errors in the search command
        content = re.sub(
            r'except Exception as e:\n\s+logger\.error\(f"Error searching: \{e\}", exc_info=True\)\n\s+await ctx\.send\(f"❌ Error during search: \{str\(e\)\}"\)',
            r'            except Exception as e:\n                logger.error(f"Error searching: {e}", exc_info=True)\n                await ctx.send(f"❌ Error during search: {str(e)}")',
            content,
        )

        # Fix the visualize command
        content = re.sub(
            r'@self\.command\(name="visualize", help="Visualize vectors in 2D space"\)\n\s+async def visualize\(ctx, query: str = None, n_vectors: int = 20\):\n\s+"""\n\s+Visualize vectors in 2D space using PCA\.\n\s+\n\s+Args:\n\s+query: Optional search query to filter vectors\n\s+n_vectors: Number of vectors to visualize\n\s+"""\n\s+if not self\.wdbx:\n\s+await ctx\.send\("❌ Bot is not connected to WDBX instance\."\)\n\s+return\n\s+\n\s+if not VISUALIZATION_AVAILABLE:\n\s+await ctx\.send\("❌ Visualization libraries not available\. Install matplotlib, numpy, and scikit-learn\."\)\n\s+return\n\s+\n\s+try:',
            r'@self.command(name="visualize", help="Visualize vectors in 2D space")\n        async def visualize(ctx, query: str = None, n_vectors: int = 20):\n            """\n            Visualize vectors in 2D space using PCA.\n            \n            Args:\n                query: Optional search query to filter vectors\n                n_vectors: Number of vectors to visualize\n            """\n            if not self.wdbx:\n                await ctx.send("❌ Bot is not connected to WDBX instance.")\n                return\n                \n            if not VISUALIZATION_AVAILABLE:\n                await ctx.send("❌ Visualization libraries not available. Install matplotlib, numpy, and scikit-learn.")\n                return\n                \n            try:',
            content,
        )

        # Fix visualize command indentation issues
        content = re.sub(
            r"else:\n\s+label = str\(v\.vector_id\)\[.*\]\n\s+labels\.append\(label\)",
            r"                            else:\n                                label = str(v.vector_id)[:10]\n                            labels.append(label)",
            content,
        )

        # Fix the expected expression and unexpected indentation in visualize command
        content = re.sub(
            r'except Exception as e:\n\s+logger\.error\(f"Error visualizing vectors: \{e\}", exc_info=True\)\n\s+await ctx\.send\(f"❌ Error visualizing vectors: \{str\(e\)\}"\)',
            r'            except Exception as e:\n                logger.error(f"Error visualizing vectors: {e}", exc_info=True)\n                await ctx.send(f"❌ Error visualizing vectors: {str(e)}")',
            content,
        )

        # Fix the admin command
        content = re.sub(
            r'@self\.command\(name="admin", help="Administrative commands"\)\n\s+@commands\.has_permissions\(administrator=True\)\n\s+async def admin\(ctx, action: str, \*args\):\n\s+"""\n\s+Perform administrative actions\.\n\s+\n\s+Args:\n\s+action: Action to perform \(status, clear, optimize\)\n\s+args: Additional arguments for the action\n\s+"""\n\s+if not self\.wdbx:\n\s+await ctx\.send\("❌ Bot is not connected to WDBX instance\."\)\n\s+return\n\s+\n\s+try:',
            r'@self.command(name="admin", help="Administrative commands")\n        @commands.has_permissions(administrator=True)\n        async def admin(ctx, action: str, *args):\n            """\n            Perform administrative actions.\n            \n            Args:\n                action: Action to perform (status, clear, optimize)\n                args: Additional arguments for the action\n            """\n            if not self.wdbx:\n                await ctx.send("❌ Bot is not connected to WDBX instance.")\n                return\n                \n            try:',
            content,
        )

        # Fix the indentation in admin confirmations
        content = re.sub(
            r'else:\n\s+await ctx\.send\("❌ Clear operation not supported by WDBX instance\."\)',
            r'                            else:\n                                await ctx.send("❌ Clear operation not supported by WDBX instance.")',
            content,
        )

        # Fix the config command
        content = re.sub(
            r'@self\.command\(name="config", help="Generate a template configuration file"\)\n\s+async def config\(ctx\):\n\s+"""Generate a template configuration file for the bot\."""\n\s+try:',
            r'@self.command(name="config", help="Generate a template configuration file")\n        async def config(ctx):\n            """Generate a template configuration file for the bot."""\n            try:',
            content,
        )

        # Fix expected expression in config command
        content = re.sub(
            r'except Exception as e:\n\s+logger\.error\(f"Error generating config: \{e\}", exc_info=True\)\n\s+await ctx\.send\(f"❌ Error generating config: \{str\(e\)\}"\)',
            r'            except Exception as e:\n                logger.error(f"Error generating config: {e}", exc_info=True)\n                await ctx.send(f"❌ Error generating config: {str(e)}")',
            content,
        )

        # Remove duplicate example methods
        content = re.sub(
            r'def _register_commands\(self\):\n\s+"""Register bot commands\."""\n\s+\n\s+@self\.command\(name="status", help="Show WDBX status"\)\n\s+async def status\(ctx\):[^@]+?@self\.command\(name="health", help="Show detailed health information"\)',
            r'def _register_commands(self):\n        """Register bot commands."""\n        \n        @self.command(name="health", help="Show detailed health information")',
            content,
        )

        # Write the fixed content back to the file - use the correct relative path
        with open("discord_bot.py", "w", encoding="utf-8") as file:
            file.write(content)

        print("Successfully fixed linter errors in discord_bot.py")
        return True
    except Exception as e:
        print(f"Error fixing Discord bot: {str(e)}")
        return False


if __name__ == "__main__":
    success = fix_discord_bot()
    sys.exit(0 if success else 1)
