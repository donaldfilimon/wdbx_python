#!/usr/bin/env python3
"""
Fix indentation issues in cmd_scrape_config function
"""


def fix_cmd_scrape_config():
    with open("wdbx_plugins/web_scraper.py", encoding="utf-8") as file:
        lines = file.readlines()

    # Find and fix the cmd_scrape_config function
    start_idx = None
    end_idx = None

    # First, find the function
    for i, line in enumerate(lines):
        if "def cmd_scrape_config(db: Any, args: str) -> None:" in line:
            start_idx = i
        elif start_idx is not None and "def cmd_scrape_url(db: Any, args: str) -> None:" in line:
            end_idx = i
            break

    if start_idx is None or end_idx is None:
        print("Could not find cmd_scrape_config function")
        return

    # Rewrite the function with proper indentation
    fixed_function = [
        "def cmd_scrape_config(db: Any, args: str) -> None:\n",
        '    """\n',
        "    Configure web scraper settings.\n",
        "\n",
        "    Args:\n",
        "        db: WDBX database instance\n",
        "        args: Configuration string in format key=value\n",
        '    """\n',
        '    print("\\033[1;35mWDBX Web Scraper Configuration\\033[0m")\n',
        "\n",
        "    if not args:\n",
        '        print("Current configuration:")\n',
        "        _print_config()\n",
        '        print("\\nTo change a setting, use: scrape:config key=value")\n',
        "        return\n",
        "\n",
        "    # Parse key=value pairs\n",
        "    updated_any = False\n",
        "    parts = args.split()\n",
        "    for part in parts:\n",
        '        if "=" in part:\n',
        '            key, value = part.split("=", 1)\n',
        "            if _update_config_value(key, value):\n",
        "                updated_any = True\n",
        "        else:\n",
        '            print(f"\\033[1;31mInvalid format: {part}. Use key=value format.\\033[0m")\n',
        "\n",
        "    if updated_any:\n",
        "        try:\n",
        "            _save_scraper_config()\n",
        "\n",
        "            # Ensure HTML directory exists if needed\n",
        "            if scraper_config.save_html:\n",
        "                _ensure_html_dir()\n",
        "\n",
        "            # Update rate limiter instance if settings changed\n",
        "            global rate_limiter\n",
        "            rate_limiter = RateLimiter(\n",
        "                scraper_config.requests_per_minute,\n",
        "                scraper_config.concurrent_requests\n",
        "            )\n",
        '            logger.info("Updated rate limiter settings")\n',
        "\n",
        "        except ConfigurationError as e:\n",
        '            logger.error(f"Error saving configuration: {e}")\n',
        '            print(f"\\033[1;31mError saving configuration: {e}\\033[0m")\n',
        "\n",
        "\n",
    ]

    # Replace the function
    lines[start_idx:end_idx] = fixed_function

    # Fix cmd_scrape_url function's duplicated lines
    for i in range(end_idx, len(lines)):
        if 'print("Usage: scrape:url <url>")' in lines[i]:
            # Found first instance, look for duplicates on next lines
            if i + 1 < len(lines) and 'print("Usage: scrape:url <url>")' in lines[i + 1]:
                # Remove the duplicate line
                lines[i] = ""
                break

    # Also fix the unterminated string in the "respect_robots_txt" lines
    for i in range(len(lines)):
        if (
            'print("You can disable robots.txt checking with: scrape:config' in lines[i]
            and "=" not in lines[i]
        ):
            lines[i] = (
                '        print("You can disable robots.txt checking with: scrape:config respect_robots_txt=false")\n'
            )

    with open("wdbx_plugins/web_scraper.py", "w", encoding="utf-8") as file:
        file.writelines(lines)

    print("Fixed cmd_scrape_config function indentation issues")


if __name__ == "__main__":
    fix_cmd_scrape_config()
