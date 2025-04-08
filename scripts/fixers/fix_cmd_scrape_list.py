#!/usr/bin/env python3
"""
Fix indentation issues in cmd_scrape_list function
"""


def fix_cmd_scrape_list():
    with open("wdbx_plugins/web_scraper.py", encoding="utf-8") as file:
        lines = file.readlines()

    # Find and fix the cmd_scrape_list function
    start_idx = None
    end_idx = None

    # First, find the function
    for i, line in enumerate(lines):
        if "def cmd_scrape_list(db: Any, args: str) -> None:" in line:
            start_idx = i
        elif start_idx is not None and "def cmd_scrape_status(db: Any, args: str) -> None:" in line:
            end_idx = i
            break

    if start_idx is None or end_idx is None:
        print("Could not find cmd_scrape_list function")
        return

    # Rewrite the function with proper indentation
    fixed_function = [
        "def cmd_scrape_list(db: Any, args: str) -> None:\n",
        '    """\n',
        "    List scraped URLs.\n",
        "\n",
        "    Args:\n",
        "        db: WDBX database instance\n",
        "        args: Optional domain filter\n",
        '    """\n',
        "    domain_filter = args.strip() if args else None\n",
        "\n",
        '    print("\\033[1;35mScraped URLs:\\033[0m")\n',
        "\n",
        "    # Get all entries from cache\n",
        "    scraped_items = []\n",
        "    for scrape_id, item in scraped_cache.items():\n",
        '        url = item.get("url", "")\n',
        '        metadata = item.get("metadata", {})\n',
        "\n",
        "        if domain_filter and domain_filter not in url:\n",
        "            continue\n",
        "\n",
        "        scraped_items.append({\n",
        '            "id": scrape_id,\n',
        '            "url": url,\n',
        '            "title": metadata.get("title", "N/A"),\n',
        '            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(item.get("timestamp", 0))),\n',
        '            "chunks": len(item.get("chunks", [])),\n',
        "        })\n",
        "\n",
        "    # Sort by timestamp (newest first)\n",
        '    scraped_items.sort(key=lambda x: x["time"], reverse=True)\n',
        "\n",
        "    if not scraped_items:\n",
        '        print("No scraped URLs found")\n',
        "        return\n",
        "\n",
        "    # Print as table\n",
        "    if domain_filter:\n",
        '        msg = f"Found {len(scraped_items)} scraped URLs for domain: {domain_filter}"\n',
        "    else:\n",
        '        msg = f"Found {len(scraped_items)} scraped URLs"\n',
        "    print(msg)\n",
        "\n",
        "    # Define table format\n",
        '    fmt = "{:<8} {:<40} {:<30} {:<20} {:<8}"\n',
        '    print(fmt.format("ID", "URL", "Title", "Scraped At", "Chunks"))\n',
        '    print("-" * 110)\n',
        "\n",
        "    for item in scraped_items:\n",
        "        # Truncate long fields\n",
        '        short_id = item["id"][:8]\n',
        '        short_url = item["url"][:40] + ("..." if len(item["url"]) > 40 else "")\n',
        '        short_title = item["title"][:30] + ("..." if len(item["title"]) > 30 else "")\n',
        "\n",
        "        print(fmt.format(\n",
        "            short_id,\n",
        "            short_url,\n",
        "            short_title,\n",
        '            item["time"],\n',
        '            item["chunks"]\n',
        "        ))\n",
        "\n",
        "\n",
    ]

    # Replace the function
    lines[start_idx:end_idx] = fixed_function

    with open("wdbx_plugins/web_scraper.py", "w", encoding="utf-8") as file:
        file.writelines(lines)

    print("Fixed cmd_scrape_list function indentation issues")


if __name__ == "__main__":
    fix_cmd_scrape_list()
