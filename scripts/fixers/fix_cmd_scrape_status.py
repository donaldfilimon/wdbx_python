#!/usr/bin/env python3
"""
Fix indentation issues in cmd_scrape_status function
"""


def fix_cmd_scrape_status():
    with open("wdbx_plugins/web_scraper.py", encoding="utf-8") as file:
        lines = file.readlines()

    # Find and fix the cmd_scrape_status function
    start_idx = None
    end_idx = None

    # First, find the function
    for i, line in enumerate(lines):
        if "def cmd_scrape_status(db: Any, args: str) -> None:" in line:
            start_idx = i
        elif start_idx is not None and "def cmd_scrape_search(db: Any, args: str) -> None:" in line:
            end_idx = i
            break

    if start_idx is None or end_idx is None:
        print("Could not find cmd_scrape_status function")
        return

    # Rewrite the function with proper indentation
    fixed_function = [
        "def cmd_scrape_status(db: Any, args: str) -> None:\n",
        '    """\n',
        "    Show detailed status of a scraped document.\n",
        "\n",
        "    Args:\n",
        "        db: WDBX database instance\n",
        "        args: Scrape ID\n",
        '    """\n',
        "    if not args:\n",
        '        print("\\033[1;31mError: Scrape ID required\\033[0m")\n',
        '        print("Usage: scrape:status <id>")\n',
        "        return\n",
        "\n",
        "    scrape_id = args.strip()\n",
        "\n",
        "    # Try partial match if exact ID not found\n",
        "    if scrape_id not in scraped_cache:\n",
        "        matches = [id for id in scraped_cache if id.startswith(scrape_id)]\n",
        "        if len(matches) == 1:\n",
        "            scrape_id = matches[0]\n",
        "        elif len(matches) > 1:\n",
        '            print(f"\\033[1;31mMultiple matches found for ID: {scrape_id}\\033[0m")\n',
        "            for match in matches:\n",
        '                print(f"  {match}")\n',
        "            return\n",
        "\n",
        "    if scrape_id not in scraped_cache:\n",
        '        print(f"\\033[1;31mNo scrape found with ID: {scrape_id}\\033[0m")\n',
        "        return\n",
        "\n",
        "    item = scraped_cache[scrape_id]\n",
        '    metadata = item.get("metadata", {})\n',
        '    chunks = item.get("chunks", [])\n',
        "\n",
        '    print(f"\\033[1;35mScrape Status for ID: {scrape_id}\\033[0m")\n',
        "    print(f\"\\033[1mURL:\\033[0m {item.get('url', 'N/A')}\")\n",
        "    print(f\"\\033[1mTitle:\\033[0m {metadata.get('title', 'N/A')}\")\n",
        "    print(\n",
        "        f\"\\033[1mScraped at:\\033[0m {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(item.get('timestamp', 0)))}\")\n",
        '    print(f"\\033[1mChunks:\\033[0m {len(chunks)}")\n',
        "\n",
        "    # Print description if available\n",
        '    if "description" in metadata:\n',
        "        print(f\"\\033[1mDescription:\\033[0m {metadata['description']}\")\n",
        "\n",
        "    # Print response info\n",
        '    if "response_info" in metadata:\n',
        '        resp = metadata["response_info"]\n',
        '        print("\\n\\033[1mResponse Info:\\033[0m")\n',
        "        print(f\"  Status: {resp.get('status_code', 'N/A')}\")\n",
        "        print(f\"  Content Type: {resp.get('content_type', 'N/A')}\")\n",
        "        print(f\"  Size: {resp.get('size_bytes', 0) / 1024:.1f} KB\")\n",
        "        print(f\"  Fetch Time: {resp.get('fetch_time_sec', 0):.2f} seconds\")\n",
        "\n",
        "    # Print extracted metadata\n",
        '    print("\\n\\033[1mMetadata:\\033[0m")\n',
        "    for key, value in metadata.items():\n",
        '        if key not in ["response_info", "links", "scrape_id", "chunk_count"]:\n',
        "            if isinstance(value, str) and len(value) > 100:\n",
        '                value = value[:100] + "..."\n',
        '            print(f"  {key}: {value}")\n',
        "\n",
        "    # Print chunk previews\n",
        '    print("\\n\\033[1mContent Chunks:\\033[0m")\n',
        "    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks\n",
        '        print(f"  Chunk {i + 1}/{len(chunks)}: {chunk[:100]}...")\n',
        "\n",
        "    if len(chunks) > 3:\n",
        '        print(f"  ... and {len(chunks) - 3} more chunks")\n',
        "\n",
        "    # Print HTML path if available\n",
        '    if "html_path" in metadata:\n',
        "        print(f\"\\n\\033[1mHTML File:\\033[0m {metadata['html_path']}\")\n",
        "\n",
        "    # Print link stats if available\n",
        '    if "links" in metadata:\n',
        '        links = metadata["links"]\n',
        '        print(f"\\n\\033[1mLinks:\\033[0m {len(links)} extracted")\n',
        "\n",
        "        # Group links by domain\n",
        "        domains = {}\n",
        "        for link in links:\n",
        "            domain = _get_domain(link)\n",
        "            domains[domain] = domains.get(domain, 0) + 1\n",
        "\n",
        "        # Print top domains\n",
        '        print("  Top domains:")\n',
        "        for domain, count in sorted(domains.items(), key=lambda x: x[1], reverse=True)[:5]:\n",
        '            print(f"    {domain}: {count} links")\n',
        "\n",
        "\n",
    ]

    # Replace the function
    lines[start_idx:end_idx] = fixed_function

    with open("wdbx_plugins/web_scraper.py", "w", encoding="utf-8") as file:
        file.writelines(lines)

    print("Fixed cmd_scrape_status function indentation issues")


if __name__ == "__main__":
    fix_cmd_scrape_status()
