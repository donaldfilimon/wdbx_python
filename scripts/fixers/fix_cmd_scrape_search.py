#!/usr/bin/env python3
"""
Fix indentation issues in cmd_scrape_search function
"""


def fix_cmd_scrape_search():
    with open("wdbx_plugins/web_scraper.py", encoding="utf-8") as file:
        lines = file.readlines()

    # Find the cmd_scrape_search function
    start_line = None
    for i, line in enumerate(lines):
        if "def cmd_scrape_search(db: Any, args: str) -> None:" in line:
            start_line = i
            break

    if start_line is None:
        print("Could not find cmd_scrape_search function")
        return

    # Find the end of the function (next function definition or end of file)
    end_line = None
    for i in range(start_line + 1, len(lines)):
        if lines[i].strip().startswith("def "):
            end_line = i
            break

    if end_line is None:
        end_line = len(lines)  # If no next function, use end of file

    print(f"Found cmd_scrape_search function from line {start_line+1} to {end_line}")

    # Rewrite the function with proper indentation
    fixed_function = [
        "def cmd_scrape_search(db: Any, args: str) -> None:\n",
        '    """\n',
        "    Search scraped content.\n",
        "\n",
        "    Args:\n",
        "        db: WDBX database instance\n",
        "        args: Search query\n",
        '    """\n',
        "    if not args:\n",
        '        print("\\033[1;31mError: Search query required\\033[0m")\n',
        '        print("Usage: scrape:search <query>")\n',
        "        return\n",
        "\n",
        "    query = args.strip()\n",
        '    print(f"\\033[1;35mSearching scraped content for: {query}\\033[0m")\n',
        "\n",
        "    # Simple text search in cache first\n",
        "    results = []\n",
        "    query_lower = query.lower()\n",
        "\n",
        "    for scrape_id, item in scraped_cache.items():\n",
        '        url = item.get("url", "")\n',
        '        metadata = item.get("metadata", {})\n',
        '        chunks = item.get("chunks", [])\n',
        "\n",
        "        # Search in title and description\n",
        '        title = metadata.get("title", "").lower()\n',
        '        description = metadata.get("description", "").lower()\n',
        "\n",
        "        title_match = query_lower in title\n",
        "        desc_match = query_lower in description\n",
        "\n",
        "        # Search in chunks\n",
        "        matched_chunks = []\n",
        "        for i, chunk in enumerate(chunks):\n",
        "            if query_lower in chunk.lower():\n",
        "                # Get context around the match\n",
        "                index = chunk.lower().find(query_lower)\n",
        "                start = max(0, index - 50)\n",
        "                end = min(len(chunk), index + len(query) + 50)\n",
        "                context = chunk[start:end]\n",
        "\n",
        "                # Replace the query with highlighted version\n",
        "                highlight = context.replace(\n",
        "                    query,\n",
        '                    f"\\033[1;33m{query}\\033[0m",\n',
        "                    1  # Only replace first occurrence\n",
        "                )\n",
        "\n",
        "                matched_chunks.append({\n",
        '                    "index": i,\n',
        '                    "highlight": highlight\n',
        "                })\n",
        "\n",
        "        if title_match or desc_match or matched_chunks:\n",
        "            results.append({\n",
        '                "id": scrape_id,\n',
        '                "url": url,\n',
        '                "title": metadata.get("title", "N/A"),\n',
        '                "title_match": title_match,\n',
        '                "desc_match": desc_match,\n',
        '                "chunks": matched_chunks\n',
        "            })\n",
        "\n",
        "    # Sort results: title matches first, then description matches, then by number of chunk matches\n",
        '    results.sort(key=lambda x: (not x["title_match"], not x["desc_match"], -len(x["chunks"])))\n',
        "\n",
        "    if not results:\n",
        '        print("No matches found in scraped content")\n',
        "\n",
        "        # Suggest using vector search if available\n",
        '        print("\\nFor semantic search, use the database search capabilities or model:embed")\n',
        "        return\n",
        "\n",
        "    # Print results\n",
        '    print(f"Found {len(results)} matches:")\n',
        "\n",
        "    for i, result in enumerate(results[:10]):  # Show top 10 results\n",
        "        print(f\"\\n\\033[1m{i + 1}. {result['title']}\\033[0m\")\n",
        "        print(f\"   URL: {result['url']}\")\n",
        "        print(f\"   ID: {result['id']}\")\n",
        "\n",
        "        # Print chunk matches\n",
        '        for j, chunk in enumerate(result["chunks"][:3]):  # Show top 3 chunk matches\n',
        "            print(f\"   Match {j + 1}: ...{chunk['highlight']}...\")\n",
        "\n",
        '        if len(result["chunks"]) > 3:\n',
        "            print(f\"   ... and {len(result['chunks']) - 3} more matches\")\n",
        "\n",
        "    if len(results) > 10:\n",
        '        print(f"\\n... and {len(results) - 10} more results")\n',
        "\n",
        '    print("\\nTo see details for a specific result, use: scrape:status <id>")\n',
        "\n",
    ]

    # Replace the function
    lines[start_line:end_line] = fixed_function

    with open("wdbx_plugins/web_scraper.py", "w", encoding="utf-8") as file:
        file.writelines(lines)

    print("Fixed cmd_scrape_search function indentation issues")


if __name__ == "__main__":
    fix_cmd_scrape_search()
