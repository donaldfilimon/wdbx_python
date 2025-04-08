#!/usr/bin/env python3
"""
Fix indentation issues in cmd_scrape_url function
"""


def fix_cmd_scrape_url():
    with open("wdbx_plugins/web_scraper.py", encoding="utf-8") as file:
        lines = file.readlines()

    # Find and fix the cmd_scrape_url function
    start_idx = None
    end_idx = None

    # First, find the function
    for i, line in enumerate(lines):
        if "def cmd_scrape_url(db: Any, args: str) -> None:" in line:
            start_idx = i
        elif start_idx is not None and "def cmd_scrape_site(db: Any, args: str) -> None:" in line:
            end_idx = i
            break

    if start_idx is None or end_idx is None:
        print("Could not find cmd_scrape_url function")
        return

    # Rewrite the function with proper indentation
    fixed_function = [
        "def cmd_scrape_url(db: Any, args: str) -> None:\n",
        '    """\n',
        "    Scrape a single URL.\n",
        "\n",
        "    Args:\n",
        "        db: WDBX database instance\n",
        "        args: URL to scrape\n",
        '    """\n',
        "    if not args:\n",
        '        print("\\033[1;31mError: URL required\\033[0m")\n',
        '        print("Usage: scrape:url <url>")\n',
        "        return\n",
        "\n",
        "    url = args.strip()\n",
        '    print(f"\\033[1;35mScraping URL: {url}\\033[0m")\n',
        "\n",
        "    try:\n",
        "        # Normalize URL to ensure proper format\n",
        "        normalized_url = _normalize_url(url)\n",
        "        if normalized_url != url:\n",
        '            print(f"Normalized URL: {normalized_url}")\n',
        "\n",
        "        scrape_id = scrape_url(db, normalized_url)\n",
        "\n",
        "        if scrape_id:\n",
        '            print(f"\\033[1;32mSuccessfully scraped URL: {normalized_url}\\033[0m")\n',
        '            print(f"Scrape ID: {scrape_id}")\n',
        "\n",
        "            # Print basic stats\n",
        "            if scrape_id in scraped_cache:\n",
        "                cache_entry = scraped_cache[scrape_id]\n",
        '                metadata = cache_entry.get("metadata", {})\n',
        '                chunks = cache_entry.get("chunks", [])\n',
        "\n",
        "                print(f\"Title: {metadata.get('title', 'N/A')}\")\n",
        "                print(f\"Description: {metadata.get('description', 'N/A')[:100]}...\")\n",
        '                print(f"Content chunks: {len(chunks)}")\n',
        '                print(f"Content size: {sum(len(c) for c in chunks)} characters")\n',
        "\n",
        '                if "links" in metadata:\n',
        "                    print(f\"Links: {len(metadata['links'])} extracted\")\n",
        "        else:\n",
        '            print(f"\\033[1;31mFailed to scrape URL: {normalized_url} (No ID returned)\\033[0m")\n',
        "            print(\"Check logs for more details. Use 'scrape:config' to adjust settings if needed\")\n",
        "\n",
        "    except FetchRobotsError as e:\n",
        '        logger.error(f"Robots.txt disallowed: {e}")\n',
        '        print(f"\\033[1;31mCannot scrape URL: {e.url} - Blocked by robots.txt\\033[0m")\n',
        '        print("You can disable robots.txt checking with: scrape:config respect_robots_txt=false")\n',
        "\n",
        "    except FetchRateLimitError as e:\n",
        '        logger.error(f"Rate limit error: {e}")\n',
        '        print(f"\\033[1;31mRate limit detected for {e.url}. Please try again later.\\033[0m")\n',
        '        print("You can decrease request frequency with: scrape:config requests_per_minute=<lower_value>")\n',
        "\n",
        "    except FetchTimeoutError as e:\n",
        '        logger.error(f"Timeout error: {e}")\n',
        '        print(f"\\033[1;31mTimeout error fetching {e.url}\\033[0m")\n',
        '        print("You can increase timeout with: scrape:config timeout=<higher_value>")\n',
        "\n",
        "    except FetchError as e:\n",
        '        logger.error(f"Fetch error: {e}")\n',
        '        print(f"\\033[1;31mError fetching {e.url}: {e.message}\\033[0m")\n',
        '        if hasattr(e, "status_code") and e.status_code:\n',
        '            print(f"HTTP Status Code: {e.status_code}")\n',
        "\n",
        "    except ScrapingError as e:\n",
        '        logger.error(f"Scraping error: {e}")\n',
        '        print(f"\\033[1;31mError during scraping: {e}\\033[0m")\n',
        "\n",
        "    except StorageError as e:\n",
        '        logger.error(f"Storage error: {e}")\n',
        '        print(f"\\033[1;31mError storing scraped content: {e}\\033[0m")\n',
        "\n",
        "    except WebScraperError as e:\n",
        '        logger.error(f"Web scraper error: {e}")\n',
        '        print(f"\\033[1;31mWeb scraper error: {e}\\033[0m")\n',
        "\n",
        "    except Exception as e:\n",
        '        logger.error(f"Unexpected error scraping {url}: {e}", exc_info=True)\n',
        '        print(f"\\033[1;31mAn unexpected error occurred: {e}\\033[0m")\n',
        '        print("Please check logs for more details.")\n',
        "\n",
        "\n",
    ]

    # Replace the function
    lines[start_idx:end_idx] = fixed_function

    # Remove duplicated print statement for respect_robots_txt
    for i in range(len(lines)):
        if (
            'print("You can disable robots.txt checking with: scrape:config respect_robots_txt=false")'
            in lines[i]
        ):
            # Check next line for duplicate
            if (
                i + 1 < len(lines)
                and 'print("You can disable robots.txt checking with: scrape:config respect_robots_txt=false")'
                in lines[i + 1]
            ):
                lines[i + 1] = ""  # Remove the duplicate line

    with open("wdbx_plugins/web_scraper.py", "w", encoding="utf-8") as file:
        file.writelines(lines)

    print("Fixed cmd_scrape_url function indentation issues")


if __name__ == "__main__":
    fix_cmd_scrape_url()
