#!/usr/bin/env python3
"""
Fix indentation issues in cmd_scrape_site function
"""


def fix_cmd_scrape_site():
    with open("wdbx_plugins/web_scraper.py", encoding="utf-8") as file:
        lines = file.readlines()

    # Find and fix the cmd_scrape_site function
    start_idx = None
    end_idx = None

    # First, find the function
    for i, line in enumerate(lines):
        if "def cmd_scrape_site(db: Any, args: str) -> None:" in line:
            start_idx = i
        elif start_idx is not None and "def cmd_scrape_list(db: Any, args: str) -> None:" in line:
            end_idx = i
            break

    if start_idx is None or end_idx is None:
        print("Could not find cmd_scrape_site function")
        return

    # Rewrite the function with proper indentation
    fixed_function = [
        "def cmd_scrape_site(db: Any, args: str) -> None:\n",
        '    """\n',
        "    Scrape a website starting from the given URL.\n",
        "\n",
        "    Args:\n",
        "        db: WDBX database instance\n",
        "        args: Start URL and optional max_pages\n",
        '    """\n',
        "    if not args:\n",
        '        print("\\033[1;31mError: URL required\\033[0m")\n',
        '        print("Usage: scrape:site <url> [max_pages]")\n',
        "        return\n",
        "\n",
        "    parts = args.strip().split()\n",
        "    url = parts[0]\n",
        "\n",
        "    max_pages = scraper_config.max_pages_per_domain\n",
        "    if len(parts) > 1:\n",
        "        try:\n",
        "            max_pages_arg = int(parts[1])\n",
        "            if max_pages_arg > 0:\n",
        "                max_pages = max_pages_arg\n",
        "            else:\n",
        "                print(\n",
        '                    f"\\033[1;33mWarning: max_pages must be positive, using default ({max_pages})\\033[0m")\n',
        "        except ValueError:\n",
        "            print(\n",
        '                f"\\033[1;31mError: max_pages must be an integer, using default ({max_pages})\\033[0m")\n',
        "\n",
        "    # Normalize URL to ensure proper format\n",
        "    normalized_url = _normalize_url(url)\n",
        "    if normalized_url != url:\n",
        '        print(f"Normalized URL: {normalized_url}")\n',
        "\n",
        '    print(f"\\033[1;35mCrawling site: {normalized_url} (max {max_pages} pages)\\033[0m")\n',
        "\n",
        "    try:\n",
        "        domain = _get_domain(normalized_url)\n",
        '        print(f"Domain: {domain}")\n',
        "\n",
        "        scraped_ids = scrape_site(db, normalized_url, max_pages)\n",
        "\n",
        "        if scraped_ids:\n",
        '            print(f"\\033[1;32mSuccessfully crawled site: {normalized_url}\\033[0m")\n',
        '            print(f"Scraped {len(scraped_ids)} pages")\n',
        "\n",
        "            # Print a summary of scraped pages\n",
        "            if len(scraped_ids) > 0:\n",
        '                print("\\nScraped pages summary:")\n',
        "                for i, scrape_id in enumerate(scraped_ids[:5]):  # Show first 5\n",
        "                    if scrape_id in scraped_cache:\n",
        "                        item = scraped_cache[scrape_id]\n",
        "                        print(f\"{i + 1}. {item.get('url', 'N/A')} ({len(item.get('chunks', []))}) chunks\")\n",
        "\n",
        "                if len(scraped_ids) > 5:\n",
        '                    print(f"...and {len(scraped_ids) - 5} more pages")\n',
        "\n",
        "                print(\"\\nUse 'scrape:list' to see all scraped URLs\")\n",
        "        else:\n",
        '            print(f"\\033[1;31mFailed to crawl site or scraped 0 pages: {normalized_url}\\033[0m")\n',
        "            print(\"Check logs for errors. Use 'scrape:config' to adjust settings if needed\")\n",
        "\n",
        "    except FetchRobotsError as e:\n",
        '        logger.error(f"Robots.txt disallowed: {e}")\n',
        '        print(f"\\033[1;31mCannot crawl site: {e.url} - Blocked by robots.txt\\033[0m")\n',
        '        print("You can disable robots.txt checking with: scrape:config respect_robots_txt=false")\n',
        "\n",
        "    except FetchRateLimitError as e:\n",
        '        logger.error(f"Rate limit error: {e}")\n',
        '        print(f"\\033[1;31mRate limit detected for {e.url}. Please try again later.\\033[0m")\n',
        '        print("You can decrease request frequency with: scrape:config set requests_per_minute <lower_value>")\n',
        "\n",
        "    except FetchTimeoutError as e:\n",
        '        logger.error(f"Timeout error: {e}")\n',
        '        print(f"\\033[1;31mTimeout error fetching {e.url}\\033[0m")\n',
        '        print("You can increase timeout with: scrape:config set timeout <higher_value>")\n',
        "\n",
        "    except FetchError as e:\n",
        '        logger.error(f"Fetch error: {e}")\n',
        '        print(f"\\033[1;31mError fetching {e.url}: {e.message}\\033[0m")\n',
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
        '        logger.error(f"Unexpected error crawling {normalized_url}: {e}", exc_info=True)\n',
        '        print(f"\\033[1;31mAn unexpected error occurred: {e}\\033[0m")\n',
        '        print("Please check logs for more details.")\n',
        "\n",
        "\n",
    ]

    # Replace the function
    lines[start_idx:end_idx] = fixed_function

    with open("wdbx_plugins/web_scraper.py", "w", encoding="utf-8") as file:
        file.writelines(lines)

    print("Fixed cmd_scrape_site function indentation issues")


if __name__ == "__main__":
    fix_cmd_scrape_site()
