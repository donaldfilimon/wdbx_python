#!/usr/bin/env python3
"""
Fix the cmd_scrape_status function to resolve unterminated string issues
"""

def fix_cmd_scrape_status():
    with open('wdbx_plugins/web_scraper.py', 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Find the start of the cmd_scrape_status function
    start_idx = content.find("def cmd_scrape_status(db: Any, args: str) -> None:")
    if start_idx == -1:
        print("Could not find cmd_scrape_status function")
        return
    
    # Find the end of the function (next function definition)
    end_idx = content.find("def cmd_scrape_search", start_idx)
    if end_idx == -1:
        print("Could not find end of cmd_scrape_status function")
        return
    
    # Replace the function with a fixed version
    fixed_function = """def cmd_scrape_status(db: Any, args: str) -> None:
    \"\"\"
    Show detailed status of a scraped document.

    Args:
        db: WDBX database instance
        args: Scrape ID
    \"\"\"
    if not args:
        print("\\033[1;31mError: Scrape ID required\\033[0m")
        print("Usage: scrape:status <id>")
        return

    scrape_id = args.strip()

    # Try partial match if exact ID not found
    if scrape_id not in scraped_cache:
        matches = [id for id in scraped_cache if id.startswith(scrape_id)]
        if len(matches) == 1:
            scrape_id = matches[0]
        elif len(matches) > 1:
            print(f"\\033[1;31mMultiple matches found for ID: {scrape_id}\\033[0m")
            for match in matches:
                print(f"  {match}")
            return

    if scrape_id not in scraped_cache:
        print(f"\\033[1;31mNo scrape found with ID: {scrape_id}\\033[0m")
        return

    item = scraped_cache[scrape_id]
    metadata = item.get("metadata", {})
    chunks = item.get("chunks", [])

    print(f"\\033[1;35mScrape Status for ID: {scrape_id}\\033[0m")
    print(f"\\033[1mURL:\\033[0m {item.get('url', 'N/A')}")
    print(f"\\033[1mTitle:\\033[0m {metadata.get('title', 'N/A')}")
    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(item.get('timestamp', 0)))
    print(f"\\033[1mScraped at:\\033[0m {time_str}")
    print(f"\\033[1mChunks:\\033[0m {len(chunks)}")

    # Print description if available
    if "description" in metadata:
        print(f"\\033[1mDescription:\\033[0m {metadata['description']}")

    # Print response info
    if "response_info" in metadata:
        resp = metadata["response_info"]
        print("\\n\\033[1mResponse Info:\\033[0m")
        print(f"  Status: {resp.get('status_code', 'N/A')}")
        print(f"  Content Type: {resp.get('content_type', 'N/A')}")
        print(f"  Size: {resp.get('size_bytes', 0) / 1024:.1f} KB")
        print(f"  Fetch Time: {resp.get('fetch_time_sec', 0):.2f} seconds")

    # Print extracted metadata
    print("\\n\\033[1mMetadata:\\033[0m")
    for key, value in metadata.items():
        if key not in ["response_info", "links", "scrape_id", "chunk_count"]:
            if isinstance(value, str) and len(value) > 100:
                value = value[:100] + "..."
            print(f"  {key}: {value}")

    # Print chunk previews
    print("\\n\\033[1mContent Chunks:\\033[0m")
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"  Chunk {i + 1}/{len(chunks)}: {chunk[:100]}...")

    if len(chunks) > 3:
        print(f"  ... and {len(chunks) - 3} more chunks")

    # Print HTML path if available
    if "html_path" in metadata:
        print(f"\\n\\033[1mHTML File:\\033[0m {metadata['html_path']}")

    # Print link stats if available
    if "links" in metadata:
        links = metadata["links"]
        print(f"\\n\\033[1mLinks:\\033[0m {len(links)} extracted")

        # Group links by domain
        domains = {}
        for link in links:
            domain = _get_domain(link)
            domains[domain] = domains.get(domain, 0) + 1

        # Print top domains
        print("  Top domains:")
        for domain, count in sorted(domains.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"    {domain}: {count} links")
"""
    
    new_content = content[:start_idx] + fixed_function + content[end_idx:]
    
    with open('wdbx_plugins/web_scraper.py', 'w', encoding='utf-8') as file:
        file.write(new_content)
    
    print("Fixed cmd_scrape_status function with unterminated strings")

if __name__ == "__main__":
    fix_cmd_scrape_status() 