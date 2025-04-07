#!/usr/bin/env python3
"""
Fix the cmd_scrape_search function which has issues with unterminated strings
"""

def fix_cmd_scrape_search():
    with open('wdbx_plugins/web_scraper.py', 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Find the start of the cmd_scrape_search function
    start_idx = content.find("def cmd_scrape_search(db: Any, args: str) -> None:")
    if start_idx == -1:
        print("Could not find cmd_scrape_search function")
        return
    
    # Find the end of the function (next function definition)
    end_idx = content.find("def ", start_idx + 10)
    if end_idx == -1:
        end_idx = len(content)
    
    # Replace the function with a fixed version
    fixed_function = """def cmd_scrape_search(db: Any, args: str) -> None:
    \"\"\"
    Search scraped content.

    Args:
        db: WDBX database instance
        args: Search query
    \"\"\"
    if not args:
        print("\\033[1;31mError: Search query required\\033[0m")
        print("Usage: scrape:search <query>")
        return

    query = args.strip()
    print(f"\\033[1;35mSearching scraped content for: {query}\\033[0m")

    # Simple text search in cache first
    results = []
    query_lower = query.lower()

    for scrape_id, item in scraped_cache.items():
        url = item.get("url", "")
        metadata = item.get("metadata", {})
        chunks = item.get("chunks", [])

        # Search in title and description
        title = metadata.get("title", "").lower()
        description = metadata.get("description", "").lower()

        title_match = query_lower in title
        desc_match = query_lower in description

        # Search in chunks
        matched_chunks = []
        for i, chunk in enumerate(chunks):
            if query_lower in chunk.lower():
                # Get context around the match
                index = chunk.lower().find(query_lower)
                start = max(0, index - 50)
                end = min(len(chunk), index + len(query) + 50)
                context = chunk[start:end]

                # Replace the query with highlighted version
                highlight = context.replace(
                    query,
                    f"\\033[1;33m{query}\\033[0m",
                    1  # Only replace first occurrence
                )

                matched_chunks.append({
                    "index": i,
                    "highlight": highlight
                })

        if title_match or desc_match or matched_chunks:
            results.append({
                "id": scrape_id,
                "url": url,
                "title": metadata.get("title", "N/A"),
                "title_match": title_match,
                "desc_match": desc_match,
                "chunks": matched_chunks
            })

    # Sort results: title matches first, then description matches, then by number of chunk matches
    results.sort(key=lambda x: (not x["title_match"], not x["desc_match"], -len(x["chunks"])))

    if not results:
        print("No matches found in scraped content")

        # Suggest using vector search if available
        print("\\nFor semantic search, use the database search capabilities or model:embed")
        return

    # Print results
    print(f"Found {len(results)} matches:")

    for i, result in enumerate(results[:10]):  # Show top 10 results
        print(f"\\n\\033[1m{i + 1}. {result['title']}\\033[0m")
        print(f"   URL: {result['url']}")
        print(f"   ID: {result['id']}")

        # Print chunk matches
        for j, chunk in enumerate(result["chunks"][:3]):  # Show top 3 chunk matches
            print(f"   Match {j + 1}: ...{chunk['highlight']}...")

        if len(result["chunks"]) > 3:
            print(f"   ... and {len(result['chunks']) - 3} more matches")

    if len(results) > 10:
        print(f"\\n... and {len(results) - 10} more results")

    print("\\nTo see details for a specific result, use: scrape:status <id>")
"""
    
    new_content = content[:start_idx] + fixed_function + content[end_idx:]
    
    with open('wdbx_plugins/web_scraper.py', 'w', encoding='utf-8') as file:
        file.write(new_content)
    
    print("Fixed cmd_scrape_search function with unterminated strings")

if __name__ == "__main__":
    fix_cmd_scrape_search() 