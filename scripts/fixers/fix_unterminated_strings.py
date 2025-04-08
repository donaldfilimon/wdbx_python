#!/usr/bin/env python3
"""
Fix unterminated string literals in web_scraper.py
"""


def fix_unterminated_strings():
    with open("wdbx_plugins/web_scraper.py", encoding="utf-8") as file:
        lines = file.readlines()

    # Fix specific unterminated string literals
    problematic_lines = [1476, 1531, 1559]

    for line_num in problematic_lines:
        idx = line_num - 1  # Convert to 0-based index
        if idx < len(lines):
            # Check if the line contains a string that wraps to the next line
            if "print(" in lines[idx] and "\n" in lines[idx]:
                # Case 1: Line 1476 - request frequency string
                if "requests_per_minute" in lines[idx]:
                    lines[idx] = (
                        'print("You can decrease request frequency with: scrape:config requests_per_minute=<lower_value>")\n'
                    )

                # Case 2: Line 1531 - request frequency in cmd_scrape_site
                elif "requests_per_minute" in lines[idx]:
                    lines[idx] = (
                        'print("You can decrease request frequency with: scrape:config set requests_per_minute <lower_value>")\n'
                    )

                # Case 3: Line 1559 - max_pages warning
                elif "max_pages" in lines[idx] and "must be positive" in lines[idx]:
                    lines[idx] = (
                        'print(f"\\033[1;33mWarning: max_pages must be positive, using default ({max_pages})\\033[0m")\n'
                    )

                # Case 4: Line 1559 - max_pages error
                elif "max_pages" in lines[idx] and "must be an integer" in lines[idx]:
                    lines[idx] = (
                        'print(f"\\033[1;31mError: max_pages must be an integer, using default ({max_pages})\\033[0m")\n'
                    )

                # Case 5: Line 1559 - chunks
                elif "chunks" in lines[idx]:
                    lines[idx] = (
                        "print(f\"{i + 1}. {item.get('url', 'N/A')} ({len(item.get('chunks', []))}) chunks\")\n"
                    )

    # Write the changes back to the file
    with open("wdbx_plugins/web_scraper.py", "w", encoding="utf-8") as file:
        file.writelines(lines)

    print("Fixed unterminated string literals in web_scraper.py")


if __name__ == "__main__":
    fix_unterminated_strings()
