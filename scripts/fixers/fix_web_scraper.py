#!/usr/bin/env python3
"""
Fix indentation issues in web_scraper.py
"""


def fix_indentation_issues():
    with open("wdbx_plugins/web_scraper.py", encoding="utf-8") as file:
        lines = file.readlines()

    # Fix _print_config function - lines 458-460
    if len(lines) > 458:
        lines[458] = '    print("\\n\\033[1;34mWeb Scraper Configuration:\\033[0m")\n'
        lines[459] = "    for key, value in scraper_config.to_dict().items():\n"
        lines[460] = '        print(f"  \\033[1m{key}\\033[0m = {value}")\n'

    # Fix _update_config_value function - lines 475-476
    if len(lines) > 475:
        lines[475] = (
            '        print(f"\\033[1;31mError: Unknown configuration key: {key}\\033[0m")\n'
        )
        lines[476] = "        return False\n"

    # Fix return False after error_msg - lines 499-500
    if len(lines) > 499:
        lines[499] = '            print(f"\\033[1;31m{error_msg}\\033[0m")\n'
        lines[500] = "            return False\n"

    # Fix return True after validation - lines 508-509
    if len(lines) > 508:
        lines[508] = '            print(f"\\033[1;32mSet {key} = {new_value}\\033[0m")\n'
        lines[509] = "            return True\n"

    # Fix return False after ValidationError - lines 513-514
    if len(lines) > 513:
        lines[513] = (
            '            print(f"\\033[1;31mError: Invalid value for {key}: {e}\\033[0m")\n'
        )
        lines[514] = "            return False\n"

    # Fix return False after ValueError - lines 518-519
    if len(lines) > 518:
        lines[518] = '        print(f"\\033[1;31m{error_msg}\\033[0m")\n'
        lines[519] = "        return False\n"

    # Fix return False after JSONDecodeError - lines 521-522
    if len(lines) > 521:
        lines[521] = (
            '        print(f"\\033[1;31mError: Invalid JSON format for list value for {key}\\033[0m")\n'
        )
        lines[522] = "        return False\n"

    # Fix cmd_scrape_help function - lines 1361-1370
    if len(lines) > 1360:
        lines[1360] = '    print("\\033[1;35mWDBX Web Scraper Plugin\\033[0m")\n'
        lines[1361] = '    print("The following web scraper commands are available:")\n'
        lines[1362] = (
            '    print("\\033[1m  scrape:url <url>\\033[0m - Scrape a single URL and add to the database")\n'
        )
        lines[1363] = (
            '    print("\\033[1m  scrape:site <url> [max_pages]\\033[0m - Crawl a site starting from URL")\n'
        )
        lines[1364] = (
            '    print("\\033[1m  scrape:list [domain]\\033[0m - List scraped URLs, optionally filtered by domain")\n'
        )
        lines[1365] = (
            '    print("\\033[1m  scrape:status <id>\\033[0m - Show detailed status of a scraped document")\n'
        )
        lines[1366] = (
            '    print("\\033[1m  scrape:search <query>\\033[0m - Search scraped content")\n'
        )
        lines[1367] = (
            '    print("\\033[1m  scrape:config [key=value]\\033[0m - Configure web scraper settings")\n'
        )
        lines[1368] = "\n"
        lines[1369] = '    print("\\n\\033[1;34mWeb Scraper Configuration:\\033[0m")\n'
        lines[1370] = "    _print_config()\n"

    # Fix cmd_scrape_config function - lines 1383-1387
    if len(lines) > 1382:
        lines[1383] = "    if not args:\n"
        lines[1384] = '        print("Current configuration:")\n'
        lines[1385] = "        _print_config()\n"
        lines[1386] = '        print("\\nTo change a setting, use: scrape:config key=value")\n'
        lines[1387] = "        return\n"

    # Fix indentation in cmd_scrape_config function - line 1400
    if len(lines) > 1399:
        lines[1400] = "    if updated_any:\n"

    # Fix cmd_scrape_url function - lines 1431-1432
    if len(lines) > 1430:
        lines[1431] = '        print("Usage: scrape:url <url>")\n'
        lines[1432] = "        return\n"

    # Fix the try section in cmd_scrape_url function
    if len(lines) > 1436:
        lines[1436] = "    try:\n"
        # Fix indentation for the content inside try block
        for i in range(1437, 1463):
            if i < len(lines):
                # Remove leading spaces/tabs and add proper indentation
                stripped = lines[i].lstrip()
                if stripped:
                    lines[i] = "        " + stripped

    # Fix the indentation for except sections in cmd_scrape_url
    if len(lines) > 1465:
        # We need to fix all the except blocks - identify them by their format
        for i in range(1465, 1475):
            if i < len(lines) and "except" in lines[i]:
                lines[i] = "    " + lines[i].lstrip()
                # Fix the next 2 lines (error message and print statement)
                if i + 1 < len(lines):
                    lines[i + 1] = "        " + lines[i + 1].lstrip()
                if i + 2 < len(lines):
                    lines[i + 2] = "        " + lines[i + 2].lstrip()

    # Fix the incomplete print statement with unterminated string on line 1469
    if len(lines) > 1469:
        lines[1469] = (
            '        print("You can disable robots.txt checking with: scrape:config respect_robots_txt=false")\n'
        )

    with open("wdbx_plugins/web_scraper.py", "w", encoding="utf-8") as file:
        file.writelines(lines)

    print("Fixed indentation issues in web_scraper.py")


if __name__ == "__main__":
    fix_indentation_issues()
