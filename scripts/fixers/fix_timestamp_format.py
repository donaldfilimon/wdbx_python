#!/usr/bin/env python3
"""
Fix timestamp format string issue in web_scraper.py
"""

def fix_timestamp_format():
    with open('wdbx_plugins/web_scraper.py', 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Look for the problematic line around line 1705
    for i in range(1703, 1708):
        if i < len(lines) and "scrape" in lines[i].lower() and "at" in lines[i].lower():
            timestamp_line = i
            break
    else:
        print("Could not find timestamp line")
        return
    
    # Replace the problematic line with a fixed version
    lines[timestamp_line] = '    print(f"\\033[1mScraped at:\\033[0m {time.strftime(\'%Y-%m-%d %H:%M:%S\', time.localtime(item.get(\'timestamp\', 0)))}")\n'
    
    with open('wdbx_plugins/web_scraper.py', 'w', encoding='utf-8') as file:
        file.writelines(lines)
    
    print(f"Fixed timestamp format on line {timestamp_line+1}")

if __name__ == "__main__":
    fix_timestamp_format() 