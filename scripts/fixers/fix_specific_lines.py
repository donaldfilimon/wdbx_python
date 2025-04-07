#!/usr/bin/env python3
"""
Fix specific long lines in web_scraper.py
"""

def fix_specific_lines():
    lines_to_fix = {
        1532: ('print(f"\\033[1;31mRate limit detected for {e.url}. Please try again later.\\033[0m")', 
               'print(f"\\033[1;31mRate limit detected for {e.url}.\\033[0m")\n        print("Please try again later.")'),
        
        1535: ('print("You can decrease request frequency with: scrape:config set requests_per_minute <lower_value>")',
               'print("You can decrease request frequency with:")\n        print("scrape:config set requests_per_minute <lower_value>")'),
        
        1560: ('print(f"{i + 1}. {item.get(\'url\', \'N/A\')} ({len(item.get(\'chunks\', []))}) chunks")',
               'print(f"{i + 1}. {item.get(\'url\', \'N/A\')}")\n                        print(f"   ({len(item.get(\'chunks\', []))}) chunks")'),
        
        1578: ('print(f"\\033[1;31mRate limit detected for {e.url}. Please try again later.\\033[0m")',
               'print(f"\\033[1;31mRate limit detected for {e.url}.\\033[0m")\n        print("Please try again later.")')
    }
    
    with open('wdbx_plugins/web_scraper.py', 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    for line_num, (old_pattern, new_replacement) in lines_to_fix.items():
        idx = line_num - 1  # Convert to 0-based index
        if idx < len(lines) and old_pattern in lines[idx]:
            lines[idx] = lines[idx].replace(old_pattern, new_replacement)
    
    with open('wdbx_plugins/web_scraper.py', 'w', encoding='utf-8') as file:
        file.writelines(lines)
    
    print("Fixed specific long lines in web_scraper.py")

if __name__ == "__main__":
    fix_specific_lines() 