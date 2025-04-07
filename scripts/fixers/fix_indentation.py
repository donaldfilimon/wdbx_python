#!/usr/bin/env python3

def fix_indentation():
    with open('wdbx_plugins/web_scraper.py', 'r') as file:
        lines = file.readlines()
    
    # Fix indentation issue at line 459 (0-indexed is 458)
    if len(lines) > 458:
        if "        for" in lines[458]:
            lines[458] = lines[458].replace("        for", "    for")
    
    with open('wdbx_plugins/web_scraper.py', 'w') as file:
        file.writelines(lines)
    
    print("Fixed indentation at line 459")

if __name__ == "__main__":
    fix_indentation() 