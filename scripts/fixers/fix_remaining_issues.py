#!/usr/bin/env python3
"""
Fix remaining linting issues in web_scraper.py
"""

def fix_remaining_issues():
    with open('wdbx_plugins/web_scraper.py', 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Fix E302: expected 2 blank lines, found 1
    # Find the function definitions at lines 463 and 1373
    for i in [463 - 1, 1373 - 1]:  # -1 because list is 0-indexed
        if i >= 0 and i < len(lines) and lines[i].strip().startswith("def "):
            # Check previous line
            blank_count = 0
            j = i - 1
            while j >= 0 and not lines[j].strip():
                blank_count += 1
                j -= 1
            
            # Add blank lines if needed
            if blank_count < 2:
                lines_to_add = 2 - blank_count
                lines[i:i] = ["\n"] * lines_to_add
    
    # Fix E501: Line too long
    # Lines to fix: 1474, 1529, 1532, 1557, 1575, 1705
    for i in [1474 - 1, 1529 - 1, 1532 - 1, 1557 - 1, 1575 - 1, 1705 - 1]:
        if i >= 0 and i < len(lines):
            line = lines[i]
            if len(line) > 100:
                # Simple fix: find the last space before 100 characters
                if "\"" in line or "'" in line:
                    # If line contains quotes, it's likely a string - split at 80 chars
                    pos = line.rfind(" ", 0, 80)
                    if pos != -1:
                        lines[i] = line[:pos] + "\n" + " " * 8 + line[pos+1:]
                else:
                    # Otherwise try to split at a reasonable point
                    pos = line.rfind(" ", 0, 90)
                    if pos != -1:
                        indent = len(line) - len(line.lstrip())
                        lines[i] = line[:pos] + "\n" + " " * indent + line[pos+1:]
    
    # Fix W391: blank line at end of file
    # Make sure there's exactly one blank line at the end
    while lines and not lines[-1].strip():
        lines.pop()
    
    lines.append("\n")  # Add exactly one newline at the end
    
    with open('wdbx_plugins/web_scraper.py', 'w', encoding='utf-8') as file:
        file.writelines(lines)
    
    print("Fixed remaining linting issues")

if __name__ == "__main__":
    fix_remaining_issues() 