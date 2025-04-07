#!/usr/bin/env python3
"""
Fix remaining style issues in web_scraper.py
"""

def fix_final_style_issues():
    with open('wdbx_plugins/web_scraper.py', 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Fix E501: Line too long
    # Lines to fix: 1476, 1530, 1532, 1557, 1575
    for line_num in [1476, 1530, 1532, 1557, 1575]:
        idx = line_num - 1  # Convert to 0-based index
        
        if idx < len(lines):
            line = lines[idx]
            
            if len(line) > 100:
                # Try to find a good breaking point
                indent = len(line) - len(line.lstrip())
                indent_str = " " * indent
                
                if "print" in line:
                    # For print statements
                    if "f" in line and "{" in line and "}" in line:
                        # For f-strings, break after a natural point like a space or comma
                        parts = line.split("f\"", 1)
                        prefix = parts[0] + "f\""
                        content = parts[1].rsplit("\"", 1)[0]
                        suffix = "\"" + parts[1].rsplit("\"", 1)[1]
                        
                        # Find a breaking point around the middle of the content
                        mid = len(content) // 2
                        break_points = []
                        for char in [" ", ",", ".", ":", ";"]:
                            pos = content.rfind(char, 0, mid + 20)
                            if pos != -1:
                                break_points.append(pos)
                        
                        if break_points:
                            break_point = max(break_points)
                            lines[idx] = prefix + content[:break_point+1] + "\"\n"
                            lines.insert(idx+1, indent_str + "f\"" + content[break_point+1:] + suffix)
                        else:
                            # If no good break point, just add a break at a reasonable point
                            lines[idx] = line[:90] + "\"\n"
                            lines.insert(idx+1, indent_str + "f\"" + line[90:].lstrip("\" "))
                    else:
                        # For regular strings, break after a natural point
                        if "\"" in line:
                            # Simple split at a space
                            pos = line.rfind(" ", 50, 90)
                            if pos != -1:
                                lines[idx] = line[:pos] + "\"\n"
                                lines.insert(idx+1, indent_str + "\"" + line[pos+1:].lstrip("\" "))
                            else:
                                # No good break point, try other split
                                lines[idx] = line[:90] + "\"\n"
                                lines.insert(idx+1, indent_str + "\"" + line[90:].lstrip("\" "))
                else:
                    # For non-print statements, break at a reasonable point
                    pos = line.rfind(" ", 50, 90)
                    if pos != -1:
                        lines[idx] = line[:pos] + "\n"
                        lines.insert(idx+1, indent_str + line[pos+1:])
                    else:
                        # No good break point, just break at 90 chars
                        lines[idx] = line[:90] + "\n"
                        lines.insert(idx+1, indent_str + line[90:])
    
    # Fix E302: Expected 2 blank lines between functions
    # Find the function definition at line 1756
    line_num = 1756 - 1  # Convert to 0-indexed
    if line_num < len(lines):
        # Count the number of blank lines before this function
        blank_count = 0
        i = line_num - 1
        while i >= 0 and not lines[i].strip():
            blank_count += 1
            i -= 1
        
        # Add blank lines if needed
        if blank_count < 2:
            lines_to_add = 2 - blank_count
            lines[line_num:line_num] = ["\n"] * lines_to_add
    
    # Write the changes back to the file
    with open('wdbx_plugins/web_scraper.py', 'w', encoding='utf-8') as file:
        file.writelines(lines)
    
    print("Fixed remaining style issues")

if __name__ == "__main__":
    fix_final_style_issues() 