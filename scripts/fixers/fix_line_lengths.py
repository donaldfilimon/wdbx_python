#!/usr/bin/env python3
"""
Fix specific long lines in web_scraper.py
"""


def fix_line_lengths():
    with open("wdbx_plugins/web_scraper.py", encoding="utf-8") as file:
        lines = file.readlines()

    # The specific lines to fix with their line numbers
    long_lines = [1476, 1531, 1534, 1559, 1577, 1707]

    # Fix each line individually
    for line_num in long_lines:
        idx = line_num - 1  # Convert to 0-based index
        if idx < len(lines):
            line = lines[idx]
            indentation = len(line) - len(line.lstrip())
            indent = " " * indentation

            if ":" in line and not line.rstrip().endswith(":"):
                # If it's a print statement or similar, split after a colon or comma
                split_points = [line.find(":", 50), line.find(",", 50)]
                split_points = [p for p in split_points if p != -1]

                if split_points:
                    split_at = min(split_points) + 1
                    lines[idx] = line[:split_at] + "\n" + indent + line[split_at:].lstrip()

            elif "(" in line and ")" in line:
                # If it's a function call, split at an argument
                try:
                    open_paren = line.find("(")
                    if open_paren != -1 and open_paren < 50:
                        # Find a comma after 50 chars
                        comma = line.find(",", 50)
                        if comma != -1:
                            lines[idx] = (
                                line[: comma + 1]
                                + "\n"
                                + indent
                                + "    "
                                + line[comma + 1 :].lstrip()
                            )
                        else:
                            # Try to split before the closing parenthesis
                            close_paren = line.rfind(")")
                            if close_paren != -1 and close_paren > 50:
                                lines[idx] = (
                                    line[:close_paren] + "\n" + indent + line[close_paren:].lstrip()
                                )
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")

            else:
                # For other lines, find a space to break at
                space_pos = line.rfind(" ", 40, 90)
                if space_pos != -1:
                    lines[idx] = line[:space_pos] + "\n" + indent + line[space_pos + 1 :]

    # Write the changes back to the file
    with open("wdbx_plugins/web_scraper.py", "w", encoding="utf-8") as file:
        file.writelines(lines)

    print("Fixed long lines in web_scraper.py")


if __name__ == "__main__":
    fix_line_lengths()
