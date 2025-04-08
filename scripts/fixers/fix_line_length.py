#!/usr/bin/env python
"""
Script to fix long lines in Python files.
"""

import sys

MAX_LINE_LENGTH = 100


def fix_long_lines(file_path):
    """Fix lines longer than MAX_LINE_LENGTH in the specified file."""
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    modified = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # If line is shorter than max length, keep it as is
        if len(line) <= MAX_LINE_LENGTH:
            modified.append(line)
            i += 1
            continue

        # Skip comment lines
        if line.strip().startswith("#"):
            modified.append(line)
            i += 1
            continue

        # Try to find a good place to split the line
        split_points = [
            line.rfind(" ", 0, MAX_LINE_LENGTH),
            line.rfind(",", 0, MAX_LINE_LENGTH),
            line.rfind("(", 0, MAX_LINE_LENGTH),
            line.rfind("[", 0, MAX_LINE_LENGTH),
            line.rfind("{", 0, MAX_LINE_LENGTH),
            line.rfind(":", 0, MAX_LINE_LENGTH),
            line.rfind("+", 0, MAX_LINE_LENGTH),
            line.rfind("-", 0, MAX_LINE_LENGTH),
            line.rfind("*", 0, MAX_LINE_LENGTH),
            line.rfind("/", 0, MAX_LINE_LENGTH),
            line.rfind("=", 0, MAX_LINE_LENGTH),
        ]

        # Filter out -1 (not found)
        valid_points = [p for p in split_points if p != -1]

        if valid_points:
            # Choose the rightmost split point
            split_at = max(valid_points)

            # Split the line
            first_part = line[: split_at + 1]
            second_part = line[split_at + 1 :].lstrip()

            # Handle indentation for the second line
            indent = len(line) - len(line.lstrip())
            additional_indent = 4  # Standard indentation for line continuation

            # Add indentation to the second part
            second_part = " " * (indent + additional_indent) + second_part

            modified.append(first_part)

            # Check if the second part is still too long
            if len(second_part) > MAX_LINE_LENGTH:
                # Insert the second part back into lines for further processing
                lines.insert(i + 1, second_part)
            else:
                modified.append(second_part)
                i += 1
        else:
            # If no good split point found, just append the line
            modified.append(line)
            i += 1

    # Join the modified lines back into content
    new_content = "\n".join(modified)

    # If content changed, write it back to the file
    if content != new_content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return True

    return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_line_length.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    if fix_long_lines(file_path):
        print(f"Fixed long lines in {file_path}")
    else:
        print(f"No changes needed in {file_path}")
