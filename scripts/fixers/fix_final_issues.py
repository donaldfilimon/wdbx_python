#!/usr/bin/env python
"""
Script to fix final line length and indentation issues.
"""

import sys


def fix_specific_lines(file_path, line_numbers):
    """Fix specific lines in the file."""
    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()

    modified = False

    for line_number in line_numbers:
        if line_number <= len(lines):
            original_line = lines[line_number - 1]

            # Skip if line is already short enough
            if len(original_line.rstrip()) <= 100:
                continue

            # For print statements with color codes, split at a good point
            if "\\033[" in original_line and "print" in original_line:
                parts = original_line.split("\\033[")
                if len(parts) >= 2:
                    # Determine indentation
                    indent = len(original_line) - len(original_line.lstrip())
                    indent_str = " " * indent

                    # Create new lines
                    new_line_1 = parts[0] + "\\033[" + parts[1].split("\\033[")[0]
                    new_line_2 = (
                        indent_str + "    " + "\\033[" + "\\033[".join(parts[1].split("\\033[")[1:])
                    )

                    # Make sure new_line_1 ends with a quote if it's not continued
                    if '"' in new_line_1 or "'" in new_line_1:
                        if '"' in new_line_1:
                            if new_line_1.rstrip()[-1] != '"':
                                new_line_1 = new_line_1.rstrip() + '" +\n'
                            else:
                                new_line_1 = new_line_1.rstrip() + " +\n"
                        elif "'" in new_line_1:
                            if new_line_1.rstrip()[-1] != "'":
                                new_line_1 = new_line_1.rstrip() + "' +\n"
                            else:
                                new_line_1 = new_line_1.rstrip() + " +\n"
                    else:
                        new_line_1 = new_line_1.rstrip() + "\n"

                    # Replace the original line
                    lines[line_number - 1] = new_line_1
                    lines.insert(line_number, new_line_2)
                    modified = True

            # For other long lines, try to find a good split point
            else:
                # Determine indentation
                indent = len(original_line) - len(original_line.lstrip())
                indent_str = " " * indent

                # Try to find a good split point
                split_points = [
                    original_line.rfind(" ", 0, 100),
                    original_line.rfind(",", 0, 100),
                    original_line.rfind("(", 0, 100),
                    original_line.rfind("[", 0, 100),
                    original_line.rfind("{", 0, 100),
                    original_line.rfind(":", 0, 100),
                    original_line.rfind("+", 0, 100),
                    original_line.rfind("-", 0, 100),
                    original_line.rfind("*", 0, 100),
                    original_line.rfind("/", 0, 100),
                    original_line.rfind("=", 0, 100),
                ]

                # Filter out -1 (not found)
                valid_points = [p for p in split_points if p != -1]

                if valid_points:
                    # Choose the rightmost split point
                    split_at = max(valid_points)

                    # Split the line
                    first_part = original_line[: split_at + 1]
                    second_part = original_line[split_at + 1 :].lstrip()

                    # Add indentation to the second part
                    second_part = indent_str + "    " + second_part

                    # Replace the original line
                    lines[line_number - 1] = first_part.rstrip() + "\n"
                    lines.insert(line_number, second_part)
                    modified = True

    # Also fix indentation issues
    for i, line in enumerate(lines):
        # Check for continuation line issues
        if line.strip().startswith(("print(", "print ", "fmt.")):
            # Check a few lines ahead for improper indentation
            for j in range(i + 1, min(i + 5, len(lines))):
                if lines[j].strip() and not lines[j].strip().startswith(("#", '"""', "'''")):
                    indent_orig = len(line) - len(line.lstrip())
                    indent_cont = len(lines[j]) - len(lines[j].lstrip())

                    # If continuation line indentation is wrong
                    if 0 < indent_cont < indent_orig + 4:
                        # Fix the indentation
                        lines[j] = " " * (indent_orig + 4) + lines[j].lstrip()
                        modified = True

                    break  # Only check the first non-empty, non-comment line

    if modified:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        return True

    return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_final_issues.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    line_numbers = [1469, 1474, 1529, 1532, 1557, 1570, 1575, 1705, 1711]

    if fix_specific_lines(file_path, line_numbers):
        print(f"Fixed specific issues in {file_path}")
    else:
        print(f"No changes needed in {file_path}")
