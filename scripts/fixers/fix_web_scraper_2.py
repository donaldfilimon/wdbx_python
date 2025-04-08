#!/usr/bin/env python3
"""
Fix syntax error in web_scraper.py
"""


def fix_syntax_error():
    with open("wdbx_plugins/web_scraper.py", encoding="utf-8") as file:
        lines = file.readlines()

    # Fix the try-except block in _update_config_value
    if len(lines) > 515:
        # We need to fix the structure - there should be an except JSONDecodeError after the ValueError
        fixed_block = [
            "    except ValueError:\n",
            '        error_msg = f"Invalid value type for {key}. Expected {expected_type.__name__}"\n',
            '        print(f"\\033[1;31m{error_msg}\\033[0m")\n',
            "        return False\n",
            "    except json.JSONDecodeError:\n",
            '        print(f"\\033[1;31mError: Invalid JSON format for list value for {key}\\033[0m")\n',
            "        return False\n",
            "\n",
        ]

        # Replace the problematic lines
        for i, line in enumerate(fixed_block):
            if 516 + i < len(lines):
                lines[516 + i] = line

    with open("wdbx_plugins/web_scraper.py", "w", encoding="utf-8") as file:
        file.writelines(lines)

    print("Fixed syntax error in web_scraper.py")


if __name__ == "__main__":
    fix_syntax_error()
