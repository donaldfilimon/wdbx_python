#!/usr/bin/env python3

def fix_line54():
    with open('wdbx_plugins/discord_bot.py', 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Fix indentation issue at line 54 (0-indexed is 53)
    if len(lines) > 53:
        if "from wdbx.core.templates" in lines[53]:
            lines[53] = "        from wdbx.core.templates import LAUNCHD_PLIST_TEMPLATE, SYSTEMD_SERVICE_TEMPLATE\n"
    
    with open('wdbx_plugins/discord_bot.py', 'w', encoding='utf-8') as file:
        file.writelines(lines)
    
    print("Fixed indentation at line 54")

if __name__ == "__main__":
    fix_line54() 