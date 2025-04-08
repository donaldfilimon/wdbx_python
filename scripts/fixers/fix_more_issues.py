#!/usr/bin/env python3


def fix_indentation_issues():
    with open("wdbx_plugins/discord_bot.py", encoding="utf-8") as file:
        lines = file.readlines()

    # Fix issue at line 616-618 (indentation in return statement)
    if len(lines) > 617:
        lines[616] = (
            "                logger.info(f\"Service '{self.SERVICE_NAME}' started successfully\")\n"
        )
        lines[617] = "                return True\n"
        lines[618] = "            time.sleep(1)\n"

    # Fix issue at line 621-622 (indentation in return statement)
    if len(lines) > 621:
        lines[621] = (
            "        logger.warning(f\"Service '{self.SERVICE_NAME}' did not start within timeout\")\n"
        )
        lines[622] = "        return False\n"

    # Fix issue at line 645-647 (indentation in return statement)
    if len(lines) > 646:
        lines[645] = (
            "                logger.info(f\"Service '{self.SERVICE_NAME}' stopped successfully\")\n"
        )
        lines[646] = "                return True\n"
        lines[647] = "            time.sleep(1)\n"

    # Fix issue at line 650-651 (indentation in return statement)
    if len(lines) > 650:
        lines[650] = (
            "        logger.warning(f\"Service '{self.SERVICE_NAME}' did not stop within timeout\")\n"
        )
        lines[651] = "        return False\n"

    # Fix issue at line 682-683 (indentation in return statement)
    if len(lines) > 682:
        lines[682] = '                logger.error("Failed to remove service file")\n'
        lines[683] = "                return False\n"

    # Fix issue at line 688-690 (indentation in return statement)
    if len(lines) > 689:
        lines[689] = (
            "        logger.info(f\"Service '{self.SERVICE_NAME}' uninstalled successfully\")\n"
        )
        lines[690] = "        return True\n"

    # Fix issue in MacServiceManager.install method
    if len(lines) > 750:
        lines[750] = "                return False\n"

    if len(lines) > 758:
        lines[758] = "                return False\n"

    with open("wdbx_plugins/discord_bot.py", "w", encoding="utf-8") as file:
        file.writelines(lines)

    print("Fixed more indentation issues")


if __name__ == "__main__":
    fix_indentation_issues()
