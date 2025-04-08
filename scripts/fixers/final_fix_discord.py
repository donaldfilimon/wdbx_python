#!/usr/bin/env python3
"""
Final fixes for specific remaining indentation issues in discord_bot.py
"""


def fix_remaining_issues():
    with open("wdbx_plugins/discord_bot.py", encoding="utf-8") as file:
        lines = file.readlines()

    # Fix MacServiceManager.service_status method indentation
    # Lines 815-820 - fix the check if plist exists section
    if len(lines) > 814:
        lines[815] = "        # Check if plist exists\n"
        lines[816] = "        if os.path.exists(self.PLIST_PATH):\n"
        lines[817] = '            return "installed (unknown state)"\n'
        lines[818] = "\n"
        lines[819] = '        return "not installed"\n'

    # Fix WindowsServiceManager.install method indentation
    # Lines 885-890 - fix indentation in self.win32serviceutil.InstallService call
    if len(lines) > 884:
        lines[885] = "            self.win32serviceutil.InstallService(\n"
        lines[886] = "                pythonClassString=None,\n"
        lines[887] = "                serviceName=self.SERVICE_NAME,\n"
        lines[888] = "                displayName=self.SERVICE_DISPLAY_NAME,\n"
        lines[889] = "                startType=self.win32service.SERVICE_AUTO_START,\n"
        lines[890] = "                commandLine=service_cmd,\n"
        lines[891] = "            )\n"

    # Fix indentation in start if requested section
    if len(lines) > 896:
        lines[896] = "            if self.config.start:\n"
        lines[897] = '                logger.info("Starting service after installation.")\n'
        lines[898] = "                if not self.start_service():\n"
        lines[899] = (
            '                    logger.warning("Service installed but failed to start.")\n'
        )
        lines[900] = "            return True\n"

    # Fix ServiceFactory._detect_platform method indentation
    if len(lines) > 1086:
        lines[1086] = "        try:\n"
        lines[1087] = "            system = platform.system()\n"
        lines[1088] = "            if system == ServiceFactory.PLATFORM_LINUX:\n"
        lines[1089] = "                # Check if running in WSL\n"
        lines[1090] = (
            '                if "microsoft-standard" in platform.uname().release.lower():\n'
        )
        lines[1091] = (
            '                    logger.warning("Detected WSL environment, using Linux service manager")\n'
        )
        lines[1092] = "                return ServiceFactory.PLATFORM_LINUX\n"
        lines[1093] = "            elif system == ServiceFactory.PLATFORM_DARWIN:\n"
        lines[1094] = "                return ServiceFactory.PLATFORM_DARWIN\n"
        lines[1095] = "            elif system == ServiceFactory.PLATFORM_WINDOWS:\n"
        lines[1096] = "                return ServiceFactory.PLATFORM_WINDOWS\n"
        lines[1097] = "            else:\n"
        lines[1098] = (
            '                raise ServicePlatformError(f"Unsupported platform: {system}")\n'
        )

    # Fix docker-compose generation - line 1189-1193
    if len(lines) > 1189:
        lines[1189] = '        logger.info(f"Successfully generated {output_path}")\n'
        lines[1190] = "        return True\n"
        lines[1191] = "    except Exception as e:\n"
        lines[1192] = '        logger.error(f"Failed to write docker-compose file: {e}")\n'
        lines[1193] = "        return False\n"

    # Fix kubernetes generation - line 1261-1265
    if len(lines) > 1261:
        lines[1261] = (
            '        logger.info(f"Successfully generated Kubernetes manifests in {output_dir}")\n'
        )
        lines[1262] = "        return True\n"
        lines[1263] = "    except Exception as e:\n"
        lines[1264] = '        logger.error(f"Failed to write Kubernetes manifests: {e}")\n'
        lines[1265] = "        return False\n"

    # Fix handle_windows_service - line 1373-1374
    if len(lines) > 1373:
        lines[1373] = '        logger.error(f"Windows service error: {e}", exc_info=True)\n'
        lines[1374] = "        return 1\n"

    with open("wdbx_plugins/discord_bot.py", "w", encoding="utf-8") as file:
        file.writelines(lines)

    print("Fixed remaining indentation issues in discord_bot.py")


if __name__ == "__main__":
    fix_remaining_issues()
