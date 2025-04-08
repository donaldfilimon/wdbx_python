#!/usr/bin/env python3
"""
Final fixes for discord_bot.py to resolve all indentation and duplicated code issues.
"""


def fix_final_issues():
    with open("wdbx_plugins/discord_bot.py", encoding="utf-8") as file:
        content = file.read()

    # Extract sections to fix
    start_section = """        # Wait for service to be fully started
        for _ in range(10):  # Wait up to 10 seconds
            status = self.service_status()
            if status == "running":
                logger.info(f"Service '{self.SERVICE_NAME}' started successfully")
            return True
                logger.info(f"Service '{self.SERVICE_NAME}' started successfully")
                return True
            time.sleep(1)
            return False

        logger.warning(f"Service '{self.SERVICE_NAME}' did not start within timeout")
        return False"""

    fixed_start_section = """        # Wait for service to be fully started
        for _ in range(10):  # Wait up to 10 seconds
            status = self.service_status()
            if status == "running":
                logger.info(f"Service '{self.SERVICE_NAME}' started successfully")
                return True
            time.sleep(1)
            
        logger.warning(f"Service '{self.SERVICE_NAME}' did not start within timeout")
        return False"""

    # Fix the start_service method section
    content = content.replace(start_section, fixed_start_section)

    # Fix the stop_service method section
    stop_section = """        # Wait for service to be fully stopped
        for _ in range(10):  # Wait up to 10 seconds
            status = self.service_status()
            if status == "stopped":
                logger.info(f"Service '{self.SERVICE_NAME}' stopped successfully")
            return True
                logger.info(f"Service '{self.SERVICE_NAME}' stopped successfully")
                return True
            time.sleep(1)
            return False

        logger.warning(f"Service '{self.SERVICE_NAME}' did not stop within timeout")
        return False"""

    fixed_stop_section = """        # Wait for service to be fully stopped
        for _ in range(10):  # Wait up to 10 seconds
            status = self.service_status()
            if status == "stopped":
                logger.info(f"Service '{self.SERVICE_NAME}' stopped successfully")
                return True
            time.sleep(1)
            
        logger.warning(f"Service '{self.SERVICE_NAME}' did not stop within timeout")
        return False"""

    content = content.replace(stop_section, fixed_stop_section)

    # Fix the uninstall method section
    uninstall_section = """        # Remove service file
        if os.path.exists(self.SERVICE_FILE_PATH):
            if not self._run_command(["rm", "-f", self.SERVICE_FILE_PATH]):
                logger.error("Failed to remove service file")
            return False
                
        # Reload systemd to apply changes
                logger.error("Failed to remove service file")
                return False
            return False
            
        logger.info(f"Service '{self.SERVICE_NAME}' uninstalled successfully")
            return True

        logger.info(f"Service '{self.SERVICE_NAME}' uninstalled successfully")
        return True"""

    fixed_uninstall_section = """        # Remove service file
        if os.path.exists(self.SERVICE_FILE_PATH):
            if not self._run_command(["rm", "-f", self.SERVICE_FILE_PATH]):
                logger.error("Failed to remove service file")
                return False
                
        # Reload systemd to apply changes
        if not self._run_systemctl(["daemon-reload"]):
            logger.error("Failed to reload systemd")
            return False
            
        logger.info(f"Service '{self.SERVICE_NAME}' uninstalled successfully")
        return True"""

    content = content.replace(uninstall_section, fixed_uninstall_section)

    # Fix MacServiceManager.install method
    mac_install_section = """            # Move file to final destination using sudo
                return False
                os.remove(temp_path)
            return False

            logger.info(f"Service file created at {self.PLIST_PATH}")

            # Load the service (-w enables it)
            if not self._run_launchctl(["load", "-w", self.PLIST_PATH]):
                return False
                self._run_command(["sudo", "rm", "-f", self.PLIST_PATH], check=False)
            return False"""

    fixed_mac_install_section = """            # Move file to final destination using sudo
            if not self._run_command(["sudo", "mv", temp_path, self.PLIST_PATH]):
                os.remove(temp_path)
                return False

            logger.info(f"Service file created at {self.PLIST_PATH}")

            # Load the service (-w enables it)
            if not self._run_launchctl(["load", "-w", self.PLIST_PATH]):
                logger.error("Failed to load service with launchctl")
                self._run_command(["sudo", "rm", "-f", self.PLIST_PATH], check=False)
                return False"""

    content = content.replace(mac_install_section, fixed_mac_install_section)

    # Fix "def stop_service" section which may be broken
    if "return False\n        Stop the systemd service" in content:
        content = content.replace(
            "return False\n        Stop the systemd service",
            'return False\n\n    def stop_service(self) -> bool:\n        """',
        )

    # Fix "def uninstall" section which may be broken
    if "return False\n        Uninstall the systemd service" in content:
        content = content.replace(
            "return False\n        Uninstall the systemd service",
            'return False\n\n    def uninstall(self) -> bool:\n        """',
        )

    # Write the fixed content back to the file
    with open("wdbx_plugins/discord_bot.py", "w", encoding="utf-8") as file:
        file.write(content)

    print("Applied final fixes to discord_bot.py")


if __name__ == "__main__":
    fix_final_issues()
