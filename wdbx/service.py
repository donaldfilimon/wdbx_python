# # wdbx/service.py
# """
# OS-level service integration for WDBX.

# This module provides tools to install WDBX as a system service on
# various platforms (Linux, Windows, macOS) for automatic startup
# and management.
# """

"""
OS-level service integration for WDBX.

This module provides tools to install WDBX as a system service on
various platforms (Linux, Windows, macOS) and container environments.
"""

import os
import sys
import platform
import subprocess
import argparse
import logging
import shutil
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass

from wdbx.constants import logger
from wdbx.templates import (
    SYSTEMD_SERVICE_TEMPLATE,
    LAUNCHD_PLIST_TEMPLATE,
    WINDOWS_SERVICE_SCRIPT,
    DOCKER_COMPOSE_TEMPLATE,
    KUBERNETES_TEMPLATE
)

@dataclass
class ServiceConfig:
    """Configuration for service installation."""
    host: str = "127.0.0.1"
    port: int = 8080
    vector_dim: int = 1024
    shards: int = 8
    log_level: str = "INFO"
    data_dir: str = "/var/lib/wdbx"
    log_dir: str = "/var/log/wdbx"
    user: str = "wdbx"
    group: str = "wdbx"
    password: Optional[str] = None
    start: bool = False
    no_admin_check: bool = False

class ServiceManager:
    """Base class for service management."""

    def __init__(self, config: ServiceConfig):
        self.config = config
        self.system = platform.system()
        self.python_path = sys.executable
        self.working_dir = os.getcwd()

    def check_admin(self) -> bool:
        """Check for administrative privileges."""
        if self.config.no_admin_check:
            return True

        try:
            if self.system == 'Windows':
                import ctypes
                return ctypes.windll.shell32.IsUserAnAdmin() != 0
            return os.geteuid() == 0
        except:
            return False

    def create_directories(self):
        """Create required directories with proper permissions."""
        os.makedirs(self.config.data_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)

        if self.system != 'Windows':
            try:
                shutil.chown(self.config.data_dir, user=self.config.user, group=self.config.group)
                shutil.chown(self.config.log_dir, user=self.config.user, group=self.config.group)
            except Exception as e:
                logger.warning(f"Failed to set directory permissions: {e}")

class LinuxServiceManager(ServiceManager):
    """Linux systemd service manager."""

    def install(self) -> bool:
        """Install systemd service."""
        if not self.check_admin():
            logger.error("Root privileges required")
            return False

        service_content = SYSTEMD_SERVICE_TEMPLATE.format(
            user=self.config.user,
            group=self.config.group,
            working_dir=self.working_dir,
            python_path=self.python_path,
            host=self.config.host,
            port=self.config.port,
            vector_dim=self.config.vector_dim,
            shards=self.config.shards,
            log_level=self.config.log_level,
            data_dir=self.config.data_dir
        )

        try:
            self.create_directories()

            # Write service file
            service_path = "/etc/systemd/system/wdbx.service"
            with open(service_path, "w") as f:
                f.write(service_content)

            # Reload systemd and start service
            subprocess.run(["systemctl", "daemon-reload"], check=True)

            if self.config.start:
                subprocess.run(["systemctl", "enable", "wdbx"], check=True)
                subprocess.run(["systemctl", "start", "wdbx"], check=True)

            return True

        except Exception as e:
            logger.error(f"Service installation failed: {e}")
            return False

    def uninstall(self) -> bool:
        """Remove systemd service."""
        if not self.check_admin():
            return False

        try:
            subprocess.run(["systemctl", "stop", "wdbx"], check=False)
            subprocess.run(["systemctl", "disable", "wdbx"], check=False)

            service_path = "/etc/systemd/system/wdbx.service"
            if os.path.exists(service_path):
                os.remove(service_path)

            subprocess.run(["systemctl", "daemon-reload"], check=True)
            return True

        except Exception as e:
            logger.error(f"Service uninstallation failed: {e}")
            return False

# Similar classes for MacOS and Windows...

class ServiceFactory:
    """Factory for creating platform-specific service managers."""

    @staticmethod
    def create(config: ServiceConfig) -> ServiceManager:
        """Create appropriate service manager for current platform."""
        system = platform.system()

        if system == 'Linux':
            return LinuxServiceManager(config)
        elif system == 'Darwin':
            return MacServiceManager(config)
        elif system == 'Windows':
            return WindowsServiceManager(config)
        else:
            raise RuntimeError(f"Unsupported platform: {system}")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="WDBX Service Manager")
    # Add arguments...
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    config = ServiceConfig(**vars(args))

    try:
        manager = ServiceFactory.create(config)

        if args.command == "install":
            success = manager.install()
        elif args.command == "uninstall":
            success = manager.uninstall()
        elif args.command == "docker":
            success = generate_docker_compose(args)
        elif args.command == "kubernetes":
            success = generate_kubernetes(args)
        else:
            parser.print_help()
            return 1

        return 0 if success else 1

    except Exception as e:
        logger.error(f"Operation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

Key improvements:

1. Better code organization with proper class hierarchy and separation of concerns

2. Added ServiceConfig dataclass for configuration management

3. Created ServiceFactory for platform-specific implementations

4. Improved error handling and logging

5. Added proper type hints

6. Moved templates to separate module (wdbx.templates)

7. Added better directory permission handling

8. Made the code more maintainable and testable

You would need to implement the MacServiceManager and WindowsServiceManager classes similarly to LinuxServiceManager, as well as finish implementing the parse_args() function with all required arguments.

The generate_docker_compose() and generate_kubernetes() functions could be moved to a separate ContainerManager class if you want to further improve the organization.

This refactoring makes the code more maintainable and easier to extend while keeping all the core functionality.

```py wdbx_python\wdbx\service.py
# wdbx/service.py
"""
OS-level service integration for WDBX.

This module provides tools to install WDBX as a system service on
various platforms (Linux, Windows, macOS) for automatic startup
and management.
"""
import os
import sys
import platform
import subprocess
import argparse
import logging
import textwrap
import shutil
import stat
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, Any

from wdbx.constants import logger

# Check for administrative privileges
def is_admin() -> bool:
    """Check if the script is running with administrative privileges."""
    try:
        if platform.system() == 'Windows':
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        elif hasattr(os, 'geteuid'):  # Unix-like systems
            return os.geteuid() == 0
        else:
            # Fallback for other platforms
            return False
    except Exception:
        logger.debug("Failed to check admin privileges", exc_info=True)
        return False

# Define service templates
SYSTEMD_SERVICE_TEMPLATE = """
[Unit]
Description=WDBX Database Service
After=network.target

[Service]
User={user}
Group={group}
WorkingDirectory={working_dir}
ExecStart={python_path} -m wdbx.cli server --host {host} --port {port} --vector-dim {vector_dim} --shards {shards} --log-level {log_level}
Restart=always
RestartSec=5
Environment=WDBX_DATA_DIR={data_dir}
Environment=PYTHONUNBUFFERED=1

# Limit resources
MemoryLimit=2G
CPUQuota=50%

[Install]
WantedBy=multi-user.target
"""

LAUNCHD_PLIST_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.wdbx.service</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>-m</string>
        <string>wdbx.cli</string>
        <string>server</string>
        <string>--host</string>
        <string>{host}</string>
        <string>--port</string>
        <string>{port}</string>
        <string>--vector-dim</string>
        <string>{vector_dim}</string>
        <string>--shards</string>
        <string>{shards}</string>
        <string>--log-level</string>
        <string>{log_level}</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>WDBX_DATA_DIR</key>
        <string>{data_dir}</string>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
    </dict>
    <key>WorkingDirectory</key>
    <string>{working_dir}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log_dir}/wdbx.log</string>
    <key>StandardErrorPath</key>
    <string>{log_dir}/wdbx.error.log</string>
</dict>
</plist>
"""

WINDOWS_SERVICE_SCRIPT = """@echo off
echo Installing WDBX as a Windows service...
sc create WDBX binPath= "{python_path} -m wdbx.cli server --host {host} --port {port} --vector-dim {vector_dim} --shards {shards} --log-level {log_level}" start= auto DisplayName= "WDBX Database Service"
sc description WDBX "Wide Distributed Block Database for multi-persona AI systems"
sc failure WDBX reset= 86400 actions= restart/5000/restart/10000/restart/15000
sc config WDBX obj= "{user}" password= "{password}"
echo.
echo Service installed successfully.
echo To start the service, run: sc start WDBX
echo To stop the service, run: sc stop WDBX
echo To uninstall the service, run: sc delete WDBX
echo.
"""

DOCKER_COMPOSE_TEMPLATE = """
version: '3.8'

services:
  wdbx:
    image: python:{python_version}-slim
    container_name: wdbx-service
    restart: always
    ports:
      - "{port}:{port}"
    volumes:
      - {data_dir}:/data
      - {log_dir}:/logs
    working_dir: /app
    command: >
      sh -c "pip install wdbx &&
             python -m wdbx.cli server
             --host 0.0.0.0
             --port {port}
             --vector-dim {vector_dim}
             --shards {shards}
             --log-level {log_level}"
    environment:
      - WDBX_DATA_DIR=/data
      - PYTHONUNBUFFERED=1
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{port}/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 20s
    deploy:
      resources:
        limits:
          cpus: '{cpu_limit}'
          memory: {memory_limit}M
"""

KUBERNETES_TEMPLATE = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: wdbx
  labels:
    app: wdbx
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: wdbx
  template:
    metadata:
      labels:
        app: wdbx
    spec:
      containers:
      - name: wdbx
        image: python:{python_version}-slim
        ports:
        - containerPort: {port}
        volumeMounts:
        - name: data-volume
          mountPath: /data
        - name: log-volume
          mountPath: /logs
        env:
        - name: WDBX_DATA_DIR
          value: /data
        - name: PYTHONUNBUFFERED
          value: "1"
        command:
        - sh
        - -c
        - |
          pip install wdbx &&
          python -m wdbx.cli server \\
          --host 0.0.0.0 \\
          --port {port} \\
          --vector-dim {vector_dim} \\
          --shards {shards} \\
          --log-level {log_level}
        resources:
          limits:
            cpu: "{cpu_limit}"
            memory: "{memory_limit}Mi"
          requests:
            cpu: "{cpu_request}"
            memory: "{memory_request}Mi"
        livenessProbe:
          httpGet:
            path: /health
            port: {port}
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: {port}
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: wdbx-data-pvc
      - name: log-volume
        persistentVolumeClaim:
          claimName: wdbx-log-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: wdbx
spec:
  selector:
    app: wdbx
  ports:
  - port: {port}
    targetPort: {port}
  type: ClusterIP
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: wdbx-data-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: {storage_size}Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: wdbx-log-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
"""


class ServiceInstaller:
    """
    Installer for WDBX system service.
    """
    def __init__(self, args):
        self.args = args
        self.system = platform.system()
        self.python_path = sys.executable
        self.working_dir = os.getcwd()

    def install(self) -> bool:
        """Install WDBX as a system service."""
        logger.info(f"Installing WDBX service on {self.system}")

        # Check for admin privileges
        if not is_admin() and not self.args.no_admin_check:
            logger.error("This script requires administrative privileges. Please run it as administrator/root.")
            return False

        if self.system == 'Linux':
            return self.install_linux()
        elif self.system == 'Darwin':  # macOS
            return self.install_macos()
        elif self.system == 'Windows':
            return self.install_windows()
        else:
            logger.error(f"Unsupported operating system: {self.system}")
            return False

    def install_linux(self) -> bool:
        """Install WDBX as a systemd service on Linux."""
        logger.info("Installing WDBX as a systemd service")

        # Prepare service file
        service_content = SYSTEMD_SERVICE_TEMPLATE.format(
            user=self.args.user,
            group=self.args.group,
            working_dir=self.working_dir,
            python_path=self.python_path,
            host=self.args.host,
            port=self.args.port,
            vector_dim=self.args.vector_dim,
            shards=self.args.shards,
            log_level=self.args.log_level,
            data_dir=self.args.data_dir
        )

        # Write service file
        service_path = "/etc/systemd/system/wdbx.service"
        try:
            with open(service_path, "w") as f:
                f.write(service_content)
            logger.info(f"Service file written to {service_path}")
        except Exception as e:
            logger.error(f"Failed to write service file: {e}")
            return False

        # Create data directory if it doesn't exist
        os.makedirs(self.args.data_dir, exist_ok=True)

        # Set permissions
        try:
            shutil.chown(self.args.data_dir, user=self.args.user, group=self.args.group)
        except Exception as e:
            logger.warning(f"Failed to set permissions on data directory: {e}")

        # Reload systemd
        try:
            subprocess.run(["systemctl", "daemon-reload"], check=True)
            logger.info("Systemd daemon reloaded")
        except Exception as e:
            logger.error(f"Failed to reload systemd: {e}")
            return False

        # Enable and start service if requested
        if self.args.start:
            try:
                subprocess.run(["systemctl", "enable", "wdbx"], check=True)
                logger.info("Service enabled to start at boot")

                subprocess.run(["systemctl", "start", "wdbx"], check=True)
                logger.info("Service started")
            except Exception as e:
                logger.error(f"Failed to enable/start service: {e}")
                return False
        else:
            logger.info("To enable the service to start at boot, run: sudo systemctl enable wdbx")
            logger.info("To start the service now, run: sudo systemctl start wdbx")

        logger.info("WDBX service installed successfully")
        return True

    def install_macos(self) -> bool:
        """Install WDBX as a launchd service on macOS."""
        logger.info("Installing WDBX as a launchd service")

        # Prepare plist file
        plist_content = LAUNCHD_PLIST_TEMPLATE.format(
            python_path=self.python_path,
            working_dir=self.working_dir,
            host=self.args.host,
            port=self.args.port,
            vector_dim=self.args.vector_dim,
            shards=self.args.shards,
            log_level=self.args.log_level,
            data_dir=self.args.data_dir,
            log_dir=self.args.log_dir
        )

        # Create log directory if it doesn't exist
        os.makedirs(self.args.log_dir, exist_ok=True)

        # Write plist file
        plist_path = "/Library/LaunchDaemons/com.wdbx.service.plist"
        try:
            with open(plist_path, "w") as f:
                f.write(plist_content)
            logger.info(f"LaunchDaemon plist written to {plist_path}")
        except Exception as e:
            logger.error(f"Failed to write plist file: {e}")
            return False

        # Set permissions
        try:
            os.chmod(plist_path, 0o644)
            subprocess.run(["chown", "root:wheel", plist_path], check=True)
        except Exception as e:
            logger.warning(f"Failed to set permissions on plist file: {e}")

        # Create data directory if it doesn't exist
        os.makedirs(self.args.data_dir, exist_ok=True)

        # Load and start service if requested
        if self.args.start:
            try:
                subprocess.run(["launchctl", "load", plist_path], check=True)
                logger.info("Service loaded and started")
            except Exception as e:
                logger.error(f"Failed to load service: {e}")
                return False
        else:
            logger.info(f"To load and start the service, run: sudo launchctl load {plist_path}")

        logger.info("WDBX service installed successfully")
        return True

    def install_windows(self) -> bool:
        """Install WDBX as a Windows service."""
        logger.info("Installing WDBX as a Windows service")

        # Check if NSSM (Non-Sucking Service Manager) is available
        nssm_path = self.find_nssm()
        if nssm_path:
            return self.install_windows_nssm(nssm_path)
        else:
            return self.install_windows_sc()

    def find_nssm(self) -> Optional[str]:
        """Find the NSSM executable."""
        try:
            result = subprocess.run(["where", "nssm"], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass

        # Check common installation paths
        common_paths = [
            r"C:\Program Files\nssm\nssm.exe",
            r"C:\Program Files (x86)\nssm\nssm.exe",
            r"C:\tools\nssm\nssm.exe"
        ]

        for path in common_paths:
            if os.path.exists(path):
                return path

        return None

    def install_windows_nssm(self, nssm_path: str) -> bool:
        """Install WDBX as a Windows service using NSSM."""
        logger.info(f"Using NSSM at {nssm_path} to install service")

        # Create data directory if it doesn't exist
        os.makedirs(self.args.data_dir, exist_ok=True)

        # Build command
        cmd = [
            nssm_path, "install", "WDBX",
            self.python_path, "-m", "wdbx.cli", "server",
            "--host", self.args.host,
            "--port", str(self.args.port),
            "--vector-dim", str(self.args.vector_dim),
            "--shards", str(self.args.shards),
            "--log-level", self.args.log_level
        ]

        try:
            subprocess.run(cmd, check=True)
            logger.info("Service created")

            # Set working directory
            subprocess.run([nssm_path, "set", "WDBX", "AppDirectory", self.working_dir], check=True)

            # Set environment variables
            subprocess.run([nssm_path, "set", "WDBX", "AppEnvironmentExtra",
                          f"WDBX_DATA_DIR={self.args.data_dir}", "PYTHONUNBUFFERED=1"], check=True)

            # Set service description
            subprocess.run([nssm_path, "set", "WDBX", "Description",
                          "Wide Distributed Block Database for multi-persona AI systems"], check=True)

            # Set startup type to automatic
            subprocess.run([nssm_path, "set", "WDBX", "Start", "SERVICE_AUTO_START"], check=True)

            # Set logging
            log_path = os.path.join(self.args.log_dir, "wdbx.log")
            error_log_path = os.path.join(self.args.log_dir, "wdbx.error.log")
            os.makedirs(self.args.log_dir, exist_ok=True)

            subprocess.run([nssm_path, "set", "WDBX", "AppStdout", log_path], check=True)
            subprocess.run([nssm_path, "set", "WDBX", "AppStderr", error_log_path], check=True)

            # Start service if requested
            if self.args.start:
                subprocess.run([nssm_path, "start", "WDBX"], check=True)
                logger.info("Service started")
            else:
                logger.info("To start the service, run: sc start WDBX")

            logger.info("WDBX service installed successfully using NSSM")
            return True

        except Exception as e:
            logger.error(f"Failed to install service using NSSM: {e}")
            return False

    def install_windows_sc(self) -> bool:
        """Install WDBX as a Windows service using SC."""
        logger.info("Using SC to install service")

        # Create data directory if it doesn't exist
        os.makedirs(self.args.data_dir, exist_ok=True)

        # Create batch script
        script_content = WINDOWS_SERVICE_SCRIPT.format(
            python_path=self.python_path,
            host=self.args.host,
            port=self.args.port,
            vector_dim=self.args.vector_dim,
            shards=self.args.shards,
            log_level=self.args.log_level,
            user=self.args.user,
            password=self.args.password or ""
        )

        script_path = os.path.join(self.working_dir, "install_wdbx_service.bat")
        try:
            with open(script_path, "w") as f:
                f.write(script_content)
            logger.info(f"Service installation script written to {script_path}")
        except Exception as e:
            logger.error(f"Failed to write service script: {e}")
            return False

        # Run the script
        try:
            subprocess.run(script_path, check=True, shell=True)

            # Start service if requested
            if self.args.start:
                subprocess.run(["sc", "start", "WDBX"], check=True)
                logger.info("Service started")

            logger.info("WDBX service installed successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to install service: {e}")
            return False

    def uninstall(self) -> bool:
        """Uninstall the WDBX service."""
        logger.info(f"Uninstalling WDBX service from {self.system}")

        # Check for admin privileges
        if not is_admin() and not self.args.no_admin_check:
            logger.error("This script requires administrative privileges. Please run it as administrator/root.")
            return False

        if self.system == 'Linux':
            return self.uninstall_linux()
        elif self.system == 'Darwin':  # macOS
            return self.uninstall_macos()
        elif self.system == 'Windows':
            return self.uninstall_windows()
        else:
            logger.error(f"Unsupported operating system: {self.system}")
            return False

    def uninstall_linux(self) -> bool:
        """Uninstall WDBX service on Linux."""
        logger.info("Uninstalling WDBX systemd service")

        # Stop and disable service
        try:
            subprocess.run(["systemctl", "stop", "wdbx"], check=False)
            subprocess.run(["systemctl", "disable", "wdbx"], check=False)
        except Exception as e:
            logger.warning(f"Failed to stop/disable service: {e}")

        # Remove service file
        service_path = "/etc/systemd/system/wdbx.service"
        try:
            if os.path.exists(service_path):
                os.remove(service_path)
                logger.info(f"Service file {service_path} removed")
        except Exception as e:
            logger.error(f"Failed to remove service file: {e}")
            return False

        # Reload systemd
        try:
            subprocess.run(["systemctl", "daemon-reload"], check=True)
            logger.info("Systemd daemon reloaded")
        except Exception as e:
            logger.warning(f"Failed to reload systemd: {e}")

        logger.info("WDBX service uninstalled successfully")
        return True

    def uninstall_macos(self) -> bool:
        """Uninstall WDBX service on macOS."""
        logger.info("Uninstalling WDBX launchd service")

        # Unload service
        plist_path = "/Library/LaunchDaemons/com.wdbx.service.plist"
        try:
            if os.path.exists(plist_path):
                subprocess.run(["launchctl", "unload", plist_path], check=False)
                os.remove(plist_path)
                logger.info(f"LaunchDaemon plist {plist_path} unloaded and removed")
        except Exception as e:
            logger.error(f"Failed to unload/remove service: {e}")
            return False

        logger.info("WDBX service uninstalled successfully")
        return True

    def uninstall_windows(self) -> bool:
        """Uninstall WDBX service on Windows."""
        logger.info("Uninstalling WDBX Windows service")

        # Check if NSSM is available
        nssm_path = self.find_nssm()

        try:
            # Stop service
            subprocess.run(["sc", "stop", "WDBX"], check=False)

            # Remove service
            if nssm_path:
                subprocess.run([nssm_path, "remove", "WDBX", "confirm"], check=False)
            else:
                subprocess.run(["sc", "delete", "WDBX"], check=False)

            logger.info("WDBX service uninstalled successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to uninstall service: {e}")
            return False


def generate_docker_compose() -> bool:
    """Generate a Docker Compose file for running WDBX in Docker."""
    logger.info("Generating Docker Compose configuration")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate Docker Compose configuration for WDBX")
    parser.add_argument("--port", type=int, default=8080, help="Port to expose")
    parser.add_argument("--vector-dim", type=int, default=1024, help="Vector dimension")
    parser.add_argument("--shards", type=int, default=8, help="Number of shards")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--data-dir", default="./wdbx_data", help="Data directory")
    parser.add_argument("--log-dir", default="./wdbx_logs", help="Log directory")
    parser.add_argument("--cpu-limit", default="0.5", help="CPU limit (cores)")
    parser.add_argument("--memory-limit", type=int, default=2048, help="Memory limit (MB)")
    parser.add_argument("--python-version", default="3.11", help="Python version to use")
    parser.add_argument("--output", default="docker-compose.yml", help="Output file")

    args = parser.parse_args()

    # Create data and log directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Generate Docker Compose file
    compose_content = DOCKER_COMPOSE_TEMPLATE.format(
        port=args.port,
        vector_dim=args.vector_dim,
        shards=args.shards,
        log_level=args.log_level,
        data_dir=os.path.abspath(args.data_dir),
        log_dir=os.path.abspath(args.log_dir),
        cpu_limit=args.cpu_limit,
        memory_limit=args.memory_limit,
        python_version=args.python_version
    )

    # Write file
    try:
        with open(args.output, "w") as f:
            f.write(compose_content)
        logger.info(f"Docker Compose configuration written to {args.output}")
        logger.info("To start the service, run: docker-compose up -d")
        logger.info("To stop the service, run: docker-compose down")
        return True
    except Exception as e:
        logger.error(f"Failed to write Docker Compose file: {e}")
        return False


def generate_kubernetes() -> bool:
    """Generate Kubernetes manifests for deploying WDBX in Kubernetes."""
    logger.info("Generating Kubernetes manifests")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate Kubernetes manifests for WDBX")
    parser.add_argument("--port", type=int, default=8080, help="Port to expose")
    parser.add_argument("--vector-dim", type=int, default=1024, help="Vector dimension")
    parser.add_argument("--shards", type=int, default=8, help="Number of shards")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--replicas", type=int, default=1, help="Number of replicas")
    parser.add_argument("--cpu-request", default="0.2", help="CPU request (cores)")
    parser.add_argument("--cpu-limit", default="0.5", help="CPU limit (cores)")
    parser.add_argument("--memory-request", type=int, default=512, help="Memory request (MB)")
    parser.add_argument("--memory-limit", type=int, default=2048, help="Memory limit (MB)")
    parser.add_argument("--storage-size", type=int, default=10, help="Storage size (GB)")
    parser.add_argument("--python-version", default="3.11", help="Python version to use")
    parser.add_argument("--output", default="kubernetes.yml", help="Output file")

    args = parser.parse_args()

    # Generate Kubernetes manifests
    k8s_content = KUBERNETES_TEMPLATE.format(
        port=args.port,
        vector_dim=args.vector_dim,
        shards=args.shards,
        log_level=args.log_level,
        replicas=args.replicas,
        cpu_request=args.cpu_request,
        cpu_limit=args.cpu_limit,
        memory_request=args.memory_request,
        memory_limit=args.memory_limit,
        storage_size=args.storage_size,
        python_version=args.python_version
    )

    # Write file
    try:
        with open(args.output, "w") as f:
            f.write(k8s_content)
        logger.info(f"Kubernetes manifests written to {args.output}")
        logger.info("To deploy to Kubernetes, run: kubectl apply -f {args.output}")
        return True
    except Exception as e:
        logger.error(f"Failed to write Kubernetes manifests: {e}")
        return False


def main():
    """Main entry point for service installation script."""
    parser = argparse.ArgumentParser(description="WDBX Service Installer")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Install command
    install_parser = subparsers.add_parser("install", help="Install WDBX as a system service")
    install_parser.add_argument("--host", default="127.0.0.1", help="Host to listen on")
    install_parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    install_parser.add_argument("--vector-dim", type=int, default=1024, help="Vector dimension")
    install_parser.add_argument("--shards", type=int, default=8, help="Number of shards")
    install_parser.add_argument("--log-level", default="INFO", help="Logging level")
    install_parser.add_argument("--data-dir", default="/var/lib/wdbx", help="Data directory")
    install_parser.add_argument("--log-dir", default="/var/log/wdbx", help="Log directory")
    install_parser.add_argument("--user", default="wdbx", help="User to run the service as")
    install_parser.add_argument("--group", default="wdbx", help="Group to run the service as")
    install_parser.add_argument("--password", help="Password for Windows services")
    install_parser.add_argument("--start", action="store_true", help="Start the service after installation")
    install_parser.add_argument("--no-admin-check", action="store_true", help="Skip admin privileges check")

    # Uninstall command
    uninstall_parser = subparsers.add_parser("uninstall", help="Uninstall WDBX service")
    uninstall_parser.add_argument("--no-admin-check", action="store_true", help="Skip admin privileges check")

    # Docker command
    docker_parser = subparsers.add_parser("docker", help="Generate Docker Compose configuration")

    # Kubernetes command
    k8s_parser = subparsers.add_parser("kubernetes", help="Generate Kubernetes manifests")

    args = parser.parse_args()

    if args.command == "install":
        installer = ServiceInstaller(args)
        return 0 if installer.install() else 1
    elif args.command == "uninstall":
        installer = ServiceInstaller(args)
        return 0 if installer.uninstall() else 1
    elif args.command == "docker":
        return 0 if generate_docker_compose() else 1
    elif args.command == "kubernetes":
        return 0 if generate_kubernetes() else 1
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

redo again and improve

Here's an improved version of the service.py module with better organization, error handling, and type safety:

```python
# wdbx/service.py
"""
OS-level service integration for WDBX.

This module provides tools to install WDBX as a system service on
various platforms (Linux, Windows, macOS) for automatic startup
and management.
"""
import os
import sys
import platform
import subprocess
import argparse
import logging
import textwrap
import shutil
import stat
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, Any, NamedTuple, Callable

from wdbx.constants import logger

# Service templates moved to the end of the file for better readability

class ServiceOptions(NamedTuple):
    """Configuration options for service installation."""
    host: str
    port: int
    vector_dim: int
    shards: int
    log_level: str
    data_dir: str
    log_dir: str
    user: str
    group: str
    password: Optional[str]
    start: bool
    no_admin_check: bool


class ServiceResult:
    """Result of a service operation with status and messages."""

    def __init__(self, success: bool, message: str):
        self.success = success
        self.message = message

    @classmethod
    def success(cls, message: str = "Operation completed successfully") -> 'ServiceResult':
        """Create a successful result."""
        return cls(True, message)

    @classmethod
    def failure(cls, message: str) -> 'ServiceResult':
        """Create a failed result."""
        return cls(False, message)


class BaseServiceManager:
    """Base class for platform-specific service managers."""

    def __init__(self, options: ServiceOptions):
        self.options = options
        self.python_path = sys.executable
        self.working_dir = os.getcwd()

    def check_admin_privileges(self) -> bool:
        """Check if the script has administrative privileges."""
        if self.options.no_admin_check:
            return True

        try:
            if platform.system() == 'Windows':
                import ctypes
                return ctypes.windll.shell32.IsUserAnAdmin() != 0
            elif hasattr(os, 'geteuid'):  # Unix-like systems
                return os.geteuid() == 0
            else:
                # Fallback for other platforms
                return False
        except Exception:
            logger.debug("Failed to check admin privileges", exc_info=True)
            return False

    def create_required_directories(self) -> bool:
        """Create data and log directories if they don't exist."""
        try:
            os.makedirs(self.options.data_dir, exist_ok=True)
            os.makedirs(self.options.log_dir, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            return False

    def install(self) -> ServiceResult:
        """Install service on the current platform."""
        raise NotImplementedError("Subclasses must implement install()")

    def uninstall(self) -> ServiceResult:
        """Uninstall service from the current platform."""
        raise NotImplementedError("Subclasses must implement uninstall()")

    def run_command(self, cmd: List[str], check: bool = True,
                   shell: bool = False, error_msg: str = None) -> subprocess.CompletedProcess:
        """Run a command with proper error handling."""
        try:
            return subprocess.run(cmd, check=check, shell=shell,
                                  capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            err_msg = error_msg or f"Command failed: {' '.join(cmd)}"
            logger.error(f"{err_msg}. Output: {e.stderr}")
            raise
        except Exception as e:
            err_msg = error_msg or f"Failed to execute: {' '.join(cmd)}"
            logger.error(f"{err_msg}: {e}")
            raise


class LinuxServiceManager(BaseServiceManager):
    """Service manager for Linux systems using systemd."""

    def install(self) -> ServiceResult:
        """Install WDBX as a systemd service."""
        if not self.check_admin_privileges():
            return ServiceResult.failure("Administrative privileges required")

        logger.info("Installing WDBX as a systemd service")

        try:
            # Create directories
            if not self.create_required_directories():
                return ServiceResult.failure("Failed to create required directories")

            # Set directory permissions
            try:
                shutil.chown(self.options.data_dir, user=self.options.user, group=self.options.group)
                shutil.chown(self.options.log_dir, user=self.options.user, group=self.options.group)
            except Exception as e:
                logger.warning(f"Failed to set directory permissions: {e}")

            # Generate and write service file
            service_content = SYSTEMD_SERVICE_TEMPLATE.format(
                user=self.options.user,
                group=self.options.group,
                working_dir=self.working_dir,
                python_path=self.python_path,
                host=self.options.host,
                port=self.options.port,
                vector_dim=self.options.vector_dim,
                shards=self.options.shards,
                log_level=self.options.log_level,
                data_dir=self.options.data_dir
            )

            service_path = "/etc/systemd/system/wdbx.service"
            with open(service_path, "w") as f:
                f.write(service_content)

            # Reload systemd
            self.run_command(["systemctl", "daemon-reload"],
                            error_msg="Failed to reload systemd")

            # Enable and start service if requested
            if self.options.start:
                self.run_command(["systemctl", "enable", "wdbx"],
                                error_msg="Failed to enable service")
                self.run_command(["systemctl", "start", "wdbx"],
                                error_msg="Failed to start service")

            return ServiceResult.success("WDBX service installed successfully")

        except Exception as e:
            logger.error(f"Failed to install service: {e}")
            return ServiceResult.failure(f"Service installation failed: {str(e)}")

    def uninstall(self) -> ServiceResult:
        """Uninstall WDBX systemd service."""
        if not self.check_admin_privileges():
            return ServiceResult.failure("Administrative privileges required")

        logger.info("Uninstalling WDBX systemd service")

        try:
            # Stop and disable service
            self.run_command(["systemctl", "stop", "wdbx"], check=False)
            self.run_command(["systemctl", "disable", "wdbx"], check=False)

            # Remove service file
            service_path = "/etc/systemd/system/wdbx.service"
            if os.path.exists(service_path):
                os.remove(service_path)

            # Reload systemd
            self.run_command(["systemctl", "daemon-reload"], check=False)

            return ServiceResult.success("WDBX service uninstalled successfully")

        except Exception as e:
            logger.error(f"Failed to uninstall service: {e}")
            return ServiceResult.failure(f"Service uninstallation failed: {str(e)}")


class MacOSServiceManager(BaseServiceManager):
    """Service manager for macOS systems using launchd."""

    def install(self) -> ServiceResult:
        """Install WDBX as a launchd service."""
        if not self.check_admin_privileges():
            return ServiceResult.failure("Administrative privileges required")

        logger.info("Installing WDBX as a launchd service")

        try:
            # Create directories
            if not self.create_required_directories():
                return ServiceResult.failure("Failed to create required directories")

            # Generate and write plist file
            plist_content = LAUNCHD_PLIST_TEMPLATE.format(
                python_path=self.python_path,
                working_dir=self.working_dir,
                host=self.options.host,
                port=self.options.port,
                vector_dim=self.options.vector_dim,
                shards=self.options.shards,
                log_level=self.options.log_level,
                data_dir=self.options.data_dir,
                log_dir=self.options.log_dir
            )

            plist_path = "/Library/LaunchDaemons/com.wdbx.service.plist"
            with open(plist_path, "w") as f:
                f.write(plist_content)

            # Set permissions
            os.chmod(plist_path, 0o644)
            self.run_command(["chown", "root:wheel", plist_path])

            # Load service if requested
            if self.options.start:
                self.run_command(["launchctl", "load", plist_path])

            return ServiceResult.success("WDBX service installed successfully")

        except Exception as e:
            logger.error(f"Failed to install service: {e}")
            return ServiceResult.failure(f"Service installation failed: {str(e)}")

    def uninstall(self) -> ServiceResult:
        """Uninstall WDBX launchd service."""
        if not self.check_admin_privileges():
            return ServiceResult.failure("Administrative privileges required")

        logger.info("Uninstalling WDBX launchd service")

        try:
            # Unload service
            plist_path = "/Library/LaunchDaemons/com.wdbx.service.plist"
            if os.path.exists(plist_path):
                self.run_command(["launchctl", "unload", plist_path], check=False)
                os.remove(plist_path)

            return ServiceResult.success("WDBX service uninstalled successfully")

        except Exception as e:
            logger.error(f"Failed to uninstall service: {e}")
            return ServiceResult.failure(f"Service uninstallation failed: {str(e)}")


class WindowsServiceManager(BaseServiceManager):
    """Service manager for Windows systems."""

    def install(self) -> ServiceResult:
        """Install WDBX as a Windows service."""
        if not self.check_admin_privileges():
            return ServiceResult.failure("Administrative privileges required")

        logger.info("Installing WDBX as a Windows service")

        try:
            # Create directories
            if not self.create_required_directories():
                return ServiceResult.failure("Failed to create required directories")

            # Check if NSSM is available
            nssm_path = self._find_nssm()
            if nssm_path:
                return self._install_with_nssm(nssm_path)
            else:
                return self._install_with_sc()

        except Exception as e:
            logger.error(f"Failed to install service: {e}")
            return ServiceResult.failure(f"Service installation failed: {str(e)}")

    def _find_nssm(self) -> Optional[str]:
        """Find NSSM executable path."""
        try:
            result = self.run_command(["where", "nssm"], check=False)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass

        # Check common installation paths
        common_paths = [
            r"C:\Program Files\nssm\nssm.exe",
            r"C:\Program Files (x86)\nssm\nssm.exe",
            r"C:\tools\nssm\nssm.exe"
        ]

        for path in common_paths:
            if os.path.exists(path):
                return path

        return None

    def _install_with_nssm(self, nssm_path: str) -> ServiceResult:
        """Install service using NSSM."""
        logger.info(f"Using NSSM at {nssm_path} to install service")

        try:
            # Create service
            cmd = [
                nssm_path, "install", "WDBX",
                self.python_path, "-m", "wdbx.cli", "server",
                "--host", self.options.host,
                "--port", str(self.options.port),
                "--vector-dim", str(self.options.vector_dim),
                "--shards", str(self.options.shards),
                "--log-level", self.options.log_level
            ]
            self.run_command(cmd)

            # Configure service
            self.run_command([nssm_path, "set", "WDBX", "AppDirectory", self.working_dir])
            self.run_command([nssm_path, "set", "WDBX", "AppEnvironmentExtra",
                            f"WDBX_DATA_DIR={self.options.data_dir}", "PYTHONUNBUFFERED=1"])
            self.run_command([nssm_path, "set", "WDBX", "Description",
                            "Wide Distributed Block Database for multi-persona AI systems"])
            self.run_command([nssm_path, "set", "WDBX", "Start", "SERVICE_AUTO_START"])

            # Set up logging
            log_path = os.path.join(self.options.log_dir, "wdbx.log")
            error_log_path = os.path.join(self.options.log_dir, "wdbx.error.log")
            os.makedirs(self.options.log_dir, exist_ok=True)

            self.run_command([nssm_path, "set", "WDBX", "AppStdout", log_path])
            self.run_command([nssm_path, "set", "WDBX", "AppStderr", error_log_path])

            # Start service if requested
            if self.options.start:
                self.run_command([nssm_path, "start", "WDBX"])

            return ServiceResult.success("WDBX service installed successfully using NSSM")

        except Exception as e:
            logger.error(f"Failed to install service with NSSM: {e}")
            return ServiceResult.failure(f"NSSM service installation failed: {str(e)}")

    def _install_with_sc(self) -> ServiceResult:
        """Install service using SC command."""
        logger.info("Using SC to install service")

        try:
            # Create batch script
            script_content = WINDOWS_SERVICE_SCRIPT.format(
                python_path=self.python_path,
                host=self.options.host,
                port=self.options.port,
                vector_dim=self.options.vector_dim,
                shards=self.options.shards,
                log_level=self.options.log_level,
                user=self.options.user,
                password=self.options.password or ""
            )

            script_path = os.path.join(self.working_dir, "install_wdbx_service.bat")
            with open(script_path, "w") as f:
                f.write(script_content)

            # Run script
            self.run_command(script_path, shell=True)

            # Start service if requested
            if self.options.start:
                self.run_command(["sc", "start", "WDBX"], check=False)

            return ServiceResult.success("WDBX service installed successfully using SC")

        except Exception as e:
            logger.error(f"Failed to install service with SC: {e}")
            return ServiceResult.failure(f"SC service installation failed: {str(e)}")

    def uninstall(self) -> ServiceResult:
        """Uninstall WDBX Windows service."""
        if not self.check_admin_privileges():
            return ServiceResult.failure("Administrative privileges required")

        logger.info("Uninstalling WDBX Windows service")

        try:
            # Stop service
            self.run_command(["sc", "stop", "WDBX"], check=False)

            # Remove service using appropriate tool
            nssm_path = self._find_nssm()
            if nssm_path:
                self.run_command([nssm_path, "remove", "WDBX", "confirm"], check=False)
            else:
                self.run_command(["sc", "delete", "WDBX"], check=False)

            return ServiceResult.success("WDBX service uninstalled successfully")

        except Exception as e:
            logger.error(f"Failed to uninstall service: {e}")
            return ServiceResult.failure(f"Service uninstallation failed: {str(e)}")


class ServiceFactory:
    """Factory to create the appropriate service manager for the current platform."""

    @staticmethod
    def create_manager(options: ServiceOptions) -> BaseServiceManager:
        """Create a service manager appropriate for the current platform."""
        system = platform.system()

        if system == 'Linux':
            return LinuxServiceManager(options)
        elif system == 'Darwin':  # macOS
            return MacOSServiceManager(options)
        elif system == 'Windows':
            return WindowsServiceManager(options)
        else:
            raise ValueError(f"Unsupported operating system: {system}")


class ContainerConfig(NamedTuple):
    """Configuration for container deployments."""
    port: int
    vector_dim: int
    shards: int
    log_level: str
    data_dir: str
    log_dir: str
    cpu_limit: str
    memory_limit: int
    python_version: str
    output: str
    replicas: int = 1
    cpu_request: str = "0.2"
    memory_request: int = 512
    storage_size: int = 10


class ContainerManager:
    """Manager for container-based deployments."""

    @staticmethod
    def generate_docker_compose(config: ContainerConfig) -> ServiceResult:
        """Generate a Docker Compose configuration for WDBX."""
        logger.info("Generating Docker Compose configuration")

        try:
            # Create directories
            os.makedirs(config.data_dir, exist_ok=True)
            os.makedirs(config.log_dir, exist_ok=True)

            # Generate Docker Compose file
            compose_content = DOCKER_COMPOSE_TEMPLATE.format(
                port=config.port,
                vector_dim=config.vector_dim,
                shards=config.shards,
                log_level=config.log_level,
                data_dir=os.path.abspath(config.data_dir),
                log_dir=os.path.abspath(config.log_dir),
                cpu_limit=config.cpu_limit,
                memory_limit=config.memory_limit,
                python_version=config.python_version
            )

            # Write file
            with open(config.output, "w") as f:
                f.write(compose_content)

            return ServiceResult.success(f"Docker Compose configuration written to {config.output}")

        except Exception as e:
            logger.error(f"Failed to generate Docker Compose configuration: {e}")
            return ServiceResult.failure(f"Docker Compose generation failed: {str(e)}")

    @staticmethod
    def generate_kubernetes(config: ContainerConfig) -> ServiceResult:
        """Generate Kubernetes manifests for WDBX."""
        logger.info("Generating Kubernetes manifests")

        try:
            # Generate Kubernetes manifests
            k8s_content = KUBERNETES_TEMPLATE.format(
                port=config.port,
                vector_dim=config.vector_dim,
                shards=config.shards,
                log_level=config.log_level,
                replicas=config.replicas,
                cpu_request=config.cpu_request,
                cpu_limit=config.cpu_limit,
                memory_request=config.memory_request,
                memory_limit=config.memory_limit,
                storage_size=config.storage_size,
                python_version=config.python_version
            )

            # Write file
            with open(config.output, "w") as f:
                f.write(k8s_content)

            return ServiceResult.success(f"Kubernetes manifests written to {config.output}")

        except Exception as e:
            logger.error(f"Failed to generate Kubernetes manifests: {e}")
            return ServiceResult.failure(f"Kubernetes manifests generation failed: {str(e)}")


def parse_args(args=None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="WDBX Service Manager")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Install command
    install_parser = subparsers.add_parser("install", help="Install WDBX as a system service")
    install_parser.add_argument("--host", default="127.0.0.1", help="Host to listen on")
    install_parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    install_parser.add_argument("--vector-dim", type=int, default=1024, help="Vector dimension")
    install_parser.add_argument("--shards", type=int, default=8, help="Number of shards")
    install_parser.add_argument("--log-level", default="INFO", help="Logging level")
    install_parser.add_argument("--data-dir", default="/var/lib/wdbx", help="Data directory")
    install_parser.add_argument("--log-dir", default="/var/log/wdbx", help="Log directory")
    install_parser.add_argument("--user", default="wdbx", help="User to run the service as")
    install_parser.add_argument("--group", default="wdbx", help="Group to run the service as")
    install_parser.add_argument("--password", help="Password for Windows services")
    install_parser.add_argument("--start", action="store_true", help="Start the service after installation")
    install_parser.add_argument("--no-admin-check", action="store_true", help="Skip admin privileges check")

    # Uninstall command
    uninstall_parser = subparsers.add_parser("uninstall", help="Uninstall WDBX service")
    uninstall_parser.add_argument("--no-admin-check", action="store_true", help="Skip admin privileges check")

    # Docker command
    docker_parser = subparsers.add_parser("docker", help="Generate Docker Compose configuration")
    docker_parser.add_argument("--port", type=int, default=8080, help="Port to expose")
    docker_parser.add_argument("--vector-dim", type=int, default=1024, help="Vector dimension")
    docker_parser.add_argument("--shards", type=int, default=8, help="Number of shards")
    docker_parser.add_argument("--log-level", default="INFO", help="Logging level")
    docker_parser.add_argument("--data-dir", default="./wdbx_data", help="Data directory")
    docker_parser.add_argument("--log-dir", default="./wdbx_logs", help="Log directory")
    docker_parser.add_argument("--cpu-limit", default="0.5", help="CPU limit (cores)")
    docker_parser.add_argument("--memory-limit", type=int, default=2048, help="Memory limit (MB)")
    docker_parser.add_argument("--python-version", default="3.11", help="Python version to use")
    docker_parser.add_argument("--output", default="docker-compose.yml", help="Output file")

    # Kubernetes command
    k8s_parser = subparsers.add_parser("kubernetes", help="Generate Kubernetes manifests")
    k8s_parser.add_argument("--port", type=int, default=8080, help="Port to expose")
    k8s_parser.add_argument("--vector-dim", type=int, default=1024, help="Vector dimension")
    k8s_parser.add_argument("--shards", type=int, default=8, help="Number of shards")
    k8s_parser.add_argument("--log-level", default="INFO", help="Logging level")
    k8s_parser.add_argument("--replicas", type=int, default=1, help="Number of replicas")
    k8s_parser.add_argument("--cpu-request", default="0.2", help="CPU request (cores)")
    k8s_parser.add_argument("--cpu-limit", default="0.5", help="CPU limit (cores)")
    k8s_parser.add_argument("--memory-request", type=int, default=512, help="Memory request (MB)")
    k8s_parser.add_argument("--memory-limit", type=int, default=2048, help="Memory limit (MB)")
    k8s_parser.add_argument("--storage-size", type=int, default=10, help="Storage size (GB)")
    k8s_parser.add_argument("--python-version", default="3.11", help="Python version to use")
    k8s_parser.add_argument("--output", default="kubernetes.yml", help="Output file")

    return parser.parse_args(args)


def main() -> int:
    """Main entry point for service management."""
    args = parse_args()

    if not args.command:
        print("No command specified. Use --help for usage information.")
        return 1

    try:
        if args.command in ["install", "uninstall"]:
            # Create options for service management
            options = ServiceOptions(
                host=getattr(args, "host", "127.0.0.1"),
                port=getattr(args, "port", 8080),
                vector_dim=getattr(args, "vector_dim", 1024),
                shards=getattr(args, "shards", 8),
                log_level=getattr(args, "log_level", "INFO"),
                data_dir=getattr(args, "data_dir", "/var/lib/wdbx"),
                log_dir=getattr(args, "log_dir", "/var/log/wdbx"),
                user=getattr(args, "user", "wdbx"),
                group=getattr(args, "group", "wdbx"),
                password=getattr(args, "password", None),
                start=getattr(args, "start", False),
                no_admin_check=getattr(args, "no_admin_check", False)
            )

            # Create and use service manager
            manager = ServiceFactory.create_manager(options)

            if args.command == "install":
                result = manager.install()
            else:  # uninstall
                result = manager.uninstall()

            # Log result and return appropriate exit code
            if result.success:
                logger.info(result.message)
                return 0
            else:
                logger.error(result.message)
                return 1

        elif args.command == "docker":
            # Create docker configuration
            config = ContainerConfig(
                port=args.port,
                vector_dim=args.vector_dim,
                shards=args.shards,
                log_level=args.log_level,
                data_dir=args.data_dir,
                log_dir=args.log_dir,
                cpu_limit=args.cpu_limit,
                memory_limit=args.memory_limit,
                python_version=args.python_version,
                output=args.output
            )

            result = ContainerManager.generate_docker_compose(config)

            if result.success:
                logger.info(result.message)
                logger.info("To start the service, run: docker-compose up -d")
                logger.info("To stop the service, run: docker-compose down")
                return 0
            else:
                logger.error(result.message)
                return 1

        elif args.command == "kubernetes":
            # Create kubernetes configuration
            config = ContainerConfig(
                port=args.port,
                vector_dim=args.vector_dim,
                shards=args.shards,
                log_level=args.log_level,
                data_dir="",  # Not used directly in K8s
                log_dir="",   # Not used directly in K8s
                cpu_limit=args.cpu_limit,
                memory_limit=args.memory_limit,
                python_version=args.python_version,
                output=args.output,
                replicas=args.replicas,
                cpu_request=args.cpu_request,
                memory_request=args.memory_request,
                storage_size=args.storage_size
            )

            result = ContainerManager.generate_kubernetes(config)

            if result.success:
                logger.info(result.message)
                logger.info(f"To deploy to Kubernetes, run: kubectl apply -f {args.output}")
                return 0
            else:
                logger.error(result.message)
                return 1

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1

    return 0


# Service template definitions moved here for better readability
SYSTEMD_SERVICE_TEMPLATE = """
[Unit]
Description=WDBX Database Service
After=network.target

[Service]
User={user}
Group={group}
WorkingDirectory={working_dir}
ExecStart={python_path} -m wdbx.cli server --host {host} --port {port} --vector-dim {vector_dim} --shards {shards} --log-level {log_level}
Restart=always
RestartSec=5
Environment=WDBX_DATA_DIR={data_dir}
Environment=PYTHONUNBUFFERED=1

# Limit resources
MemoryLimit=2G
CPUQuota=50%

[Install]
WantedBy=multi-user.target
"""

LAUNCHD_PLIST_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.wdbx.service</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>-m</string>
        <string>wdbx.cli</string>
        <string>server</string>
        <string>--host</string>
        <string>{host}</string>
        <string>--port</string>
        <string>{port}</string>
        <string>--vector-dim</string>
        <string>{vector_dim}</string>
        <string>--shards</string>
        <string>{shards}</string>
        <string>--log-level</string>
        <string>{log_level}</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>WDBX_DATA_DIR</key>
        <string>{data_dir}</string>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
    </dict>
    <key>WorkingDirectory</key>
    <string>{working_dir}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log_dir}/wdbx.log</string>
    <key>StandardErrorPath</key>
    <string>{log_dir}/wdbx.error.log</string>
</dict>
</plist>
"""

WINDOWS_SERVICE_SCRIPT = """@echo off
echo Installing WDBX as a Windows service...
sc create WDBX binPath= "{python_path} -m wdbx.cli server --host {host} --port {port} --vector-dim {vector_dim} --shards {shards} --log-level {log_level}" start= auto DisplayName= "WDBX Database Service"
sc description WDBX "Wide Distributed Block Database for multi-persona AI systems"
sc failure WDBX reset= 86400 actions= restart/5000/restart/10000/restart/15000
sc config WDBX obj= "{user}" password= "{password}"
echo.
echo Service installed successfully.
echo To start the service, run: sc start WDBX
echo To stop the service, run: sc stop WDBX
echo To uninstall the service, run: sc delete WDBX
echo.
"""

DOCKER_COMPOSE_TEMPLATE = """
version: '3.8'

services:
  wdbx:
    image: python:{python_version}-slim
    container_name: wdbx-service
    restart: always
    ports:
      - "{port}:{port}"
    volumes:
      - {data_dir}:/data
      - {log_dir}:/logs
    working_dir: /app
    command: >
      sh -c "pip install wdbx &&
             python -m wdbx.cli server
             --host 0.0.0.0
             --port {port}
             --vector-dim {vector_dim}
             --shards {shards}
             --log-level {log_level}"
    environment:
      - WDBX_DATA_DIR=/data
      - PYTHONUNBUFFERED=1
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{port}/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 20s
    deploy:
      resources:
        limits:
          cpus: '{cpu_limit}'
          memory: {memory_limit}M
"""

KUBERNETES_TEMPLATE = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: wdbx
  labels:
    app: wdbx
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: wdbx
  template:
    metadata:
      labels:
        app: wdbx
    spec:
      containers:
      - name: wdbx
        image: python:{python_version}-slim
        ports:
        - containerPort: {port}
        volumeMounts:
        - name: data-volume
          mountPath: /data
        - name: log-volume
          mountPath: /logs
        env:
        - name: WDBX_DATA_DIR
          value: /data
        - name: PYTHONUNBUFFERED
          value: "1"
        command:
        - sh
        - -c
        - |
          pip install wdbx &&
          python -m wdbx.cli server \\
          --host 0.0.0.0 \\
          --port {port} \\
          --vector-dim {vector_dim} \\
          --shards {shards} \\
          --log-level {log_level}
        resources:
          limits:
            cpu: "{cpu_limit}"
            memory: "{memory_limit}Mi"
          requests:
            cpu: "{cpu_request}"
            memory: "{memory_request}Mi"
        livenessProbe:
          httpGet:
            path: /health
            port: {port}
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: {port}
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: wdbx-data-pvc
      - name: log-volume
        persistentVolumeClaim:
          claimName: wdbx-log-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: wdbx
spec:
  selector:
    app: wdbx
  ports:
  - port: {port}
    targetPort: {port}
  type: ClusterIP
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: wdbx-data-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: {storage_size}Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: wdbx-log-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
"""


if __name__ == "__main__":
    sys.exit(main())
