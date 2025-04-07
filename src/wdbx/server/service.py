"""
OS-level service integration for WDBX.

This module provides tools to install WDBX as a system service on
various platforms (Linux, Windows, macOS) and container environments.
"""

import argparse
import os
import platform
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from ..constants import logger
from ..templates import SYSTEMD_SERVICE_TEMPLATE


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


class ServiceWorker(ABC):
    """Abstract base class for background service workers."""

    @abstractmethod
    def start(self) -> None:
        """Start the worker process."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the worker process."""
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """Check if the worker is running."""
        pass


class ServiceManager:
    """Manages WDBX as a background service."""

    def __init__(self, wdbx_instance=None):
        self.wdbx = wdbx_instance
        self.workers: List[ServiceWorker] = []
        self.monitor_thread = None
        self.running = False
        logger.info("ServiceManager initialized")

    def add_worker(self, worker: ServiceWorker) -> None:
        """Add a worker to be managed."""
        self.workers.append(worker)
        logger.info(f"Added worker: {type(worker).__name__}")

    def start_all(self) -> None:
        """Start all managed workers."""
        if self.running:
            logger.warning("ServiceManager already running")
            return

        logger.info("Starting all service workers...")
        self.running = True
        for worker in self.workers:
            try:
                worker.start()
                logger.info(f"Started worker: {type(worker).__name__}")
            except Exception as e:
                logger.error(f"Failed to start worker {type(worker).__name__}: {e}")
        self._start_monitoring()
        logger.info("All service workers started.")

    def stop_all(self) -> None:
        """Stop all managed workers."""
        if not self.running:
            logger.warning("ServiceManager is not running")
            return

        logger.info("Stopping all service workers...")
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()

        for worker in reversed(self.workers):  # Stop in reverse order
            try:
                worker.stop()
                logger.info(f"Stopped worker: {type(worker).__name__}")
            except Exception as e:
                logger.error(f"Failed to stop worker {type(worker).__name__}: {e}")
        logger.info("All service workers stopped.")

    def _start_monitoring(self) -> None:
        """Start the monitoring thread."""
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _monitor_loop(self) -> None:
        """Monitor worker health and restart if necessary."""
        logger.info("Starting service monitor loop")
        while self.running:
            for worker in self.workers:
                try:
                    if not worker.is_running():
                        logger.warning(f"Worker {type(worker).__name__} is not running. Attempting restart.")
                        worker.stop()  # Ensure clean stop before restart
                        worker.start()
                        logger.info(f"Restarted worker: {type(worker).__name__}")
                except Exception as e:
                    logger.error(f"Error monitoring worker {type(worker).__name__}: {e}")
            time.sleep(10)  # Check every 10 seconds
        logger.info("Stopping service monitor loop")

    def install_systemd_service(self, service_name="wdbx", user=None, group=None) -> bool:
        """Install WDBX as a systemd service (Linux only)."""
        if platform.system() != "Linux":
            logger.error("Systemd service installation is only supported on Linux.")
            return False

        if os.geteuid() != 0:
            logger.error("Root privileges are required to install systemd service.")
            return False

        python_executable = sys.executable
        script_path = os.path.abspath(sys.argv[0]) # Assuming the entry point script
        working_directory = os.getcwd()

        # Get effective user/group if not specified
        try:
            import grp
            import pwd
            effective_user = pwd.getpwuid(os.geteuid()).pw_name
            effective_group = grp.getgrgid(os.getegid()).gr_name
            user = user or effective_user
            group = group or effective_group
        except ImportError:
            logger.warning("pwd/grp modules not available on this system. Using default user/group.")
            user = user or "root"
            group = group or "root"

        service_content = SYSTEMD_SERVICE_TEMPLATE.format(
            description="WDBX Database Service",
            user=user,
            group=group,
            working_directory=working_directory,
            python_executable=python_executable,
            script_path=script_path,
            service_args="--server"  # Example: run in server mode
        )

        service_file_path = f"/etc/systemd/system/{service_name}.service"

        try:
            with open(service_file_path, "w") as f:
                f.write(service_content)
            logger.info(f"Created systemd service file: {service_file_path}")

            # Reload systemd daemon
            subprocess.check_call(["systemctl", "daemon-reload"])
            logger.info("Systemd daemon reloaded.")

            # Enable the service
            subprocess.check_call(["systemctl", "enable", service_name])
            logger.info(f"Enabled systemd service: {service_name}")

            print(f"Systemd service '{service_name}' installed successfully.")
            print(f"Start with: sudo systemctl start {service_name}")
            print(f"Check status: sudo systemctl status {service_name}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Systemd command failed: {e}")
            return False
        except OSError as e:
            logger.error(f"Failed to write service file: {e}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during service installation: {e}")
            return False

    def uninstall_systemd_service(self, service_name="wdbx") -> bool:
        """Uninstall the WDBX systemd service (Linux only)."""
        if platform.system() != "Linux":
            logger.error("Systemd service management is only supported on Linux.")
            return False
        
        if os.geteuid() != 0:
            logger.error("Root privileges are required to uninstall systemd service.")
            return False

        service_file_path = f"/etc/systemd/system/{service_name}.service"

        try:
            # Stop the service if running
            logger.info(f"Stopping service {service_name}...")
            subprocess.run(["systemctl", "stop", service_name], check=False)

            # Disable the service
            logger.info(f"Disabling service {service_name}...")
            subprocess.run(["systemctl", "disable", service_name], check=False)

            # Remove the service file
            if os.path.exists(service_file_path):
                os.remove(service_file_path)
                logger.info(f"Removed service file: {service_file_path}")

            # Reload systemd daemon
            subprocess.check_call(["systemctl", "daemon-reload"])
            logger.info("Systemd daemon reloaded.")

            print(f"Systemd service '{service_name}' uninstalled successfully.")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Systemd command failed during uninstall: {e}")
            return False
        except OSError as e:
            logger.error(f"Failed to remove service file: {e}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during service uninstallation: {e}")
            return False


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


class MacServiceManager(ServiceManager):
    """macOS launchd service manager."""

    def install(self) -> bool:
        """Install launchd service."""
        # Implementation would go here
        return False

    def uninstall(self) -> bool:
        """Remove launchd service."""
        # Implementation would go here
        return False


class WindowsServiceManager(ServiceManager):
    """Windows service manager."""

    def install(self) -> bool:
        """Install Windows service."""
        # Implementation would go here
        return False

    def uninstall(self) -> bool:
        """Remove Windows service."""
        # Implementation would go here
        return False


class ServiceFactory:
    """Factory for creating platform-specific service managers."""

    @staticmethod
    def create(config: ServiceConfig) -> ServiceManager:
        """Create appropriate service manager for current platform."""
        system = platform.system()

        if system == "Linux":
            return LinuxServiceManager(config)
        if system == "Darwin":
            return MacServiceManager(config)
        if system == "Windows":
            return WindowsServiceManager(config)
        raise RuntimeError(f"Unsupported platform: {system}")


def generate_docker_compose(args: argparse.Namespace) -> bool:
    """Generate Docker Compose configuration file."""
    # Implementation would go here
    return False


def generate_kubernetes(args: argparse.Namespace) -> bool:
    """Generate Kubernetes configuration files."""
    # Implementation would go here
    return False


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
            parser = argparse.ArgumentParser(description="WDBX Service Manager")
            parser.print_help()
            return 1

        return 0 if success else 1

    except Exception as e:
        logger.error(f"Operation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())


def is_admin() -> bool:
    """Check if the script is running with administrative privileges."""
    # This function seems to be a duplicate of ServiceManager.check_admin
    # Implementation would go here
    return False
