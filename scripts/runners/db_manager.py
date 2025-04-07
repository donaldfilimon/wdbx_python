#!/usr/bin/env python3
"""
WDBX Database Manager

This script provides utilities for managing WDBX database operations:
- Initialize database
- Check database status
- Backup database
- Restore database
- Clean up and optimize database
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("db_manager")

# Add project root to Python path
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    # Import WDBX components
    from wdbx.config import WDBXConfig
    from wdbx.data_structures import EmbeddingVector
    from wdbx.storage.vector_store import VectorStore
except ImportError as e:
    logger.error(f"Failed to import WDBX modules: {e}")
    logger.error("Make sure the WDBX package is installed or in your Python path")
    sys.exit(1)

def get_data_directory():
    """Get the data directory from environment or default."""
    data_dir = os.environ.get("WDBX_DATA_DIR")
    if not data_dir:
        data_dir = PROJECT_ROOT / "data"
    return Path(data_dir)

def get_db_info(data_dir):
    """Get information about the database."""
    try:
        db_info = {
            "data_directory": str(data_dir),
            "exists": data_dir.exists(),
            "size_bytes": 0,
            "last_modified": None,
            "shard_count": 0,
            "vector_files": 0,
            "index_files": 0,
            "metadata_files": 0,
            "total_files": 0
        }
        
        if not db_info["exists"]:
            return db_info
            
        # Get directory stats
        total_size = 0
        latest_mtime = 0
        for root, dirs, files in os.walk(data_dir):
            root_path = Path(root)
            for file in files:
                file_path = root_path / file
                total_size += file_path.stat().st_size
                mtime = file_path.stat().st_mtime
                if mtime > latest_mtime:
                    latest_mtime = mtime
                
                # Count file types
                if file.endswith(".vector"):
                    db_info["vector_files"] += 1
                elif file.endswith(".index"):
                    db_info["index_files"] += 1
                elif file.endswith(".meta") or file.endswith(".json"):
                    db_info["metadata_files"] += 1
                
                db_info["total_files"] += 1
        
        # Update database info
        db_info["size_bytes"] = total_size
        db_info["size_human"] = format_size(total_size)
        if latest_mtime > 0:
            db_info["last_modified"] = datetime.fromtimestamp(latest_mtime).isoformat()
        
        # Count shards if shard directory exists
        shard_dir = data_dir / "shards"
        if shard_dir.exists():
            db_info["shard_count"] = sum(1 for _ in shard_dir.glob("shard_*"))
        
        return db_info
    
    except Exception as e:
        logger.error(f"Error getting database info: {e}")
        return {"error": str(e)}

def format_size(size_bytes):
    """Format size in bytes to human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024 or unit == "TB":
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024

def initialize_database(data_dir, force=False):
    """Initialize the database directory structure."""
    try:
        # Create main directories
        directories = [
            data_dir,
            data_dir / "shards",
            data_dir / "indexes",
            data_dir / "metadata",
            data_dir / "cache"
        ]
        
        for directory in directories:
            if directory.exists() and not force:
                logger.info(f"Directory already exists: {directory}")
            else:
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {directory}")
        
        # Create metadata files
        meta_file = data_dir / "metadata" / "db_info.json"
        meta = {
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "description": "WDBX Vector Database",
            "shard_count": 0,
            "vector_dimension": 768  # Default dimension
        }
        
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)
            
        logger.info(f"Initialized database at {data_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False

def backup_database(data_dir, target_file=None):
    """Backup the database to a compressed archive."""
    try:
        # Create backup filename if not provided
        if not target_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target_file = PROJECT_ROOT / f"wdbx_backup_{timestamp}.zip"
        else:
            target_file = Path(target_file)
        
        # Make sure the target directory exists
        target_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create the backup
        logger.info(f"Backing up database from {data_dir} to {target_file}")
        
        # Using shutil.make_archive
        backup_path = shutil.make_archive(
            str(target_file).replace(".zip", ""),
            "zip",
            root_dir=str(data_dir.parent),
            base_dir=data_dir.name
        )
        
        logger.info(f"Backup completed: {backup_path}")
        backup_size = Path(backup_path).stat().st_size
        logger.info(f"Backup size: {format_size(backup_size)}")
        
        return backup_path
        
    except Exception as e:
        logger.error(f"Error backing up database: {e}")
        return None

def restore_database(target_file, data_dir):
    """Restore database from a compressed archive."""
    try:
        target_file = Path(target_file)
        if not target_file.exists():
            logger.error(f"Backup file not found: {target_file}")
            return False
            
        # Create a temporary extraction directory
        temp_dir = PROJECT_ROOT / "tmp_restore"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True)
        
        logger.info(f"Restoring database from {target_file} to {data_dir}")
        
        # Extract the backup
        shutil.unpack_archive(str(target_file), str(temp_dir))
        
        # Find the extracted data directory
        extracted_dirs = list(temp_dir.glob("*"))
        if not extracted_dirs:
            logger.error("No data found in backup")
            return False
            
        # If data directory already exists, make a backup
        if data_dir.exists():
            backup_dir = Path(str(data_dir) + f".bak_{int(time.time())}")
            logger.info(f"Moving existing data directory to {backup_dir}")
            shutil.move(str(data_dir), str(backup_dir))
        
        # Move extracted data to the target location
        if len(extracted_dirs) == 1 and extracted_dirs[0].is_dir():
            # Single directory in the archive
            shutil.move(str(extracted_dirs[0]), str(data_dir))
        else:
            # Multiple files/directories in the archive
            data_dir.mkdir(parents=True, exist_ok=True)
            for item in extracted_dirs:
                destination = data_dir / item.name
                shutil.move(str(item), str(destination))
        
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        logger.info(f"Restored database to {data_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error restoring database: {e}")
        return False

def cleanup_database(data_dir):
    """Clean up and optimize the database."""
    try:
        # Import vector store and initialize
        logger.info("Loading vector store to perform cleanup...")
        
        # First check if db exists
        if not data_dir.exists():
            logger.error(f"Database directory does not exist: {data_dir}")
            return False
            
        # Get initial database info for comparison
        initial_info = get_db_info(data_dir)
        
        # Load vector store
        config = WDBXConfig(data_dir=str(data_dir))
        vector_store = VectorStore(config=config)
        
        # Perform optimization
        logger.info("Running store optimization...")
        optimized = vector_store.optimize()
        
        # Remove temporary files
        temp_files = list(data_dir.glob("*.tmp"))
        for tmp in temp_files:
            logger.info(f"Removing temporary file: {tmp}")
            tmp.unlink()
            
        # Remove empty directories
        for root, dirs, files in os.walk(data_dir, topdown=False):
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                if not any(dir_path.iterdir()):
                    logger.info(f"Removing empty directory: {dir_path}")
                    dir_path.rmdir()
        
        # Get final database info
        final_info = get_db_info(data_dir)
        
        # Calculate space saved
        bytes_saved = initial_info["size_bytes"] - final_info["size_bytes"]
        if bytes_saved > 0:
            logger.info(f"Cleanup complete. Saved {format_size(bytes_saved)}")
        else:
            logger.info("Cleanup complete. No space saved.")
            
        return True
        
    except Exception as e:
        logger.error(f"Error cleaning up database: {e}")
        return False

def main():
    """Main entry point for the database manager."""
    parser = argparse.ArgumentParser(description="WDBX Database Manager")
    
    # Command argument
    parser.add_argument("command", choices=["init", "status", "backup", "restore", "cleanup"],
                        help="Database command to execute")
                        
    # Optional arguments
    parser.add_argument("--target", help="Target file for backup/restore")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing data")
    parser.add_argument("--data-dir", help="Override data directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = get_data_directory()
    
    logger.debug(f"Using data directory: {data_dir}")
    
    # Execute command
    if args.command == "init":
        if initialize_database(data_dir, args.force):
            logger.info("Database initialization successful")
            return 0
        logger.error("Database initialization failed")
        return 1
    
    if args.command == "status":
        db_info = get_db_info(data_dir)
        
        # Print formatted status
        print("\n=== WDBX Database Status ===\n")
        print(f"Data Directory: {db_info['data_directory']}")
        
        if not db_info.get("exists", False):
            print("Status: Not initialized")
            print("\nRun 'wdbx_tool.py db init' to initialize the database")
            return 1
            
        print(f"Size: {db_info.get('size_human', 'Unknown')}")
        print(f"Last Modified: {db_info.get('last_modified', 'Unknown')}")
        print(f"Shard Count: {db_info.get('shard_count', 0)}")
        print("File Counts:")
        print(f"  Vector Files: {db_info.get('vector_files', 0)}")
        print(f"  Index Files: {db_info.get('index_files', 0)}")
        print(f"  Metadata Files: {db_info.get('metadata_files', 0)}")
        print(f"  Total Files: {db_info.get('total_files', 0)}")
        print()
        
        return 0
    
    if args.command == "backup":
        backup_path = backup_database(data_dir, args.target)
        if backup_path:
            logger.info(f"Backup created successfully: {backup_path}")
            return 0
        logger.error("Backup failed")
        return 1
    
    if args.command == "restore":
        if not args.target:
            logger.error("You must specify a target backup file with --target")
            return 1
            
        if restore_database(args.target, data_dir):
            logger.info("Database restored successfully")
            return 0
        logger.error("Database restoration failed")
        return 1
    
    if args.command == "cleanup":
        if cleanup_database(data_dir):
            logger.info("Database cleanup successful")
            return 0
        logger.error("Database cleanup failed")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 