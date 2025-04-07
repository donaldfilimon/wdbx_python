#!/usr/bin/env python
"""
WDBX Command Line Interface.

This module provides a command-line interface for interacting with WDBX,
enabling users to perform operations like creating vectors, searching,
and managing data directly from the terminal.
"""

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .client import WDBXClient
from .utils.logging_utils import setup_logging, get_logger

# Initialize logger
logger = get_logger("wdbx.cli")


def setup_parser() -> argparse.ArgumentParser:
    """
    Set up the command-line argument parser.
    
    Returns:
        Configured argument parser
    """
    # Create the main parser
    parser = argparse.ArgumentParser(
        description="WDBX Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          wdbx --version
          wdbx init --data-dir ./data
          wdbx create-vector --data "[0.1, 0.2, 0.3]" --metadata '{"description": "Test vector"}'
          wdbx search --query-id 12345 --top-k 5
          wdbx export --output-dir ./export
        """)
    )
    
    # Add global arguments
    parser.add_argument('--version', action='store_true', help='Show version information')
    parser.add_argument('--data-dir', type=str, help='Data directory')
    parser.add_argument('--vector-dimension', type=int, default=1536, help='Vector dimension')
    parser.add_argument('--log-level', type=str, default='info', choices=['debug', 'info', 'warning', 'error', 'critical'], help='Logging level')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # 'init' command
    init_parser = subparsers.add_parser('init', help='Initialize WDBX')
    init_parser.add_argument('--force', action='store_true', help='Force initialization even if data directory exists')
    
    # 'create-vector' command
    create_vector_parser = subparsers.add_parser('create-vector', help='Create a new vector')
    create_vector_parser.add_argument('--data', type=str, required=True, help='Vector data as JSON array')
    create_vector_parser.add_argument('--metadata', type=str, help='Vector metadata as JSON object')
    create_vector_parser.add_argument('--id', type=str, help='Vector ID (generated if not provided)')
    create_vector_parser.add_argument('--save', action='store_true', help='Save the vector to disk')
    
    # 'create-block' command
    create_block_parser = subparsers.add_parser('create-block', help='Create a new block')
    create_block_parser.add_argument('--data', type=str, required=True, help='Block data as JSON object')
    create_block_parser.add_argument('--vectors', type=str, help='List of vector IDs to include in the block')
    create_block_parser.add_argument('--id', type=str, help='Block ID (generated if not provided)')
    create_block_parser.add_argument('--save', action='store_true', help='Save the block to disk')
    
    # 'search' command
    search_parser = subparsers.add_parser('search', help='Search for similar vectors')
    search_group = search_parser.add_mutually_exclusive_group(required=True)
    search_group.add_argument('--query-id', type=str, help='ID of the query vector')
    search_group.add_argument('--query-data', type=str, help='Query vector data as JSON array')
    search_parser.add_argument('--top-k', type=int, default=10, help='Number of results to return')
    search_parser.add_argument('--threshold', type=float, default=0.0, help='Minimum similarity threshold')
    search_parser.add_argument('--output-format', type=str, default='text', choices=['text', 'json'], help='Output format')
    
    # 'search-blocks' command
    search_blocks_parser = subparsers.add_parser('search-blocks', help='Search for relevant blocks')
    search_blocks_group = search_blocks_parser.add_mutually_exclusive_group(required=True)
    search_blocks_group.add_argument('--query-id', type=str, help='ID of the query vector')
    search_blocks_group.add_argument('--query-data', type=str, help='Query vector data as JSON array')
    search_blocks_group.add_argument('--query-text', type=str, help='Query text')
    search_blocks_parser.add_argument('--top-k', type=int, default=10, help='Number of results to return')
    search_blocks_parser.add_argument('--threshold', type=float, default=0.0, help='Minimum similarity threshold')
    search_blocks_parser.add_argument('--output-format', type=str, default='text', choices=['text', 'json'], help='Output format')
    
    # 'get' command
    get_parser = subparsers.add_parser('get', help='Get a vector or block by ID')
    get_parser.add_argument('--id', type=str, required=True, help='ID of the vector or block')
    get_parser.add_argument('--type', type=str, required=True, choices=['vector', 'block'], help='Type of object to get')
    get_parser.add_argument('--output-format', type=str, default='text', choices=['text', 'json'], help='Output format')
    
    # 'export' command
    export_parser = subparsers.add_parser('export', help='Export data to a directory')
    export_parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    export_parser.add_argument('--format', type=str, default='json', choices=['json', 'binary'], help='Export format')
    
    # 'import' command
    import_parser = subparsers.add_parser('import', help='Import data from a directory')
    import_parser.add_argument('--input-dir', type=str, required=True, help='Input directory')
    import_parser.add_argument('--format', type=str, default='json', choices=['json', 'binary'], help='Import format')
    
    # 'stats' command
    stats_parser = subparsers.add_parser('stats', help='Show statistics')
    stats_parser.add_argument('--output-format', type=str, default='text', choices=['text', 'json'], help='Output format')
    
    # 'clear' command
    clear_parser = subparsers.add_parser('clear', help='Clear all in-memory data')
    clear_parser.add_argument('--confirm', action='store_true', help='Confirm clearing data')
    
    # 'optimize' command
    optimize_parser = subparsers.add_parser('optimize', help='Optimize memory usage')
    
    # 'server' command
    server_parser = subparsers.add_parser('server', help='Start the WDBX server')
    server_parser.add_argument('--host', type=str, default='127.0.0.1', help='Server host')
    server_parser.add_argument('--port', type=int, default=8000, help='Server port')
    server_parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
    
    return parser


def create_client(args: argparse.Namespace) -> WDBXClient:
    """
    Create a WDBX client from command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        WDBX client
    """
    # Create the client
    client = WDBXClient(
        data_dir=args.data_dir,
        vector_dimension=args.vector_dimension,
        enable_memory_optimization=True,
        config_path=args.config,
    )
    
    return client


def handle_init(args: argparse.Namespace) -> int:
    """
    Handle the 'init' command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    if args.data_dir is None:
        logger.error("Data directory is required for 'init' command")
        return 1
    
    data_dir = Path(args.data_dir)
    
    # Check if data directory exists
    if data_dir.exists() and not args.force:
        logger.error(f"Data directory '{data_dir}' already exists, use --force to initialize anyway")
        return 1
    
    # Create data directory
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (data_dir / "vectors").mkdir(exist_ok=True)
    (data_dir / "blocks").mkdir(exist_ok=True)
    
    # Create a client to initialize the database
    client = create_client(args)
    client.connect()
    client.disconnect()
    
    logger.info(f"Initialized WDBX at {data_dir}")
    return 0


def handle_create_vector(args: argparse.Namespace) -> int:
    """
    Handle the 'create-vector' command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    try:
        # Parse vector data
        vector_data = json.loads(args.data)
        if not isinstance(vector_data, list):
            logger.error("Vector data must be a JSON array")
            return 1
        
        # Parse metadata
        metadata = None
        if args.metadata:
            metadata = json.loads(args.metadata)
            if not isinstance(metadata, dict):
                logger.error("Metadata must be a JSON object")
                return 1
        
        # Create client
        client = create_client(args)
        client.connect()
        
        # Create vector
        vector = client.create_vector(
            vector_data=vector_data,
            metadata=metadata,
            vector_id=args.id,
        )
        
        # Save vector if requested
        if args.save:
            client.save_vector(vector)
            logger.info(f"Vector saved with ID: {vector.vector_id}")
        else:
            logger.info(f"Vector created with ID: {vector.vector_id}")
        
        # Print vector details
        print(json.dumps({
            "id": vector.vector_id,
            "dimension": len(vector.vector),
            "metadata": vector.metadata,
            "timestamp": vector.timestamp,
        }, indent=2))
        
        client.disconnect()
        return 0
    except Exception as e:
        logger.error(f"Failed to create vector: {str(e)}")
        return 1


def handle_create_block(args: argparse.Namespace) -> int:
    """
    Handle the 'create-block' command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    try:
        # Parse block data
        data = json.loads(args.data)
        if not isinstance(data, dict):
            logger.error("Block data must be a JSON object")
            return 1
        
        # Parse vector IDs
        vector_ids = []
        if args.vectors:
            vector_ids = json.loads(args.vectors)
            if not isinstance(vector_ids, list):
                logger.error("Vector IDs must be a JSON array")
                return 1
        
        # Create client
        client = create_client(args)
        client.connect()
        
        # Get vectors
        vectors = []
        for vector_id in vector_ids:
            vector = client.get_vector(vector_id)
            if vector is None:
                logger.error(f"Vector not found: {vector_id}")
                client.disconnect()
                return 1
            vectors.append(vector)
        
        # Create block
        block = client.create_block(
            data=data,
            embeddings=vectors,
            block_id=args.id,
        )
        
        # Save block if requested
        if args.save:
            client.save_block(block)
            logger.info(f"Block saved with ID: {block.block_id}")
        else:
            logger.info(f"Block created with ID: {block.block_id}")
        
        # Print block details
        print(json.dumps({
            "id": block.block_id,
            "data": block.data,
            "vectors": [v.vector_id for v in block.embeddings] if block.embeddings else [],
            "timestamp": block.timestamp,
        }, indent=2))
        
        client.disconnect()
        return 0
            except Exception as e:
        logger.error(f"Failed to create block: {str(e)}")
        return 1


def handle_search(args: argparse.Namespace) -> int:
    """
    Handle the 'search' command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    try:
        # Create client
        client = create_client(args)
        client.connect()
        
        # Prepare query
        if args.query_id:
            query = args.query_id
        else:
            query_data = json.loads(args.query_data)
            if not isinstance(query_data, list):
                logger.error("Query data must be a JSON array")
                client.disconnect()
                return 1
            query = np.array(query_data, dtype=np.float32)
        
        # Search
        results = client.find_similar_vectors(
            query=query,
            top_k=args.top_k,
            threshold=args.threshold,
        )
        
        # Format output
        if args.output_format == 'json':
            output = []
            for vector_id, similarity in results:
                vector = client.get_vector(vector_id)
                metadata = vector.metadata if vector else None
                output.append({
                    "id": vector_id,
                    "similarity": similarity,
                    "metadata": metadata,
                })
            print(json.dumps(output, indent=2))
        else:
            print(f"Found {len(results)} similar vectors:")
            for i, (vector_id, similarity) in enumerate(results):
                print(f"{i+1}. ID: {vector_id}, Similarity: {similarity:.4f}")
        
        client.disconnect()
        return 0
    except Exception as e:
        logger.error(f"Failed to search: {str(e)}")
        return 1


def handle_search_blocks(args: argparse.Namespace) -> int:
    """
    Handle the 'search-blocks' command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    try:
        # Create client
        client = create_client(args)
        client.connect()
        
        # Prepare query
        if args.query_id:
            query = args.query_id
        elif args.query_data:
            query_data = json.loads(args.query_data)
            if not isinstance(query_data, list):
                logger.error("Query data must be a JSON array")
                client.disconnect()
                return 1
            query = np.array(query_data, dtype=np.float32)
        else:
            query = args.query_text
        
        # Search
        results = client.search_blocks(
            query=query,
            top_k=args.top_k,
            threshold=args.threshold,
        )
        
        # Format output
        if args.output_format == 'json':
            output = []
            for block, similarity in results:
                output.append({
                    "id": block.block_id,
                    "similarity": similarity,
                    "data": block.data,
                    "vectors": [v.vector_id for v in block.embeddings] if block.embeddings else [],
                })
            print(json.dumps(output, indent=2))
        else:
            print(f"Found {len(results)} relevant blocks:")
            for i, (block, similarity) in enumerate(results):
                print(f"{i+1}. ID: {block.block_id}, Similarity: {similarity:.4f}")
                if block.data and isinstance(block.data, dict) and 'name' in block.data:
                    print(f"   Name: {block.data['name']}")
        
        client.disconnect()
        return 0
    except Exception as e:
        logger.error(f"Failed to search blocks: {str(e)}")
        return 1


def handle_get(args: argparse.Namespace) -> int:
    """
    Handle the 'get' command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    try:
        # Create client
        client = create_client(args)
        client.connect()
        
        # Get object
        if args.type == 'vector':
            obj = client.get_vector(args.id)
            if obj is None:
                logger.error(f"Vector not found: {args.id}")
                client.disconnect()
                return 1
            
            # Format output
            if args.output_format == 'json':
                output = obj.to_dict()
                print(json.dumps(output, indent=2))
            else:
                print(f"Vector ID: {obj.vector_id}")
                print(f"Dimension: {len(obj.vector)}")
                print(f"Metadata: {obj.metadata}")
                print(f"Timestamp: {obj.timestamp}")
    else:
            obj = client.get_block(args.id)
            if obj is None:
                logger.error(f"Block not found: {args.id}")
                client.disconnect()
                return 1
            
            # Format output
            if args.output_format == 'json':
                output = obj.to_dict()
                print(json.dumps(output, indent=2))
            else:
                print(f"Block ID: {obj.block_id}")
                print(f"Data: {obj.data}")
                if obj.embeddings:
                    print(f"Vectors: {[v.vector_id for v in obj.embeddings]}")
                print(f"Timestamp: {obj.timestamp}")
        
        client.disconnect()
        return 0
    except Exception as e:
        logger.error(f"Failed to get object: {str(e)}")
        return 1


def handle_export(args: argparse.Namespace) -> int:
    """
    Handle the 'export' command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    try:
        # Create client
        client = create_client(args)
        client.connect()
        
        # Export data
        success = client.export_data(
            output_dir=args.output_dir,
            format=args.format,
        )
        
        if success:
            logger.info(f"Data exported to {args.output_dir}")
        else:
            logger.error("Failed to export data")
            client.disconnect()
            return 1
        
        client.disconnect()
        return 0
    except Exception as e:
        logger.error(f"Failed to export data: {str(e)}")
        return 1


def handle_import(args: argparse.Namespace) -> int:
    """
    Handle the 'import' command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    try:
        # Create client
        client = create_client(args)
        client.connect()
        
        # Import data
        success = client.import_data(
            input_dir=args.input_dir,
            format=args.format,
        )
        
        if success:
            logger.info(f"Data imported from {args.input_dir}")
        else:
            logger.error("Failed to import data")
            client.disconnect()
            return 1
        
        client.disconnect()
        return 0
    except Exception as e:
        logger.error(f"Failed to import data: {str(e)}")
        return 1


def handle_stats(args: argparse.Namespace) -> int:
    """
    Handle the 'stats' command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    try:
        # Create client
        client = create_client(args)
        client.connect()
        
        # Get stats
        stats = client.get_stats()
        
        # Format output
        if args.output_format == 'json':
            print(json.dumps(stats, indent=2))
        else:
            print("WDBX Statistics:")
            for key, value in stats.items():
                print(f"{key}: {value}")
        
        client.disconnect()
        return 0
    except Exception as e:
        logger.error(f"Failed to get statistics: {str(e)}")
        return 1


def handle_clear(args: argparse.Namespace) -> int:
    """
    Handle the 'clear' command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    try:
        # Check confirmation
        if not args.confirm:
            print("Warning: This will clear all in-memory data.")
            response = input("Are you sure you want to continue? [y/N] ")
            if response.lower() != 'y':
                print("Operation cancelled.")
                return 0
        
        # Create client
        client = create_client(args)
        client.connect()
        
        # Clear data
        client.clear()
        logger.info("All in-memory data cleared")
        
        client.disconnect()
        return 0
    except Exception as e:
        logger.error(f"Failed to clear data: {str(e)}")
        return 1


def handle_optimize(args: argparse.Namespace) -> int:
    """
    Handle the 'optimize' command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    try:
        # Create client
        client = create_client(args)
        client.connect()
        
        # Get stats before optimization
        stats_before = client.get_stats()
        if 'memory_usage' in stats_before:
            memory_before = stats_before['memory_usage']
            logger.info(f"Memory usage before optimization: {memory_before / 1024 / 1024:.2f} MB")
        
        # Optimize memory
        client.optimize_memory()
        logger.info("Memory optimization completed")
        
        # Get stats after optimization
        stats_after = client.get_stats()
        if 'memory_usage' in stats_after:
            memory_after = stats_after['memory_usage']
            logger.info(f"Memory usage after optimization: {memory_after / 1024 / 1024:.2f} MB")
            if 'memory_usage' in stats_before:
                reduction = memory_before - memory_after
                percentage = (reduction / memory_before) * 100 if memory_before > 0 else 0
                logger.info(f"Memory reduction: {reduction / 1024 / 1024:.2f} MB ({percentage:.2f}%)")
        
        client.disconnect()
        return 0
    except Exception as e:
        logger.error(f"Failed to optimize memory: {str(e)}")
        return 1


def handle_server(args: argparse.Namespace) -> int:
    """
    Handle the 'server' command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    try:
        from .server import run_server
        
        # Run server
        run_server(
            host=args.host,
            port=args.port,
            data_dir=args.data_dir,
            vector_dimension=args.vector_dimension,
            config_path=args.config,
            workers=args.workers,
        )
        
        return 0
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        return 1


def main() -> int:
    """
    Main entry point for the CLI.
    
    Returns:
        Exit code
    """
    # Parse arguments
    parser = setup_parser()
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(log_level=args.log_level.upper())
    
    # Handle version command
    if args.version:
        from . import __version__
        print(f"WDBX version {__version__}")
        return 0
    
    # Handle command
    if args.command == 'init':
        return handle_init(args)
    elif args.command == 'create-vector':
        return handle_create_vector(args)
    elif args.command == 'create-block':
        return handle_create_block(args)
    elif args.command == 'search':
        return handle_search(args)
    elif args.command == 'search-blocks':
        return handle_search_blocks(args)
    elif args.command == 'get':
        return handle_get(args)
    elif args.command == 'export':
        return handle_export(args)
    elif args.command == 'import':
        return handle_import(args)
    elif args.command == 'stats':
        return handle_stats(args)
    elif args.command == 'clear':
        return handle_clear(args)
    elif args.command == 'optimize':
        return handle_optimize(args)
    elif args.command == 'server':
        return handle_server(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main()) 