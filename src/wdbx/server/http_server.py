# wdbx/server.py
import asyncio
import json
import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, cast

import numpy as np
from aiohttp import web
from aiohttp.web import (
    Application,
    AppRunner,
    Request,
    Response,
    StreamResponse,
    TCPSite,
)

from wdbx import WDBX, create_wdbx

from ..core.config import WDBXConfig, load_config
from ..core.data_structures import EmbeddingVector

# Removed PersonaManager as it's accessed via wdbx_instance
# from wdbx.persona import PersonaManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("wdbx.server")

# API version prefix
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"

# Custom response helper functions


def success_response(data: Dict[str, Any], status: int = 200) -> Response:
    """Create a standardized success JSON response."""
    return web.json_response({
        "status": "success",
        "data": data,
        "timestamp": time.time()
    }, status=status)


def error_response(message: str, status: int = 400) -> Response:
    """Create a standardized error JSON response."""
    return web.json_response({
        "status": "error",
        "error": message,
        "timestamp": time.time()
    }, status=status)

# Request handlers


async def health_check(request: Request) -> Response:
    """Endpoint to check server health status."""
    return success_response({
        "status": "healthy",
        "version": request.app["config"]["version"],
        "uptime": time.time() - request.app["start_time"]
    })


async def system_stats(request: Request) -> Response:
    """Endpoint to retrieve system statistics."""
    try:
        stats = request.app["wdbx"].get_system_stats()
        return success_response(stats)
    except Exception as e:
        logger.error(f"Error fetching system stats: {str(e)}")
        return error_response(f"Failed to retrieve system statistics: {str(e)}", 500)


async def process_input(request: Request) -> Response:
    """Process user input through the persona manager."""
    try:
        # Cast the JSON data to provide type hints
        data = cast(Dict[str, Any], await request.json())
        user_input = data.get("input")
        context = data.get("context", {})
        session_id = data.get("session_id")

        if not user_input:
            return error_response("Missing required field: 'input'")

        persona_manager = request.app["persona_manager"]
        # Add check if persona_manager is None (if it wasn't enabled)
        if persona_manager is None:
            return error_response(
                "Persona management is not enabled on the server",
                501)  # 501 Not Implemented

        response, block_id = persona_manager.process_user_input(
            user_input,
            context,
            session_id=session_id
        )

        return success_response({
            "response": response,
            "block_id": block_id,
            "session_id": session_id
        })
    except json.JSONDecodeError:
        return error_response("Invalid JSON in request body")
    except Exception as e:
        logger.error(f"Error processing input: {str(e)}", exc_info=True)
        return error_response(f"Error processing input: {str(e)}", 500)


async def store_embedding(request: Request) -> Response:
    """Store a vector embedding with optional metadata."""
    try:
        data = await request.json()
        vector_data = data.get("vector")
        metadata = data.get("metadata", {})

        if not vector_data:
            return error_response("Missing required field: 'vector'")

        if not isinstance(vector_data, list):
            return error_response("Vector data must be a list of floating point values")

        try:
            vector = np.array(vector_data, dtype=np.float32)
        except ValueError:
            return error_response("Invalid vector format: must contain numeric values")

        # Check vector dimension
        expected_dim = request.app["wdbx"].vector_dimension
        if vector.shape[0] != expected_dim:
            return error_response(
                f"Vector dimension mismatch: expected {expected_dim}, got {
                    vector.shape[0]}")

        embedding_obj = EmbeddingVector(vector=vector, metadata=metadata)
        vector_id = request.app["wdbx"].store_embedding(embedding_obj)

        return success_response({"vector_id": vector_id})
    except json.JSONDecodeError:
        return error_response("Invalid JSON in request body")
    except Exception as e:
        logger.error(f"Error storing embedding: {str(e)}", exc_info=True)
        return error_response(f"Error storing embedding: {str(e)}", 500)


async def search_embeddings(request: Request) -> Response:
    """Search for similar vectors by vector or id."""
    try:
        data = await request.json()
        vector_data = data.get("vector")
        vector_id = data.get("vector_id")
        limit = int(data.get("limit", 10))
        threshold = float(data.get("threshold", 0.7))

        if not vector_data and not vector_id:
            return error_response(
                "Missing required field: either 'vector' or 'vector_id' must be provided")

        if limit < 1 or limit > 100:
            return error_response("Invalid limit: must be between 1 and 100")

        if threshold < 0 or threshold > 1:
            return error_response("Invalid threshold: must be between 0 and 1")

        wdbx = request.app["wdbx"]

        if vector_id:
            # Search by vector ID
            results = wdbx.search_by_id(vector_id, limit=limit, threshold=threshold)
        else:
            # Search by vector
            try:
                vector = np.array(vector_data, dtype=np.float32)
            except ValueError:
                return error_response("Invalid vector format: must contain numeric values")

            # Check vector dimension
            expected_dim = wdbx.vector_dimension
            if vector.shape[0] != expected_dim:
                return error_response(
                    f"Vector dimension mismatch: expected {expected_dim}, got {
                        vector.shape[0]}")

            results = wdbx.search_similar_vectors(vector, limit=limit, threshold=threshold)

        return success_response({"results": results})
    except json.JSONDecodeError:
        return error_response("Invalid JSON in request body")
    except Exception as e:
        logger.error(f"Error searching embeddings: {str(e)}", exc_info=True)
        return error_response(f"Error searching embeddings: {str(e)}", 500)


async def batch_process(request: Request) -> Response:
    """Process multiple inputs in a single batch request."""
    try:
        data = cast(Dict[str, Any], await request.json())
        inputs = cast(List[Dict[str, Any]], data.get("inputs", []))

        # Simplified check after casting
        if not inputs:
            return error_response("Missing or invalid 'inputs' field: must be a non-empty list")

        results: List[Dict[str, Any]] = []
        persona_manager = request.app["persona_manager"]

        if persona_manager is None:
            return error_response("Persona management is not enabled on the server", 501)

        for item in inputs:
            user_input = item.get("input")
            context = item.get("context", {})
            session_id = item.get("session_id")

            if not user_input:
                results.append({"status": "error", "error": "Missing input"})
                continue

            try:
                response, block_id = persona_manager.process_user_input(
                    user_input,
                    context,
                    session_id=session_id
                )
                results.append({
                    "status": "success",
                    "response": response,
                    "block_id": block_id,
                    "session_id": session_id
                })
            except Exception as e:
                results.append({
                    "status": "error",
                    "error": str(e),
                    "input": user_input  # Include input for context
                })

        return success_response({"results": results})

    except json.JSONDecodeError:
        return error_response("Invalid JSON in request body")
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}", exc_info=True)
        return error_response(f"Error in batch processing: {str(e)}", 500)

# Middleware for request logging and error handling
# Define the type for the handler callable
Handler = Callable[[Request], Awaitable[StreamResponse]]


@web.middleware
async def error_middleware(request: Request, handler: Handler) -> StreamResponse:
    """Middleware for logging requests and handling unexpected errors."""
    start_time = time.time()
    request_id = request.headers.get("X-Request-ID", "-")

    logger.info(f"Request started: {request.method} {request.path} (ID: {request_id})")

    try:
        response: StreamResponse = await handler(request)
        elapsed = time.time() - start_time
        # Use getattr for status in case response isn't always a Response object
        # (though it should be)
        status_code = getattr(response, "status", "N/A")
        logger.info(f"Request completed: {request.method} {request.path} "
                    f"- Status: {status_code} - {elapsed:.4f}s (ID: {request_id})")
        return response
    except web.HTTPException as ex:
        elapsed = time.time() - start_time
        logger.warning(f"HTTP exception: {request.method} {request.path} "
                       f"- Status: {ex.status} - {elapsed:.4f}s (ID: {request_id})")
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Unhandled exception in {request.method} {request.path} - {elapsed:.4f}s "
                     f"(ID: {request_id}): {str(e)}", exc_info=True)
        # Return our standardized error response
        return error_response("Internal server error", 500)


def create_app(config: Dict[str, Any], wdbx_instance: WDBX) -> Application:
    """Creates and configures the aiohttp web application.

    Args:
        config: Configuration dictionary.
        wdbx_instance: The initialized WDBX instance.

    Returns:
        Configured aiohttp Application.
    """
    app = Application(middlewares=[error_middleware])
    app["config"] = config
    app["start_time"] = time.time()
    app["wdbx"] = wdbx_instance  # Use the passed instance

    # Initialize PersonaManager using the passed WDBX instance
    if wdbx_instance.persona_manager:  # Check if persona management is enabled
        app["persona_manager"] = wdbx_instance.persona_manager
    else:
        # Handle case where persona manager might be needed but wasn't enabled
        logger.warning("Persona management not enabled in WDBX instance, /process endpoint may fail.")
        app["persona_manager"] = None  # Or provide a dummy/error handler

    # Add routes with API versioning prefix
    app.add_routes([
        web.get("/health", health_check),  # Keep health check at root level
        web.get(f"{API_PREFIX}/stats", system_stats),
        web.post(f"{API_PREFIX}/process", process_input),
        web.post(f"{API_PREFIX}/embedding", store_embedding),
        web.post(f"{API_PREFIX}/search", search_embeddings),
        web.post(f"{API_PREFIX}/batch", batch_process)
    ])

    # Add backward compatibility routes (without version prefix)
    # This allows existing clients to continue working
    app.add_routes([
        web.get("/stats", system_stats),
        web.post("/process", process_input),
        web.post("/embedding", store_embedding),
        web.post("/search", search_embeddings),
        web.post("/batch", batch_process)
    ])

    # Add documentation route
    async def api_docs(request):
        """Return API documentation"""
        docs = {
            "api_version": API_VERSION,
            "endpoints": [
                {"path": f"{API_PREFIX}/stats", "method": "GET", "description": "Get system statistics"},
                {"path": f"{API_PREFIX}/process", "method": "POST", "description": "Process user input"},
                {"path": f"{API_PREFIX}/embedding", "method": "POST",
                    "description": "Store embedding vector"},
                {"path": f"{API_PREFIX}/search", "method": "POST",
                    "description": "Search for similar vectors"},
                {"path": f"{API_PREFIX}/batch", "method": "POST",
                    "description": "Process multiple inputs in batch"}
            ],
            "deprecated_endpoints": [
                "/stats", "/process", "/embedding", "/search", "/batch"
            ]
        }
        return web.json_response(docs)

    app.add_routes([web.get(f"{API_PREFIX}", api_docs)])

    logger.info(f"Aiohttp application configured with API version {API_VERSION}.")
    return app


async def start_server(app: Application, host: str, port: int) -> None:
    """Starts the configured aiohttp server."""
    runner = AppRunner(app)
    await runner.setup()
    site = TCPSite(runner, host, port)
    await site.start()
    logger.info(f"Server started successfully on http://{host}:{port}")
    logger.info(f"API endpoints available at http://{host}:{port}{API_PREFIX}")
    # Keep server running (the loop will handle this outside this function)
    # while True:
    #     await asyncio.sleep(3600)


def run_server(
    wdbx_instance: Optional[WDBX] = None,
    host: str = "0.0.0.0",
    port: int = 8080,
    config: Optional[WDBXConfig] = None,
    # Additional parameters
    vector_dimension: Optional[int] = None,
    num_shards: Optional[int] = None,
    data_dir: Optional[str] = None,
) -> None:
    """
    Run the WDBX server.

    Args:
        wdbx_instance: Optional pre-initialized WDBX instance.
        host: The host address to bind the server to.
        port: The port to listen on.
        config: Optional WDBXConfig for initializing WDBX.
        vector_dimension: Optional vector dimension (used only if no config or instance).
        num_shards: Optional shard count (used only if no config or instance).
        data_dir: Optional data directory (used only if no config or instance).
    """
    # If no instance is provided, create one using config or parameters
    if wdbx_instance is None:
        if config is None:
            # Create config from individual parameters
            logger.info("Creating WDBXConfig from parameters")
            params = {}
            if vector_dimension is not None:
                params["vector_dimension"] = vector_dimension
            if num_shards is not None:
                params["num_shards"] = num_shards
            if data_dir is not None:
                params["data_dir"] = data_dir
            config = WDBXConfig(**params)

        logger.info(f"Initializing WDBX with config: {config.to_dict()}")
        wdbx_instance = create_wdbx(config=config)

    app_config = {
        "host": host,
        "port": port,
        "vector_dim": wdbx_instance.vector_dimension,
        "num_shards": wdbx_instance.num_shards,
        "version": getattr(wdbx_instance, "__version__", "unknown")
    }

    logger.info(f"Initializing WDBX server with config: {app_config}")
    logger.info(f"Using WDBX instance: {wdbx_instance}")

    loop = asyncio.get_event_loop()
    try:
        app = create_app(app_config, wdbx_instance)
        loop.run_until_complete(start_server(app, host, port))
        # Keep the server running until interrupted
        loop.run_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped by user (KeyboardInterrupt).")
    except Exception as e:
        logger.critical(f"Server failed to run: {e}", exc_info=True)
    finally:
        # Cleanup tasks if necessary (aiohttp runner cleanup is handled in start_server)
        if loop.is_running():
            logger.info("Stopping server loop...")
            loop.stop()
        pass  # Further cleanup might be needed depending on application structure


if __name__ == "__main__":
    import argparse
    # Local testing requires creating an instance first
    parser = argparse.ArgumentParser(description="WDBX Server (Local Test)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--vector-dim", type=int, default=128,
                        help="Dimension of embedding vectors for test instance")
    parser.add_argument("--num-shards", type=int, default=4,
                        help="Number of shards for test instance")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./test_server_data",
        help="Data directory for test instance")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file to use instead of command line arguments")
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Start in standalone mode"
    )

    args = parser.parse_args()

    if args.config:
        config_data = load_config(args.config)
        config = WDBXConfig.from_dict(config_data)

        print(f"Creating WDBX instance from config file: {args.config}")
        test_db = create_wdbx(config=config)
    elif args.standalone:
        logger.info("Starting in standalone mode")
        config = WDBXConfig(
            vector_dimension=args.vector_dim,
            num_shards=args.num_shards,
            data_dir=args.data_dir
        )
        db = WDBX(config=config)
    else:
        print("Creating temporary WDBX instance for local server test...")
        test_db = create_wdbx(
            vector_dimension=args.vector_dim,
            num_shards=args.num_shards,
            data_dir=args.data_dir
        )

    print(f"WDBX instance created. Data dir: {test_db.data_dir}")

    run_server(
        wdbx_instance=test_db,
        host=args.host,
        port=args.port
    )
