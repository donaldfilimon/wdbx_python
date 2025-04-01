# # wdbx/server.py
# import asyncio
# import json
# import logging
# from typing import Dict, Any, Optional, Tuple, List
# import numpy as np
# from aiohttp import web
# from wdbx import WDBX
# from wdbx.persona import PersonaManager
# from wdbx.data_structures import EmbeddingVector

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# async def health(request: web.Request) -> web.Response:
#     """Health check endpoint."""
#     return web.json_response({"status": "healthy"})

# async def stats(request: web.Request) -> web.Response:
#     """Get system statistics."""
#     try:
#         stats = request.app["wdbx"].get_system_stats()
#         return web.json_response(stats)
#     except Exception as e:
#         logger.error(f"Error retrieving stats: {str(e)}")
#         return web.json_response({"error": str(e)}, status=500)

# async def process(request: web.Request) -> web.Response:
#     """Process user input through persona manager."""
#     try:
#         data = await request.json()
#         user_input = data.get("input")
#         context = data.get("context", {})

#         if not user_input:
#             return web.json_response({"error": "Missing input"}, status=400)

#         persona_manager = request.app["persona_manager"]
#         response, block_id = persona_manager.process_user_input(user_input, context)
#         return web.json_response({"response": response, "block_id": block_id})
#     except json.JSONDecodeError:
#         return web.json_response({"error": "Invalid JSON"}, status=400)
#     except Exception as e:
#         logger.error(f"Error processing input: {str(e)}")
#         return web.json_response({"error": str(e)}, status=500)

# async def embedding(request: web.Request) -> web.Response:
#     """Store vector embeddings with metadata."""
#     try:
#         data = await request.json()
#         vector_data = data.get("vector")
#         metadata = data.get("metadata", {})

#         if not vector_data:
#             return web.json_response({"error": "Missing vector data"}, status=400)

#         # Validate vector format
#         if not isinstance(vector_data, list):
#             return web.json_response({"error": "Vector must be a list of numbers"}, status=400)

#         vector = np.array(vector_data, dtype=np.float32)
#         embedding_obj = EmbeddingVector(vector=vector, metadata=metadata)
#         vector_id = request.app["wdbx"].store_embedding(embedding_obj)
#         return web.json_response({"vector_id": vector_id})
#     except json.JSONDecodeError:
#         return web.json_response({"error": "Invalid JSON"}, status=400)
#     except Exception as e:
#         logger.error(f"Error storing embedding: {str(e)}")
#         return web.json_response({"error": str(e)}, status=500)

# async def search(request: web.Request) -> web.Response:
#     """Search for similar vectors."""
#     try:
#         data = await request.json()
#         vector_data = data.get("vector")
#         limit = data.get("limit", 10)  # Default to 10 results

#         if not vector_data:
#             return web.json_response({"error": "Missing vector data"}, status=400)

#         # Validate vector format
#         if not isinstance(vector_data, list):
#             return web.json_response({"error": "Vector must be a list of numbers"}, status=400)

#         vector = np.array(vector_data, dtype=np.float32)
#         results = request.app["wdbx"].search_similar_vectors(vector, limit=limit)
#         return web.json_response({"results": results})
#     except json.JSONDecodeError:
#         return web.json_response({"error": "Invalid JSON"}, status=400)
#     except Exception as e:
#         logger.error(f"Error searching vectors: {str(e)}")
#         return web.json_response({"error": str(e)}, status=500)

# def create_app(wdbx_instance: WDBX, persona_manager: PersonaManager) -> web.Application:
#     """Create and configure the application."""
#     app = web.Application()
#     app["wdbx"] = wdbx_instance
#     app["persona_manager"] = persona_manager

#     # Add routes
#     app.add_routes([
#         web.get("/health", health),
#         web.get("/stats", stats),
#         web.post("/process", process),
#         web.post("/embedding", embedding),
#         web.post("/search", search)
#     ])

#     # Add middleware for CORS if needed
#     # app.add_middleware(...)

#     return app

# def run_server(host: str, port: int, vector_dim: int, num_shards: int) -> None:
#     """Run the WDBX server with the specified configuration."""
#     try:
#         logger.info(f"Starting WDBX server on {host}:{port}")
#         logger.info(f"Vector dimension: {vector_dim}, Shards: {num_shards}")

#         wdbx_instance = WDBX(vector_dimension=vector_dim, num_shards=num_shards)
#         persona_manager = PersonaManager(wdbx_instance)

#         app = create_app(wdbx_instance, persona_manager)
#         web.run_app(app, host=host, port=port, access_log=logger)
#     except Exception as e:
#         logger.critical(f"Failed to start server: {str(e)}")
#         raise

# wdbx/server.py
import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
from aiohttp import web
from aiohttp.web import Request, Response, Application, AppRunner, TCPSite

from wdbx import WDBX
from wdbx.data_structures import EmbeddingVector
from wdbx.persona import PersonaManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("wdbx.server")

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
        data = await request.json()
        user_input = data.get("input")
        context = data.get("context", {})
        session_id = data.get("session_id")

        if not user_input:
            return error_response("Missing required field: 'input'")

        persona_manager = request.app["persona_manager"]
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
            return error_response(f"Vector dimension mismatch: expected {expected_dim}, got {vector.shape[0]}")

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
            return error_response("Missing required field: either 'vector' or 'vector_id' must be provided")

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
                return error_response(f"Vector dimension mismatch: expected {expected_dim}, got {vector.shape[0]}")

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
        data = await request.json()
        inputs = data.get("inputs", [])

        if not inputs or not isinstance(inputs, list):
            return error_response("Missing or invalid 'inputs' field: must be a non-empty list")

        results = []
        persona_manager = request.app["persona_manager"]

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
                    "input": user_input
                })

        return success_response({"results": results})
    except json.JSONDecodeError:
        return error_response("Invalid JSON in request body")
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}", exc_info=True)
        return error_response(f"Error in batch processing: {str(e)}", 500)

# Middleware for request logging and error handling
@web.middleware
async def error_middleware(request: Request, handler):
    """Middleware for logging requests and handling unexpected errors."""
    start_time = time.time()
    request_id = request.headers.get('X-Request-ID', '-')

    logger.info(f"Request started: {request.method} {request.path} (ID: {request_id})")

    try:
        response = await handler(request)
        elapsed = time.time() - start_time
        logger.info(f"Request completed: {request.method} {request.path} "
                   f"- Status: {response.status} - {elapsed:.4f}s (ID: {request_id})")
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
        return error_response("Internal server error", 500)

async def create_app(config: Dict[str, Any]) -> Application:
    """Create and configure the AIOHTTP application."""
    # Initialize the WDBX instance
    wdbx_instance = WDBX(
        vector_dimension=config["vector_dim"],
        num_shards=config["num_shards"]
    )

    # Initialize the persona manager
    persona_manager = PersonaManager(wdbx_instance)

    # Create the application
    app = web.Application(middlewares=[error_middleware])

    # Store shared objects and configuration
    app["wdbx"] = wdbx_instance
    app["persona_manager"] = persona_manager
    app["config"] = config
    app["start_time"] = time.time()

    # Register routes
    app.add_routes([
        web.get("/api/health", health_check),
        web.get("/api/stats", system_stats),
        web.post("/api/process", process_input),
        web.post("/api/embedding", store_embedding),
        web.post("/api/search", search_embeddings),
        web.post("/api/batch", batch_process)
    ])

    logger.info(f"Application initialized with vector dimension {config['vector_dim']} "
               f"and {config['num_shards']} shards")

    return app

async def start_server(app: Application, host: str, port: int) -> None:
    """Start the server with graceful shutdown capability."""
    runner = AppRunner(app)
    await runner.setup()
    site = TCPSite(runner, host, port)

    logger.info(f"Starting WDBX server on http://{host}:{port}")
    await site.start()

    # Setup signal handlers for graceful shutdown
    try:
        # Keep the server running
        while True:
            await asyncio.sleep(3600)  # Just keep the task alive
    finally:
        logger.info("Shutting down server...")
        await runner.cleanup()
        logger.info("Server shutdown complete")

def run_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    vector_dim: int = 768,
    num_shards: int = 4
) -> None:
    """
    Run the WDBX server with the specified configuration.

    Args:
        host: The host address to bind the server to
        port: The port to listen on
        vector_dim: The dimension of embedding vectors
        num_shards: The number of shards for the vector database
    """
    config = {
        "host": host,
        "port": port,
        "vector_dim": vector_dim,
        "num_shards": num_shards,
        "version": "1.0.0"  # Should be imported from package metadata
    }

    logger.info(f"Initializing WDBX server with config: {config}")

    loop = asyncio.get_event_loop()
    app = loop.run_until_complete(create_app(config))

    try:
        loop.run_until_complete(start_server(app, host, port))
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    finally:
        loop.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WDBX Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--vector-dim", type=int, default=768, help="Dimension of embedding vectors")
    parser.add_argument("--num-shards", type=int, default=4, help="Number of shards for the vector database")

    args = parser.parse_args()

    run_server(
        host=args.host,
        port=args.port,
        vector_dim=args.vector_dim,
        num_shards=args.num_shards
    )
