# wdbx/server.py
import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional, cast

import numpy as np
from aiohttp import web
from aiohttp.web import (
    Application,
    AppRunner,
    Request,
    Response,
    TCPSite,
)

from wdbx import WDBX

# Try to import JAX for faster vector operations
try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
    logger = logging.getLogger("wdbx.server")
    logger.info("JAX is available and will be used for vector operations")
except ImportError:
    HAS_JAX = False
    logger = logging.getLogger("wdbx.server")
    logger.info("JAX not available, falling back to NumPy")

# Import security components
try:
    from ..security import ML_AVAILABLE as SECURITY_ML_AVAILABLE
    from ..security import (
        ContentFilter,
        ContentSafetyLevel,
        JWTAuthProvider,
        SecurityManager,
    )

    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False
    logging.warning("Security module not available. Running with reduced security features.")

# Import monitoring components
try:
    from ..monitoring import BenchmarkRunner, MonitoringSystem, PerformanceMonitor

    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    logging.warning("Monitoring module not available. Performance tracking disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("wdbx.server")

# API version prefix
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"


# CORS and error handling middleware
@web.middleware
async def cors_middleware(request, handler):
    """
    Middleware for handling CORS (Cross-Origin Resource Sharing).

    Adds appropriate CORS headers to responses and handles preflight requests.
    """
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        response = web.Response()
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, Authorization, X-Requested-With"
        )
        response.headers["Access-Control-Max-Age"] = "86400"  # 24 hours
    else:
        # Handle the request normally
        response = await handler(request)

    # Add CORS headers to all responses
    response.headers["Access-Control-Allow-Origin"] = (
        "*"  # Replace with specific origin in production
    )
    response.headers["Access-Control-Allow-Credentials"] = "true"

    return response


@web.middleware
async def error_middleware(request, handler):
    """
    Middleware for handling errors and exceptions.

    Catches exceptions and returns appropriate error responses.
    Also logs errors and optionally sends them to monitoring systems.
    """
    try:
        # Process the request
        response = await handler(request)
        return response
    except web.HTTPException as ex:
        # Handle HTTP exceptions (e.g., 404, 403)
        message = str(ex) or "HTTP Error"
        status = ex.status
        status_message = ex.reason

        # Log the error
        logger.warning(f"HTTP {status} error: {message}")

        # Return a JSON error response
        return error_response(
            message=message,
            status=status,
            error_type="http_error",
            error_detail={
                "path": request.path,
                "method": request.method,
                "status": status,
                "status_message": status_message,
            },
        )
    except json.JSONDecodeError:
        # Handle JSON parse errors
        logger.warning(f"JSON parse error for {request.path}")
        return error_response(
            message="Invalid JSON in request body",
            status=400,
            error_type="json_error",
            error_detail={"path": request.path, "method": request.method},
        )
    except (KeyError, ValueError, TypeError) as ex:
        # Handle client errors (bad input)
        error_message = str(ex)
        logger.warning(f"Client error: {error_message}")
        return error_response(
            message=f"Invalid request: {error_message}",
            status=400,
            error_type="client_error",
            error_detail={"path": request.path, "method": request.method, "error": error_message},
        )
    except Exception as ex:
        # Handle unexpected errors
        error_message = str(ex)
        logger.exception(f"Unexpected error: {error_message}")

        # Report to monitoring system if available
        try:
            if MONITORING_AVAILABLE and "monitoring_system" in request.app:
                monitoring_system = request.app["monitoring_system"]
                monitoring_system.report_error(
                    error_type="server_error",
                    error_message=error_message,
                    context={
                        "path": request.path,
                        "method": request.method,
                        "remote": request.remote,
                        "user_agent": request.headers.get("User-Agent", "unknown"),
                    },
                )
        except Exception as monitor_ex:
            logger.error(f"Failed to report error to monitoring system: {str(monitor_ex)}")

        # Return a server error response
        return error_response(
            message="Internal server error",
            status=500,
            error_type="server_error",
            error_detail={
                "path": request.path,
                "method": request.method,
                "request_id": request.headers.get("X-Request-ID", "unknown"),
            },
        )


# Custom response helper functions


def success_response(data: Dict[str, Any], status: int = 200) -> Response:
    """Create a standardized success JSON response."""
    return web.json_response(
        {"status": "success", "data": data, "timestamp": time.time()}, status=status
    )


def error_response(
    message: str,
    status: int = 400,
    error_type: str = "general_error",
    error_detail: Optional[Dict[str, Any]] = None,
) -> Response:
    """
    Create a standardized error JSON response.

    Args:
        message: Error message to return
        status: HTTP status code
        error_type: Type of error (for categorization)
        error_detail: Additional error details

    Returns:
        JSON response with error information
    """
    response_data = {
        "status": "error",
        "error": message,
        "error_type": error_type,
        "timestamp": time.time(),
    }

    if error_detail:
        response_data["error_detail"] = error_detail

    return web.json_response(response_data, status=status)


# Request handlers


async def health_check(request: Request) -> Response:
    """Endpoint to check server health status."""
    if MONITORING_AVAILABLE:
        # Get extended health metrics from monitoring system
        monitoring = request.app.get("monitoring_system")
        if monitoring:
            health_data = monitoring.get_health_metrics()
            health_data.update(
                {
                    "version": request.app["config"]["version"],
                    "uptime": time.time() - request.app["start_time"],
                }
            )
            return success_response(health_data)

    # Basic health data if monitoring not available
    return success_response(
        {
            "status": "healthy",
            "version": request.app["config"]["version"],
            "uptime": time.time() - request.app["start_time"],
        }
    )


async def system_stats(request: Request) -> Response:
    """Endpoint to retrieve system statistics."""
    try:
        # Start performance profile if monitoring is available
        profiler = None
        if MONITORING_AVAILABLE and "performance_monitor" in request.app:
            profiler = request.app["performance_monitor"].profile("system_stats_endpoint")

        stats = request.app["wdbx"].get_system_stats()

        # Add ML backend stats if available
        if SECURITY_ML_AVAILABLE:
            try:
                from ..ml.backend import MLBackend

                ml_backend = MLBackend()
                stats["ml_backend"] = {
                    "backend_type": ml_backend.backend_type,
                    "device": ml_backend.device,
                    "memory_usage": (
                        ml_backend.get_memory_usage()
                        if hasattr(ml_backend, "get_memory_usage")
                        else "unknown"
                    ),
                }
            except ImportError:
                stats["ml_backend"] = {"status": "unavailable"}

        # Add security stats if available
        if SECURITY_AVAILABLE and "security_manager" in request.app:
            security_manager = request.app["security_manager"]
            stats["security"] = {
                "active_tokens": (
                    security_manager.get_active_token_count()
                    if hasattr(security_manager, "get_active_token_count")
                    else 0
                ),
                "auth_providers": (
                    list(security_manager.providers.keys())
                    if hasattr(security_manager, "providers")
                    else ["default"]
                ),
            }

        # End profiler if active
        if profiler:
            profiler.__exit__(None, None, None)

        return success_response(stats)
    except Exception as e:
        logger.error(f"Error fetching system stats: {str(e)}")
        return error_response(f"Failed to retrieve system statistics: {str(e)}", 500)


async def process_input(request: Request) -> Response:
    """Process user input through the persona manager."""
    try:
        # Start performance profile if monitoring is available
        profiler = None
        if MONITORING_AVAILABLE and "performance_monitor" in request.app:
            profiler = request.app["performance_monitor"].profile("process_input_endpoint")

        # Cast the JSON data to provide type hints
        data = cast(Dict[str, Any], await request.json())
        user_input = data.get("input")
        context = data.get("context", {})
        session_id = data.get("session_id")

        if not user_input:
            return error_response("Missing required field: 'input'")

        # Authenticate if security is available and token is provided
        token = data.get("token") or request.headers.get("Authorization", "").replace("Bearer ", "")
        if SECURITY_AVAILABLE and "security_manager" in request.app and token:
            security_manager = request.app["security_manager"]
            user_info = security_manager.validate_token(token)
            if not user_info:
                return error_response("Invalid or expired authentication token", 401)

            # Add user context to request for personalization
            context["user"] = user_info

        persona_manager = request.app["persona_manager"]
        # Add check if persona_manager is None (if it wasn't enabled)
        if persona_manager is None:
            return error_response(
                "Persona management is not enabled on the server", 501
            )  # 501 Not Implemented

        # Check content safety if filter is enabled
        if SECURITY_AVAILABLE and "content_filter" in request.app:
            content_filter = request.app["content_filter"]
            safety_check = content_filter.check_safety(user_input)
            if not safety_check.get("safe", True):
                # End profiler if active
                if profiler:
                    profiler.__exit__(None, None, None)
                return error_response(
                    "Input contains unsafe content that violates content policy", 403
                )

        response, block_id = persona_manager.process_user_input(
            user_input, context, session_id=session_id
        )

        # End profiler if active
        if profiler:
            profiler.__exit__(None, None, None)

        return success_response(
            {"response": response, "block_id": block_id, "session_id": session_id}
        )
    except json.JSONDecodeError:
        return error_response("Invalid JSON in request body")
    except Exception as e:
        logger.error(f"Error processing input: {str(e)}", exc_info=True)
        return error_response(f"Error processing input: {str(e)}", 500)


async def store_embedding(request: Request) -> Response:
    """
    Store a vector embedding in the database.

    Expects the following JSON payload:
    {
        "vector": [float, float, ...],
        "metadata": {...},
        "id": "optional-id"
    }
    """
    try:
        # Start performance profile if monitoring is available
        profiler = None
        if MONITORING_AVAILABLE and "performance_monitor" in request.app:
            profiler = request.app["performance_monitor"].profile("store_embedding_endpoint")

        # Check authentication if security is enabled
        if SECURITY_AVAILABLE and "security_manager" in request.app:
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return error_response("Authentication required", status=401)

            token = auth_header[7:]  # Remove "Bearer " prefix
            security_manager = request.app["security_manager"]
            verify_result = security_manager.verify_token(token)

            if not verify_result.valid:
                return error_response("Invalid token", status=401)

        # Parse request data
        data = await request.json()

        # Validate required fields
        if "vector" not in data:
            return error_response("Missing required field: 'vector'", status=400)

        # Parse vector data
        vector_data = data["vector"]

        # Validate vector data is a list of floats
        if not isinstance(vector_data, list):
            return error_response("Vector must be a list of floats", status=400)

        try:
            # Convert to NumPy array for validation
            vector = np.array(vector_data, dtype=np.float32)
        except (ValueError, TypeError):
            return error_response("Vector must contain only numeric values", status=400)

        # Get metadata and optional ID
        metadata = data.get("metadata", {})
        embedding_id = data.get("id")

        # Content safety check if enabled
        if SECURITY_AVAILABLE and "content_filter" in request.app:
            # Check any text fields in metadata
            for key, value in metadata.items():
                if isinstance(value, str) and key != "id":
                    content_filter = request.app["content_filter"]
                    safety_check = content_filter.check_safety(value)
                    if not safety_check.get("safe", True):
                        return error_response(
                            f"Content safety check failed for metadata field: {key}",
                            status=400,
                            error_type="content_safety_violation",
                            error_detail=safety_check.get("details", {}),
                        )

        # Store the embedding
        wdbx = request.app["wdbx"]
        result = wdbx.store_embedding(vector, metadata, embedding_id)

        # End profiler if active
        if profiler:
            profiler.__exit__(None, None, None)

        return success_response(
            {"id": result.get("id"), "status": "stored", "vector_dim": len(vector)}
        )
    except json.JSONDecodeError:
        return error_response("Invalid JSON in request body", status=400)
    except Exception as e:
        logger.exception(f"Error storing embedding: {e}")
        return error_response(f"Error storing embedding: {str(e)}", status=500)


async def search_embeddings(request: Request) -> Response:
    """
    Search for similar embeddings in the database.

    Expects the following JSON payload:
    {
        "vector": [float, float, ...],
        "top_k": 10,
        "filter": {...}
    }
    """
    try:
        # Start performance profile if monitoring is available
        profiler = None
        if MONITORING_AVAILABLE and "performance_monitor" in request.app:
            profiler = request.app["performance_monitor"].profile("search_embeddings_endpoint")

        # Check authentication if security is enabled
        if SECURITY_AVAILABLE and "security_manager" in request.app:
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return error_response("Authentication required", status=401)

            token = auth_header[7:]  # Remove "Bearer " prefix
            security_manager = request.app["security_manager"]
            verify_result = security_manager.verify_token(token)

            if not verify_result.valid:
                return error_response("Invalid token", status=401)

        # Parse request data
        data = await request.json()

        # Validate required fields
        if "vector" not in data:
            return error_response("Missing required field: 'vector'", status=400)

        # Parse vector data
        vector_data = data["vector"]

        # Validate vector data is a list of floats
        if not isinstance(vector_data, list):
            return error_response("Vector must be a list of floats", status=400)

        try:
            # Convert to NumPy array for validation
            vector = np.array(vector_data, dtype=np.float32)
        except (ValueError, TypeError):
            return error_response("Vector must contain only numeric values", status=400)

        # Get optional parameters
        top_k = int(data.get("top_k", 10))
        filter_criteria = data.get("filter", {})

        # Perform the search
        wdbx = request.app["wdbx"]
        results = wdbx.search_similar(vector, top_k, filter_criteria)

        # Format results to ensure JSON serializable
        formatted_results = []
        for result in results:
            formatted_result = {
                "id": result.get("id"),
                "score": float(result.get("score", 0.0)),  # Ensure float is JSON serializable
                "metadata": result.get("metadata", {}),
            }
            formatted_results.append(formatted_result)

        # End profiler if active
        if profiler:
            profiler.__exit__(None, None, None)

        return success_response(
            {
                "results": formatted_results,
                "count": len(formatted_results),
                "query_vector_dim": len(vector),
            }
        )
    except json.JSONDecodeError:
        return error_response("Invalid JSON in request body", status=400)
    except Exception as e:
        logger.exception(f"Error searching embeddings: {e}")
        return error_response(f"Error searching embeddings: {str(e)}", status=500)


async def batch_process(request: Request) -> Response:
    """
    Process multiple requests in a single batch operation.

    Expects the following JSON payload:
    {
        "operations": [
            {
                "type": "embedding",
                "data": {...}
            },
            {
                "type": "search",
                "data": {...}
            },
            ...
        ]
    }
    """
    try:
        # Start performance profile if monitoring is available
        profiler = None
        if MONITORING_AVAILABLE and "performance_monitor" in request.app:
            profiler = request.app["performance_monitor"].profile("batch_process_endpoint")

        # Check authentication if security is enabled
        if SECURITY_AVAILABLE and "security_manager" in request.app:
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return error_response("Authentication required", status=401)

            token = auth_header[7:]  # Remove "Bearer " prefix
            security_manager = request.app["security_manager"]
            verify_result = security_manager.verify_token(token)

            if not verify_result.valid:
                return error_response("Invalid token", status=401)

        # Parse request data
        data = await request.json()

        # Validate required fields
        if "operations" not in data or not isinstance(data["operations"], list):
            return error_response("Missing or invalid 'operations' field", status=400)

        operations = data["operations"]
        results = []

        # Create mock request objects for each operation
        wdbx = request.app["wdbx"]

        for i, operation in enumerate(operations):
            try:
                op_type = operation.get("type")
                op_data = operation.get("data", {})

                if op_type == "embedding":
                    # Store embedding
                    vector_data = op_data.get("vector")
                    if not vector_data or not isinstance(vector_data, list):
                        results.append(
                            {
                                "status": "error",
                                "index": i,
                                "error": "Invalid or missing vector data",
                            }
                        )
                        continue

                    # Convert vector
                    vector = np.array(vector_data, dtype=np.float32)
                    metadata = op_data.get("metadata", {})
                    embedding_id = op_data.get("id")

                    # Store embedding
                    store_result = wdbx.store_embedding(vector, metadata, embedding_id)
                    results.append(
                        {
                            "status": "success",
                            "index": i,
                            "operation": "embedding",
                            "id": store_result.get("id"),
                        }
                    )

                elif op_type == "search":
                    # Search embedding
                    vector_data = op_data.get("vector")
                    if not vector_data or not isinstance(vector_data, list):
                        results.append(
                            {
                                "status": "error",
                                "index": i,
                                "error": "Invalid or missing vector data",
                            }
                        )
                        continue

                    # Convert vector
                    vector = np.array(vector_data, dtype=np.float32)
                    top_k = op_data.get("top_k", 10)
                    filter_criteria = op_data.get("filter", {})

                    # Search similar vectors
                    search_results = wdbx.search_similar(vector, top_k, filter_criteria)

                    # Format results
                    formatted_results = []
                    for result in search_results:
                        formatted_result = {
                            "id": result.get("id"),
                            "score": float(result.get("score", 0.0)),
                            "metadata": result.get("metadata", {}),
                        }
                        formatted_results.append(formatted_result)

                    results.append(
                        {
                            "status": "success",
                            "index": i,
                            "operation": "search",
                            "results": formatted_results,
                            "count": len(formatted_results),
                        }
                    )

                elif op_type == "process":
                    # Process input with persona manager
                    if "input" not in op_data:
                        results.append(
                            {
                                "status": "error",
                                "index": i,
                                "error": "Missing required field: 'input'",
                            }
                        )
                        continue

                    # Get input data
                    user_input = op_data.get("input")
                    context = op_data.get("context", {})
                    session_id = op_data.get("session_id")

                    # Check content safety if enabled
                    if SECURITY_AVAILABLE and "content_filter" in request.app:
                        content_filter = request.app["content_filter"]
                        safety_check = content_filter.check_safety(user_input)
                        if not safety_check.get("safe", True):
                            results.append(
                                {
                                    "status": "error",
                                    "index": i,
                                    "error": "Content safety check failed",
                                    "details": safety_check.get("details", {}),
                                }
                            )
                            continue

                    # Process with persona manager
                    persona_manager = request.app["persona_manager"]
                    if persona_manager is None:
                        results.append(
                            {
                                "status": "error",
                                "index": i,
                                "error": "Persona management is not enabled on the server",
                            }
                        )
                        continue

                    # Process input
                    persona = op_data.get("persona", "default")
                    response = await persona_manager.process_input(
                        user_input, persona=persona, context=context, session_id=session_id
                    )

                    results.append(
                        {
                            "status": "success",
                            "index": i,
                            "operation": "process",
                            "response": response,
                        }
                    )

                else:
                    # Unknown operation type
                    results.append(
                        {
                            "status": "error",
                            "index": i,
                            "error": f"Unknown operation type: {op_type}",
                        }
                    )

            except Exception as op_error:
                logger.exception(f"Error in batch operation {i}: {op_error}")
                results.append({"status": "error", "index": i, "error": str(op_error)})

        # End profiler if active
        if profiler:
            profiler.__exit__(None, None, None)

        return success_response(
            {
                "results": results,
                "total": len(operations),
                "successful": sum(1 for r in results if r.get("status") == "success"),
                "failed": sum(1 for r in results if r.get("status") == "error"),
            }
        )
    except json.JSONDecodeError:
        return error_response("Invalid JSON in request body", status=400)
    except Exception as e:
        logger.exception(f"Error in batch processing: {e}")
        return error_response(f"Error in batch processing: {str(e)}", status=500)


def create_app(config: Dict[str, Any], wdbx_instance: WDBX) -> Application:
    """
    Create and configure the AIOHTTP application.

    Args:
        config: Configuration dictionary
        wdbx_instance: Instance of WDBX

    Returns:
        Configured Application instance
    """
    # Add CORS middleware to the list
    app = web.Application(middlewares=[cors_middleware, error_middleware])

    # Store configuration
    app["config"] = config
    app["wdbx"] = wdbx_instance
    app["start_time"] = time.time()

    # Initialize persona manager if available in WDBX instance
    app["persona_manager"] = getattr(wdbx_instance, "persona_manager", None)

    # Initialize security components
    if SECURITY_AVAILABLE:
        # Create security manager
        security_config = config.get("security", {})
        secret_key = security_config.get("secret_key", os.environ.get("WDBX_SECRET_KEY", None))

        if not secret_key:
            logger.warning("No secret key provided. Generating a temporary key for this session.")
            import secrets

            secret_key = secrets.token_hex(32)

        security_manager = SecurityManager()

        # Register JWT provider
        jwt_provider = JWTAuthProvider(
            secret_key=secret_key,
            algorithm=security_config.get("jwt_algorithm", "HS256"),
            expires_delta_seconds=security_config.get("token_expiration", 3600),
            issuer=security_config.get("jwt_issuer", "wdbx-server"),
            audience=security_config.get("jwt_audience", None),
        )
        security_manager.register_provider("jwt", jwt_provider)
        security_manager.set_default_provider("jwt")

        app["security_manager"] = security_manager

        # Initialize content filter
        content_filter = ContentFilter(
            safety_level=ContentSafetyLevel.MEDIUM, enable_ml=SECURITY_ML_AVAILABLE
        )
        app["content_filter"] = content_filter

    # Initialize monitoring components
    if MONITORING_AVAILABLE:
        monitoring_system = MonitoringSystem()
        performance_monitor = PerformanceMonitor()
        benchmark_runner = BenchmarkRunner()

        app["monitoring_system"] = monitoring_system
        app["performance_monitor"] = performance_monitor
        app["benchmark_runner"] = benchmark_runner

        logger.info("Monitoring system initialized")

    # Define routes with versioned API prefix
    app.router.add_get(f"{API_PREFIX}/health", health_check)
    app.router.add_get(f"{API_PREFIX}/stats", system_stats)
    app.router.add_post(f"{API_PREFIX}/process", process_input)
    app.router.add_post(f"{API_PREFIX}/embedding", store_embedding)
    app.router.add_post(f"{API_PREFIX}/search", search_embeddings)

    # Add security-specific endpoints if security is available
    if SECURITY_AVAILABLE:
        app.router.add_post(f"{API_PREFIX}/auth/token", authenticate_user)
        app.router.add_get(f"{API_PREFIX}/auth/verify", verify_token)

    # Add monitoring-specific endpoints if monitoring is available
    if MONITORING_AVAILABLE:
        app.router.add_get(f"{API_PREFIX}/monitor/metrics", get_metrics)
        app.router.add_get(f"{API_PREFIX}/monitor/performance", get_performance)

    # Simplified API documentation endpoint
    async def api_docs(request):
        """Return API documentation."""
        docs = {
            "api_version": API_VERSION,
            "endpoints": [
                {
                    "path": f"{API_PREFIX}/health",
                    "method": "GET",
                    "description": "Server health check",
                },
                {
                    "path": f"{API_PREFIX}/stats",
                    "method": "GET",
                    "description": "System statistics",
                },
                {
                    "path": f"{API_PREFIX}/process",
                    "method": "POST",
                    "description": "Process input with persona",
                },
                {
                    "path": f"{API_PREFIX}/embedding",
                    "method": "POST",
                    "description": "Store vector embedding",
                },
                {
                    "path": f"{API_PREFIX}/search",
                    "method": "POST",
                    "description": "Search similar vectors",
                },
            ],
            "auth_required": SECURITY_AVAILABLE,
            "monitoring_available": MONITORING_AVAILABLE,
        }

        # Add security endpoints if available
        if SECURITY_AVAILABLE:
            docs["endpoints"].extend(
                [
                    {
                        "path": f"{API_PREFIX}/auth/token",
                        "method": "POST",
                        "description": "Authenticate and get token",
                    },
                    {
                        "path": f"{API_PREFIX}/auth/verify",
                        "method": "GET",
                        "description": "Verify authentication token",
                    },
                ]
            )

        # Add monitoring endpoints if available
        if MONITORING_AVAILABLE:
            docs["endpoints"].extend(
                [
                    {
                        "path": f"{API_PREFIX}/monitor/metrics",
                        "method": "GET",
                        "description": "Get monitoring metrics",
                    },
                    {
                        "path": f"{API_PREFIX}/monitor/performance",
                        "method": "GET",
                        "description": "Get performance data",
                    },
                ]
            )

        return success_response(docs)

    app.router.add_get(f"{API_PREFIX}/docs", api_docs)
    app.router.add_get("/api", api_docs)  # Shorthand route

    return app


async def start_server(app: Application, host: str, port: int) -> None:
    """
    Start the HTTP server.

    Args:
        app: Application instance
        host: Host to bind to
        port: Port to listen on
    """
    runner = AppRunner(app)
    await runner.setup()
    site = TCPSite(runner, host, port)

    # Log startup message with security and ML status
    security_status = "enabled" if SECURITY_AVAILABLE else "disabled"
    ml_status = "available" if SECURITY_ML_AVAILABLE else "unavailable"
    monitoring_status = "enabled" if MONITORING_AVAILABLE else "disabled"
    jax_status = "enabled" if HAS_JAX else "disabled"

    logger.info(f"Starting WDBX server on http://{host}:{port}")
    logger.info(f"API available at http://{host}:{port}{API_PREFIX}")
    logger.info(
        f"Security: {security_status}, ML: {ml_status}, Monitoring: {monitoring_status}, JAX: {jax_status}"
    )

    await site.start()

    # This keeps the server running until it's cancelled
    while True:
        await asyncio.sleep(3600)  # Sleep for an hour (or until cancelled)


async def run_server(
    wdbx_instance=None,
    host: str = "0.0.0.0",
    port: int = 8080,
    config: Dict[str, Any] = None,
    enable_security: bool = True,
    enable_monitoring: bool = True,
    enable_cors: bool = True,
    log_level: str = "INFO",
) -> None:
    """
    Run the WDBX HTTP server.

    Args:
        wdbx_instance: WDBX instance to serve, or None to create a new one
        host: Host address to bind the server to
        port: Port to listen on
        config: Server configuration dictionary
        enable_security: Whether to enable security features
        enable_monitoring: Whether to enable ML monitoring
        enable_cors: Whether to enable CORS support
        log_level: Logging level
    """
    # Create and configure the server
    server = WDBXHttpServer(
        wdbx_instance=wdbx_instance,
        host=host,
        port=port,
        config=config,
        enable_security=enable_security,
        enable_monitoring=enable_monitoring,
        enable_cors=enable_cors,
        log_level=log_level,
    )

    # Start the server
    await server.start()

    # Keep the server running until interrupted
    try:
        while True:
            await asyncio.sleep(3600)  # Sleep for 1 hour
    except asyncio.CancelledError:
        logger.info("Server task cancelled")
    finally:
        await server.stop()


def start_server(
    wdbx_instance=None,
    host: str = "0.0.0.0",
    port: int = 8080,
    config: Dict[str, Any] = None,
    enable_security: bool = True,
    enable_monitoring: bool = True,
    enable_cors: bool = True,
    log_level: str = "INFO",
) -> None:
    """
    Start the WDBX HTTP server in a blocking call.

    This is a convenience function that runs the server in the default event loop.
    For more control, use the async run_server function.

    Args:
        wdbx_instance: WDBX instance to serve, or None to create a new one
        host: Host address to bind the server to
        port: Port to listen on
        config: Server configuration dictionary
        enable_security: Whether to enable security features
        enable_monitoring: Whether to enable ML monitoring
        enable_cors: Whether to enable CORS support
        log_level: Logging level
    """
    asyncio.run(
        run_server(
            wdbx_instance=wdbx_instance,
            host=host,
            port=port,
            config=config,
            enable_security=enable_security,
            enable_monitoring=enable_monitoring,
            enable_cors=enable_cors,
            log_level=log_level,
        )
    )


# New security endpoints


async def authenticate_user(request: Request) -> Response:
    """
    Authenticate user and issue a JWT token.

    Requires security module to be available.
    """
    if not SECURITY_AVAILABLE:
        return error_response("Security module not available", status=501)

    if "security_manager" not in request.app:
        return error_response("Security manager not configured", status=500)

    try:
        data = await request.json()

        # Validate required fields
        if "username" not in data or "password" not in data:
            return error_response("Missing required fields", status=400)

        username = data["username"]
        password = data["password"]

        # Authenticate user
        security_manager = request.app["security_manager"]
        auth_result = await security_manager.authenticate(username, password)

        if not auth_result.success:
            # Apply exponential backoff for failed login attempts
            security_manager.record_failed_login(username, request.remote)
            return error_response("Authentication failed", status=401)

        # Generate token
        token = security_manager.generate_token(
            user_id=auth_result.user_id, username=username, scopes=auth_result.scopes
        )

        # Record successful login
        security_manager.record_successful_login(username, request.remote)

        # Return token information
        return success_response(
            {
                "token": token.token,
                "expires_at": token.expires_at.isoformat() if token.expires_at else None,
                "user_id": auth_result.user_id,
                "username": username,
                "scopes": auth_result.scopes,
            }
        )
    except json.JSONDecodeError:
        return error_response("Invalid JSON", status=400)
    except Exception as e:
        logger.exception("Error during authentication")
        return error_response(f"Authentication error: {str(e)}", status=500)


async def verify_token(request: Request) -> Response:
    """
    Verify JWT token and return user information.

    Requires security module to be available.
    """
    if not SECURITY_AVAILABLE:
        return error_response("Security module not available", status=501)

    if "security_manager" not in request.app:
        return error_response("Security manager not configured", status=500)

    # Get token from authorization header
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return error_response("Missing or invalid Authorization header", status=401)

    token = auth_header[7:]  # Remove "Bearer " prefix

    try:
        # Verify token
        security_manager = request.app["security_manager"]
        verify_result = security_manager.verify_token(token)

        if not verify_result.valid:
            return error_response("Invalid token", status=401)

        # Return user information
        return success_response(
            {
                "valid": True,
                "user_id": verify_result.user_id,
                "username": verify_result.username,
                "scopes": verify_result.scopes,
                "expires_at": (
                    verify_result.expires_at.isoformat() if verify_result.expires_at else None
                ),
            }
        )
    except Exception as e:
        logger.exception("Error during token verification")
        return error_response(f"Token verification error: {str(e)}", status=500)


# New monitoring endpoints


async def get_metrics(request: Request) -> Response:
    """
    Get monitoring metrics.

    Requires monitoring module to be available.
    """
    if not MONITORING_AVAILABLE:
        return error_response("Monitoring module not available", status=501)

    if "monitoring_system" not in request.app:
        return error_response("Monitoring system not configured", status=500)

    try:
        # Get monitoring metrics
        monitoring_system = request.app["monitoring_system"]
        metrics = monitoring_system.get_metrics()

        # Add server uptime
        start_time = request.app.get("start_time", time.time())
        uptime = time.time() - start_time
        metrics["uptime_seconds"] = uptime

        return success_response(metrics)
    except Exception as e:
        logger.exception("Error retrieving metrics")
        return error_response(f"Error retrieving metrics: {str(e)}", status=500)


async def get_performance(request: Request) -> Response:
    """
    Get performance data.

    Requires monitoring module to be available.
    """
    if not MONITORING_AVAILABLE:
        return error_response("Monitoring module not available", status=501)

    if "performance_monitor" not in request.app:
        return error_response("Performance monitor not configured", status=500)

    try:
        # Get performance data
        performance_monitor = request.app["performance_monitor"]
        performance_data = performance_monitor.get_performance_data()

        # Add ML anomaly detection results if available
        ml_backend = request.app.get("ml_backend")
        if ml_backend and hasattr(ml_backend, "detect_anomalies"):
            try:
                anomalies = ml_backend.detect_anomalies(performance_data)
                performance_data["anomalies"] = anomalies
            except Exception as e:
                logger.warning(f"Error detecting anomalies: {e}")

        return success_response(performance_data)
    except Exception as e:
        logger.exception("Error retrieving performance data")
        return error_response(f"Error retrieving performance data: {str(e)}", status=500)


def main():
    """Command line entry point for the WDBX HTTP server."""
    parser = argparse.ArgumentParser(description="WDBX HTTP Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--vector-dim", type=int, help="Vector dimension")
    parser.add_argument("--shards", type=int, help="Number of shards")
    parser.add_argument("--data-dir", type=str, help="Data directory")
    parser.add_argument("--no-security", action="store_true", help="Disable security features")
    parser.add_argument(
        "--no-monitoring", action="store_true", help="Disable performance monitoring"
    )
    parser.add_argument("--no-cors", action="store_true", help="Disable CORS support")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument("--test", action="store_true", help="Run a test instance with sample data")
    args = parser.parse_args()

    # Handle configuration
    config = None
    if args.config:
        try:
            with open(args.config) as f:
                config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading configuration file: {e}")
            sys.exit(1)
    else:
        config = {}

    # Override configuration with command line arguments
    if args.vector_dim:
        config["vector_dim"] = args.vector_dim
    if args.shards:
        config["shards"] = args.shards
    if args.data_dir:
        config["data_dir"] = args.data_dir

    # Setup test instance if requested
    wdbx_instance = None
    if args.test:
        try:
            from ..testing import create_test_db

            wdbx_instance = create_test_db()
            print("Created test database with sample data")
        except ImportError:
            print("Could not create test database: testing module not available")

    # Start the server
    try:
        start_server(
            wdbx_instance=wdbx_instance,
            host=args.host,
            port=args.port,
            config=config,
            enable_security=not args.no_security,
            enable_monitoring=not args.no_monitoring,
            enable_cors=not args.no_cors,
            log_level=args.log_level,
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")


if __name__ == "__main__":
    main()


class WDBXHttpServer:
    """
    WDBX HTTP server class for managing the server lifecycle.

    This class encapsulates the HTTP server functionality and provides
    methods for starting, stopping, and managing the server instance.
    It integrates with security, monitoring, and ML components when available.
    """

    def __init__(
        self,
        wdbx_instance=None,
        host: str = "0.0.0.0",
        port: int = 8080,
        config: Optional[Dict[str, Any]] = None,
        enable_security: bool = True,
        enable_monitoring: bool = True,
        enable_cors: bool = True,
        log_level: str = "INFO",
    ):
        """
        Initialize a WDBX HTTP server.

        Args:
            wdbx_instance: WDBX instance to serve, or None to create a new one
            host: Host address to bind the server to
            port: Port to listen on
            config: Server configuration dictionary
            enable_security: Whether to enable security features
            enable_monitoring: Whether to enable monitoring features
            enable_cors: Whether to enable CORS support
            log_level: Logging level
        """
        self.host = host
        self.port = port
        self.config = config or {}
        self.wdbx_instance = wdbx_instance
        self.enable_security = enable_security
        self.enable_monitoring = enable_monitoring
        self.enable_cors = enable_cors

        # Server state
        self.runner = None
        self.site = None
        self.app = None
        self.running = False
        self.start_time = None

        # Setup logging
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            numeric_level = logging.INFO
        logging.basicConfig(
            level=numeric_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Set up shutdown handler
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        if sys.platform != "win32":  # Not supported on Windows
            try:
                import signal

                for sig in (signal.SIGINT, signal.SIGTERM):
                    signal.signal(sig, self._signal_handler)
                logger.info("Signal handlers installed for graceful shutdown")
            except (ImportError, ValueError) as e:
                logger.warning(f"Could not set up signal handlers: {e}")

    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down server...")
        asyncio.create_task(self.stop())

    async def create_app(self) -> web.Application:
        """
        Create and configure the AIOHTTP application.

        Returns:
            Configured Application instance
        """
        # Determine which middlewares to use
        middlewares = []
        if self.enable_cors:
            middlewares.append(cors_middleware)
        middlewares.append(error_middleware)

        # Create the application
        app = web.Application(middlewares=middlewares)

        # Store configuration
        app["config"] = self.config
        app["wdbx"] = self.wdbx_instance
        app["start_time"] = time.time()
        app["server"] = self

        # Initialize persona manager if available in WDBX instance
        app["persona_manager"] = getattr(self.wdbx_instance, "persona_manager", None)

        # Initialize security components
        if self.enable_security and SECURITY_AVAILABLE:
            # Create security manager
            security_config = self.config.get("security", {})
            secret_key = security_config.get("secret_key", os.environ.get("WDBX_SECRET_KEY", None))

            if not secret_key:
                logger.warning(
                    "No secret key provided. Generating a temporary key for this session."
                )
                import secrets

                secret_key = secrets.token_hex(32)

            security_manager = SecurityManager()

            # Register JWT provider
            jwt_provider = JWTAuthProvider(
                secret_key=secret_key,
                algorithm=security_config.get("jwt_algorithm", "HS256"),
                expires_delta_seconds=security_config.get("token_expiration", 3600),
                issuer=security_config.get("jwt_issuer", "wdbx-server"),
                audience=security_config.get("jwt_audience", None),
            )
            security_manager.register_provider("jwt", jwt_provider)
            security_manager.set_default_provider("jwt")

            app["security_manager"] = security_manager

            # Initialize content filter
            content_filter = ContentFilter(
                safety_level=ContentSafetyLevel.MEDIUM, enable_ml=SECURITY_ML_AVAILABLE
            )
            app["content_filter"] = content_filter

        # Initialize monitoring components
        if self.enable_monitoring and MONITORING_AVAILABLE:
            monitoring_system = MonitoringSystem()
            performance_monitor = PerformanceMonitor()
            benchmark_runner = BenchmarkRunner()

            app["monitoring_system"] = monitoring_system
            app["performance_monitor"] = performance_monitor
            app["benchmark_runner"] = benchmark_runner

            logger.info("Monitoring system initialized")

        # Define routes with versioned API prefix
        app.router.add_get(f"{API_PREFIX}/health", health_check)
        app.router.add_get(f"{API_PREFIX}/stats", system_stats)
        app.router.add_post(f"{API_PREFIX}/process", process_input)
        app.router.add_post(f"{API_PREFIX}/embedding", store_embedding)
        app.router.add_post(f"{API_PREFIX}/search", search_embeddings)
        app.router.add_post(f"{API_PREFIX}/batch", batch_process)

        # Add security-specific endpoints if security is available
        if self.enable_security and SECURITY_AVAILABLE:
            app.router.add_post(f"{API_PREFIX}/auth/token", authenticate_user)
            app.router.add_get(f"{API_PREFIX}/auth/verify", verify_token)

        # Add monitoring-specific endpoints if monitoring is available
        if self.enable_monitoring and MONITORING_AVAILABLE:
            app.router.add_get(f"{API_PREFIX}/monitor/metrics", get_metrics)
            app.router.add_get(f"{API_PREFIX}/monitor/performance", get_performance)

        # Simplified API documentation endpoint
        async def api_docs(request):
            """Return API documentation."""
            docs = {
                "api_version": API_VERSION,
                "endpoints": [
                    {
                        "path": f"{API_PREFIX}/health",
                        "method": "GET",
                        "description": "Server health check",
                    },
                    {
                        "path": f"{API_PREFIX}/stats",
                        "method": "GET",
                        "description": "System statistics",
                    },
                    {
                        "path": f"{API_PREFIX}/process",
                        "method": "POST",
                        "description": "Process input with persona",
                    },
                    {
                        "path": f"{API_PREFIX}/embedding",
                        "method": "POST",
                        "description": "Store vector embedding",
                    },
                    {
                        "path": f"{API_PREFIX}/search",
                        "method": "POST",
                        "description": "Search similar vectors",
                    },
                    {
                        "path": f"{API_PREFIX}/batch",
                        "method": "POST",
                        "description": "Batch process inputs",
                    },
                ],
                "auth_required": SECURITY_AVAILABLE and self.enable_security,
                "monitoring_available": MONITORING_AVAILABLE and self.enable_monitoring,
            }

            # Add security endpoints if available
            if self.enable_security and SECURITY_AVAILABLE:
                docs["endpoints"].extend(
                    [
                        {
                            "path": f"{API_PREFIX}/auth/token",
                            "method": "POST",
                            "description": "Authenticate and get token",
                        },
                        {
                            "path": f"{API_PREFIX}/auth/verify",
                            "method": "GET",
                            "description": "Verify authentication token",
                        },
                    ]
                )

            # Add monitoring endpoints if available
            if self.enable_monitoring and MONITORING_AVAILABLE:
                docs["endpoints"].extend(
                    [
                        {
                            "path": f"{API_PREFIX}/monitor/metrics",
                            "method": "GET",
                            "description": "Get monitoring metrics",
                        },
                        {
                            "path": f"{API_PREFIX}/monitor/performance",
                            "method": "GET",
                            "description": "Get performance data",
                        },
                    ]
                )

            return success_response(docs)

        app.router.add_get(f"{API_PREFIX}/docs", api_docs)
        app.router.add_get("/api", api_docs)  # Shorthand route

        # Add WDBX version endpoint
        async def version(request):
            """Return WDBX version information."""
            return success_response(
                {
                    "version": self.config.get("version", "unknown"),
                    "api_version": API_VERSION,
                    "build_date": self.config.get("build_date", "unknown"),
                    "py_version": sys.version,
                    "jax_available": HAS_JAX,
                    "security_available": SECURITY_AVAILABLE and self.enable_security,
                    "monitoring_available": MONITORING_AVAILABLE and self.enable_monitoring,
                }
            )

        app.router.add_get(f"{API_PREFIX}/version", version)

        # Set up application startup and cleanup
        app.on_startup.append(self._on_startup)
        app.on_shutdown.append(self._on_shutdown)

        return app

    async def _on_startup(self, app):
        """Perform initialization when the application starts."""
        # Record start time
        self.start_time = time.time()
        app["start_time"] = self.start_time

        # Start monitoring if available
        if self.enable_monitoring and MONITORING_AVAILABLE and "performance_monitor" in app:
            app["performance_monitor"].start()
            logger.info("Performance monitoring started")

        # Initialize ML components if available
        try:
            from ..ml.backend import MLBackend

            ml_backend = MLBackend()
            app["ml_backend"] = ml_backend
            logger.info(f"ML backend initialized: {ml_backend.backend_type}")
        except ImportError:
            logger.info("ML backend not available")

        logger.info(f"Server started at http://{self.host}:{self.port}")
        logger.info(f"API available at http://{self.host}:{self.port}{API_PREFIX}")

    async def _on_shutdown(self, app):
        """Perform cleanup when the application is shutting down."""
        # Stop monitoring if available
        if self.enable_monitoring and MONITORING_AVAILABLE and "performance_monitor" in app:
            app["performance_monitor"].stop()
            logger.info("Performance monitoring stopped")

        # Log shutdown
        uptime = time.time() - self.start_time if self.start_time else 0
        logger.info(f"Server shutting down. Uptime: {uptime:.2f} seconds")

    async def start(self) -> None:
        """Start the HTTP server."""
        if self.running:
            logger.warning("Server is already running")
            return

        # Create the application
        self.app = await self.create_app()

        # Set up the server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)

        # Start the site
        await self.site.start()
        self.running = True

        # Log startup
        security_status = "enabled" if SECURITY_AVAILABLE and self.enable_security else "disabled"
        ml_status = "available" if SECURITY_ML_AVAILABLE else "unavailable"
        monitoring_status = (
            "enabled" if MONITORING_AVAILABLE and self.enable_monitoring else "disabled"
        )
        jax_status = "enabled" if HAS_JAX else "disabled"

        logger.info(f"WDBX HTTP server started on http://{self.host}:{self.port}")
        logger.info(
            f"Security: {security_status}, ML: {ml_status}, Monitoring: {monitoring_status}, JAX: {jax_status}"
        )

    async def stop(self) -> None:
        """Stop the HTTP server."""
        if not self.running:
            logger.warning("Server is not running")
            return

        # Stop the site and runner
        if self.site:
            await self.site.stop()

        if self.runner:
            await self.runner.cleanup()

        self.running = False
        logger.info("WDBX HTTP server stopped")

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current server status.

        Returns:
            Dictionary with server status information
        """
        status = {
            "running": self.running,
            "host": self.host,
            "port": self.port,
            "uptime": time.time() - self.start_time if self.start_time else 0,
            "security_enabled": SECURITY_AVAILABLE and self.enable_security,
            "monitoring_enabled": MONITORING_AVAILABLE and self.enable_monitoring,
            "cors_enabled": self.enable_cors,
            "jax_available": HAS_JAX,
        }

        # Add monitoring stats if available
        if (
            self.enable_monitoring
            and MONITORING_AVAILABLE
            and self.app
            and "monitoring_system" in self.app
        ):
            status["monitoring"] = self.app["monitoring_system"].get_metrics()

        return status
