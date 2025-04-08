"""
OpenAPI documentation for the WDBX API.

This module provides tools for generating OpenAPI documentation for the WDBX API,
making it easier for developers to understand and interact with the API.
"""

import json
import os
from typing import Any, Dict

import yaml


class OpenAPIDocumentation:
    """
    OpenAPI documentation generator for WDBX API.

    This class provides functionality to generate, validate, and serve
    OpenAPI documentation for the WDBX HTTP API.
    """

    def __init__(
        self,
        api_version: str = "v1",
        title: str = "WDBX API",
        description: str = "API for the WDBX vector database and processing system",
    ):
        """
        Initialize the OpenAPI documentation.

        Args:
            api_version: API version string
            title: API title
            description: API description
        """
        self.api_version = api_version
        self.title = title
        self.description = description
        self._spec = self._build_base_spec()

    def _build_base_spec(self) -> Dict[str, Any]:
        """
        Build the base OpenAPI specification.

        Returns:
            Base OpenAPI specification as a dictionary
        """
        return {
            "openapi": "3.0.0",
            "info": {
                "title": self.title,
                "description": self.description,
                "version": self.api_version,
                "contact": {"name": "WDBX Support", "email": "support@wdbx.example.com"},
                "license": {"name": "MIT License", "url": "https://opensource.org/licenses/MIT"},
            },
            "servers": [
                {"url": f"/api/{self.api_version}", "description": f"WDBX API {self.api_version}"}
            ],
            "paths": self._build_paths(),
            "components": self._build_components(),
            "tags": [
                {"name": "System", "description": "System management endpoints"},
                {"name": "Vectors", "description": "Vector embedding operations"},
                {"name": "Blocks", "description": "Data block operations"},
                {"name": "Processing", "description": "Data processing endpoints"},
            ],
        }

    def _build_paths(self) -> Dict[str, Any]:
        """
        Build the paths section of the OpenAPI specification.

        Returns:
            Paths section as a dictionary
        """
        return {
            "/health": {
                "get": {
                    "summary": "Health check",
                    "description": "Check the health status of the WDBX server",
                    "operationId": "healthCheck",
                    "tags": ["System"],
                    "responses": {
                        "200": {
                            "description": "Server is healthy",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/HealthResponse"}
                                }
                            },
                        }
                    },
                }
            },
            "/stats": {
                "get": {
                    "summary": "System statistics",
                    "description": "Get system statistics and performance metrics",
                    "operationId": "getSystemStats",
                    "tags": ["System"],
                    "responses": {
                        "200": {
                            "description": "System statistics retrieved successfully",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/StatsResponse"}
                                }
                            },
                        },
                        "500": {
                            "description": "Internal server error",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                    },
                }
            },
            "/vectors": {
                "post": {
                    "summary": "Store vector embedding",
                    "description": "Store a vector embedding with optional metadata",
                    "operationId": "storeEmbedding",
                    "tags": ["Vectors"],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/StoreEmbeddingRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Vector stored successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/StoreEmbeddingResponse"
                                    }
                                }
                            },
                        },
                        "400": {
                            "description": "Invalid input",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                        "500": {
                            "description": "Internal server error",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                    },
                }
            },
            "/vectors/search": {
                "post": {
                    "summary": "Search similar vectors",
                    "description": "Search for similar vectors by vector data or vector ID",
                    "operationId": "searchEmbeddings",
                    "tags": ["Vectors"],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/SearchRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Search completed successfully",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/SearchResponse"}
                                }
                            },
                        },
                        "400": {
                            "description": "Invalid input",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                        "500": {
                            "description": "Internal server error",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                    },
                }
            },
            "/process": {
                "post": {
                    "summary": "Process input",
                    "description": "Process user input through the persona manager",
                    "operationId": "processInput",
                    "tags": ["Processing"],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ProcessInputRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Input processed successfully",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ProcessInputResponse"}
                                }
                            },
                        },
                        "400": {
                            "description": "Invalid input",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                        "500": {
                            "description": "Internal server error",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                    },
                }
            },
            "/batch": {
                "post": {
                    "summary": "Batch process",
                    "description": "Process multiple inputs in a single request",
                    "operationId": "batchProcess",
                    "tags": ["Processing"],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/BatchProcessRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Batch processed successfully",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/BatchProcessResponse"}
                                }
                            },
                        },
                        "400": {
                            "description": "Invalid input",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                        "500": {
                            "description": "Internal server error",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            },
                        },
                    },
                }
            },
        }

    def _build_components(self) -> Dict[str, Any]:
        """
        Build the components section of the OpenAPI specification.

        Returns:
            Components section as a dictionary
        """
        return {
            "schemas": {
                "SuccessResponse": {
                    "type": "object",
                    "required": ["status", "data", "timestamp"],
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["success"],
                            "description": "Response status",
                        },
                        "data": {"type": "object", "description": "Response data"},
                        "timestamp": {
                            "type": "number",
                            "format": "float",
                            "description": "Response timestamp (Unix timestamp)",
                        },
                    },
                },
                "ErrorResponse": {
                    "type": "object",
                    "required": ["status", "error", "timestamp"],
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["error"],
                            "description": "Error status",
                        },
                        "error": {"type": "string", "description": "Error message"},
                        "timestamp": {
                            "type": "number",
                            "format": "float",
                            "description": "Response timestamp (Unix timestamp)",
                        },
                    },
                },
                "HealthResponse": {
                    "allOf": [
                        {"$ref": "#/components/schemas/SuccessResponse"},
                        {
                            "type": "object",
                            "properties": {
                                "data": {
                                    "type": "object",
                                    "required": ["status", "version", "uptime"],
                                    "properties": {
                                        "status": {
                                            "type": "string",
                                            "description": "Health status",
                                        },
                                        "version": {
                                            "type": "string",
                                            "description": "WDBX version",
                                        },
                                        "uptime": {
                                            "type": "number",
                                            "format": "float",
                                            "description": "Server uptime in seconds",
                                        },
                                    },
                                }
                            },
                        },
                    ]
                },
                "StatsResponse": {
                    "allOf": [
                        {"$ref": "#/components/schemas/SuccessResponse"},
                        {
                            "type": "object",
                            "properties": {
                                "data": {
                                    "type": "object",
                                    "description": "System statistics",
                                    "properties": {
                                        "memory_usage_mb": {
                                            "type": "number",
                                            "format": "float",
                                            "description": "Memory usage in MB",
                                        },
                                        "vector_count": {
                                            "type": "integer",
                                            "description": "Number of vectors stored",
                                        },
                                        "block_count": {
                                            "type": "integer",
                                            "description": "Number of blocks stored",
                                        },
                                        "processing_time_ms": {
                                            "type": "number",
                                            "format": "float",
                                            "description": "Average processing time in milliseconds",
                                        },
                                    },
                                }
                            },
                        },
                    ]
                },
                "Vector": {
                    "type": "array",
                    "items": {"type": "number", "format": "float"},
                    "description": "Vector embedding as an array of floating-point numbers",
                },
                "Metadata": {"type": "object", "description": "Metadata associated with a vector"},
                "StoreEmbeddingRequest": {
                    "type": "object",
                    "required": ["vector"],
                    "properties": {
                        "vector": {"$ref": "#/components/schemas/Vector"},
                        "metadata": {"$ref": "#/components/schemas/Metadata"},
                    },
                },
                "StoreEmbeddingResponse": {
                    "allOf": [
                        {"$ref": "#/components/schemas/SuccessResponse"},
                        {
                            "type": "object",
                            "properties": {
                                "data": {
                                    "type": "object",
                                    "required": ["vector_id"],
                                    "properties": {
                                        "vector_id": {
                                            "type": "string",
                                            "description": "ID of the stored vector",
                                        }
                                    },
                                }
                            },
                        },
                    ]
                },
                "SearchRequest": {
                    "type": "object",
                    "properties": {
                        "vector": {"$ref": "#/components/schemas/Vector"},
                        "vector_id": {
                            "type": "string",
                            "description": "ID of the vector to use as a query",
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                            "default": 10,
                            "description": "Maximum number of results to return",
                        },
                        "threshold": {
                            "type": "number",
                            "format": "float",
                            "minimum": 0,
                            "maximum": 1,
                            "default": 0.7,
                            "description": "Minimum similarity threshold",
                        },
                    },
                    "oneOf": [{"required": ["vector"]}, {"required": ["vector_id"]}],
                },
                "SearchResult": {
                    "type": "object",
                    "required": ["vector_id", "similarity"],
                    "properties": {
                        "vector_id": {"type": "string", "description": "ID of the similar vector"},
                        "similarity": {
                            "type": "number",
                            "format": "float",
                            "description": "Similarity score between 0 and 1",
                        },
                        "metadata": {"$ref": "#/components/schemas/Metadata"},
                    },
                },
                "SearchResponse": {
                    "allOf": [
                        {"$ref": "#/components/schemas/SuccessResponse"},
                        {
                            "type": "object",
                            "properties": {
                                "data": {
                                    "type": "object",
                                    "required": ["results"],
                                    "properties": {
                                        "results": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/SearchResult"},
                                        }
                                    },
                                }
                            },
                        },
                    ]
                },
                "ProcessInputRequest": {
                    "type": "object",
                    "required": ["input"],
                    "properties": {
                        "input": {"type": "string", "description": "User input to process"},
                        "context": {
                            "type": "object",
                            "description": "Additional context for processing",
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Session ID for continuous interactions",
                        },
                    },
                },
                "ProcessInputResponse": {
                    "allOf": [
                        {"$ref": "#/components/schemas/SuccessResponse"},
                        {
                            "type": "object",
                            "properties": {
                                "data": {
                                    "type": "object",
                                    "required": ["response", "block_id", "session_id"],
                                    "properties": {
                                        "response": {
                                            "type": "string",
                                            "description": "Processed response",
                                        },
                                        "block_id": {
                                            "type": "string",
                                            "description": "ID of the created block",
                                        },
                                        "session_id": {
                                            "type": "string",
                                            "description": "Session ID for continuous interactions",
                                        },
                                    },
                                }
                            },
                        },
                    ]
                },
                "BatchProcessRequest": {
                    "type": "object",
                    "required": ["inputs"],
                    "properties": {
                        "inputs": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/ProcessInputRequest"},
                            "description": "List of inputs to process",
                        }
                    },
                },
                "BatchProcessResult": {
                    "type": "object",
                    "required": ["status"],
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["success", "error"],
                            "description": "Status of this batch item",
                        },
                        "response": {
                            "type": "string",
                            "description": "Processed response (for successful items)",
                        },
                        "block_id": {
                            "type": "string",
                            "description": "ID of the created block (for successful items)",
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Session ID (for successful items)",
                        },
                        "error": {
                            "type": "string",
                            "description": "Error message (for failed items)",
                        },
                        "input": {
                            "type": "string",
                            "description": "Original input (for failed items)",
                        },
                    },
                },
                "BatchProcessResponse": {
                    "allOf": [
                        {"$ref": "#/components/schemas/SuccessResponse"},
                        {
                            "type": "object",
                            "properties": {
                                "data": {
                                    "type": "object",
                                    "required": ["results"],
                                    "properties": {
                                        "results": {
                                            "type": "array",
                                            "items": {
                                                "$ref": "#/components/schemas/BatchProcessResult"
                                            },
                                        }
                                    },
                                }
                            },
                        },
                    ]
                },
            },
            "securitySchemes": {
                "ApiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key",
                    "description": "API key for authentication",
                }
            },
        }

    def get_spec(self) -> Dict[str, Any]:
        """
        Get the complete OpenAPI specification.

        Returns:
            Complete OpenAPI specification as a dictionary
        """
        return self._spec

    def to_json(self) -> str:
        """
        Convert the OpenAPI specification to JSON.

        Returns:
            JSON string representation of the specification
        """
        return json.dumps(self._spec, indent=2)

    def to_yaml(self) -> str:
        """
        Convert the OpenAPI specification to YAML.

        Returns:
            YAML string representation of the specification
        """
        return yaml.dump(self._spec, sort_keys=False)

    def save_to_file(self, file_path: str, format: str = "yaml") -> None:
        """
        Save the OpenAPI specification to a file.

        Args:
            file_path: Path to save the file
            format: Format to save (yaml or json)
        """
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        if format.lower() == "json":
            with open(file_path, "w") as f:
                f.write(self.to_json())
        elif format.lower() == "yaml":
            with open(file_path, "w") as f:
                f.write(self.to_yaml())
        else:
            raise ValueError(f"Unsupported format: {format}")
