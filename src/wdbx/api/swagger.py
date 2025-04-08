"""
Swagger UI integration for WDBX API.

This module provides functionality to integrate Swagger UI with the WDBX HTTP server,
making it easier to explore and test the API endpoints interactively.
"""

from aiohttp import web
from aiohttp.web import Application, Request, Response

from .openapi import OpenAPIDocumentation


async def swagger_ui_handler(request: Request) -> Response:
    """
    Handle requests for the Swagger UI page.

    Args:
        request: HTTP request

    Returns:
        HTTP response with Swagger UI HTML
    """
    # Get the base URL for the API
    request.app.get("swagger_ui_path", "/api/docs")
    api_spec_path = request.app.get("api_spec_path", "/api/openapi.json")
    request.app.get("api_version", "v1")

    # Create Swagger UI HTML
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>WDBX API Documentation</title>
        <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@4.5.0/swagger-ui.css">
        <style>
            html {{
                box-sizing: border-box;
                overflow: -moz-scrollbars-vertical;
                overflow-y: scroll;
            }}
            
            *,
            *:before,
            *:after {{
                box-sizing: inherit;
            }}
            
            body {{
                margin: 0;
                background: #fafafa;
            }}
            
            .swagger-ui .topbar {{
                background-color: #1f2937;
            }}
            
            .swagger-ui .info .title {{
                color: #1f2937;
            }}
        </style>
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script src="https://unpkg.com/swagger-ui-dist@4.5.0/swagger-ui-bundle.js"></script>
        <script src="https://unpkg.com/swagger-ui-dist@4.5.0/swagger-ui-standalone-preset.js"></script>
        <script>
            window.onload = function() {{
                const ui = SwaggerUIBundle({{
                    url: "{api_spec_path}",
                    dom_id: '#swagger-ui',
                    deepLinking: true,
                    presets: [
                        SwaggerUIBundle.presets.apis,
                        SwaggerUIStandalonePreset
                    ],
                    plugins: [
                        SwaggerUIBundle.plugins.DownloadUrl
                    ],
                    layout: "BaseLayout",
                    tagsSorter: 'alpha',
                    operationsSorter: 'alpha',
                    docExpansion: 'list',
                    defaultModelsExpandDepth: 1,
                    defaultModelExpandDepth: 1,
                    displayRequestDuration: true,
                    filter: true,
                    withCredentials: true
                }});
                window.ui = ui;
            }};
        </script>
    </body>
    </html>
    """

    return web.Response(text=html, content_type="text/html")


async def openapi_json_handler(request: Request) -> Response:
    """
    Handle requests for the OpenAPI specification in JSON format.

    Args:
        request: HTTP request

    Returns:
        HTTP response with OpenAPI specification in JSON format
    """
    api_documentation = request.app.get("api_documentation")

    if not api_documentation:
        api_version = request.app.get("api_version", "v1")
        api_documentation = OpenAPIDocumentation(api_version=api_version)
        request.app["api_documentation"] = api_documentation

    return web.json_response(api_documentation.get_spec())


async def openapi_yaml_handler(request: Request) -> Response:
    """
    Handle requests for the OpenAPI specification in YAML format.

    Args:
        request: HTTP request

    Returns:
        HTTP response with OpenAPI specification in YAML format
    """
    api_documentation = request.app.get("api_documentation")

    if not api_documentation:
        api_version = request.app.get("api_version", "v1")
        api_documentation = OpenAPIDocumentation(api_version=api_version)
        request.app["api_documentation"] = api_documentation

    return web.Response(text=api_documentation.to_yaml(), content_type="text/yaml")


def setup_swagger(
    app: Application,
    api_version: str = "v1",
    swagger_ui_path: str = "/api/docs",
    api_spec_json_path: str = "/api/openapi.json",
    api_spec_yaml_path: str = "/api/openapi.yaml",
    title: str = "WDBX API",
    description: str = "API for the WDBX vector database and processing system",
) -> None:
    """
    Set up Swagger UI for the AIOHTTP application.

    Args:
        app: AIOHTTP application
        api_version: API version string
        swagger_ui_path: Path for the Swagger UI
        api_spec_json_path: Path for the OpenAPI specification in JSON format
        api_spec_yaml_path: Path for the OpenAPI specification in YAML format
        title: API title
        description: API description
    """
    # Create OpenAPI documentation
    api_documentation = OpenAPIDocumentation(
        api_version=api_version, title=title, description=description
    )

    # Store in application
    app["api_documentation"] = api_documentation
    app["api_version"] = api_version
    app["swagger_ui_path"] = swagger_ui_path
    app["api_spec_path"] = api_spec_json_path

    # Add routes
    app.router.add_get(swagger_ui_path, swagger_ui_handler)
    app.router.add_get(api_spec_json_path, openapi_json_handler)
    app.router.add_get(api_spec_yaml_path, openapi_yaml_handler)
