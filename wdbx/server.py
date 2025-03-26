# wdbx/server.py
import asyncio
import json
from aiohttp import web
from wdbx import WDBX
from wdbx.persona import PersonaManager

async def health(request):
    return web.json_response({"status": "healthy"})

async def stats(request):
    stats = request.app["wdbx"].get_system_stats()
    return web.json_response(stats)

async def process(request):
    data = await request.json()
    user_input = data.get("input")
    context = data.get("context", {})
    if not user_input:
        return web.json_response({"error": "Missing input"}, status=400)
    persona_manager = request.app["persona_manager"]
    response, block_id = persona_manager.process_user_input(user_input, context)
    return web.json_response({"response": response, "block_id": block_id})

async def embedding(request):
    data = await request.json()
    vector_data = data.get("vector")
    metadata = data.get("metadata", {})
    if not vector_data:
        return web.json_response({"error": "Missing vector data"}, status=400)
    import numpy as np
    from wdbx.data_structures import EmbeddingVector
    vector = np.array(vector_data, dtype=np.float32)
    embedding_obj = EmbeddingVector(vector=vector, metadata=metadata)
    vector_id = request.app["wdbx"].store_embedding(embedding_obj)
    return web.json_response({"vector_id": vector_id})

async def search(request):
    data = await request.json()
    vector_data = data.get("vector")
    if not vector_data:
        return web.json_response({"error": "Missing vector data"}, status=400)
    import numpy as np
    vector = np.array(vector_data, dtype=np.float32)
    results = request.app["wdbx"].search_similar_vectors(vector)
    return web.json_response({"results": results})

def run_server(host: str, port: int, vector_dim: int, num_shards: int) -> None:
    wdbx_instance = WDBX(vector_dimension=vector_dim, num_shards=num_shards)
    persona_manager = PersonaManager(wdbx_instance)
    app = web.Application()
    app["wdbx"] = wdbx_instance
    app["persona_manager"] = persona_manager
    app.add_routes([
        web.get("/health", health),
        web.get("/stats", stats),
        web.post("/process", process),
        web.post("/embedding", embedding),
        web.post("/search", search)
    ])
    web.run_app(app, host=host, port=port)

