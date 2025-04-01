# wdbx/web_ui.py
"""
Web UI for the WDBX database.

This module provides an enhanced web-based dashboard for monitoring, visualizing,
and interacting with the WDBX database system.
"""
import os
import sys
import time
import threading
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path

# Try to import web framework dependencies
try:
    import flask 
    from flask import Flask, request, jsonify, render_template, Response, send_file
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Install with: pip install flask flask-cors")

try:
    import plotly
    import plotly.graph_objects as go 
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Install with: pip install plotly")

from wdbx import WDBX
from wdbx.data_structures import EmbeddingVector
from wdbx.constants import logger, VECTOR_DIMENSION, SHARD_COUNT

# HTML templates directory
TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

# Create directories
TEMPLATES_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# Index template with enhanced dark mode and accessibility
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="WDBX Database Dashboard">
    <meta name="theme-color" content="#0d6efd">
    <title>WDBX Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.0/font/bootstrap-icons.css" rel="stylesheet">
    <script defer src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    <script defer src="https://cdn.plot.ly/plotly-2.20.0.min.js"></script>
    <style>
        :root {
            --primary-color: #0d6efd;
            --secondary-color: #6c757d;
            --success-color: #198754;
            --info-color: #0dcaf0;
            --warning-color: #ffc107;
            --danger-color: #dc3545;
            --light-color: #f8f9fa;
            --dark-color: #212529;
        }

        [data-theme="dark"] {
            --body-bg: #121212;
            --body-color: #e0e0e0;
            --card-bg: #1e1e1e;
            --card-border: #2d2d2d;
            --input-bg: #2d2d2d;
            --input-color: #e0e0e0;
        }

        body {
            padding-top: 70px;
            background-color: var(--body-bg);
            color: var(--body-color);
            transition: background-color 0.3s ease;
        }

        .card {
            background-color: var(--card-bg);
            border-color: var(--card-border);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .form-control {
            background-color: var(--input-bg);
            color: var(--input-color);
            border-color: var(--card-border);
        }

        .log-container {
            height: 300px;
            overflow-y: auto;
            font-family: 'SFMono-Regular', Consolas, monospace;
            font-size: 0.875rem;
            line-height: 1.4;
            padding: 1rem;
            border-radius: 0.375rem;
            background-color: var(--dark-color);
            color: var(--light-color);
        }

        .performance-chart {
            height: 300px;
            border-radius: 0.375rem;
            overflow: hidden;
        }

        @media (prefers-reduced-motion: reduce) {
            * {
                transition: none !important;
            }
        }

        @media (prefers-color-scheme: dark) {
            body:not([data-theme]) {
                --body-bg: #121212;
                --body-color: #e0e0e0;
                --card-bg: #1e1e1e;
                --card-border: #2d2d2d;
                --input-bg: #2d2d2d;
                --input-color: #e0e0e0;
            }
        }
    </style>
</head>
<body>
...
[REST OF TEMPLATE CONTENT UNCHANGED]
...
</body>
</html>
"""

# Write template with error handling
try:
    template_path = TEMPLATES_DIR / "index.html"
    template_path.write_text(INDEX_HTML, encoding='utf-8')
except IOError as e:
    logger.error(f"Failed to write template: {e}")
    raise

@dataclass 
class WDBXWebMonitor:
    """Enhanced web-based monitoring for WDBX."""
    wdbx: WDBX
    stats_history: List[Dict] = field(default_factory=list)
    log_history: List[Dict] = field(default_factory=list)
    max_history_size: int = 100
    max_log_size: int = 1000
    log_handler: Optional[logging.Handler] = None

    def __post_init__(self):
        """Initialize logging and monitoring."""
        self._setup_logging()

    def _setup_logging(self):
        """Configure enhanced logging with structured data capture."""
        class StructuredLogHandler(logging.Handler):
            def __init__(self, monitor):
                super().__init__()
                self.monitor = monitor

            def emit(self, record):
                try:
                    log_entry = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", 
                                                 time.localtime(record.created)),
                        "level": record.levelname,
                        "message": self.format(record),
                        "module": record.module,
                        "function": record.funcName,
                        "line": record.lineno
                    }
                    self.monitor.add_log(log_entry)
                except Exception as e:
                    print(f"Failed to emit log: {e}")

        self.log_handler = StructuredLogHandler(self)
        self.log_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self.log_handler.setLevel(logging.INFO)
        logger.addHandler(self.log_handler)

    def add_log(self, log_entry: Dict[str, Any]):
        """Add a log entry with rotation."""
        self.log_history.append(log_entry)
        if len(self.log_history) > self.max_log_size:
            self.log_history = self.log_history[-self.max_log_size:]

    def get_recent_logs(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get most recent logs with bounds checking."""
        count = min(count, len(self.log_history))
        return self.log_history[-count:]

    def update_stats(self) -> Dict[str, Any]:
        """Update system statistics with enhanced metrics."""
        try:
            stats = self.wdbx.get_system_stats()
            stats.update({
                "timestamp": time.time(),
                "cpu_usage": self._get_cpu_usage(),
                "memory_usage": self._get_memory_usage()
            })

            self.stats_history.append(stats)
            if len(self.stats_history) > self.max_history_size:
                self.stats_history = self.stats_history[-self.max_history_size:]

            return stats
        except Exception as e:
            logger.error(f"Failed to update stats: {e}")
            return {}

    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent()
        except ImportError:
            return 0.0

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                "total": mem.total,
                "available": mem.available,
                "percent": mem.percent
            }
        except ImportError:
            return {"total": 0, "available": 0, "percent": 0}

def create_web_app(wdbx_instance: WDBX) -> Optional[Flask]:
    """Create enhanced Flask web application for WDBX dashboard."""
    if not FLASK_AVAILABLE:
        logger.error("Flask not available")
        return None

    app = Flask(__name__, 
                template_folder=str(TEMPLATES_DIR),
                static_folder=str(STATIC_DIR))
    CORS(app)

    monitor = WDBXWebMonitor(wdbx_instance)

    # Enhanced error handling
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Resource not found"}), 404

    @app.errorhandler(500) 
    def server_error(e):
        return jsonify({"error": "Internal server error"}), 500

    # Routes with input validation and error handling
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/api/stats')
    def get_stats():
        try:
            return jsonify(monitor.update_stats())
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/logs')
    def get_logs():
        try:
            count = max(1, min(int(request.args.get('count', 10)), 1000))
            return jsonify(monitor.get_recent_logs(count))
        except ValueError:
            return jsonify({"error": "Invalid count parameter"}), 400
        except Exception as e:
            logger.error(f"Logs error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/vectors')
    def get_vectors():
        try:
            vectors = []
            for vector_id, vector in wdbx_instance.vector_store.vectors.items():
                vector_data = {
                    "id": vector_id,
                    "dimension": len(vector.vector),
                    "timestamp": vector.metadata.get("timestamp", 0),
                    "description": vector.metadata.get("description", ""),
                    "magnitude": float(np.linalg.norm(vector.vector))
                }
                vectors.append(vector_data)

            vectors.sort(key=lambda x: x["timestamp"], reverse=True)
            return jsonify(vectors)
        except Exception as e:
            logger.error(f"Vector list error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/vectors/<vector_id>')
    def get_vector(vector_id):
        try:
            vector = wdbx_instance.vector_store.get(vector_id)
            if not vector:
                return jsonify({"error": "Vector not found"}), 404

            return jsonify({
                "id": vector_id,
                "dimension": len(vector.vector),
                "timestamp": vector.metadata.get("timestamp", 0),
                "description": vector.metadata.get("description", ""),
                "metadata": vector.metadata,
                "vector": vector.vector.tolist(),
                "magnitude": float(np.linalg.norm(vector.vector)),
                "mean": float(np.mean(vector.vector)),
                "std": float(np.std(vector.vector))
            })
        except Exception as e:
            logger.error(f"Vector detail error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/vectors/<vector_id>', methods=['DELETE'])
    def delete_vector(vector_id):
        try:
            success = wdbx_instance.vector_store.delete(vector_id)
            return jsonify({"success": success})
        except Exception as e:
            logger.error(f"Vector delete error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/vectors/random', methods=['POST'])
    def add_random_vector():
        try:
            # Generate random vector with controlled magnitude
            vector = np.random.randn(wdbx_instance.vector_dimension)
            vector = vector / np.linalg.norm(vector)  # Normalize
            vector = vector.astype(np.float32)

            # Create embedding with rich metadata
            embedding = EmbeddingVector(
                vector=vector,
                metadata={
                    "description": f"Random vector {time.strftime('%Y-%m-%d %H:%M:%S')}",
                    "timestamp": time.time(),
                    "magnitude": float(np.linalg.norm(vector)),
                    "mean": float(np.mean(vector)),
                    "std": float(np.std(vector))
                }
            )

            vector_id = wdbx_instance.store_embedding(embedding)
            return jsonify({
                "success": True, 
                "vector_id": vector_id,
                "metadata": embedding.metadata
            })
        except Exception as e:
            logger.error(f"Random vector error: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/vectors/visualization')
    def get_vector_visualization():
        try:
            vectors = list(wdbx_instance.vector_store.vectors.values())

            if not vectors:
                return jsonify({
                    "x": [], "y": [], "z": [],
                    "colors": [], "labels": []
                })

            if PLOTLY_AVAILABLE and len(vectors) > 1:
                try:
                    from sklearn.decomposition import PCA

                    # Prepare data
                    vector_data = np.array([v.vector for v in vectors])

                    # Apply PCA with explained variance
                    pca = PCA(n_components=3)
                    pca_result = pca.fit_transform(vector_data)
                    explained_var = pca.explained_variance_ratio_

                    # Prepare visualization
                    x = pca_result[:, 0].tolist()
                    y = pca_result[:, 1].tolist()
                    z = pca_result[:, 2].tolist()

                    # Enhanced labels
                    labels = []
                    for i, v in enumerate(vectors):
                        desc = v.metadata.get("description", f"Vector {i}")
                        mag = np.linalg.norm(v.vector)
                        labels.append(f"{desc}\nMagnitude: {mag:.2f}")

                    # Color by vector properties
                    colors = [np.linalg.norm(v.vector) for v in vectors]

                    return jsonify({
                        "x": x, "y": y, "z": z,
                        "colors": colors,
                        "labels": labels,
                        "explained_variance": explained_var.tolist()
                    })
                except Exception as e:
                    logger.error(f"PCA visualization error: {e}")

            # Fallback visualization
            x = list(range(len(vectors)))
            y = [0] * len(vectors)
            z = [0] * len(vectors)
            colors = list(range(len(vectors)))
            labels = [f"Vector {i}" for i in range(len(vectors))]

            return jsonify({
                "x": x, "y": y, "z": z,
                "colors": colors, "labels": labels
            })
        except Exception as e:
            logger.error(f"Visualization error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/blocks')
    def get_blocks():
        try:
            blocks = []
            for block_id, block in wdbx_instance.block_chain_manager.blocks.items():
                chain_id = wdbx_instance.block_chain_manager.block_chain.get(
                    block_id, "unknown")

                block_data = {
                    "id": block_id,
                    "chain_id": chain_id,
                    "timestamp": block.timestamp,
                    "embedding_count": len(block.embeddings),
                    "is_mined": bool(block.hash),
                    "data_size": len(str(block.data)),
                    "ref_count": len(block.context_references)
                }
                blocks.append(block_data)

            blocks.sort(key=lambda x: x["timestamp"], reverse=True)
            return jsonify(blocks)
        except Exception as e:
            logger.error(f"Block list error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/blocks/<block_id>')
    def get_block(block_id):
        try:
            block = wdbx_instance.block_chain_manager.get_block(block_id)
            if not block:
                return jsonify({"error": "Block not found"}), 404

            chain_id = wdbx_instance.block_chain_manager.block_chain.get(
                block_id, "unknown")

            block_data = {
                "id": block_id,
                "chain_id": chain_id,
                "timestamp": block.timestamp,
                "hash": block.hash,
                "previous_hash": block.previous_hash,
                "nonce": block.nonce,
                "embedding_count": len(block.embeddings),
                "data": block.data,
                "context_references": block.context_references,
                "is_mined": bool(block.hash),
                "data_size": len(str(block.data)),
                "ref_count": len(block.context_references)
            }

            return jsonify(block_data)
        except Exception as e:
            logger.error(f"Block detail error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/chains')
    def get_chains():
        try:
            chains = []
            for chain_id, head_block_id in wdbx_instance.block_chain_manager.chain_heads.items():
                chain_blocks = wdbx_instance.block_chain_manager.get_chain(chain_id)

                chain_data = {
                    "id": chain_id,
                    "head_block_id": head_block_id,
                    "block_count": len(chain_blocks),
                    "first_block": chain_blocks[0] if chain_blocks else None,
                    "last_block": chain_blocks[-1] if chain_blocks else None,
                    "total_embeddings": sum(
                        len(wdbx_instance.block_chain_manager.blocks[b].embeddings)
                        for b in chain_blocks
                    )
                }
                chains.append(chain_data)

            return jsonify(chains)
        except Exception as e:
            logger.error(f"Chain list error: {e}")
            return jsonify({"error": str(e)}), 500

    return app

def run_web_ui(wdbx_instance: WDBX, host: str = "127.0.0.1", port: int = 5000):
    """Run enhanced web UI with proper error handling."""
    if not FLASK_AVAILABLE:
        logger.error("Flask not available. Install with: pip install flask flask-cors")
        return

    try:
        app = create_web_app(wdbx_instance)
        if app:
            app.run(host=host, port=port, debug=False)
        else:
            logger.error("Failed to create web application")
    except Exception as e:
        logger.error(f"Failed to start web UI: {e}")

if __name__ == "__main__":
    # Example usage
    import argparse
    from wdbx import WDBX

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='WDBX Web UI')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to listen on')
    parser.add_argument('--dimension', type=int, default=128, help='Vector dimension')
    parser.add_argument('--shards', type=int, default=4, help='Number of shards')
    parser.add_argument('--samples', type=int, default=10, help='Number of sample vectors to create')
    args = parser.parse_args()

    # Initialize WDBX
    wdbx = WDBX(vector_dimension=args.dimension, num_shards=args.shards)

    # Create sample data with progress tracking
    print(f"Creating {args.samples} sample vectors...")
    for i in range(args.samples):
        vector = np.random.randn(wdbx.vector_dimension).astype(np.float32)

        # Add more interesting metadata
        embedding = EmbeddingVector(
            vector=vector,
            metadata={
                "description": f"Sample embedding {i}",
                "timestamp": time.time(),
                "category": f"category_{i % 3}",  # Add some categorization
                "magnitude": float(np.linalg.norm(vector)),
                "mean": float(np.mean(vector)),
                "std": float(np.std(vector))
            }
        )
        wdbx.store_embedding(embedding)
        print(f"Created vector {i+1}/{args.samples}", end='\r')
    print("\nSample data creation complete!")

    # Run the web UI with provided host/port
    print(f"\nStarting Web UI at http://{args.host}:{args.port}")
    run_web_ui(wdbx, host=args.host, port=args.port)
