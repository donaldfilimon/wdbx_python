# wdbx/web_ui.py
"""
Web UI for the WDBX database.

This module provides a web-based dashboard for monitoring, visualizing,
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

# Try to import web framework dependencies
try:
    import flask
    from flask import Flask, request, jsonify, render_template, Response
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Install with: pip install flask")

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
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Create templates directory if it doesn't exist
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Create index.html template
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WDBX Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-2.20.0.min.js"></script>
    <style>
        body {
            padding-top: 70px;
            background-color: #f8f9fa;
        }
        .card {
            margin-bottom: 20px;
            box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
            border-radius: 0.5rem;
        }
        .card-header {
            background-color: #f1f8ff;
            border-bottom: 1px solid rgba(0,0,0,.125);
        }
        .stats-value {
            font-size: 2rem;
            font-weight: bold;
        }
        .stats-label {
            font-size: 0.9rem;
            color: #6c757d;
        }
        .navbar-brand {
            font-weight: bold;
        }
        .log-container {
            height: 300px;
            overflow-y: auto;
            background-color: #212529;
            color: #f8f9fa;
            padding: 10px;
            font-family: monospace;
            font-size: 0.9rem;
        }
        .log-info {
            color: #0dcaf0;
        }
        .log-warning {
            color: #ffc107;
        }
        .log-error {
            color: #dc3545;
        }
        .log-debug {
            color: #6c757d;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark fixed-top">
        <div class="container-fluid">
            <a class="navbar-brand" href="#">WDBX Dashboard</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav">
                    <li class="nav-item">
                        <a class="nav-link active" href="#" id="nav-dashboard">Dashboard</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#" id="nav-vectors">Vectors</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#" id="nav-blocks">Blocks</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#" id="nav-settings">Settings</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <div class="container" id="dashboard-container">
        <div class="row">
            <div class="col-md-3">
                <div class="card">
                    <div class="card-header">
                        <h5 class="card-title mb-0">Blocks</h5>
                    </div>
                    <div class="card-body text-center">
                        <div class="stats-value" id="blocks-count">-</div>
                        <div class="stats-label">Total blocks</div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card">
                    <div class="card-header">
                        <h5 class="card-title mb-0">Vectors</h5>
                    </div>
                    <div class="card-body text-center">
                        <div class="stats-value" id="vectors-count">-</div>
                        <div class="stats-label">Total vectors</div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card">
                    <div class="card-header">
                        <h5 class="card-title mb-0">Transactions</h5>
                    </div>
                    <div class="card-body text-center">
                        <div class="stats-value" id="transactions-count">-</div>
                        <div class="stats-label">Total transactions</div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card">
                    <div class="card-header">
                        <h5 class="card-title mb-0">Uptime</h5>
                    </div>
                    <div class="card-body text-center">
                        <div class="stats-value" id="uptime">-</div>
                        <div class="stats-label">Seconds</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5 class="card-title mb-0">Performance</h5>
                    </div>
                    <div class="card-body">
                        <div id="performance-chart" style="height: 300px;"></div>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5 class="card-title mb-0">System Log</h5>
                    </div>
                    <div class="card-body p-0">
                        <div class="log-container" id="log-container">
                            <div class="log-entry log-info">System initialized</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        <h5 class="card-title mb-0">Vector Space Visualization</h5>
                    </div>
                    <div class="card-body">
                        <div id="vector-viz" style="height: 400px;"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="container d-none" id="vectors-container">
        <div class="card">
            <div class="card-header">
                <h5 class="card-title mb-0">Vector Management</h5>
            </div>
            <div class="card-body">
                <div class="row mb-3">
                    <div class="col-md-6">
                        <button class="btn btn-primary" id="btn-add-vector">Add Random Vector</button>
                    </div>
                    <div class="col-md-6 text-end">
                        <input type="text" class="form-control d-inline-block w-50" id="search-vector" placeholder="Search vectors...">
                    </div>
                </div>
                <div class="table-responsive">
                    <table class="table table-striped table-hover">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Description</th>
                                <th>Created</th>
                                <th>Dimension</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody id="vectors-table-body">
                            <!-- Vector data will be loaded here -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <div class="container d-none" id="blocks-container">
        <div class="card">
            <div class="card-header">
                <h5 class="card-title mb-0">Block Explorer</h5>
            </div>
            <div class="card-body">
                <div class="row mb-3">
                    <div class="col-md-6">
                        <select class="form-select d-inline-block w-50" id="chain-selector">
                            <option value="">All chains</option>
                        </select>
                    </div>
                    <div class="col-md-6 text-end">
                        <input type="text" class="form-control d-inline-block w-50" id="search-block" placeholder="Search blocks...">
                    </div>
                </div>
                <div class="table-responsive">
                    <table class="table table-striped table-hover">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Chain</th>
                                <th>Created</th>
                                <th>Embeddings</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody id="blocks-table-body">
                            <!-- Block data will be loaded here -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <div class="container d-none" id="settings-container">
        <div class="card">
            <div class="card-header">
                <h5 class="card-title mb-0">System Settings</h5>
            </div>
            <div class="card-body">
                <form id="settings-form">
                    <div class="mb-3">
                        <label for="setting-vector-dimension" class="form-label">Vector Dimension</label>
                        <input type="number" class="form-control" id="setting-vector-dimension" readonly>
                    </div>
                    <div class="mb-3">
                        <label for="setting-shard-count" class="form-label">Shard Count</label>
                        <input type="number" class="form-control" id="setting-shard-count" readonly>
                    </div>
                    <div class="mb-3">
                        <label for="setting-refresh-interval" class="form-label">Dashboard Refresh Interval (ms)</label>
                        <input type="number" class="form-control" id="setting-refresh-interval" min="500" max="10000" step="100" value="1000">
                    </div>
                    <button type="submit" class="btn btn-primary">Save Settings</button>
                </form>
            </div>
        </div>
    </div>

    <!-- Modal for viewing vector details -->
    <div class="modal fade" id="vectorDetailsModal" tabindex="-1">
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Vector Details</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <div id="vector-details-content"></div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                </div>
            </div>
        </div>
    </div>

    <!-- Modal for viewing block details -->
    <div class="modal fade" id="blockDetailsModal" tabindex="-1">
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Block Details</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <div id="block-details-content"></div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Dashboard state
        let refreshInterval = 1000; // 1 second by default
        let refreshTimer = null;
        let statsHistory = [];
        const MAX_HISTORY_POINTS = 50;

        // Performance charts
        let performanceChart = null;

        // Initialize the dashboard
        document.addEventListener('DOMContentLoaded', function() {
            // Set up navigation
            document.getElementById('nav-dashboard').addEventListener('click', function(e) {
                e.preventDefault();
                showPage('dashboard');
            });
            document.getElementById('nav-vectors').addEventListener('click', function(e) {
                e.preventDefault();
                showPage('vectors');
            });
            document.getElementById('nav-blocks').addEventListener('click', function(e) {
                e.preventDefault();
                showPage('blocks');
            });
            document.getElementById('nav-settings').addEventListener('click', function(e) {
                e.preventDefault();
                showPage('settings');
            });

            // Initialize charts
            initializeCharts();

            // Add random vector button
            document.getElementById('btn-add-vector').addEventListener('click', addRandomVector);

            // Settings form
            document.getElementById('settings-form').addEventListener('submit', function(e) {
                e.preventDefault();
                saveSettings();
            });

            // Start the dashboard
            refreshDashboard();
            startRefreshTimer();
        });

        function showPage(pageId) {
            // Hide all containers
            document.getElementById('dashboard-container').classList.add('d-none');
            document.getElementById('vectors-container').classList.add('d-none');
            document.getElementById('blocks-container').classList.add('d-none');
            document.getElementById('settings-container').classList.add('d-none');

            // Reset active nav links
            document.getElementById('nav-dashboard').classList.remove('active');
            document.getElementById('nav-vectors').classList.remove('active');
            document.getElementById('nav-blocks').classList.remove('active');
            document.getElementById('nav-settings').classList.remove('active');

            // Show selected container and set active nav link
            document.getElementById(`${pageId}-container`).classList.remove('d-none');
            document.getElementById(`nav-${pageId}`).classList.add('active');

            // Load page-specific data
            if (pageId === 'vectors') {
                loadVectors();
            } else if (pageId === 'blocks') {
                loadBlocks();
            } else if (pageId === 'settings') {
                loadSettings();
            }
        }

        function startRefreshTimer() {
            if (refreshTimer) {
                clearInterval(refreshTimer);
            }
            refreshTimer = setInterval(refreshDashboard, refreshInterval);
        }

        function refreshDashboard() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => updateDashboard(data))
                .catch(error => console.error('Error fetching stats:', error));

            // Get recent logs
            fetch('/api/logs')
                .then(response => response.json())
                .then(data => updateLogs(data))
                .catch(error => console.error('Error fetching logs:', error));
        }

        function updateDashboard(stats) {
            // Update stats history
            statsHistory.push(stats);
            if (statsHistory.length > MAX_HISTORY_POINTS) {
                statsHistory.shift();
            }

            // Update counters
            document.getElementById('blocks-count').textContent = stats.blocks_created;
            document.getElementById('vectors-count').textContent = stats.vectors_stored;
            document.getElementById('transactions-count').textContent = stats.transactions_processed;
            document.getElementById('uptime').textContent = stats.uptime.toFixed(1);

            // Update charts
            updatePerformanceChart();
        }

        function updateLogs(logs) {
            const logContainer = document.getElementById('log-container');

            // Add new logs
            logs.forEach(log => {
                const logEntry = document.createElement('div');
                logEntry.className = `log-entry log-${log.level.toLowerCase()}`;
                logEntry.textContent = `[${log.timestamp}] ${log.level}: ${log.message}`;
                logContainer.appendChild(logEntry);
            });

            // Auto-scroll to bottom
            logContainer.scrollTop = logContainer.scrollHeight;

            // Limit number of log entries
            while (logContainer.children.length > 100) {
                logContainer.removeChild(logContainer.firstChild);
            }
        }

        function initializeCharts() {
            // Initialize performance chart with Plotly
            const performanceDiv = document.getElementById('performance-chart');
            Plotly.newPlot(performanceDiv, [
                {
                    x: [],
                    y: [],
                    name: 'Blocks/s',
                    type: 'line',
                    line: { color: 'rgb(0, 123, 255)' }
                },
                {
                    x: [],
                    y: [],
                    name: 'Vectors/s',
                    type: 'line',
                    line: { color: 'rgb(40, 167, 69)' }
                }
            ], {
                margin: { t: 10, r: 10, b: 40, l: 40 },
                xaxis: { title: 'Time' },
                yaxis: { title: 'Operations per second' },
                showlegend: true,
                legend: { orientation: 'h', y: 1.1 }
            });

            // Initialize vector visualization
            const vectorVizDiv = document.getElementById('vector-viz');
            Plotly.newPlot(vectorVizDiv, [
                {
                    x: [],
                    y: [],
                    z: [],
                    mode: 'markers',
                    type: 'scatter3d',
                    marker: {
                        size: 5,
                        color: [],
                        colorscale: 'Viridis',
                        opacity: 0.8
                    },
                    text: []
                }
            ], {
                margin: { t: 10, r: 10, b: 10, l: 10 },
                scene: {
                    xaxis: { title: 'Component 1' },
                    yaxis: { title: 'Component 2' },
                    zaxis: { title: 'Component 3' }
                }
            });

            // Load vector visualization data
            loadVectorVisualization();
        }

        function updatePerformanceChart() {
            if (statsHistory.length < 2) return;

            const timestamps = statsHistory.map(s => new Date(s.timestamp * 1000).toLocaleTimeString());
            const blocksPerSecond = statsHistory.map(s => s.blocks_per_second);
            const vectorsPerSecond = statsHistory.map(s => s.vectors_per_second);

            const performanceDiv = document.getElementById('performance-chart');

            Plotly.update(performanceDiv, {
                x: [timestamps, timestamps],
                y: [blocksPerSecond, vectorsPerSecond]
            }, {}, [0, 1]);
        }

        function loadVectorVisualization() {
            fetch('/api/vectors/visualization')
                .then(response => response.json())
                .then(data => {
                    const vectorVizDiv = document.getElementById('vector-viz');

                    Plotly.update(vectorVizDiv, {
                        x: [data.x],
                        y: [data.y],
                        z: [data.z],
                        'marker.color': [data.colors],
                        text: [data.labels]
                    }, {}, [0]);
                })
                .catch(error => console.error('Error loading vector visualization:', error));
        }

        function loadVectors() {
            fetch('/api/vectors')
                .then(response => response.json())
                .then(data => {
                    const tableBody = document.getElementById('vectors-table-body');
                    tableBody.innerHTML = '';

                    data.forEach(vector => {
                        const row = document.createElement('tr');
                        row.innerHTML = `
                            <td>${vector.id.substring(0, 8)}...</td>
                            <td>${vector.description || 'N/A'}</td>
                            <td>${new Date(vector.timestamp * 1000).toLocaleString()}</td>
                            <td>${vector.dimension}</td>
                            <td>
                                <button class="btn btn-sm btn-primary" onclick="viewVectorDetails('${vector.id}')">View</button>
                                <button class="btn btn-sm btn-danger" onclick="deleteVector('${vector.id}')">Delete</button>
                            </td>
                        `;
                        tableBody.appendChild(row);
                    });
                })
                .catch(error => console.error('Error loading vectors:', error));
        }

        function loadBlocks() {
            // Load chains for the selector
            fetch('/api/chains')
                .then(response => response.json())
                .then(data => {
                    const selector = document.getElementById('chain-selector');
                    // Clear options except the first one
                    while (selector.options.length > 1) {
                        selector.remove(1);
                    }

                    data.forEach(chain => {
                        const option = document.createElement('option');
                        option.value = chain.id;
                        option.textContent = `Chain ${chain.id.substring(0, 8)}... (${chain.block_count} blocks)`;
                        selector.appendChild(option);
                    });
                })
                .catch(error => console.error('Error loading chains:', error));

            // Load blocks
            fetch('/api/blocks')
                .then(response => response.json())
                .then(data => {
                    const tableBody = document.getElementById('blocks-table-body');
                    tableBody.innerHTML = '';

                    data.forEach(block => {
                        const row = document.createElement('tr');
                        row.innerHTML = `
                            <td>${block.id.substring(0, 8)}...</td>
                            <td>${block.chain_id.substring(0, 8)}...</td>
                            <td>${new Date(block.timestamp * 1000).toLocaleString()}</td>
                            <td>${block.embedding_count}</td>
                            <td>
                                <button class="btn btn-sm btn-primary" onclick="viewBlockDetails('${block.id}')">View</button>
                            </td>
                        `;
                        tableBody.appendChild(row);
                    });
                })
                .catch(error => console.error('Error loading blocks:', error));
        }

        function loadSettings() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('setting-vector-dimension').value = data.vector_dimension;
                    document.getElementById('setting-shard-count').value = data.shard_count;
                })
                .catch(error => console.error('Error loading settings:', error));
        }

        function saveSettings() {
            const refreshInterval = parseInt(document.getElementById('setting-refresh-interval').value);
            window.refreshInterval = refreshInterval;
            startRefreshTimer();
            alert('Settings saved!');
        }

        function addRandomVector() {
            fetch('/api/vectors/random', {
                method: 'POST'
            })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert(`Vector added with ID: ${data.vector_id}`);
                        loadVectors();
                    } else {
                        alert('Failed to add vector');
                    }
                })
                .catch(error => console.error('Error adding vector:', error));
        }

        function viewVectorDetails(vectorId) {
            fetch(`/api/vectors/${vectorId}`)
                .then(response => response.json())
                .then(data => {
                    const content = document.getElementById('vector-details-content');
                    content.innerHTML = `
                        <div class="table-responsive">
                            <table class="table table-bordered">
                                <tr>
                                    <th>ID</th>
                                    <td>${data.id}</td>
                                </tr>
                                <tr>
                                    <th>Description</th>
                                    <td>${data.description || 'N/A'}</td>
                                </tr>
                                <tr>
                                    <th>Created</th>
                                    <td>${new Date(data.timestamp * 1000).toLocaleString()}</td>
                                </tr>
                                <tr>
                                    <th>Dimension</th>
                                    <td>${data.dimension}</td>
                                </tr>
                            </table>
                        </div>
                        <h6>Metadata</h6>
                        <pre class="bg-light p-3">${JSON.stringify(data.metadata, null, 2)}</pre>
                        <h6>Vector Preview (first 10 components)</h6>
                        <div id="vector-preview" style="height: 200px;"></div>
                    `;

                    // Show the modal
                    const modal = new bootstrap.Modal(document.getElementById('vectorDetailsModal'));
                    modal.show();

                    // Render vector preview
                    const previewDiv = document.getElementById('vector-preview');
                    const previewData = data.vector.slice(0, 10);
                    Plotly.newPlot(previewDiv, [
                        {
                            y: previewData,
                            type: 'bar',
                            marker: {
                                color: 'rgb(0, 123, 255)'
                            }
                        }
                    ], {
                        margin: { t: 10, r: 10, b: 30, l: 40 },
                        xaxis: { title: 'Component' },
                        yaxis: { title: 'Value' }
                    });
                })
                .catch(error => console.error('Error loading vector details:', error));
        }

        function viewBlockDetails(blockId) {
            fetch(`/api/blocks/${blockId}`)
                .then(response => response.json())
                .then(data => {
                    const content = document.getElementById('block-details-content');
                    content.innerHTML = `
                        <div class="table-responsive">
                            <table class="table table-bordered">
                                <tr>
                                    <th>ID</th>
                                    <td>${data.id}</td>
                                </tr>
                                <tr>
                                    <th>Chain ID</th>
                                    <td>${data.chain_id}</td>
                                </tr>
                                <tr>
                                    <th>Created</th>
                                    <td>${new Date(data.timestamp * 1000).toLocaleString()}</td>
                                </tr>
                                <tr>
                                    <th>Hash</th>
                                    <td>${data.hash}</td>
                                </tr>
                                <tr>
                                    <th>Previous Hash</th>
                                    <td>${data.previous_hash || 'None'}</td>
                                </tr>
                                <tr>
                                    <th>Nonce</th>
                                    <td>${data.nonce}</td>
                                </tr>
                                <tr>
                                    <th>Embedding Count</th>
                                    <td>${data.embedding_count}</td>
                                </tr>
                            </table>
                        </div>
                        <h6>Data</h6>
                        <pre class="bg-light p-3">${JSON.stringify(data.data, null, 2)}</pre>
                        <h6>Context References</h6>
                        <pre class="bg-light p-3">${JSON.stringify(data.context_references, null, 2)}</pre>
                    `;

                    // Show the modal
                    const modal = new bootstrap.Modal(document.getElementById('blockDetailsModal'));
                    modal.show();
                })
                .catch(error => console.error('Error loading block details:', error));
        }

        function deleteVector(vectorId) {
            if (confirm('Are you sure you want to delete this vector?')) {
                fetch(`/api/vectors/${vectorId}`, {
                    method: 'DELETE'
                })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            alert('Vector deleted');
                            loadVectors();
                        } else {
                            alert('Failed to delete vector');
                        }
                    })
                    .catch(error => console.error('Error deleting vector:', error));
            }
        }
    </script>
</body>
</html>
"""

# Write the HTML template to file
with open(os.path.join(TEMPLATES_DIR, "index.html"), "w") as f:
    f.write(INDEX_HTML)


class WDBXWebMonitor:
    """
    Web-based monitoring for WDBX.
    """
    def __init__(self, wdbx_instance: WDBX):
        self.wdbx = wdbx_instance
        self.stats_history = []
        self.max_history_size = 100
        self.log_history = []
        self.max_log_size = 1000
        self.log_handler = None

        # Set up logging
        self._setup_logging()

    def _setup_logging(self):
        """Set up logging to capture log messages."""
        class WebUILogHandler(logging.Handler):
            def __init__(self, monitor):
                super().__init__()
                self.monitor = monitor

            def emit(self, record):
                log_entry = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created)),
                    "level": record.levelname,
                    "message": self.format(record)
                }
                self.monitor.add_log(log_entry)

        self.log_handler = WebUILogHandler(self)
        self.log_handler.setLevel(logging.INFO)
        self.log_handler.setFormatter(logging.Formatter("%(message)s"))

        logger.addHandler(self.log_handler)

    def add_log(self, log_entry):
        """Add a log entry to the history."""
        self.log_history.append(log_entry)
        if len(self.log_history) > self.max_log_size:
            self.log_history = self.log_history[-self.max_log_size:]

    def get_recent_logs(self, count=10):
        """Get the most recent log entries."""
        return self.log_history[-count:]

    def update_stats(self):
        """Update stats history with current stats."""
        stats = self.wdbx.get_system_stats()

        # Add timestamp if not present
        if "timestamp" not in stats:
            stats["timestamp"] = time.time()

        # Add to history
        self.stats_history.append(stats)

        # Trim history if needed
        if len(self.stats_history) > self.max_history_size:
            self.stats_history = self.stats_history[-self.max_history_size:]

        return stats


def create_web_app(wdbx_instance: WDBX) -> Optional[Flask]:
    """
    Create a Flask web application for the WDBX dashboard.

    Args:
        wdbx_instance: WDBX instance to monitor

    Returns:
        Flask application, or None if Flask is not available
    """
    if not FLASK_AVAILABLE:
        return None

    app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
    monitor = WDBXWebMonitor(wdbx_instance)

    # Routes
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/api/stats')
    def get_stats():
        return jsonify(monitor.update_stats())

    @app.route('/api/logs')
    def get_logs():
        count = request.args.get('count', 10, type=int)
        return jsonify(monitor.get_recent_logs(count))

    @app.route('/api/vectors')
    def get_vectors():
        # Get vectors from the vector store
        vectors = []
        for vector_id, vector in wdbx_instance.vector_store.vectors.items():
            vector_data = {
                "id": vector_id,
                "dimension": len(vector.vector),
                "timestamp": vector.metadata.get("timestamp", 0),
                "description": vector.metadata.get("description", "")
            }
            vectors.append(vector_data)

        # Sort by timestamp, newest first
        vectors.sort(key=lambda x: x["timestamp"], reverse=True)

        return jsonify(vectors)

    @app.route('/api/vectors/<vector_id>')
    def get_vector(vector_id):
        vector = wdbx_instance.vector_store.get(vector_id)
        if not vector:
            return jsonify({"error": "Vector not found"}), 404

        vector_data = {
            "id": vector_id,
            "dimension": len(vector.vector),
            "timestamp": vector.metadata.get("timestamp", 0),
            "description": vector.metadata.get("description", ""),
            "metadata": vector.metadata,
            "vector": vector.vector.tolist()
        }

        return jsonify(vector_data)

    @app.route('/api/vectors/<vector_id>', methods=['DELETE'])
    def delete_vector(vector_id):
        success = wdbx_instance.vector_store.delete(vector_id)
        return jsonify({"success": success})

    @app.route('/api/vectors/random', methods=['POST'])
    def add_random_vector():
        # Create a random vector
        vector = np.random.randn(wdbx_instance.vector_dimension).astype(np.float32)

        # Create an embedding
        embedding = EmbeddingVector(
            vector=vector,
            metadata={
                "description": f"Random vector {time.strftime('%Y-%m-%d %H:%M:%S')}",
                "timestamp": time.time()
            }
        )

        # Store the embedding
        try:
            vector_id = wdbx_instance.store_embedding(embedding)
            return jsonify({"success": True, "vector_id": vector_id})
        except Exception as e:
            logger.error(f"Error adding random vector: {e}")
            return jsonify({"success": False, "error": str(e)})

    @app.route('/api/vectors/visualization')
    def get_vector_visualization():
        # Get vectors from the vector store
        vectors = list(wdbx_instance.vector_store.vectors.values())

        if not vectors:
            return jsonify({
                "x": [],
                "y": [],
                "z": [],
                "colors": [],
                "labels": []
            })

        # If we have Plotly and more than 1 vector, use PCA for visualization
        if PLOTLY_AVAILABLE and len(vectors) > 1:
            try:
                from sklearn.decomposition import PCA

                # Get vector data
                vector_data = np.array([v.vector for v in vectors])

                # Apply PCA
                pca = PCA(n_components=3)
                pca_result = pca.fit_transform(vector_data)

                # Prepare visualization data
                x = pca_result[:, 0].tolist()
                y = pca_result[:, 1].tolist()
                z = pca_result[:, 2].tolist()

                # Generate labels
                labels = []
                for i, v in enumerate(vectors):
                    desc = v.metadata.get("description", f"Vector {i}")
                    labels.append(desc)

                # Generate colors based on vector norm
                norms = [np.linalg.norm(v.vector) for v in vectors]
                colors = norms

                return jsonify({
                    "x": x,
                    "y": y,
                    "z": z,
                    "colors": colors,
                    "labels": labels
                })
            except Exception as e:
                logger.error(f"Error generating vector visualization: {e}")

        # Fallback to simple visualization
        x = [i for i in range(len(vectors))]
        y = [0 for _ in range(len(vectors))]
        z = [0 for _ in range(len(vectors))]
        colors = [i for i in range(len(vectors))]
        labels = [f"Vector {i}" for i in range(len(vectors))]

        return jsonify({
            "x": x,
            "y": y,
            "z": z,
            "colors": colors,
            "labels": labels
        })

    @app.route('/api/blocks')
    def get_blocks():
        # Get blocks from the blockchain manager
        blocks = []
        for block_id, block in wdbx_instance.block_chain_manager.blocks.items():
            chain_id = wdbx_instance.block_chain_manager.block_chain.get(block_id, "unknown")
            block_data = {
                "id": block_id,
                "chain_id": chain_id,
                "timestamp": block.timestamp,
                "embedding_count": len(block.embeddings)
            }
            blocks.append(block_data)

        # Sort by timestamp, newest first
        blocks.sort(key=lambda x: x["timestamp"], reverse=True)

        return jsonify(blocks)

    @app.route('/api/blocks/<block_id>')
    def get_block(block_id):
        block = wdbx_instance.block_chain_manager.get_block(block_id)
        if not block:
            return jsonify({"error": "Block not found"}), 404

        chain_id = wdbx_instance.block_chain_manager.block_chain.get(block_id, "unknown")

        block_data = {
            "id": block_id,
            "chain_id": chain_id,
            "timestamp": block.timestamp,
            "hash": block.hash,
            "previous_hash": block.previous_hash,
            "nonce": block.nonce,
            "embedding_count": len(block.embeddings),
            "data": block.data,
            "context_references": block.context_references
        }

        return jsonify(block_data)

    @app.route('/api/chains')
    def get_chains():
        chains = []
        for chain_id, head_block_id in wdbx_instance.block_chain_manager.chain_heads.items():
            chain_blocks = wdbx_instance.block_chain_manager.get_chain(chain_id)
            chain_data = {
                "id": chain_id,
                "head_block_id": head_block_id,
                "block_count": len(chain_blocks)
            }
            chains.append(chain_data)

        return jsonify(chains)

    return app


def run_web_ui(wdbx_instance: WDBX, host: str = "127.0.0.1", port: int = 5000):
    """
    Run the web UI.

    Args:
        wdbx_instance: WDBX instance to monitor
        host: Host to listen on
        port: Port to listen on
    """
    if not FLASK_AVAILABLE:
        print("Flask not available. Install with: pip install flask")
        return

    app = create_web_app(wdbx_instance)
    if app:
        app.run(host=host, port=port, debug=False)
    else:
        print("Failed to create web application")


if __name__ == "__main__":
    # Example usage
    from wdbx import WDBX

    wdbx = WDBX(vector_dimension=128, num_shards=4)

    # Create some sample data
    for i in range(10):
        vector = np.random.randn(wdbx.vector_dimension).astype(np.float32)
        embedding = EmbeddingVector(
            vector=vector,
            metadata={"description": f"Sample embedding {i}", "timestamp": time.time()}
        )
        wdbx.store_embedding(embedding)

    # Run the web UI
    run_web_ui(wdbx)
