============
Monitoring
============

WDBX provides robust monitoring capabilities to help you observe the health, performance, and resource usage of your WDBX deployment. This documentation covers the built-in health checks and Prometheus metrics integration.

Health Checks
============

WDBX includes a comprehensive health monitoring system that allows operators to assess system health and detect problems early. The health monitoring system is designed to be used in production environments to ensure that WDBX is functioning correctly.

Health Status Levels
-------------------

Health checks in WDBX use the following status levels:

* **OK**: The component is functioning normally
* **WARNING**: The component is functioning but with reduced performance or potential issues
* **ERROR**: The component is not functioning correctly
* **UNKNOWN**: The component's status could not be determined

Built-in Health Checks
---------------------

WDBX includes the following built-in health checks:

* **Disk Space**: Monitors available disk space in the data directory
* **Memory Usage**: Monitors process and system memory usage
* **Vector Store**: Checks the health of the vector storage
* **Data Directory**: Verifies data directory exists and is accessible
* **ML Backend**: Validates that the ML backend is functioning correctly

Using Health Checks
------------------

You can access health checks in several ways:

1. **HTTP Endpoint**: When running the WDBX server, a health endpoint is available at ``/health``
2. **Direct API Access**: You can use the health module directly in your code
3. **Command Line**: The WDBX CLI provides a ``health`` command

Example - HTTP Health Check
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Request the health status from a running WDBX server
    curl http://localhost:8000/health

Example - Direct API Usage
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from wdbx.health import HealthMonitor, print_health_report
    
    # Create a health monitor for your WDBX instance
    monitor = HealthMonitor(wdbx_instance)
    
    # Get a comprehensive health report
    report = monitor.get_health_report()
    
    # Check the overall status
    if report.status.value == "error":
        print("System has critical issues!")
    
    # Print a formatted report to the console
    print_health_report(wdbx_instance)

Example - CLI Usage
~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Get a health report from the command line
    wdbx health

    # Get detailed health information
    wdbx health --verbose


Prometheus Metrics
=================

WDBX provides integration with Prometheus for collecting and exposing metrics. This allows you to monitor WDBX using industry-standard tools like Prometheus and Grafana.

Available Metrics
----------------

WDBX exposes the following categories of metrics:

* **Vector Metrics**: Count, creation, search operations
* **Block Metrics**: Count, creation, search operations
* **Memory Metrics**: Process memory usage, optimization operations
* **Performance Metrics**: Latency of search and create operations
* **API Metrics**: HTTP request count and latency
* **Background Task Metrics**: Active tasks and errors
* **I/O Metrics**: Persistence operations count and latency

Enabling Prometheus Metrics
--------------------------

To use Prometheus metrics, you need to:

1. Install the required dependencies:

.. code-block:: bash

    pip install prometheus-client

2. Start the metrics server in your application:

.. code-block:: python

    from wdbx.prometheus import get_metrics
    
    # Get the metrics instance
    metrics = get_metrics(prefix="wdbx")
    
    # Start the metrics server on port 9090
    metrics.start_server(port=9090)

3. Configure Prometheus to scrape metrics from your WDBX instance:

.. code-block:: yaml

    # Example prometheus.yml configuration
    scrape_configs:
      - job_name: 'wdbx'
        scrape_interval: 15s
        static_configs:
          - targets: ['localhost:9090']

Instrumenting Your Code
---------------------

You can instrument your own functions using the provided decorator:

.. code-block:: python

    from wdbx.prometheus import instrument
    
    # Instrument a function
    @instrument(name="my_custom_operation", kind="search")
    def my_search_function(query):
        # Function implementation
        pass
    
    # Instrument an async function
    @instrument(name="my_async_operation")
    async def my_async_function(data):
        # Async function implementation
        pass

HTTP Server Integration
---------------------

If you're using AIOHTTP for your HTTP server, you can use the provided middleware:

.. code-block:: python

    from aiohttp import web
    from wdbx.prometheus import prometheus_middleware
    
    # Create your app
    app = web.Application(middlewares=[prometheus_middleware])

Monitoring System Resources
-------------------------

WDBX can automatically monitor system resources like memory usage:

.. code-block:: python

    from wdbx.prometheus import get_metrics
    
    # Get the metrics instance
    metrics = get_metrics()
    
    # Collect memory usage
    metrics.collect_memory_usage(wdbx_instance)
    
    # Collect all stats (memory, vectors, blocks)
    metrics.collect_all_stats(wdbx_instance)

Example Dashboard
---------------

WDBX includes example Grafana dashboards in the ``monitoring/grafana`` directory:

* ``wdbx-overview.json``: General overview of WDBX metrics
* ``wdbx-performance.json``: Detailed performance metrics
* ``wdbx-api.json``: API usage and latency metrics

Best Practices
=============

1. **Regular Health Checks**: Implement regular health checks in your infrastructure monitoring
2. **Alerting**: Set up alerts for warning and error conditions
3. **Dashboards**: Create dashboards to visualize metrics over time
4. **Log Correlation**: Correlate metrics with logs for better troubleshooting
5. **Resource Limits**: Set appropriate resource limits based on monitored usage
6. **Performance Baselines**: Establish performance baselines to detect degradation

Troubleshooting
==============

Common Issues
-----------

* **High Memory Usage**: Check vector count, dimension size, and consider memory optimization
* **Slow Search Performance**: Monitor search latency metrics, consider index optimizations
* **Disk Space Warnings**: Implement data lifecycle policies to manage storage growth

Diagnostic Steps
--------------

1. Check the health report for specific errors
2. Review metrics for anomalies in performance or resource usage
3. Correlate issues with recent changes or load patterns
4. Examine logs for error messages

For more information, refer to the :doc:`troubleshooting` section. 