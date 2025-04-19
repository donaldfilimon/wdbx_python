"""
Vector Database Comparison Tool.

This module provides tools for comparing WDBX with other vector databases
like FAISS, Qdrant, and Milvus on metrics such as query speed, memory usage,
and result quality.
"""

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..utils.logging_utils import get_logger

# Initialize logger
logger = get_logger("wdbx.benchmarks.compare")

# Try to import vector database libraries
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, FAISS benchmark will be skipped")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("Qdrant not available, Qdrant benchmark will be skipped")

try:
    from pymilvus import Collection, connections, utility

    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    logger.warning("Milvus not available, Milvus benchmark will be skipped")


@dataclass
class ComparisonConfig:
    """Configuration for database comparison."""

    # Test data parameters
    vector_dim: int = 1536
    num_vectors: int = 100000
    query_count: int = 100

    # Batch sizes for insertion and query tests
    batch_sizes: List[int] = field(default_factory=lambda: [1, 10, 100, 1000])

    # Top-k values for query tests
    top_k_values: List[int] = field(default_factory=lambda: [1, 10, 50, 100])

    # Databases to benchmark
    databases: List[str] = field(default_factory=lambda: ["wdbx", "faiss", "qdrant", "milvus"])

    # Test iterations
    iterations: int = 3

    # Output directory
    output_dir: str = "comparison_results"

    # WDBX-specific settings
    wdbx_data_dir: str = "wdbx_compare_data"

    # Qdrant-specific settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "wdbx_compare"

    # Milvus-specific settings
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "wdbx_compare"


@dataclass
class BenchmarkResult:
    """Results from a single benchmark test."""

    database: str
    operation: str
    batch_size: int
    top_k: Optional[int] = None

    latency_ms: float = 0.0
    throughput: float = 0.0
    memory_mb: float = 0.0


class DatabaseBenchmarker:
    """Base class for database-specific benchmarkers."""

    def __init__(self, config: ComparisonConfig):
        """
        Initialize the benchmarker.

        Args:
            config: Comparison configuration
        """
        self.config = config
        self.name = "base"

    async def setup(self) -> None:
        """Set up the database for benchmarking."""

    async def teardown(self) -> None:
        """Clean up the database after benchmarking."""

    async def insert_vectors(
        self, vectors: NDArray[np.float32], batch_size: int
    ) -> BenchmarkResult:
        """
        Benchmark vector insertion.

        Args:
            vectors: Vectors to insert
            batch_size: Size of each batch

        Returns:
            Benchmark result
        """
        raise NotImplementedError("Subclasses must implement insert_vectors")

    async def query_vectors(
        self, query_vectors: NDArray[np.float32], top_k: int, batch_size: int
    ) -> BenchmarkResult:
        """
        Benchmark vector queries.

        Args:
            query_vectors: Vectors to query
            top_k: Number of results to return
            batch_size: Size of each batch

        Returns:
            Benchmark result
        """
        raise NotImplementedError("Subclasses must implement query_vectors")

    def get_memory_usage(self) -> float:
        """
        Get the current memory usage in MB.

        Returns:
            Memory usage in MB
        """
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)
        except ImportError:
            logger.warning("psutil not available, memory usage will not be tracked")
            return 0.0


class WDBXBenchmarker(DatabaseBenchmarker):
    """WDBX-specific benchmarker implementation."""

    def __init__(self, config: ComparisonConfig):
        """Initialize WDBX benchmarker."""
        super().__init__(config)
        self.name = "wdbx"
        self.wdbx = None

    async def setup(self) -> None:
        """Set up WDBX for benchmarking."""
        from ..core.wdbx_core import WDBXCore

        # Create data directory
        os.makedirs(self.config.wdbx_data_dir, exist_ok=True)

        # Initialize WDBX
        self.wdbx = WDBXCore(
            data_dir=self.config.wdbx_data_dir,
            vector_dimension=self.config.vector_dim,
            enable_memory_optimization=True,
        )

        logger.info("WDBX initialized for benchmarking")

    async def teardown(self) -> None:
        """Clean up WDBX after benchmarking."""
        if self.wdbx:
            self.wdbx.clear()
            logger.info("WDBX data cleared")

    async def insert_vectors(
        self, vectors: NDArray[np.float32], batch_size: int
    ) -> BenchmarkResult:
        """Benchmark vector insertion in WDBX."""
        if not self.wdbx:
            raise RuntimeError("WDBX not initialized")

        num_vectors = len(vectors)
        start_memory = self.get_memory_usage()
        start_time = time.time()

        for i in range(0, num_vectors, batch_size):
            batch = vectors[i: i + batch_size]
            for vector in batch:
                self.wdbx.create_vector(vector_data=vector)

        elapsed_time = time.time() - start_time
        memory_used = self.get_memory_usage() - start_memory

        # Calculate metrics
        latency_ms = (elapsed_time / (num_vectors / batch_size)) * 1000
        throughput = num_vectors / elapsed_time if elapsed_time > 0 else 0

        return BenchmarkResult(
            database=self.name,
            operation="insert",
            batch_size=batch_size,
            latency_ms=latency_ms,
            throughput=throughput,
            memory_mb=memory_used,
        )

    async def query_vectors(
        self, query_vectors: NDArray[np.float32], top_k: int, batch_size: int
    ) -> BenchmarkResult:
        """Benchmark vector queries in WDBX."""
        if not self.wdbx:
            raise RuntimeError("WDBX not initialized")

        num_queries = len(query_vectors)
        start_memory = self.get_memory_usage()
        start_time = time.time()

        for i in range(0, num_queries, batch_size):
            batch = query_vectors[i: i + batch_size]
            for query in batch:
                self.wdbx.find_similar_vectors(query_vector=query, top_k=top_k)

        elapsed_time = time.time() - start_time
        memory_used = self.get_memory_usage() - start_memory

        # Calculate metrics
        latency_ms = (elapsed_time / (num_queries / batch_size)) * 1000
        throughput = num_queries / elapsed_time if elapsed_time > 0 else 0

        return BenchmarkResult(
            database=self.name,
            operation="query",
            batch_size=batch_size,
            top_k=top_k,
            latency_ms=latency_ms,
            throughput=throughput,
            memory_mb=memory_used,
        )


class FAISSBenchmarker(DatabaseBenchmarker):
    """FAISS-specific benchmarker implementation."""

    def __init__(self, config: ComparisonConfig):
        """Initialize FAISS benchmarker."""
        super().__init__(config)
        self.name = "faiss"
        self.index = None

    async def setup(self) -> None:
        """Set up FAISS for benchmarking."""
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS is not available")

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.config.vector_dim)
        logger.info("FAISS index initialized for benchmarking")

    async def teardown(self) -> None:
        """Clean up FAISS after benchmarking."""
        self.index = None
        logger.info("FAISS index cleared")

    async def insert_vectors(
        self, vectors: NDArray[np.float32], batch_size: int
    ) -> BenchmarkResult:
        """Benchmark vector insertion in FAISS."""
        if self.index is None:
            raise RuntimeError("FAISS index not initialized")

        vectors = vectors.astype(np.float32)
        num_vectors = len(vectors)
        start_memory = self.get_memory_usage()
        start_time = time.time()

        for i in range(0, num_vectors, batch_size):
            batch = vectors[i: i + batch_size]
            self.index.add(batch)

        elapsed_time = time.time() - start_time
        memory_used = self.get_memory_usage() - start_memory

        # Calculate metrics
        latency_ms = (elapsed_time / (num_vectors / batch_size)) * 1000
        throughput = num_vectors / elapsed_time if elapsed_time > 0 else 0

        return BenchmarkResult(
            database=self.name,
            operation="insert",
            batch_size=batch_size,
            latency_ms=latency_ms,
            throughput=throughput,
            memory_mb=memory_used,
        )

    async def query_vectors(
        self, query_vectors: NDArray[np.float32], top_k: int, batch_size: int
    ) -> BenchmarkResult:
        """Benchmark vector queries in FAISS."""
        if self.index is None:
            raise RuntimeError("FAISS index not initialized")

        query_vectors = query_vectors.astype(np.float32)
        num_queries = len(query_vectors)
        start_memory = self.get_memory_usage()
        start_time = time.time()

        for i in range(0, num_queries, batch_size):
            batch = query_vectors[i: i + batch_size]
            _, _ = self.index.search(batch, top_k)

        elapsed_time = time.time() - start_time
        memory_used = self.get_memory_usage() - start_memory

        # Calculate metrics
        latency_ms = (elapsed_time / (num_queries / batch_size)) * 1000
        throughput = num_queries / elapsed_time if elapsed_time > 0 else 0

        return BenchmarkResult(
            database=self.name,
            operation="query",
            batch_size=batch_size,
            top_k=top_k,
            latency_ms=latency_ms,
            throughput=throughput,
            memory_mb=memory_used,
        )


class QdrantBenchmarker(DatabaseBenchmarker):
    """Qdrant-specific benchmarker implementation."""

    def __init__(self, config: ComparisonConfig):
        """Initialize Qdrant benchmarker."""
        super().__init__(config)
        self.name = "qdrant"
        self.client = None

    async def setup(self) -> None:
        """Set up Qdrant for benchmarking."""
        if not QDRANT_AVAILABLE:
            raise RuntimeError("Qdrant is not available")

        # Initialize Qdrant client
        self.client = QdrantClient(host=self.config.qdrant_host, port=self.config.qdrant_port)

        # Check if collection exists and recreate it
        try:
            self.client.delete_collection(self.config.qdrant_collection)
        except Exception:
            pass

        # Create collection
        self.client.create_collection(
            collection_name=self.config.qdrant_collection,
            vectors_config=models.VectorParams(
                size=self.config.vector_dim, distance=models.Distance.COSINE
            ),
        )

        logger.info("Qdrant collection initialized for benchmarking")

    async def teardown(self) -> None:
        """Clean up Qdrant after benchmarking."""
        if self.client:
            try:
                self.client.delete_collection(self.config.qdrant_collection)
                logger.info("Qdrant collection deleted")
            except Exception as e:
                logger.error(f"Error deleting Qdrant collection: {str(e)}")

    async def insert_vectors(
        self, vectors: NDArray[np.float32], batch_size: int
    ) -> BenchmarkResult:
        """Benchmark vector insertion in Qdrant."""
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")

        num_vectors = len(vectors)
        start_memory = self.get_memory_usage()
        start_time = time.time()

        for i in range(0, num_vectors, batch_size):
            batch = vectors[i: i + batch_size]
            points = [
                models.PointStruct(id=i + j, vector=batch[j].tolist(), payload={"index": i + j})
                for j in range(len(batch))
            ]
            self.client.upsert(collection_name=self.config.qdrant_collection, points=points)

        elapsed_time = time.time() - start_time
        memory_used = self.get_memory_usage() - start_memory

        # Calculate metrics
        latency_ms = (elapsed_time / (num_vectors / batch_size)) * 1000
        throughput = num_vectors / elapsed_time if elapsed_time > 0 else 0

        return BenchmarkResult(
            database=self.name,
            operation="insert",
            batch_size=batch_size,
            latency_ms=latency_ms,
            throughput=throughput,
            memory_mb=memory_used,
        )

    async def query_vectors(
        self, query_vectors: NDArray[np.float32], top_k: int, batch_size: int
    ) -> BenchmarkResult:
        """Benchmark vector queries in Qdrant."""
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")

        num_queries = len(query_vectors)
        start_memory = self.get_memory_usage()
        start_time = time.time()

        for i in range(0, num_queries, batch_size):
            batch = query_vectors[i: i + batch_size]
            for query in batch:
                self.client.search(
                    collection_name=self.config.qdrant_collection,
                    query_vector=query.tolist(),
                    limit=top_k,
                )

        elapsed_time = time.time() - start_time
        memory_used = self.get_memory_usage() - start_memory

        # Calculate metrics
        latency_ms = (elapsed_time / (num_queries / batch_size)) * 1000
        throughput = num_queries / elapsed_time if elapsed_time > 0 else 0

        return BenchmarkResult(
            database=self.name,
            operation="query",
            batch_size=batch_size,
            top_k=top_k,
            latency_ms=latency_ms,
            throughput=throughput,
            memory_mb=memory_used,
        )


class MilvusBenchmarker(DatabaseBenchmarker):
    """Milvus-specific benchmarker implementation."""

    def __init__(self, config: ComparisonConfig):
        """Initialize Milvus benchmarker."""
        super().__init__(config)
        self.name = "milvus"
        self.collection = None

    async def setup(self) -> None:
        """Set up Milvus for benchmarking."""
        if not MILVUS_AVAILABLE:
            raise RuntimeError("Milvus is not available")

        # Connect to Milvus
        connections.connect(host=self.config.milvus_host, port=self.config.milvus_port)

        # Drop collection if it exists
        if utility.has_collection(self.config.milvus_collection):
            utility.drop_collection(self.config.milvus_collection)

        # Create collection schema
        from pymilvus import CollectionSchema, DataType, FieldSchema

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.config.vector_dim),
        ]
        schema = CollectionSchema(fields=fields)

        # Create collection
        self.collection = Collection(name=self.config.milvus_collection, schema=schema)

        # Create index
        self.collection.create_index(
            field_name="vector", index_params={"metric_type": "COSINE", "index_type": "FLAT"}
        )

        # Load collection
        self.collection.load()

        logger.info("Milvus collection initialized for benchmarking")

    async def teardown(self) -> None:
        """Clean up Milvus after benchmarking."""
        if utility.has_collection(self.config.milvus_collection):
            utility.drop_collection(self.config.milvus_collection)
            logger.info("Milvus collection deleted")

        connections.disconnect()

    async def insert_vectors(
        self, vectors: NDArray[np.float32], batch_size: int
    ) -> BenchmarkResult:
        """Benchmark vector insertion in Milvus."""
        if not self.collection:
            raise RuntimeError("Milvus collection not initialized")

        num_vectors = len(vectors)
        start_memory = self.get_memory_usage()
        start_time = time.time()

        for i in range(0, num_vectors, batch_size):
            end_idx = min(i + batch_size, num_vectors)
            batch = vectors[i:end_idx]
            entities = [[j for j in range(i, end_idx)], batch.tolist()]  # id  # vector
            self.collection.insert(entities)

        # Flush the collection to ensure data is persisted
        self.collection.flush()

        elapsed_time = time.time() - start_time
        memory_used = self.get_memory_usage() - start_memory

        # Calculate metrics
        latency_ms = (elapsed_time / (num_vectors / batch_size)) * 1000
        throughput = num_vectors / elapsed_time if elapsed_time > 0 else 0

        return BenchmarkResult(
            database=self.name,
            operation="insert",
            batch_size=batch_size,
            latency_ms=latency_ms,
            throughput=throughput,
            memory_mb=memory_used,
        )

    async def query_vectors(
        self, query_vectors: NDArray[np.float32], top_k: int, batch_size: int
    ) -> BenchmarkResult:
        """Benchmark vector queries in Milvus."""
        if not self.collection:
            raise RuntimeError("Milvus collection not initialized")

        num_queries = len(query_vectors)
        start_memory = self.get_memory_usage()
        start_time = time.time()

        for i in range(0, num_queries, batch_size):
            batch = query_vectors[i: i + batch_size]
            for query in batch:
                search_params = {"metric_type": "COSINE"}
                self.collection.search(
                    data=[query.tolist()], anns_field="vector", param=search_params, limit=top_k
                )

        elapsed_time = time.time() - start_time
        memory_used = self.get_memory_usage() - start_memory

        # Calculate metrics
        latency_ms = (elapsed_time / (num_queries / batch_size)) * 1000
        throughput = num_queries / elapsed_time if elapsed_time > 0 else 0

        return BenchmarkResult(
            database=self.name,
            operation="query",
            batch_size=batch_size,
            top_k=top_k,
            latency_ms=latency_ms,
            throughput=throughput,
            memory_mb=memory_used,
        )


class ComparisonRunner:
    """Run comparison benchmarks across multiple vector databases."""

    def __init__(self, config: ComparisonConfig):
        """Initialize the comparison runner."""
        self.config = config
        self.results: List[BenchmarkResult] = []
        self.benchmarkers: Dict[str, DatabaseBenchmarker] = {}

    async def setup(self) -> None:
        """Set up all benchmarkers."""
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Initialize benchmarkers
        if "wdbx" in self.config.databases:
            self.benchmarkers["wdbx"] = WDBXBenchmarker(self.config)

        if "faiss" in self.config.databases and FAISS_AVAILABLE:
            self.benchmarkers["faiss"] = FAISSBenchmarker(self.config)

        if "qdrant" in self.config.databases and QDRANT_AVAILABLE:
            self.benchmarkers["qdrant"] = QdrantBenchmarker(self.config)

        if "milvus" in self.config.databases and MILVUS_AVAILABLE:
            self.benchmarkers["milvus"] = MilvusBenchmarker(self.config)

        # Set up each benchmarker
        for name, benchmarker in self.benchmarkers.items():
            try:
                logger.info(f"Setting up {name} benchmarker")
                await benchmarker.setup()
            except Exception as e:
                logger.error(f"Error setting up {name} benchmarker: {str(e)}")
                del self.benchmarkers[name]

    async def teardown(self) -> None:
        """Clean up all benchmarkers."""
        for name, benchmarker in self.benchmarkers.items():
            try:
                logger.info(f"Tearing down {name} benchmarker")
                await benchmarker.teardown()
            except Exception as e:
                logger.error(f"Error tearing down {name} benchmarker: {str(e)}")

    def generate_vectors(self, count: int) -> NDArray[np.float32]:
        """
        Generate random vectors for testing.

        Args:
            count: Number of vectors to generate

        Returns:
            Array of random vectors
        """
        logger.info(f"Generating {count} random vectors")
        return np.random.rand(count, self.config.vector_dim).astype(np.float32)

    async def run_insert_benchmarks(self) -> None:
        """Run insertion benchmarks for all databases."""
        for batch_size in self.config.batch_sizes:
            logger.info(f"Running insertion benchmarks with batch size {batch_size}")

            # Generate vectors for this batch size
            vectors = self.generate_vectors(batch_size * 10)  # 10 batches

            for name, benchmarker in self.benchmarkers.items():
                # Run multiple iterations
                for i in range(self.config.iterations):
                    try:
                        # Set up benchmarker for this iteration
                        await benchmarker.setup()

                        # Run benchmark
                        logger.info(
                            f"Running {name} insertion benchmark (iteration {i+1}/{self.config.iterations})"
                        )
                        result = await benchmarker.insert_vectors(vectors, batch_size)
                        self.results.append(result)

                        # Tear down benchmarker
                        await benchmarker.teardown()
                    except Exception as e:
                        logger.error(f"Error running {name} insertion benchmark: {str(e)}")

    async def run_query_benchmarks(self) -> None:
        """Run query benchmarks for all databases."""
        for db_name, benchmarker in self.benchmarkers.items():
            # Set up database with data for querying
            await benchmarker.setup()

            # Insert a significant number of vectors first
            vectors = self.generate_vectors(10000)
            try:
                logger.info(f"Inserting vectors into {db_name} for query benchmarks")
                await benchmarker.insert_vectors(vectors, 1000)
            except Exception as e:
                logger.error(f"Error inserting vectors into {db_name}: {str(e)}")
                await benchmarker.teardown()
                continue

            # Generate query vectors
            query_vectors = self.generate_vectors(self.config.query_count)

            # Run benchmarks for each batch size and top-k combination
            for batch_size in self.config.batch_sizes:
                for top_k in self.config.top_k_values:
                    logger.info(
                        f"Running {db_name} query benchmark with batch size {batch_size}, top-k {top_k}"
                    )

                    # Run multiple iterations
                    for i in range(self.config.iterations):
                        try:
                            logger.info(
                                f"Running {db_name} query benchmark (iteration {i+1}/{self.config.iterations})"
                            )
                            result = await benchmarker.query_vectors(
                                query_vectors, top_k, batch_size
                            )
                            self.results.append(result)
                        except Exception as e:
                            logger.error(f"Error running {db_name} query benchmark: {str(e)}")

            # Clean up
            await benchmarker.teardown()

    async def run_all_benchmarks(self) -> None:
        """Run all benchmarks."""
        await self.setup()
        await self.run_insert_benchmarks()
        await self.run_query_benchmarks()
        await self.teardown()
        self.save_results()
        self.generate_charts()

    def save_results(self) -> None:
        """Save benchmark results to a file."""
        result_file = os.path.join(self.config.output_dir, "benchmark_results.json")

        # Convert results to dictionaries
        result_dicts = []
        for result in self.results:
            result_dict = {
                "database": result.database,
                "operation": result.operation,
                "batch_size": result.batch_size,
                "top_k": result.top_k,
                "latency_ms": result.latency_ms,
                "throughput": result.throughput,
                "memory_mb": result.memory_mb,
            }
            result_dicts.append(result_dict)

        # Save results
        with open(result_file, "w") as f:
            json.dump(result_dicts, f, indent=2)

        logger.info(f"Benchmark results saved to {result_file}")

        # Save as CSV as well
        csv_file = os.path.join(self.config.output_dir, "benchmark_results.csv")
        df = pd.DataFrame(result_dicts)
        df.to_csv(csv_file, index=False)
        logger.info(f"Benchmark results saved to {csv_file}")

    def generate_charts(self) -> None:
        """Generate charts from benchmark results."""
        # Make sure we have results
        if not self.results:
            logger.warning("No benchmark results to generate charts from")
            return

        # Convert results to a DataFrame
        df = pd.DataFrame(
            [
                {
                    "database": r.database,
                    "operation": r.operation,
                    "batch_size": r.batch_size,
                    "top_k": r.top_k,
                    "latency_ms": r.latency_ms,
                    "throughput": r.throughput,
                    "memory_mb": r.memory_mb,
                }
                for r in self.results
            ]
        )

        # Generate latency comparison charts
        self._generate_operation_charts(df, "latency_ms", "Latency (ms)")

        # Generate throughput comparison charts
        self._generate_operation_charts(df, "throughput", "Throughput (ops/sec)")

        # Generate memory usage comparison charts
        self._generate_operation_charts(df, "memory_mb", "Memory Usage (MB)")

        logger.info(f"Charts saved to {self.config.output_dir}")

    def _generate_operation_charts(self, df: pd.DataFrame, metric: str, metric_label: str) -> None:
        """
        Generate charts for a specific operation and metric.

        Args:
            df: DataFrame with benchmark results
            metric: Metric to plot
            metric_label: Label for the metric
        """
        # Create charts directory
        charts_dir = os.path.join(self.config.output_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)

        # Generate insert operation charts
        insert_df = df[df["operation"] == "insert"]
        if not insert_df.empty:
            plt.figure(figsize=(10, 6))

            for db_name in insert_df["database"].unique():
                db_data = insert_df[insert_df["database"] == db_name]
                plt.plot(db_data["batch_size"], db_data[metric], marker="o", label=db_name)

            plt.xscale("log")
            plt.xlabel("Batch Size")
            plt.ylabel(metric_label)
            plt.title(f"Insert Operation {metric_label} by Batch Size")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            chart_file = os.path.join(charts_dir, f"insert_{metric}.png")
            plt.savefig(chart_file)
            plt.close()

        # Generate query operation charts for each top-k value
        query_df = df[df["operation"] == "query"]
        if not query_df.empty:
            for top_k in query_df["top_k"].unique():
                top_k_df = query_df[query_df["top_k"] == top_k]

                plt.figure(figsize=(10, 6))

                for db_name in top_k_df["database"].unique():
                    db_data = top_k_df[top_k_df["database"] == db_name]
                    plt.plot(db_data["batch_size"], db_data[metric], marker="o", label=db_name)

                plt.xscale("log")
                plt.xlabel("Batch Size")
                plt.ylabel(metric_label)
                plt.title(f"Query Operation {metric_label} by Batch Size (top-k={top_k})")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                chart_file = os.path.join(charts_dir, f"query_top{top_k}_{metric}.png")
                plt.savefig(chart_file)
                plt.close()

        # Generate bar charts comparing databases
        plt.figure(figsize=(12, 6))

        # Group by database and operation, and calculate mean of the metric
        grouped = df.groupby(["database", "operation"])[metric].mean().unstack()

        if "insert" in grouped.columns:
            insert_values = grouped["insert"]
            plt.bar(np.arange(len(insert_values)) - 0.2, insert_values, width=0.4, label="Insert")

        if "query" in grouped.columns:
            query_values = grouped["query"]
            plt.bar(np.arange(len(query_values)) + 0.2, query_values, width=0.4, label="Query")

        plt.xlabel("Database")
        plt.ylabel(metric_label)
        plt.title(f"Average {metric_label} by Database and Operation")
        plt.xticks(range(len(grouped.index)), grouped.index)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        chart_file = os.path.join(charts_dir, f"comparison_{metric}.png")
        plt.savefig(chart_file)
        plt.close()


async def main() -> None:
    """Main entry point for the comparison tool."""
    parser = argparse.ArgumentParser(description="Vector Database Comparison Tool")

    parser.add_argument("--vector-dim", type=int, default=1536, help="Vector dimension")
    parser.add_argument(
        "--num-vectors", type=int, default=100000, help="Number of vectors to benchmark with"
    )
    parser.add_argument("--query-count", type=int, default=100, help="Number of queries to run")
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,10,100,1000",
        help="Comma-separated list of batch sizes",
    )
    parser.add_argument(
        "--top-k-values",
        type=str,
        default="1,10,50,100",
        help="Comma-separated list of top-k values",
    )
    parser.add_argument(
        "--databases",
        type=str,
        default="wdbx,faiss,qdrant,milvus",
        help="Comma-separated list of databases to benchmark",
    )
    parser.add_argument(
        "--iterations", type=int, default=3, help="Number of iterations for each benchmark"
    )
    parser.add_argument(
        "--output-dir", type=str, default="comparison_results", help="Output directory for results"
    )

    args = parser.parse_args()

    # Parse lists
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    top_k_values = [int(x) for x in args.top_k_values.split(",")]
    databases = [x.strip() for x in args.databases.split(",")]

    # Create configuration
    config = ComparisonConfig(
        vector_dim=args.vector_dim,
        num_vectors=args.num_vectors,
        query_count=args.query_count,
        batch_sizes=batch_sizes,
        top_k_values=top_k_values,
        databases=databases,
        iterations=args.iterations,
        output_dir=args.output_dir,
    )

    # Run benchmarks
    runner = ComparisonRunner(config)
    await runner.run_all_benchmarks()


if __name__ == "__main__":
    asyncio.run(main())
