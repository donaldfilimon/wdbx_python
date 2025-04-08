"""
WDBX Benchmarking Tool.

This module provides tools for benchmarking WDBX against other vector databases,
measuring performance metrics such as query speed, memory usage, and accuracy.
"""

import argparse
import asyncio
import csv
import datetime
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ..core.wdbx_core import WDBXCore
from ..utils.logging_utils import LogContext, get_logger

# Initialize logger
logger = get_logger("wdbx.benchmarks")


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests."""

    # Data generation settings
    num_vectors: int = 10000
    vector_dimension: int = 1536
    batch_sizes: List[int] = field(default_factory=lambda: [1, 10, 100, 1000])

    # Benchmark settings
    num_iterations: int = 5
    num_queries: int = 100
    top_k_values: List[int] = field(default_factory=lambda: [1, 10, 50, 100])

    # Vector database options
    benchmarks: List[str] = field(default_factory=lambda: ["wdbx", "faiss", "qdrant", "milvus"])

    # Output settings
    output_dir: str = "benchmark_results"
    generate_charts: bool = True
    save_raw_data: bool = True


@dataclass
class BenchmarkResult:
    """Results of a benchmark test."""

    database: str
    operation: str
    batch_size: int
    top_k: Optional[int] = None

    times: List[float] = field(default_factory=list)
    memory_usages: List[float] = field(default_factory=list)

    @property
    def avg_time(self) -> float:
        """Average operation time in milliseconds."""
        return statistics.mean(self.times) * 1000 if self.times else 0

    @property
    def median_time(self) -> float:
        """Median operation time in milliseconds."""
        return statistics.median(self.times) * 1000 if self.times else 0

    @property
    def stddev_time(self) -> float:
        """Standard deviation of operation time in milliseconds."""
        return statistics.stdev(self.times) * 1000 if len(self.times) > 1 else 0

    @property
    def avg_memory(self) -> float:
        """Average memory usage in MB."""
        return statistics.mean(self.memory_usages) if self.memory_usages else 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "database": self.database,
            "operation": self.operation,
            "batch_size": self.batch_size,
            "top_k": self.top_k,
            "avg_time_ms": self.avg_time,
            "median_time_ms": self.median_time,
            "stddev_time_ms": self.stddev_time,
            "avg_memory_mb": self.avg_memory,
            "num_samples": len(self.times),
        }


class VectorDatabaseBenchmark:
    """
    Base class for vector database benchmarks.

    This class defines the interface that all database-specific benchmark
    implementations must follow.
    """

    def __init__(self, config: BenchmarkConfig):
        """
        Initialize the benchmark.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.name = "base"
        self.results: List[BenchmarkResult] = []

    async def setup(self) -> None:
        """Set up the benchmark environment."""

    async def teardown(self) -> None:
        """Clean up the benchmark environment."""

    async def benchmark_insert(self, vectors: List[NDArray[np.float32]]) -> BenchmarkResult:
        """
        Benchmark vector insertion.

        Args:
            vectors: List of vectors to insert

        Returns:
            Benchmark result
        """
        raise NotImplementedError("Subclasses must implement benchmark_insert")

    async def benchmark_query(
        self, query_vectors: List[NDArray[np.float32]], top_k: int
    ) -> BenchmarkResult:
        """
        Benchmark vector querying.

        Args:
            query_vectors: List of vectors to query
            top_k: Number of results to return

        Returns:
            Benchmark result
        """
        raise NotImplementedError("Subclasses must implement benchmark_query")

    async def benchmark_update(
        self, vector_ids: List[str], new_vectors: List[NDArray[np.float32]]
    ) -> BenchmarkResult:
        """
        Benchmark vector updates.

        Args:
            vector_ids: IDs of vectors to update
            new_vectors: New vector values

        Returns:
            Benchmark result
        """
        raise NotImplementedError("Subclasses must implement benchmark_update")

    async def benchmark_delete(self, vector_ids: List[str]) -> BenchmarkResult:
        """
        Benchmark vector deletion.

        Args:
            vector_ids: IDs of vectors to delete

        Returns:
            Benchmark result
        """
        raise NotImplementedError("Subclasses must implement benchmark_delete")


class WDBXBenchmark(VectorDatabaseBenchmark):
    """Benchmark implementation for WDBX."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize WDBX benchmark."""
        super().__init__(config)
        self.name = "wdbx"
        self.wdbx: Optional[WDBXCore] = None
        self.vector_ids: List[str] = []

    async def setup(self) -> None:
        """Set up WDBX benchmark environment."""
        # Create a temporary directory for the benchmark
        os.makedirs("benchmark_data/wdbx", exist_ok=True)

        # Initialize WDBX with benchmark settings
        self.wdbx = WDBXCore(
            data_dir="benchmark_data/wdbx",
            vector_dimension=self.config.vector_dimension,
            enable_memory_optimization=True,
        )

        logger.info(f"Initialized WDBX with dimension {self.config.vector_dimension}")

    async def teardown(self) -> None:
        """Clean up WDBX benchmark environment."""
        if self.wdbx:
            self.wdbx.shutdown()
            logger.info("WDBX shutdown complete")

    async def benchmark_insert(self, vectors: List[NDArray[np.float32]]) -> BenchmarkResult:
        """Benchmark vector insertion in WDBX."""
        assert self.wdbx is not None, "WDBX must be initialized before benchmarking"

        result = BenchmarkResult(database=self.name, operation="insert", batch_size=len(vectors))

        # Run the benchmark
        for _ in range(self.config.num_iterations):
            # Clean up before each iteration
            self.wdbx.clear()
            self.vector_ids = []

            # Measure insertion time
            memory_before = self.wdbx.get_stats()["memory_peak_mb"]
            start_time = time.time()

            # Insert vectors
            for vector in vectors:
                embedding = self.wdbx.create_vector(vector_data=vector)
                self.vector_ids.append(embedding.vector_id)

            # Record results
            elapsed_time = time.time() - start_time
            memory_after = self.wdbx.get_stats()["memory_peak_mb"]
            memory_usage = memory_after - memory_before

            result.times.append(elapsed_time)
            result.memory_usages.append(memory_usage)

            # Add a short delay between iterations
            await asyncio.sleep(0.1)

        logger.info(
            f"WDBX insert benchmark: {result.avg_time:.2f}ms per batch of {len(vectors)} vectors"
        )
        self.results.append(result)
        return result

    async def benchmark_query(
        self, query_vectors: List[NDArray[np.float32]], top_k: int
    ) -> BenchmarkResult:
        """Benchmark vector querying in WDBX."""
        assert self.wdbx is not None, "WDBX must be initialized before benchmarking"

        result = BenchmarkResult(
            database=self.name, operation="query", batch_size=len(query_vectors), top_k=top_k
        )

        # Ensure we have vectors to query against
        if not self.vector_ids:
            # Insert some vectors if none exist
            for _ in range(max(self.config.num_vectors - len(self.vector_ids), 0)):
                vector = np.random.rand(self.config.vector_dimension).astype(np.float32)
                embedding = self.wdbx.create_vector(vector_data=vector)
                self.vector_ids.append(embedding.vector_id)

        # Run the benchmark
        for _ in range(self.config.num_iterations):
            memory_before = self.wdbx.get_stats()["memory_peak_mb"]
            start_time = time.time()

            # Query vectors
            for query_vector in query_vectors:
                self.wdbx.find_similar_vectors(query_vector=query_vector, top_k=top_k)

            # Record results
            elapsed_time = time.time() - start_time
            memory_after = self.wdbx.get_stats()["memory_peak_mb"]
            memory_usage = memory_after - memory_before

            result.times.append(elapsed_time)
            result.memory_usages.append(memory_usage)

            # Add a short delay between iterations
            await asyncio.sleep(0.1)

        logger.info(
            f"WDBX query benchmark: {result.avg_time:.2f}ms per batch of {len(query_vectors)} "
            f"queries with top_k={top_k}"
        )
        self.results.append(result)
        return result


class FAISSBenchmark(VectorDatabaseBenchmark):
    """Benchmark implementation for FAISS."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize FAISS benchmark."""
        super().__init__(config)
        self.name = "faiss"
        self.faiss_index = None
        self.vector_ids: List[str] = []

    async def setup(self) -> None:
        """Set up FAISS benchmark environment."""
        try:
            import faiss

            self.faiss_index = faiss.IndexFlatL2(self.config.vector_dimension)
            logger.info(f"Initialized FAISS with dimension {self.config.vector_dimension}")
        except ImportError:
            logger.error("FAISS not available, skipping benchmark")
            raise

    async def teardown(self) -> None:
        """Clean up FAISS benchmark environment."""
        self.faiss_index = None

    async def benchmark_insert(self, vectors: List[NDArray[np.float32]]) -> BenchmarkResult:
        """Benchmark vector insertion in FAISS."""
        import faiss

        result = BenchmarkResult(database=self.name, operation="insert", batch_size=len(vectors))

        # Run the benchmark
        for _ in range(self.config.num_iterations):
            # Reset the index
            self.faiss_index = faiss.IndexFlatL2(self.config.vector_dimension)

            # Convert vectors to numpy array
            vectors_np = np.stack(vectors).astype(np.float32)

            # Measure insertion time
            start_time = time.time()

            # Insert vectors
            self.faiss_index.add(vectors_np)

            # Record results
            elapsed_time = time.time() - start_time

            result.times.append(elapsed_time)
            # FAISS doesn't provide memory usage info, so we use 0
            result.memory_usages.append(0)

            # Add a short delay between iterations
            await asyncio.sleep(0.1)

        logger.info(
            f"FAISS insert benchmark: {result.avg_time:.2f}ms per batch of {len(vectors)} vectors"
        )
        self.results.append(result)
        return result

    async def benchmark_query(
        self, query_vectors: List[NDArray[np.float32]], top_k: int
    ) -> BenchmarkResult:
        """Benchmark vector querying in FAISS."""

        result = BenchmarkResult(
            database=self.name, operation="query", batch_size=len(query_vectors), top_k=top_k
        )

        # Ensure we have vectors to query against
        if self.faiss_index.ntotal < self.config.num_vectors:
            # Insert random vectors if needed
            random_vectors = np.random.rand(
                self.config.num_vectors - self.faiss_index.ntotal, self.config.vector_dimension
            ).astype(np.float32)
            self.faiss_index.add(random_vectors)

        # Convert query vectors to numpy array
        query_np = np.stack(query_vectors).astype(np.float32)

        # Run the benchmark
        for _ in range(self.config.num_iterations):
            start_time = time.time()

            # Query vectors
            distances, indices = self.faiss_index.search(query_np, top_k)

            # Record results
            elapsed_time = time.time() - start_time

            result.times.append(elapsed_time)
            # FAISS doesn't provide memory usage info
            result.memory_usages.append(0)

            # Add a short delay between iterations
            await asyncio.sleep(0.1)

        logger.info(
            f"FAISS query benchmark: {result.avg_time:.2f}ms per batch of {len(query_vectors)} "
            f"queries with top_k={top_k}"
        )
        self.results.append(result)
        return result


class BenchmarkRunner:
    """Runs benchmarks for multiple vector databases and compares results."""

    def __init__(self, config: BenchmarkConfig):
        """
        Initialize the benchmark runner.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.benchmarks: Dict[str, VectorDatabaseBenchmark] = {}
        self.all_results: List[BenchmarkResult] = []

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize random seed for reproducibility
        np.random.seed(42)

    async def setup_benchmarks(self) -> None:
        """Set up all benchmark implementations."""
        # Initialize benchmark implementations
        if "wdbx" in self.config.benchmarks:
            try:
                self.benchmarks["wdbx"] = WDBXBenchmark(self.config)
                await self.benchmarks["wdbx"].setup()
            except Exception as e:
                logger.error(f"Failed to set up WDBX benchmark: {e}")

        if "faiss" in self.config.benchmarks:
            try:
                self.benchmarks["faiss"] = FAISSBenchmark(self.config)
                await self.benchmarks["faiss"].setup()
            except Exception as e:
                logger.error(f"Failed to set up FAISS benchmark: {e}")

        # Add other benchmark implementations as needed

        logger.info(f"Set up {len(self.benchmarks)} benchmarks: {list(self.benchmarks.keys())}")

    async def generate_test_data(
        self,
    ) -> Tuple[List[NDArray[np.float32]], List[NDArray[np.float32]]]:
        """
        Generate test data for benchmarks.

        Returns:
            Tuple of (vectors for insertion, vectors for querying)
        """
        # Generate vectors for insertion
        logger.info(f"Generating {self.config.num_vectors} test vectors...")
        test_vectors = [
            np.random.rand(self.config.vector_dimension).astype(np.float32)
            for _ in range(self.config.num_vectors)
        ]

        # Generate query vectors (a subset of the test vectors with some noise)
        logger.info(f"Generating {self.config.num_queries} query vectors...")
        query_indices = np.random.choice(
            len(test_vectors), size=self.config.num_queries, replace=False
        )
        query_vectors = []
        for idx in query_indices:
            # Add some noise to make the queries more realistic
            vector = test_vectors[idx].copy()
            noise = np.random.normal(0, 0.1, self.config.vector_dimension).astype(np.float32)
            vector += noise
            # Normalize
            vector = vector / np.linalg.norm(vector)
            query_vectors.append(vector)

        return test_vectors, query_vectors

    async def run_benchmark(self) -> None:
        """Run all benchmarks."""
        logger.info("Starting benchmark run...")

        # Set up benchmarks
        await self.setup_benchmarks()

        # Generate test data
        test_vectors, query_vectors = await self.generate_test_data()

        # Run insertion benchmarks for different batch sizes
        for batch_size in self.config.batch_sizes:
            if batch_size > len(test_vectors):
                continue

            logger.info(f"Running insertion benchmark with batch size {batch_size}...")
            for db_name, benchmark in self.benchmarks.items():
                try:
                    # Take a batch of vectors
                    vector_batch = test_vectors[:batch_size]
                    result = await benchmark.benchmark_insert(vector_batch)
                    self.all_results.append(result)
                except Exception as e:
                    logger.error(f"Error running insertion benchmark for {db_name}: {e}")

        # Run query benchmarks for different batch sizes and top_k values
        for batch_size in self.config.batch_sizes:
            if batch_size > len(query_vectors):
                continue

            for top_k in self.config.top_k_values:
                logger.info(
                    f"Running query benchmark with batch size {batch_size} and top_k {top_k}..."
                )
                for db_name, benchmark in self.benchmarks.items():
                    try:
                        # Take a batch of query vectors
                        query_batch = query_vectors[:batch_size]
                        result = await benchmark.benchmark_query(query_batch, top_k)
                        self.all_results.append(result)
                    except Exception as e:
                        logger.error(f"Error running query benchmark for {db_name}: {e}")

        # Clean up benchmarks
        for db_name, benchmark in self.benchmarks.items():
            try:
                await benchmark.teardown()
            except Exception as e:
                logger.error(f"Error tearing down benchmark for {db_name}: {e}")

        # Save and visualize results
        self.save_results()
        if self.config.generate_charts:
            self.generate_charts()

    def save_results(self) -> None:
        """Save benchmark results to files."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save results as JSON
        json_path = self.output_dir / f"benchmark_results_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(
                {
                    "config": {
                        k: v for k, v in self.config.__dict__.items() if not k.startswith("_")
                    },
                    "results": [r.to_dict() for r in self.all_results],
                },
                f,
                indent=2,
            )
        logger.info(f"Saved benchmark results to {json_path}")

        # Save results as CSV
        csv_path = self.output_dir / f"benchmark_results_{timestamp}.csv"
        with open(csv_path, "w", newline="") as f:
            fieldnames = [
                "database",
                "operation",
                "batch_size",
                "top_k",
                "avg_time_ms",
                "median_time_ms",
                "stddev_time_ms",
                "avg_memory_mb",
                "num_samples",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.all_results:
                writer.writerow(result.to_dict())
        logger.info(f"Saved benchmark results to {csv_path}")

    def generate_charts(self) -> None:
        """Generate charts from benchmark results."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Group results by operation
        insert_results = [r for r in self.all_results if r.operation == "insert"]
        query_results = [r for r in self.all_results if r.operation == "query"]

        # Create charts directory
        charts_dir = self.output_dir / "charts"
        charts_dir.mkdir(exist_ok=True)

        # Generate insertion time comparison chart
        self._generate_time_chart(
            insert_results,
            charts_dir / f"insert_time_comparison_{timestamp}.png",
            "Vector Insertion Time Comparison",
            "Batch Size",
            "Average Time (ms)",
        )

        # Generate query time comparison charts for each top_k
        for top_k in self.config.top_k_values:
            top_k_results = [r for r in query_results if r.top_k == top_k]
            if not top_k_results:
                continue

            self._generate_time_chart(
                top_k_results,
                charts_dir / f"query_time_comparison_top{top_k}_{timestamp}.png",
                f"Vector Query Time Comparison (top_k={top_k})",
                "Batch Size",
                "Average Time (ms)",
            )

        # Generate memory usage comparison chart for insertion
        if any(r.avg_memory > 0 for r in insert_results):
            self._generate_memory_chart(
                insert_results,
                charts_dir / f"insert_memory_comparison_{timestamp}.png",
                "Vector Insertion Memory Usage Comparison",
                "Batch Size",
                "Average Memory (MB)",
            )

        logger.info(f"Generated benchmark charts in {charts_dir}")

    def _generate_time_chart(
        self,
        results: List[BenchmarkResult],
        output_path: Path,
        title: str,
        x_label: str,
        y_label: str,
    ) -> None:
        """Generate a time comparison chart."""
        if not results:
            return

        plt.figure(figsize=(10, 6))

        # Group results by database
        databases = {r.database for r in results}
        for db in databases:
            db_results = [r for r in results if r.database == db]
            # Sort by batch size
            db_results.sort(key=lambda r: r.batch_size)

            x = [r.batch_size for r in db_results]
            y = [r.avg_time for r in db_results]
            err = [r.stddev_time for r in db_results]

            plt.errorbar(x, y, yerr=err, label=db, marker="o", capsize=5)

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()

        # Save the chart
        plt.savefig(output_path)
        plt.close()

    def _generate_memory_chart(
        self,
        results: List[BenchmarkResult],
        output_path: Path,
        title: str,
        x_label: str,
        y_label: str,
    ) -> None:
        """Generate a memory usage comparison chart."""
        if not results:
            return

        plt.figure(figsize=(10, 6))

        # Group results by database
        databases = {r.database for r in results}
        for db in databases:
            db_results = [r for r in results if r.database == db]
            # Filter out results with no memory data
            db_results = [r for r in db_results if r.avg_memory > 0]
            if not db_results:
                continue

            # Sort by batch size
            db_results.sort(key=lambda r: r.batch_size)

            x = [r.batch_size for r in db_results]
            y = [r.avg_memory for r in db_results]

            plt.plot(x, y, label=db, marker="o")

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()

        # Save the chart
        plt.savefig(output_path)
        plt.close()


async def main():
    """Run the benchmark as a command-line tool."""
    parser = argparse.ArgumentParser(description="WDBX Benchmarking Tool")
    parser.add_argument(
        "--num-vectors", type=int, default=10000, help="Number of vectors to use in the benchmark"
    )
    parser.add_argument("--vector-dimension", type=int, default=1536, help="Dimension of vectors")
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
        help="Comma-separated list of top_k values",
    )
    parser.add_argument(
        "--num-iterations", type=int, default=5, help="Number of iterations for each benchmark"
    )
    parser.add_argument("--num-queries", type=int, default=100, help="Number of query vectors")
    parser.add_argument(
        "--databases",
        type=str,
        default="wdbx,faiss",
        help="Comma-separated list of databases to benchmark",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save benchmark results",
    )
    parser.add_argument("--no-charts", action="store_true", help="Disable chart generation")
    parser.add_argument("--no-raw-data", action="store_true", help="Don't save raw benchmark data")

    args = parser.parse_args()

    # Parse lists
    batch_sizes = [int(s) for s in args.batch_sizes.split(",")]
    top_k_values = [int(k) for k in args.top_k_values.split(",")]
    databases = [db.strip() for db in args.databases.split(",")]

    # Configure benchmark
    config = BenchmarkConfig(
        num_vectors=args.num_vectors,
        vector_dimension=args.vector_dimension,
        batch_sizes=batch_sizes,
        top_k_values=top_k_values,
        num_iterations=args.num_iterations,
        num_queries=args.num_queries,
        benchmarks=databases,
        output_dir=args.output_dir,
        generate_charts=not args.no_charts,
        save_raw_data=not args.no_raw_data,
    )

    # Run benchmark
    with LogContext(component="Benchmark"):
        runner = BenchmarkRunner(config)
        await runner.run_benchmark()


if __name__ == "__main__":
    asyncio.run(main())
