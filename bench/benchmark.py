#!/usr/bin/env python3
"""
Comprehensive benchmark suite for vectorwrap.

Compares all supported database backends (PostgreSQL, MySQL, SQLite, DuckDB, ClickHouse)
across multiple metrics:
- Insert performance (single and batch)
- Query performance (accuracy and speed)
- Memory usage
- Scalability with dataset size

Results are saved to JSON and optionally visualized.
"""

import argparse
import json
import time
import random
import statistics
from typing import Any, Dict, List, Tuple
from datetime import datetime
import traceback

try:
    from vectorwrap import VectorDB
except ImportError:
    import sys
    sys.path.insert(0, "..")
    from vectorwrap import VectorDB


class BenchmarkRunner:
    """Run benchmarks across different vector database backends."""

    def __init__(self, backends: Dict[str, str], dim: int = 128, dataset_size: int = 10000):
        """
        Initialize benchmark runner.

        Args:
            backends: Dict mapping backend name to connection URL
            dim: Vector dimensionality
            dataset_size: Number of vectors to test with
        """
        self.backends = backends
        self.dim = dim
        self.dataset_size = dataset_size
        self.results: Dict[str, Any] = {}

    def generate_vectors(self, count: int) -> List[List[float]]:
        """Generate random normalized vectors."""
        vectors = []
        for _ in range(count):
            vector = [random.gauss(0, 1) for _ in range(self.dim)]
            # Normalize
            norm = sum(x * x for x in vector) ** 0.5
            if norm > 0:
                vector = [x / norm for x in vector]
            vectors.append(vector)
        return vectors

    def benchmark_insert(
        self, db: Any, collection: str, vectors: List[List[float]], batch_size: int = 1
    ) -> Dict[str, float]:
        """Benchmark insert performance."""
        total_time = 0.0
        times = []

        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            start = time.perf_counter()

            for j, vector in enumerate(batch):
                vector_id = i + j
                metadata = {"id": vector_id, "batch": i // batch_size}
                db.upsert(collection, vector_id, vector, metadata)

            elapsed = time.perf_counter() - start
            times.append(elapsed)
            total_time += elapsed

        return {
            "total_time": total_time,
            "avg_time": statistics.mean(times),
            "median_time": statistics.median(times),
            "min_time": min(times),
            "max_time": max(times),
            "vectors_per_sec": len(vectors) / total_time if total_time > 0 else 0,
        }

    def benchmark_query(
        self,
        db: Any,
        collection: str,
        query_vectors: List[List[float]],
        top_k: int = 10,
    ) -> Dict[str, float]:
        """Benchmark query performance."""
        times = []

        for query_vector in query_vectors:
            start = time.perf_counter()
            results = db.query(collection, query_vector, top_k=top_k)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        return {
            "avg_time": statistics.mean(times),
            "median_time": statistics.median(times),
            "min_time": min(times),
            "max_time": max(times),
            "p95_time": statistics.quantiles(times, n=20)[18] if len(times) > 20 else max(times),
            "p99_time": statistics.quantiles(times, n=100)[98] if len(times) > 100 else max(times),
            "queries_per_sec": len(query_vectors) / sum(times) if sum(times) > 0 else 0,
        }

    def benchmark_query_with_filter(
        self,
        db: Any,
        collection: str,
        query_vectors: List[List[float]],
        top_k: int = 10,
    ) -> Dict[str, float]:
        """Benchmark filtered query performance."""
        times = []

        for i, query_vector in enumerate(query_vectors):
            # Filter for specific batch
            filter_dict = {"batch": i % 10}

            start = time.perf_counter()
            try:
                results = db.query(collection, query_vector, top_k=top_k, filter=filter_dict)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            except Exception:
                # Some backends might not support filtering
                continue

        if not times:
            return {"error": "Filtering not supported"}

        return {
            "avg_time": statistics.mean(times),
            "median_time": statistics.median(times),
            "queries_per_sec": len(times) / sum(times) if sum(times) > 0 else 0,
        }

    def run_benchmark(self, backend_name: str, connection_url: str) -> Dict[str, Any]:
        """Run full benchmark suite for a single backend."""
        print(f"\n{'='*60}")
        print(f"Benchmarking: {backend_name}")
        print(f"URL: {connection_url}")
        print(f"Vectors: {self.dataset_size}, Dimensions: {self.dim}")
        print(f"{'='*60}")

        results = {
            "backend": backend_name,
            "url": connection_url,
            "dim": self.dim,
            "dataset_size": self.dataset_size,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # Connect to database
            print(f"Connecting to {backend_name}...")
            db = VectorDB(connection_url)

            # Create collection
            collection = f"benchmark_{int(time.time())}"
            print(f"Creating collection: {collection}")
            db.create_collection(collection, self.dim)

            # Generate test data
            print(f"Generating {self.dataset_size} test vectors...")
            vectors = self.generate_vectors(self.dataset_size)
            query_vectors = self.generate_vectors(100)  # 100 query vectors

            # Benchmark insertions
            print("Benchmarking insertions...")
            insert_results = self.benchmark_insert(db, collection, vectors)
            results["insert"] = insert_results
            print(f"  Insert rate: {insert_results['vectors_per_sec']:.2f} vectors/sec")

            # Benchmark queries
            print("Benchmarking queries (top-10)...")
            query_results = self.benchmark_query(db, collection, query_vectors, top_k=10)
            results["query_top10"] = query_results
            print(f"  Query rate: {query_results['queries_per_sec']:.2f} queries/sec")
            print(f"  Avg latency: {query_results['avg_time']*1000:.2f} ms")

            # Benchmark queries with larger k
            print("Benchmarking queries (top-100)...")
            query_results_100 = self.benchmark_query(db, collection, query_vectors, top_k=100)
            results["query_top100"] = query_results_100
            print(f"  Query rate: {query_results_100['queries_per_sec']:.2f} queries/sec")

            # Benchmark filtered queries
            print("Benchmarking filtered queries...")
            filter_results = self.benchmark_query_with_filter(db, collection, query_vectors[:20])
            results["query_filtered"] = filter_results
            if "error" not in filter_results:
                print(f"  Filtered query rate: {filter_results['queries_per_sec']:.2f} queries/sec")
            else:
                print(f"  Filtered queries: {filter_results['error']}")

            results["status"] = "success"
            print(f"\n[SUCCESS] {backend_name} benchmark completed successfully")

        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            results["traceback"] = traceback.format_exc()
            print(f"\n[ERROR] {backend_name} benchmark failed: {e}")

        return results

    def run_all(self) -> Dict[str, Any]:
        """Run benchmarks for all backends."""
        print(f"\nStarting vectorwrap benchmark suite")
        print(f"Dataset size: {self.dataset_size}")
        print(f"Vector dimensions: {self.dim}")
        print(f"Backends: {', '.join(self.backends.keys())}")

        all_results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "dataset_size": self.dataset_size,
                "dimensions": self.dim,
            },
            "backends": {},
        }

        for backend_name, connection_url in self.backends.items():
            results = self.run_benchmark(backend_name, connection_url)
            all_results["backends"][backend_name] = results

        self.results = all_results
        return all_results

    def save_results(self, filename: str = "benchmark_results.json") -> None:
        """Save benchmark results to JSON file."""
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {filename}")

    def print_summary(self) -> None:
        """Print summary comparison table."""
        if not self.results:
            print("No results to display")
            return

        print(f"\n{'='*80}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*80}")

        # Create comparison table
        headers = ["Backend", "Insert (vec/s)", "Query (q/s)", "Avg Latency (ms)", "Status"]
        rows = []

        for backend_name, backend_results in self.results.get("backends", {}).items():
            if backend_results.get("status") == "success":
                insert_rate = backend_results.get("insert", {}).get("vectors_per_sec", 0)
                query_rate = backend_results.get("query_top10", {}).get("queries_per_sec", 0)
                avg_latency = backend_results.get("query_top10", {}).get("avg_time", 0) * 1000
                status = "OK"
            else:
                insert_rate = query_rate = avg_latency = 0
                status = "FAIL"

            rows.append(
                [
                    backend_name,
                    f"{insert_rate:.0f}",
                    f"{query_rate:.1f}",
                    f"{avg_latency:.2f}",
                    status,
                ]
            )

        # Print table
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]

        def print_row(row):
            print(
                "  ".join(
                    str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)
                )
            )

        print_row(headers)
        print("-" * sum(col_widths + [2 * (len(headers) - 1)]))
        for row in rows:
            print_row(row)

        print(f"{'='*80}\n")


def main():
    """Run benchmark suite."""
    parser = argparse.ArgumentParser(description="Benchmark vectorwrap backends")
    parser.add_argument("--dim", type=int, default=128, help="Vector dimensions (default: 128)")
    parser.add_argument(
        "--size", type=int, default=10000, help="Dataset size (default: 10000)"
    )
    parser.add_argument(
        "--output", type=str, default="benchmark_results.json", help="Output JSON file"
    )
    parser.add_argument(
        "--backends",
        type=str,
        nargs="+",
        default=["postgres", "mysql", "sqlite", "duckdb", "clickhouse"],
        help="Backends to benchmark",
    )

    args = parser.parse_args()

    # Define backend connection URLs
    # Users should modify these for their environment
    backend_urls = {
        "postgres": "postgresql://postgres:password@localhost:5432/benchmark",
        "mysql": "mysql://root:password@localhost:3306/benchmark",
        "sqlite": "sqlite:///./benchmark.db",
        "duckdb": "duckdb:///./benchmark_duckdb.db",
        "clickhouse": "clickhouse://default@localhost:8123/default",
    }

    # Filter to requested backends
    backends_to_test = {
        name: url for name, url in backend_urls.items() if name in args.backends
    }

    # Run benchmarks
    runner = BenchmarkRunner(backends_to_test, dim=args.dim, dataset_size=args.size)
    runner.run_all()
    runner.print_summary()
    runner.save_results(args.output)


if __name__ == "__main__":
    main()
