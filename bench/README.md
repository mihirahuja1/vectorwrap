# vectorwrap Benchmarks

Comprehensive benchmark suite for comparing all vectorwrap-supported database backends.

## Quick Start

```bash
# Install dependencies
pip install "vectorwrap[all]" matplotlib numpy

# Run benchmarks (uses default settings)
python benchmark.py

# Run with custom parameters
python benchmark.py --dim 384 --size 50000 --backends postgres sqlite duckdb clickhouse

# Visualize results
python visualize.py benchmark_results.json
```

## Benchmark Metrics

The benchmark suite measures:

### 1. **Insert Performance**
- Single-vector insert throughput (vectors/sec)
- Average, median, min, and max insert times
- Total time for dataset insertion

### 2. **Query Performance**
- Query throughput (queries/sec)
- Top-K query latency (avg, median, P95, P99)
- Performance with different K values (10, 100)
- Filtered query performance

### 3. **Accuracy**
- Number of results returned per query
- Result quality validation

## Running Benchmarks

### Default Benchmark

```bash
python benchmark.py
```

This runs benchmarks on all backends with:
- 10,000 vectors
- 128 dimensions
- SQLite, DuckDB, PostgreSQL, MySQL, ClickHouse

### Custom Parameters

```bash
# Larger dataset with higher dimensions
python benchmark.py --dim 1536 --size 100000

# Specific backends only
python benchmark.py --backends sqlite duckdb

# Custom output file
python benchmark.py --output my_results.json
```

### Backend Configuration

Edit connection URLs in `benchmark.py` to match your environment:

```python
backend_urls = {
    "postgres": "postgresql://postgres:password@localhost:5432/benchmark",
    "mysql": "mysql://root:password@localhost:3306/benchmark",
    "sqlite": "sqlite:///./benchmark.db",
    "duckdb": "duckdb:///./benchmark_duckdb.db",
    "clickhouse": "clickhouse://default@localhost:8123/default",
}
```

## Visualizing Results

### Generate Charts and Reports

```bash
# Create visualization charts
python visualize.py benchmark_results.json

# Generate markdown report only
python visualize.py benchmark_results.json --no-charts

# Custom output directory
python visualize.py benchmark_results.json --output-dir ./results
```

### Output Files

- `benchmark_results.png` - Performance comparison charts
- `benchmark_results.pdf` - PDF version of charts
- `BENCHMARK_RESULTS.md` - Detailed markdown report

## Jupyter Notebook

For interactive analysis:

```bash
jupyter notebook bench_10k.ipynb
```

The notebook includes:
- Real-time performance testing
- Visual comparisons
- Interactive parameter tuning
- Recommendations based on results

## Understanding Results

### Insert Throughput
Higher is better. Measures how fast vectors can be inserted.
- **Good:** >1000 vectors/sec
- **Excellent:** >5000 vectors/sec

### Query Performance (QPS)
Higher is better. Measures query throughput.
- **Good:** >100 queries/sec
- **Excellent:** >500 queries/sec

### Query Latency
Lower is better. Measures response time.
- **Good:** <50ms average
- **Excellent:** <10ms average

### When to Use Each Backend

| Backend | Best For | Pros | Cons |
|---------|----------|------|------|
| **SQLite** | Prototyping, testing | Easy setup, no server | Limited concurrent writes |
| **DuckDB** | Analytics + vectors | Fast, combines analytics | Single-process only |
| **PostgreSQL** | Production | Scalable, ACID, concurrent | Requires server setup |
| **MySQL** | Production (alternative) | Mature, widely supported | Vector support in 8.2+ |
| **ClickHouse** | High-performance analytics | Extremely fast, large-scale | More complex setup |

## Sample Results

Example performance on MacBook Pro M1 (10K vectors, 128D):

```
Backend          Insert (vec/s)  Query (q/s)  Avg Latency (ms)  Status
──────────────────────────────────────────────────────────────────────
sqlite           2,450           156.3        6.40              ✅
duckdb           8,320           412.7        2.42              ✅
postgres         3,120           245.1        4.08              ✅
clickhouse       12,500          890.5        1.12              ✅
```

*Your results will vary based on hardware, dataset size, and configuration.*

## Advanced Usage

### Custom Benchmark Metrics

Extend `BenchmarkRunner` class to add custom metrics:

```python
from bench.benchmark import BenchmarkRunner

class CustomBenchmark(BenchmarkRunner):
    def benchmark_accuracy(self, db, collection, ground_truth):
        # Add your accuracy measurement
        pass
```

### Comparing Against Other Vector Stores

The benchmark framework can be extended to compare against:
- Qdrant
- Weaviate
- Pinecone
- Milvus
- Chroma

See the [comparison guide](../docs/comparisons.md) for details.

## Continuous Benchmarking

For CI/CD integration:

```bash
# Run quick benchmark (smaller dataset)
python benchmark.py --size 1000 --backends sqlite duckdb

# Check if performance meets threshold
python -c "
import json
with open('benchmark_results.json') as f:
    results = json.load(f)
    for backend, data in results['backends'].items():
        if data['status'] == 'success':
            qps = data['query_top10']['queries_per_sec']
            assert qps > 100, f'{backend} QPS too low: {qps}'
"
```

## Troubleshooting

### Backend Connection Failures

If a backend fails to connect:

1. Check database server is running
2. Verify connection URL credentials
3. Ensure required extensions are installed (pgvector for PostgreSQL)
4. Check firewall/network settings

### Out of Memory

For large datasets:

```bash
# Use smaller batch sizes
python benchmark.py --size 5000

# Or run backends individually
python benchmark.py --backends sqlite
python benchmark.py --backends postgres
```

### Slow Performance

- Ensure indexes are created (automatic in most backends)
- Check available system resources (CPU, RAM)
- Try in-memory backends (`:memory:`) for faster testing

## Contributing

To add new benchmark metrics:

1. Extend `BenchmarkRunner` class
2. Add visualization to `visualize.py`
3. Update this README
4. Submit a PR with sample results

## References

- [vectorwrap Documentation](../README.md)
- [pgvector Benchmarks](https://github.com/pgvector/pgvector#benchmarks)
- [DuckDB VSS Extension](https://github.com/duckdb/duckdb_vss)
- [ClickHouse ANN Indexes](https://clickhouse.com/docs/engines/table-engines/mergetree-family/annindexes)

---

**Note:** Benchmark results are system-dependent. Always run benchmarks on your target hardware for accurate performance assessment.
