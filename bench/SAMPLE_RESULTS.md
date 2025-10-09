# vectorwrap Benchmark Results

**Date:** 2025-01-07

**Dataset Size:** 5,000 vectors

**Vector Dimensions:** 128

**Hardware:** Apple M-series (representative results)

## Performance Summary

| Backend | Insert (vec/s) | Query Top-10 (q/s) | Query Top-100 (q/s) | Avg Latency (ms) | Status |
|---------|----------------|--------------------|--------------------|------------------|--------|
| sqlite | 388 | 5424.3 | 4662.3 | 0.18 | ✅ |
| duckdb | 744 | 551.0 | 737.9 | 1.81 | ✅ |

## Detailed Results

### sqlite

**Connection URL:** `sqlite:///./benchmark.db`

**Insert Performance:**
- Total time: 12.89s
- Throughput: 388 vectors/sec
- Avg time: 2.58ms per batch

**Query Performance (Top-10):**
- Throughput: 5424.3 queries/sec
- Avg latency: 0.18ms
- Median latency: 0.18ms
- P95 latency: 0.21ms
- P99 latency: 0.23ms

**Filtered Query Performance:**
- Throughput: 3786.9 queries/sec
- Avg latency: 0.26ms

### duckdb

**Connection URL:** `duckdb:///./benchmark_duckdb.db`

**Insert Performance:**
- Total time: 6.72s
- Throughput: 744 vectors/sec
- Avg time: 1.34ms per batch

**Query Performance (Top-10):**
- Throughput: 551.0 queries/sec
- Avg latency: 1.81ms
- Median latency: 1.77ms
- P95 latency: 2.14ms
- P99 latency: 2.39ms

**Filtered Query Performance:**
- Throughput: 585.5 queries/sec
- Avg latency: 1.71ms

## Key Findings

### Performance Highlights

1. **SQLite** excels at query throughput with over 5,400 queries/sec
2. **DuckDB** has better insert performance (~2x faster than SQLite)
3. **SQLite** offers sub-millisecond query latency (0.18ms average)
4. Both backends support filtered queries efficiently

### Performance Comparison

**Best Insert Performance:** DuckDB (744 vec/s)
- 92% faster than SQLite for insertions

**Best Query Performance:** SQLite (5,424 q/s)
- 10x faster than DuckDB for queries
- Ultra-low latency (0.18ms)

## Use Case Recommendations

### When to Use SQLite
✅ **Best for:**
- High-volume query workloads
- Low-latency requirements (<1ms)
- Simple embedded applications
- Single-user scenarios

❌ **Not ideal for:**
- Concurrent write-heavy workloads
- Very large datasets (>100M vectors)

### When to Use DuckDB
✅ **Best for:**
- Analytical queries + vector search
- Better insert performance
- OLAP workloads with vector similarity
- Data science notebooks

❌ **Not ideal for:**
- Ultra-low latency requirements
- Web APIs with strict latency SLAs

## Production Backend Comparison

For production workloads, consider:

| Backend | Typical Performance | Best For |
|---------|-------------------|----------|
| **PostgreSQL + pgvector** | 200-500 q/s | Production, ACID, concurrent access |
| **MySQL 8.2+** | 150-400 q/s | Production, existing MySQL infrastructure |
| **ClickHouse** | 500-2000 q/s | High-performance analytics at scale |
| **SQLite** | 2000-10000 q/s | Embedded, single-user, prototyping |
| **DuckDB** | 300-1000 q/s | Analytics + vectors, data science |

## Benchmark Configuration

- Vector dimensions: 128
- Dataset size: 5,000 vectors
- Query count: 100 queries
- Top-K: 10 (primary), 100 (secondary)
- Filtered queries: 20 queries with metadata filters

## Running Your Own Benchmarks

Results vary based on:
- Hardware (CPU, RAM, SSD speed)
- Dataset size and dimensions
- Index configuration
- Concurrent load

**Run benchmarks on your hardware:**

```bash
cd bench
python benchmark.py --dim 128 --size 10000 --backends sqlite duckdb
python visualize.py benchmark_results.json
```

## Notes

- **Accuracy:** All backends return exact results for these dataset sizes
- **Index:** SQLite uses VSS HNSW index, DuckDB uses VSS extension
- **Memory:** Both backends tested with file-based storage (not :memory:)
- **Normalization:** All vectors are L2-normalized

## Recommendations

### For Prototyping
Start with **SQLite** or **DuckDB** - both offer excellent performance without server setup.

### For Production
- **PostgreSQL + pgvector** for general production use
- **ClickHouse** for high-performance analytics workloads
- **MySQL 8.2+** if already using MySQL infrastructure

### For Analytics
**DuckDB** or **ClickHouse** - combine vector search with analytical queries efficiently.

---

*These results are representative. Always benchmark on your target environment for accurate performance assessment.*

*For full benchmark methodology, see [bench/README.md](README.md)*
