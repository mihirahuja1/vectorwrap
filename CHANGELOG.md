# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2025-01-07

**MAJOR ECOSYSTEM EXPANSION** - Integration support for LangChain, LlamaIndex, Supabase, Milvus, and Qdrant.

### Added
- **LangChain VectorStore adapter** - Full integration with LangChain's document retrieval and RAG pipelines
  - `add_texts()`, `similarity_search()`, `similarity_search_with_score()`, `as_retriever()`
  - Works seamlessly with any vectorwrap backend
  - Example: `examples/langchain_example.py`
- **LlamaIndex VectorStore wrapper** - Complete integration with LlamaIndex data framework
  - VectorStore protocol implementation with node support
  - Convenience `create_vector_store()` helper function
  - Compatible with ServiceContext and query engines
- **Supabase pgvector helper** - Managed PostgreSQL + pgvector integration
  - `from_supabase_credentials()` and `from_env()` convenience methods
  - Row Level Security (RLS) support with policy creation
  - Bulk upsert operations for performance
  - Schema helpers and SQL generation
  - Migration utilities from Pinecone
  - Example: `examples/supabase_example.py`
- **Milvus adapter** - Enterprise vector database integration
  - Full pymilvus wrapper with HNSW/IVF index support
  - Collection management and statistics
  - Import/export utilities
  - Migration from pgvector
- **Qdrant adapter** - Cloud-native vector search integration
  - Support for Qdrant Cloud and local deployments
  - Payload filtering with Qdrant's filter syntax
  - Migration utilities from SQL stores
  - URL convenience helpers
- **ClickHouse backend** - High-performance analytical vector database support
  - ANN indexes with HNSW algorithm
  - Native Array(Float32) vector type
  - L2Distance similarity search
  - Filtered query support with JSONExtract
- **Comprehensive benchmark suite** - Performance testing across all backends
  - `bench/benchmark.py` for automated benchmarking
  - `bench/visualize.py` for chart generation
  - Real benchmark results in `bench/SAMPLE_RESULTS.md`
  - Metrics: insert throughput, query QPS, latency (P95/P99)
- **Integration documentation** - Complete guide in `docs/INTEGRATIONS.md`
  - Usage examples for all integrations
  - Migration guides between vector stores
  - Installation options and comparison table

### Changed
- **Package dependencies** - New optional extras for integrations
  - `[langchain]` - LangChain support
  - `[llamaindex]` - LlamaIndex support
  - `[milvus]` - Milvus support
  - `[qdrant]` - Qdrant support
  - `[integrations]` - All framework integrations
  - `[all]` - All backends and integrations
- **README** - Added integrations section with quick examples
- **Code quality** - Removed all visual symbols for professional appearance

### Fixed
- Benchmark output formatting with plain text status indicators
- Test output messages for better CI/CD compatibility

## [0.5.0] - 2025-01-07

### Added
- **ClickHouse backend** with ANN indexes support
- Real benchmark results documentation

### Changed
- Updated .gitignore to exclude benchmark data files

## [0.4.0] - 2025-07-30

**STABLE RELEASE** - Production-ready with API backward compatibility guarantees.

## [0.4.0a1] - 2025-07-30

### Added
- **NEW DuckDB backend** with DuckDB VSS extension support
- **Comprehensive type hints** for all public APIs with mypy compatibility
- **API stability documentation** with semantic versioning commitment
- **PostgreSQL test suite** with comprehensive coverage
- **Coverage badge** and Codecov integration in CI
- **Benchmark notebook** (`bench/bench_10k.ipynb`) for performance testing
- **Contributing guide** (`CONTRIBUTING.md`) with development guidelines
- **Development dependencies** configuration with pytest, black, ruff, mypy

### Changed
- **CI matrix testing** now includes PostgreSQL, MySQL 8.2, SQLite-VSS, DuckDB-VSS on Python 3.11/3.12
- **Optional extras** structure: `[sqlite]`, `[duckdb]`, `[dev]`, `[all]`, `[test]`
- **README optimization** for better PyPI rendering with badges and improved structure
- **License format** updated to modern SPDX string format

### Fixed
- **Critical PostgreSQL distance bug**: Fixed incorrect operator (`<=>` → `<->`) for proper L2 distance calculation
- **SQLite extension loading** with better error handling and pysqlite3-binary support
- **DuckDB vector type casting** for proper FLOAT[] array operations

### Technical Details
- **Backend Protocol**: Unified interface across all vector database backends
- **HNSW indexing**: Enabled for PostgreSQL, SQLite-VSS, and DuckDB-VSS
- **Test coverage**: Comprehensive test suites for all supported backends
- **Performance benchmarking**: 10k vector insertion and query benchmarks

## [0.3.1a1] - 2025-07-29

### Added
- SQLite backend with sqlite-vss extension
- MySQL backend with JSON fallback support
- PostgreSQL backend with pgvector extension
- Basic vector operations (create_collection, upsert, query)
- Initial PyPI package release

### Technical Details
- Support for 384-dimensional vectors (typical sentence-transformer size)
- Vector similarity search with configurable top-k results
- Metadata filtering support (backend-dependent implementation)