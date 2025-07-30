# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2025-07-30

🎉 **STABLE RELEASE** - Production-ready with API backward compatibility guarantees.

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