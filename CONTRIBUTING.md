# Contributing to vectorwrap

Thank you for contributing to vectorwrap! This guide will help you get started with development.

## Development Setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/your-username/vectorwrap.git
   cd vectorwrap
   ```

2. **Install in development mode**:
   ```bash
   pip install -e ".[sqlite,duckdb]"  # Install all backends
   pip install pytest black isort mypy  # Development tools
   ```

3. **Set up test databases** (optional, for full testing):
   ```bash
   # PostgreSQL with pgvector
   docker run -d --name postgres-test -p 5432:5432 \
     -e POSTGRES_PASSWORD=secret \
     pgvector/pgvector:pg16
   
   # MySQL 8.2+ with vector support
   docker run -d --name mysql-test -p 3306:3306 \
     -e MYSQL_ROOT_PASSWORD=secret \
     mysql:8.2
   ```

## Running Tests

```bash
# Run all tests
pytest

# Run specific backend tests
pytest tests/test_sqlite.py
pytest tests/test_duckdb.py

# Run with coverage
pytest --cov=vectorwrap
```

**Note**: Some tests require running databases. Tests will skip if databases are unavailable.

## Code Style

We use **black** for code formatting and **isort** for import sorting:

```bash
# Format code
black vectorwrap/ tests/
isort vectorwrap/ tests/

# Check formatting (CI will enforce this)
black --check vectorwrap/ tests/
isort --check-only vectorwrap/ tests/

# Type checking
mypy vectorwrap/
```

## Branch Naming

Use descriptive branch names with prefixes:

- `feature/add-redis-backend` - New features
- `fix/sqlite-extension-loading` - Bug fixes  
- `docs/update-readme` - Documentation changes
- `refactor/cleanup-backends` - Code refactoring

## Pull Request Guidelines

1. **Create an issue first** for significant changes
2. **Write tests** for new functionality
3. **Update documentation** if adding new features
4. **Run the full test suite** before submitting
5. **Keep commits focused** - one logical change per commit
6. **Write clear commit messages**:
   ```
   Add DuckDB backend support
   
   - Implement DuckDB VSS extension integration
   - Add HNSW indexing for fast similarity search
   - Include comprehensive test coverage
   ```

## Code Guidelines

- **Follow existing patterns** in the codebase
- **Add type hints** to all public functions
- **Keep backends consistent** - implement the same interface
- **Handle errors gracefully** with informative messages
- **Document public APIs** with docstrings

## Testing New Backends

When adding a new vector database backend:

1. **Implement the Protocol**: Follow `vectorwrap/__init__.py:VectorBackend`
2. **Add factory support**: Update `VectorDB()` constructor  
3. **Write comprehensive tests**: Cover all CRUD operations
4. **Update documentation**: Add to README.md and examples
5. **Add optional dependency**: Update `pyproject.toml`

## Questions?

- **Open an issue** for bugs or feature requests
- **Start a discussion** for design questions
- **Check existing issues** before creating new ones

We appreciate your contributions! 🚀