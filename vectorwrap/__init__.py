# vectorwrap/__init__.py
from __future__ import annotations

from typing import Any, Union
from typing_extensions import Protocol

from .pg_backend import PgBackend
from .mysql_backend import MySQLBackend
from .sqlite_backend import SQLiteBackend
from .duckdb_backend import DuckDBBackend


class VectorBackend(Protocol):
    """Protocol defining the interface for vector database backends."""
    
    def create_collection(self, name: str, dim: int) -> None:
        """Create a new collection for vectors of dimension `dim`."""
        ...
    
    def upsert(self, name: str, _id: int, emb: list[float], meta: dict[str, Any] | None = None) -> None:
        """Insert or update a vector with optional metadata."""
        ...
    
    def query(
        self, 
        name: str, 
        emb: list[float], 
        top_k: int = 5, 
        filter: dict[str, Any] | None = None,
        **kwargs: Any
    ) -> list[tuple[int, float]]:
        """Find the `top_k` most similar vectors. Returns list of (id, distance) tuples."""
        ...


def VectorDB(url: str) -> VectorBackend:
    """
    Create a vector database connection.
    
    Args:
        url: Database connection string (e.g., "postgresql://...", "mysql://...", 
             "sqlite:///...", "duckdb:///...")
    
    Returns:
        A vector database backend instance
        
    Raises:
        ValueError: If the URL scheme is not supported
    """
    if url.startswith("postgres"):
        return PgBackend(url)
    if url.startswith("mysql"):
        return MySQLBackend(url)
    if url.startswith("sqlite"):
        return SQLiteBackend(url)
    if url.startswith("duckdb"):
        return DuckDBBackend(url)
    raise ValueError(f"Unsupported URL scheme. URL must start with postgres, mysql, sqlite, or duckdb, got: {url}")


__all__ = ["VectorDB", "VectorBackend"]
