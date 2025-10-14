# vectorwrap/mariadb_backend.py
from __future__ import annotations

from typing import Any, cast
import json
import numpy as np

try:
    import mysql.connector
    from mysql.connector import Error
except ImportError:
    mysql = None  # type: ignore


def _cosine_distance(v1: list[float], v2: list[float]) -> float:
    """Calculate cosine distance between two vectors."""
    a = np.array(v1, dtype=float)
    b = np.array(v2, dtype=float)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - (dot_product / (norm_a * norm_b))


def _euclidean_distance(v1: list[float], v2: list[float]) -> float:
    """Calculate Euclidean (L2) distance between two vectors."""
    a = np.array(v1, dtype=float)
    b = np.array(v2, dtype=float)
    return float(np.linalg.norm(a - b))


class MariaDBBackend:
    """
    Backend for MariaDB with JSON-based vector storage.

    MariaDB doesn't have native vector support, so we store vectors as JSON
    and compute distances in Python. While not as fast as native vector types,
    this provides a simple way to use MariaDB for vector search.

    Note: For production workloads with large datasets, consider MySQL 8.2+
    with native VECTOR type, or PostgreSQL with pgvector.

    Example:
        ```python
        from vectorwrap import VectorDB

        # Connect to MariaDB
        db = VectorDB("mysql://user:pass@localhost:3306/vectordb")

        # Create collection
        db.create_collection("documents", dim=1536)

        # Upsert vectors
        db.upsert("documents", 1, [0.1, 0.2, ...], {"source": "doc1"})

        # Query (computed in Python)
        results = db.query("documents", query_vector, top_k=10)
        ```
    """

    def __init__(self, url: str) -> None:
        """
        Initialize MariaDB connection.

        Args:
            url: Database connection URL (mysql://user:pass@host:port/database)
        """
        if mysql is None:
            raise RuntimeError(
                "mysql-connector-python not installed. "
                "Install with: pip install mysql-connector-python"
            )

        # Parse connection URL
        # Format: mysql://user:password@host:port/database
        if not url.startswith("mysql://") and not url.startswith("mariadb://"):
            raise ValueError("URL must start with 'mysql://' or 'mariadb://'")

        url_clean = url.replace("mysql://", "").replace("mariadb://", "")

        # Parse components
        if "@" in url_clean:
            auth_part, host_part = url_clean.split("@", 1)
            if ":" in auth_part:
                user, password = auth_part.split(":", 1)
            else:
                user = auth_part
                password = ""
        else:
            user = "root"
            password = ""
            host_part = url_clean

        if "/" in host_part:
            host_port, database = host_part.split("/", 1)
        else:
            host_port = host_part
            database = "vectordb"

        if ":" in host_port:
            host, port_str = host_port.split(":", 1)
            port = int(port_str)
        else:
            host = host_port
            port = 3306

        # Connect to MariaDB
        self.conn = mysql.connector.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            autocommit=True
        )

        # Check if this is actually MariaDB (optional - works with both)
        cursor = self.conn.cursor()
        cursor.execute("SELECT VERSION()")
        version = cursor.fetchone()[0]
        self.is_mariadb = "MariaDB" in version
        cursor.close()

    def create_collection(self, name: str, dim: int) -> None:
        """
        Create a collection with JSON vector storage.

        Args:
            name: Collection name
            dim: Vector dimension (stored for validation)
        """
        cursor = self.conn.cursor()

        # Create table with JSON column for vectors
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {name} (
                id BIGINT PRIMARY KEY,
                vector JSON NOT NULL,
                metadata JSON,
                dim INT NOT NULL
            )
        """)

        # Create index on ID for faster lookups
        try:
            cursor.execute(f"CREATE INDEX idx_{name}_id ON {name}(id)")
        except Error:
            # Index might already exist
            pass

        cursor.close()

    def upsert(
        self,
        name: str,
        _id: int,
        emb: list[float],
        meta: dict[str, Any] | None = None,
    ) -> None:
        """
        Insert or update a vector.

        Args:
            name: Collection name
            _id: Vector ID
            emb: Embedding vector
            meta: Optional metadata
        """
        cursor = self.conn.cursor()

        # Convert vector to JSON
        vector_json = json.dumps(emb)
        metadata_json = json.dumps(meta) if meta else None
        dim = len(emb)

        # Upsert using REPLACE or INSERT ... ON DUPLICATE KEY UPDATE
        if metadata_json:
            cursor.execute(
                f"""
                INSERT INTO {name} (id, vector, metadata, dim)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    vector = VALUES(vector),
                    metadata = VALUES(metadata),
                    dim = VALUES(dim)
                """,
                (_id, vector_json, metadata_json, dim)
            )
        else:
            cursor.execute(
                f"""
                INSERT INTO {name} (id, vector, metadata, dim)
                VALUES (%s, %s, NULL, %s)
                ON DUPLICATE KEY UPDATE
                    vector = VALUES(vector),
                    dim = VALUES(dim)
                """,
                (_id, vector_json, dim)
            )

        cursor.close()

    def query(
        self,
        name: str,
        emb: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
        metric: str = "cosine",
        **kwargs: Any,
    ) -> list[tuple[int, float]]:
        """
        Query for similar vectors using Python-based distance calculation.

        Note: This fetches all vectors and computes distances in Python.
        For large datasets, this will be slow. Consider using MySQL 8.2+
        with native VECTOR type or PostgreSQL with pgvector for better performance.

        Args:
            name: Collection name
            emb: Query embedding
            top_k: Number of results
            filter: Optional metadata filter
            metric: Distance metric ("cosine" or "euclidean", default: "cosine")
            **kwargs: Additional arguments

        Returns:
            List of (id, distance) tuples
        """
        cursor = self.conn.cursor()

        # Build filter clause
        where_clause = ""
        params = []

        if filter:
            conditions = []
            for key, value in filter.items():
                if isinstance(value, str):
                    conditions.append(f"JSON_EXTRACT(metadata, '$.{key}') = %s")
                else:
                    conditions.append(f"JSON_EXTRACT(metadata, '$.{key}') = %s")
                params.append(value)

            if conditions:
                where_clause = " WHERE " + " AND ".join(conditions)

        # Fetch all vectors (with optional filter)
        query_sql = f"SELECT id, vector FROM {name}{where_clause}"
        cursor.execute(query_sql, params)

        rows = cursor.fetchall()
        cursor.close()

        if not rows:
            return []

        # Calculate distances in Python
        results = []
        for row_id, vector_json in rows:
            stored_vector = json.loads(vector_json)

            # Calculate distance based on metric
            if metric == "cosine":
                distance = _cosine_distance(emb, stored_vector)
            elif metric == "euclidean" or metric == "l2":
                distance = _euclidean_distance(emb, stored_vector)
            else:
                raise ValueError(f"Unsupported metric: {metric}. Use 'cosine' or 'euclidean'")

            results.append((row_id, distance))

        # Sort by distance and return top_k
        results.sort(key=lambda x: x[1])
        return results[:top_k]

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self, 'conn') and self.conn.is_connected():
            self.conn.close()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass


# Add support for mariadb:// URLs
def MariaDBBackendFromURL(url: str) -> MariaDBBackend:
    """
    Create MariaDBBackend from mariadb:// URL.

    Args:
        url: Connection URL (mariadb://user:pass@host:port/database)

    Returns:
        MariaDBBackend instance
    """
    return MariaDBBackend(url)
