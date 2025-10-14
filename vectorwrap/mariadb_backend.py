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
    """Calculate cosine distance between two vectors (for legacy mode)."""
    a = np.array(v1, dtype=float)
    b = np.array(v2, dtype=float)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - (dot_product / (norm_a * norm_b))


def _euclidean_distance(v1: list[float], v2: list[float]) -> float:
    """Calculate Euclidean (L2) distance between two vectors (for legacy mode)."""
    a = np.array(v1, dtype=float)
    b = np.array(v2, dtype=float)
    return float(np.linalg.norm(a - b))


class MariaDBBackend:
    """
    Backend for MariaDB with native VECTOR support (11.8+) or JSON fallback.

    MariaDB 11.8 GA LTS introduced native VECTOR data type with HNSW indexing,
    similar to pgvector. For older versions, falls back to JSON storage with
    Python-based distance calculations.

    Features (MariaDB 11.8+):
    - Native VECTOR(n) data type
    - HNSW indexing for fast similarity search
    - Distance functions: VEC_DISTANCE_EUCLIDEAN, VEC_DISTANCE_COSINE
    - Production-ready performance

    Example:
        ```python
        from vectorwrap import VectorDB

        # Connect to MariaDB 11.8+
        db = VectorDB("mariadb://user:pass@localhost:3306/vectordb")

        # Create collection with native VECTOR type
        db.create_collection("documents", dim=1536)

        # Upsert vectors
        db.upsert("documents", 1, [0.1, 0.2, ...], {"source": "doc1"})

        # Query with native vector similarity
        results = db.query("documents", query_vector, top_k=10)
        ```
    """

    def __init__(self, url: str) -> None:
        """
        Initialize MariaDB connection.

        Args:
            url: Database connection URL (mariadb://user:pass@host:port/database)
        """
        if mysql is None:
            raise RuntimeError(
                "mysql-connector-python not installed. "
                "Install with: pip install mysql-connector-python"
            )

        # Parse connection URL
        # Format: mariadb://user:password@host:port/database
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

        # Detect MariaDB version and native vector support
        cursor = self.conn.cursor()
        cursor.execute("SELECT VERSION()")
        version_str = cursor.fetchone()[0]
        cursor.close()

        self.is_mariadb = "MariaDB" in version_str
        self.native_vector_support = False

        if self.is_mariadb:
            # Parse version (e.g., "11.8.0-MariaDB")
            version_parts = version_str.split("-")[0].split(".")
            major = int(version_parts[0])
            minor = int(version_parts[1]) if len(version_parts) > 1 else 0

            # Native VECTOR support in MariaDB 11.8+
            if major > 11 or (major == 11 and minor >= 8):
                self.native_vector_support = True

    def create_collection(self, name: str, dim: int) -> None:
        """
        Create a collection with native VECTOR type (11.8+) or JSON fallback.

        Args:
            name: Collection name
            dim: Vector dimension
        """
        cursor = self.conn.cursor()

        if self.native_vector_support:
            # Use native VECTOR type with HNSW index
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {name} (
                    id BIGINT PRIMARY KEY,
                    vector VECTOR({dim}) NOT NULL,
                    metadata JSON
                )
            """)

            # Create HNSW index for fast similarity search
            try:
                cursor.execute(f"""
                    CREATE INDEX idx_{name}_vector
                    ON {name}(vector)
                    USING HNSW
                """)
            except Error as e:
                # Index might already exist or HNSW not available
                if "Duplicate key name" not in str(e):
                    # Try without USING HNSW
                    try:
                        cursor.execute(f"CREATE INDEX idx_{name}_vector ON {name}(vector)")
                    except Error:
                        pass
        else:
            # Fallback to JSON storage for older versions
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {name} (
                    id BIGINT PRIMARY KEY,
                    vector JSON NOT NULL,
                    metadata JSON,
                    dim INT NOT NULL
                )
            """)

            # Create index on ID
            try:
                cursor.execute(f"CREATE INDEX idx_{name}_id ON {name}(id)")
            except Error:
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
        metadata_json = json.dumps(meta) if meta else None

        if self.native_vector_support:
            # Use native VECTOR type
            # Convert list to VECTOR literal: VEC_FromText('[1,2,3]')
            vector_literal = "[" + ",".join(str(x) for x in emb) + "]"

            if metadata_json:
                cursor.execute(
                    f"""
                    INSERT INTO {name} (id, vector, metadata)
                    VALUES (%s, VEC_FromText(%s), %s)
                    ON DUPLICATE KEY UPDATE
                        vector = VALUES(vector),
                        metadata = VALUES(metadata)
                    """,
                    (_id, vector_literal, metadata_json)
                )
            else:
                cursor.execute(
                    f"""
                    INSERT INTO {name} (id, vector, metadata)
                    VALUES (%s, VEC_FromText(%s), NULL)
                    ON DUPLICATE KEY UPDATE
                        vector = VALUES(vector)
                    """,
                    (_id, vector_literal)
                )
        else:
            # Fallback to JSON storage
            vector_json = json.dumps(emb)
            dim = len(emb)

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
        Query for similar vectors using native or fallback implementation.

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

        if self.native_vector_support:
            # Use native vector similarity functions
            vector_literal = "[" + ",".join(str(x) for x in emb) + "]"

            # Choose distance function
            if metric == "cosine":
                distance_func = "VEC_DISTANCE_COSINE"
            elif metric == "euclidean" or metric == "l2":
                distance_func = "VEC_DISTANCE_EUCLIDEAN"
            else:
                raise ValueError(f"Unsupported metric: {metric}. Use 'cosine' or 'euclidean'")

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

            # Query with native distance function
            query_sql = f"""
                SELECT id, {distance_func}(vector, VEC_FromText(%s)) as distance
                FROM {name}
                {where_clause}
                ORDER BY distance ASC
                LIMIT %s
            """
            params = [vector_literal] + params + [top_k]
            cursor.execute(query_sql, params)

            results = [(row[0], float(row[1])) for row in cursor.fetchall()]
            cursor.close()
            return results

        else:
            # Fallback to Python-based distance calculation
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

            # Fetch all vectors
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

                if metric == "cosine":
                    distance = _cosine_distance(emb, stored_vector)
                elif metric == "euclidean" or metric == "l2":
                    distance = _euclidean_distance(emb, stored_vector)
                else:
                    raise ValueError(f"Unsupported metric: {metric}. Use 'cosine' or 'euclidean'")

                results.append((row_id, distance))

            # Sort and return top_k
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


# Support mariadb:// URLs
def MariaDBBackendFromURL(url: str) -> MariaDBBackend:
    """
    Create MariaDBBackend from mariadb:// URL.

    Args:
        url: Connection URL (mariadb://user:pass@host:port/database)

    Returns:
        MariaDBBackend instance
    """
    return MariaDBBackend(url)
