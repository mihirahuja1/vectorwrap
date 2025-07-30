from __future__ import annotations

from typing import Any
import urllib.parse as up
import mysql.connector
import numpy as np
import json


def _euclidean_distance(v1: list[float] | np.ndarray[Any, np.dtype[np.floating[Any]]], v2: list[float] | np.ndarray[Any, np.dtype[np.floating[Any]]]) -> float:
    """Calculate Euclidean distance between two vectors."""
    v1, v2 = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float)
    return float(np.sqrt(np.sum((v1 - v2) ** 2)))


def _where(flt: dict[str, Any]) -> tuple[str, list[Any]]:
    """Build WHERE clause from filter dictionary using JSON operators."""
    if not flt:
        return "", []
    clauses, vals = [], []
    for key, val in flt.items():
        # Use JSON_UNQUOTE and JSON_EXTRACT for filtering
        clauses.append("JSON_UNQUOTE(JSON_EXTRACT(metadata, %s)) = %s")
        vals.extend([f"$.{key}", str(val)])  # JSON path and string value
    return " WHERE " + " AND ".join(clauses), vals


class MySQLBackend:
    """Backend for MySQL with JSON-based vector storage."""

    def __init__(self, url: str) -> None:
        """Initialize MySQL connection."""
        p = up.urlparse(url)
        self.db = p.path.lstrip("/")
        self.conn = mysql.connector.connect(
            host=p.hostname, port=p.port or 3306,
            user=p.username, password=p.password,
            database=self.db, autocommit=True
        )

    def create_collection(self, name: str, dim: int) -> None:
        """Create a new collection with JSON vector storage and metadata."""
        cur = self.conn.cursor()
        # Create table with basic structure first
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {name}("
            f"id BIGINT PRIMARY KEY, "
            f"emb JSON NOT NULL, "
            f"INDEX(id));"
        )
        
        # Add metadata column if it doesn't exist (for backward compatibility)
        try:
            cur.execute(f"ALTER TABLE {name} ADD COLUMN metadata JSON;")
        except Exception:
            # Column already exists or other error - ignore
            pass
        
        cur.close()

    def upsert(self, name: str, _id: int, emb: list[float], meta: dict[str, Any] | None = None) -> None:
        """Insert or update a vector with optional metadata."""
        cur = self.conn.cursor()
        emb_json = json.dumps(list(np.asarray(emb, dtype=float)))
        metadata_json = json.dumps(meta) if meta is not None else None
        cur.execute(
            f"REPLACE INTO {name}(id, emb, metadata) VALUES (%s, %s, %s)",
            (_id, emb_json, metadata_json)
        )
        cur.close()

    def query(
        self, 
        name: str, 
        emb: list[float], 
        top_k: int = 5, 
        filter: dict[str, Any] | None = None, 
        **_: Any
    ) -> list[tuple[int, float]]:
        """Query for similar vectors using Python-based distance calculation."""
        where_sql, vals = _where(filter or {})
        
        # Fetch all vectors and calculate distances in Python
        # For large datasets, this should be optimized with proper indexing
        cur = self.conn.cursor()
        cur.execute(f"SELECT id, emb FROM {name}{where_sql}", vals)
        rows = cur.fetchall()
        cur.close()
        
        # Calculate distances and sort
        query_vec = np.asarray(emb, dtype=float)
        results = []
        
        for row_id, emb_json in rows:
            stored_vec = json.loads(str(emb_json))  # Ensure emb_json is a string
            distance = _euclidean_distance(query_vec, stored_vec)
            results.append((int(str(row_id)), distance))  # Ensure row_id is int
        
        # Sort by distance and return top_k
        results.sort(key=lambda x: x[1])
        return results[:top_k]