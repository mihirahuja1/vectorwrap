from __future__ import annotations

from typing import Any
import urllib.parse as up
import mysql.connector
import numpy as np
import json


def _euclidean_distance(v1: list[float] | np.ndarray, v2: list[float] | np.ndarray) -> float:
    """Calculate Euclidean distance between two vectors."""
    v1, v2 = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float)
    return float(np.sqrt(np.sum((v1 - v2) ** 2)))


def _where(flt: dict[str, Any]) -> tuple[str, list[Any]]:
    """Build WHERE clause from filter dictionary."""
    if not flt:
        return "", []
    clauses, vals = [], []
    for col, val in flt.items():
        clauses.append(f"{col} = %s")
        vals.append(val)
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
        """Create a new collection with JSON vector storage."""
        cur = self.conn.cursor()
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {name}("
            f"id BIGINT PRIMARY KEY, "
            f"emb JSON NOT NULL, "
            f"INDEX(id));"
        )
        cur.close()

    def upsert(self, name: str, _id: int, emb: list[float], meta: dict[str, Any] | None = None) -> None:
        """Insert or update a vector with optional metadata."""
        cur = self.conn.cursor()
        emb_json = json.dumps(list(np.asarray(emb, dtype=float)))
        cur.execute(
            f"REPLACE INTO {name}(id, emb) VALUES (%s, %s)",
            (_id, emb_json)
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