# vectorwrap/duckdb_backend.py
from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import duckdb
else:
    try:
        import duckdb  # type: ignore
    except ImportError:
        duckdb = None

import numpy as np


def _array_literal(v: list[float]) -> str:
    """Convert vector to DuckDB ARRAY literal format."""
    return "[" + ",".join(map(str, np.asarray(v, dtype=float))) + "]"


class DuckDBBackend:
    """Backend for DuckDB with VSS (Vector Similarity Search) extension."""

    def __init__(self, url: str) -> None:
        if duckdb is None:
            raise RuntimeError(
                "duckdb not installed. Install with: pip install 'vectorwrap[duckdb]'"
            )
        
        # url pattern: duckdb:///path/to/file.db or duckdb:///:memory:
        if url.startswith("duckdb:///"):
            path = url.replace("duckdb:///", "", 1)
            if path == ":memory:":
                self.conn = duckdb.connect(":memory:")
            else:
                self.conn = duckdb.connect(path)
        else:
            raise ValueError("DuckDB URL must start with 'duckdb:///'")
        
        # Install and load VSS extension
        try:
            self.conn.execute("INSTALL vss")
            self.conn.execute("LOAD vss")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load VSS extension: {e}. "
                "Make sure you're using DuckDB v0.10.2+ and the VSS extension is available."
            )

    def create_collection(self, name: str, dim: int) -> None:
        """Create a table with vector column and HNSW index."""
        # Create table with ARRAY column for vectors
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {name} (
                id INTEGER PRIMARY KEY,
                vector FLOAT[{dim}],
                metadata JSON
            )
        """)
        
        # Create HNSW index on the vector column
        # Note: It's more efficient to create index after data is loaded
        try:
            self.conn.execute(f"CREATE INDEX IF NOT EXISTS {name}_hnsw_idx ON {name} USING HNSW (vector)")
        except Exception:
            # Index creation might fail if table is empty, that's okay
            pass

    def upsert(self, name: str, _id: int, emb: list[float], meta: dict[str, Any] | None = None) -> None:
        """Insert or update a vector with optional metadata."""
        import json
        
        # First delete any existing row with this ID, then insert
        self.conn.execute(f"DELETE FROM {name} WHERE id = ?", [_id])
        
        if meta is None:
            # Use parameterized query for NULL
            self.conn.execute(f"""
                INSERT INTO {name} (id, vector, metadata) 
                VALUES (?, {_array_literal(emb)}, ?)
            """, [_id, None])
        else:
            # Use parameterized query for JSON
            self.conn.execute(f"""
                INSERT INTO {name} (id, vector, metadata) 
                VALUES (?, {_array_literal(emb)}, ?)
            """, [_id, json.dumps(meta)])
        
        # Try to create/recreate index if it doesn't exist
        try:
            self.conn.execute(f"CREATE INDEX IF NOT EXISTS {name}_hnsw_idx ON {name} USING HNSW (vector)")
        except Exception:
            pass

    def query(
        self, 
        name: str, 
        emb: list[float], 
        top_k: int = 5, 
        filter: dict[str, Any] | None = None, 
        **kwargs: Any
    ) -> list[tuple[int, float]]:
        """Query for similar vectors using array_distance"""
        
        # Get vector dimension from query vector
        dim = len(emb)
        
        # Build the base query
        if filter:
            # Extract filter conditions
            filter_conditions = []
            filter_params = []
            for key, value in filter.items():
                # DuckDB JSON_EXTRACT returns JSON values, so we need to compare with JSON representation
                if isinstance(value, str):
                    filter_conditions.append(f"JSON_EXTRACT_STRING(metadata, '$.{key}') = ?")
                else:
                    filter_conditions.append(f"JSON_EXTRACT(metadata, '$.{key}') = ?")
                filter_params.append(value)
            
            filter_clause = " AND " + " AND ".join(filter_conditions)
        else:
            filter_clause = ""
            filter_params = []
        
        # Use array_distance for similarity search with explicit cast to correct dimension
        query = f"""
            SELECT id, array_distance(vector, {_array_literal(emb)}::FLOAT[{dim}]) as distance
            FROM {name}
            WHERE 1=1{filter_clause}
            ORDER BY distance ASC
            LIMIT ?
        """
        
        params = filter_params + [top_k]
        result = self.conn.execute(query, params).fetchall()
        
        return result

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close()