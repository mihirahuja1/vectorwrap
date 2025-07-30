# vectorwrap/sqlite_backend.py
from __future__ import annotations

import json
from typing import Any
try:
    import pysqlite3 as sqlite3  # type: ignore
except ImportError:
    import sqlite3

import numpy as np


def _lit(v: list[float]) -> str:
    """Convert vector to SQLite-VSS array literal format."""
    return "[" + ",".join(map(str, np.asarray(v, dtype=float))) + "]"


class SQLiteBackend:
    """Backend for local prototype databases using sqlite-vss (HNSW)."""

    def __init__(self, url: str) -> None:
        # url pattern: sqlite:///absolute/path.db  or  sqlite:///:memory:
        path = url.replace("sqlite:///", "", 1)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        try:
            self.conn.enable_load_extension(True)
        except AttributeError:
            # SQLite was compiled without extension loading support
            pass
        
        # load vss extension (bundled with package)
        try:
            import sqlite_vss
            sqlite_vss.load(conn=self.conn)
        except ImportError:
            raise RuntimeError(
                "sqlite-vss not installed. Install with: pip install 'vectorwrap[sqlite]'"
            )
        except AttributeError:
            # SQLite was compiled without extension loading support
            raise RuntimeError(
                "SQLite was compiled without extension loading support. "
                "Install with: pip install 'vectorwrap[sqlite]'"
            )

    def create_collection(self, name: str, dim: int) -> None:
        """Create a new collection with VSS virtual table and metadata table."""
        cur = self.conn.cursor()
        # Create VSS virtual table for vector search
        cur.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {name} "
            f"USING vss0(emb({dim}));"
        )
        # Create companion table for metadata
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {name}_metadata ("
            f"id INTEGER PRIMARY KEY, "
            f"metadata TEXT"
            f");"
        )
        cur.close()

    def upsert(self, name: str, _id: int, emb: list[float], meta: dict[str, Any] | None = None) -> None:
        """Insert or update a vector with optional metadata."""
        cur = self.conn.cursor()
        
        # For VSS tables, we need to DELETE + INSERT instead of REPLACE
        # Check if the record exists first
        cur.execute(f"SELECT 1 FROM {name} WHERE rowid = ? LIMIT 1;", (_id,))
        exists = cur.fetchone() is not None
        
        if exists:
            # Delete existing record
            cur.execute(f"DELETE FROM {name} WHERE rowid = ?;", (_id,))
        
        # Insert new/updated vector
        cur.execute(f"INSERT INTO {name}(rowid, emb) VALUES (?, ?);", (_id, _lit(emb)))
        
        # Insert/update metadata in companion table if provided
        if meta is not None:
            metadata_json = json.dumps(meta)
            cur.execute(f"REPLACE INTO {name}_metadata(id, metadata) VALUES (?, ?);", (_id, metadata_json))
        
        cur.close()
        self.conn.commit()

    def query(
        self, 
        name: str, 
        emb: list[float], 
        top_k: int = 5, 
        filter: dict[str, Any] | None = None, 
        **_: Any
    ) -> list[tuple[int, float]]:
        """Query for similar vectors. Uses adaptive oversampling for filtering."""
        cur = self.conn.cursor()
        
        if filter:
            # Adaptive oversampling: fetch more results, then filter in Python
            oversample_factor = 3  # Fetch 3x more results for filtering
            fetch_limit = max(top_k * oversample_factor, 50)  # At least 50 for small top_k
            
            # Get vector results with metadata using JOIN
            cur.execute(
                f"SELECT v.rowid, v.distance, m.metadata "
                f"FROM (SELECT rowid, distance FROM {name} WHERE vss_search(emb, ?) LIMIT {fetch_limit}) v "
                f"LEFT JOIN {name}_metadata m ON v.rowid = m.id",
                (_lit(emb),),
            )
            rows = cur.fetchall()
            
            # Filter results in Python
            filtered_results = []
            for row_id, distance, metadata_json in rows:
                if metadata_json:
                    try:
                        metadata = json.loads(metadata_json)
                        # Check if metadata matches all filter criteria
                        match = all(
                            metadata.get(key) == value 
                            for key, value in filter.items()
                        )
                        if match:
                            filtered_results.append((row_id, distance))
                    except (json.JSONDecodeError, TypeError):
                        # Skip rows with invalid metadata
                        continue
                
                # Stop if we have enough results
                if len(filtered_results) >= top_k:
                    break
            
            cur.close()
            return filtered_results[:top_k]
        else:
            # No filtering - use direct query
            cur.execute(
                f"SELECT rowid, distance "
                f"FROM {name} WHERE vss_search(emb, ?) LIMIT {top_k};",
                (_lit(emb),),
            )
            rows = cur.fetchall()
            cur.close()
            return list(rows)  # Ensure proper type
