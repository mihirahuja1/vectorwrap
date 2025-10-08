# vectorwrap/clickhouse_backend.py
from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import clickhouse_connect
else:
    try:
        import clickhouse_connect  # type: ignore
    except ImportError:
        clickhouse_connect = None

import numpy as np


def _array_literal(v: list[float]) -> str:
    """Convert vector to ClickHouse Array literal format."""
    return "[" + ",".join(map(str, np.asarray(v, dtype=np.float32))) + "]"


class ClickHouseBackend:
    """Backend for ClickHouse with ANN (Approximate Nearest Neighbor) indexes."""

    def __init__(self, url: str) -> None:
        if clickhouse_connect is None:
            raise RuntimeError(
                "clickhouse-connect not installed. Install with: pip install 'vectorwrap[clickhouse]'"
            )

        # url pattern: clickhouse://[user[:password]@]host[:port][/database]
        # or clickhouse+native://...
        if url.startswith("clickhouse://") or url.startswith("clickhouse+native://"):
            # Parse the URL manually
            from urllib.parse import urlparse

            parsed = urlparse(url)

            # Extract connection parameters
            host = parsed.hostname or "localhost"
            port = parsed.port or 8123  # Default HTTP port
            username = parsed.username or "default"
            password = parsed.password or ""
            database = parsed.path.lstrip("/") if parsed.path else "default"

            # Use native protocol if specified
            if url.startswith("clickhouse+native://"):
                port = port if parsed.port else 9000
                self.conn = clickhouse_connect.get_client(
                    host=host,
                    port=port,
                    username=username,
                    password=password,
                    database=database
                )
            else:
                self.conn = clickhouse_connect.get_client(
                    host=host,
                    port=port,
                    username=username,
                    password=password,
                    database=database
                )
        else:
            raise ValueError("ClickHouse URL must start with 'clickhouse://' or 'clickhouse+native://'")

        # Enable experimental features if needed
        try:
            self.conn.command("SET allow_experimental_vector_similarity_index = 1")
        except Exception:
            # Older versions might not need this setting
            pass

    def create_collection(self, name: str, dim: int) -> None:
        """Create a table with vector column and ANN index."""
        # Create table with Array(Float32) for vectors
        # Using MergeTree engine with vector_similarity index
        self.conn.command(f"""
            CREATE TABLE IF NOT EXISTS {name} (
                id UInt64,
                vector Array(Float32),
                metadata String
            ) ENGINE = MergeTree()
            ORDER BY id
        """)

        # Create vector similarity index (HNSW) on the vector column
        # This is an experimental feature in ClickHouse
        try:
            self.conn.command(f"""
                ALTER TABLE {name}
                ADD INDEX IF NOT EXISTS {name}_vector_idx vector
                TYPE vector_similarity('hnsw', 'L2Distance', {dim})
                GRANULARITY 1000
            """)
        except Exception as e:
            # Index creation might fail if not supported or already exists
            # Continue without index - queries will still work but slower
            pass

    def upsert(self, name: str, _id: int, emb: list[float], meta: dict[str, Any] | None = None) -> None:
        """Insert or update a vector with optional metadata."""
        import json

        # ClickHouse doesn't support UPSERT directly, use ReplacingMergeTree pattern
        # For simplicity, we'll use the deduplication on insert approach
        # First, we need to convert metadata to JSON string
        metadata_json = json.dumps(meta) if meta is not None else "{}"

        # Delete existing record with this ID (if any)
        self.conn.command(f"ALTER TABLE {name} DELETE WHERE id = {_id}")

        # Insert new record
        vector_str = _array_literal(emb)
        self.conn.command(f"""
            INSERT INTO {name} (id, vector, metadata)
            VALUES ({_id}, {vector_str}, '{metadata_json}')
        """)

    def query(
        self,
        name: str,
        emb: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
        **kwargs: Any
    ) -> list[tuple[int, float]]:
        """Query for similar vectors using L2Distance."""

        vector_str = _array_literal(emb)

        # Build the filter clause if provided
        if filter:
            import json
            filter_conditions = []
            for key, value in filter.items():
                # Parse metadata JSON and filter
                if isinstance(value, str):
                    filter_conditions.append(f"JSONExtractString(metadata, '{key}') = '{value}'")
                elif isinstance(value, (int, float)):
                    filter_conditions.append(f"JSONExtractFloat(metadata, '{key}') = {value}")
                else:
                    filter_conditions.append(f"JSONExtractString(metadata, '{key}') = '{value}'")

            filter_clause = " AND " + " AND ".join(filter_conditions)
        else:
            filter_clause = ""

        # Use L2Distance for similarity search
        query = f"""
            SELECT id, L2Distance(vector, {vector_str}) as distance
            FROM {name}
            WHERE 1=1{filter_clause}
            ORDER BY distance ASC
            LIMIT {top_k}
        """

        result = self.conn.query(query)

        # Convert result to list of tuples
        return [(int(row[0]), float(row[1])) for row in result.result_rows]

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass
