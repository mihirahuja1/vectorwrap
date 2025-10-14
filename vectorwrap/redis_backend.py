# vectorwrap/redis_backend.py
from __future__ import annotations

from typing import Any, List, Tuple, Optional, Dict
import json
import numpy as np

try:
    import redis
    from redis.commands.search.field import VectorField, TagField, TextField
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
    from redis.commands.search.query import Query
except ImportError:
    redis = None  # type: ignore


class RedisBackend:
    """
    Backend for Redis with RediSearch vector similarity.

    Provides a simplified interface to Redis vector search capabilities,
    making it as easy to use as other vectorwrap backends.

    Requires Redis Stack or Redis with RediSearch module.

    Example:
        ```python
        from vectorwrap import VectorDB

        # Connect to Redis
        db = VectorDB("redis://localhost:6379/0")

        # Create collection with HNSW index
        db.create_collection("documents", dim=1536)

        # Upsert vectors
        db.upsert("documents", 1, [0.1, 0.2, ...], {"source": "doc1"})

        # Query with fast HNSW search
        results = db.query("documents", query_vector, top_k=10)
        ```
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        **kwargs: Any
    ) -> None:
        """
        Initialize Redis connection.

        Args:
            url: Redis connection URL (redis://[[username]:password@]host:port/db)
            **kwargs: Additional Redis connection parameters
        """
        if redis is None:
            raise RuntimeError(
                "redis and redis-py not installed. "
                "Install with: pip install 'vectorwrap[redis]'"
            )

        # Parse Redis URL
        # Format: redis://[[username]:password@]host:port/db
        self.client = redis.from_url(url, decode_responses=False, **kwargs)

        # Test connection
        try:
            self.client.ping()
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Redis: {e}")

        # Check if RediSearch is available
        try:
            modules = self.client.execute_command("MODULE LIST")
            has_search = any(b"search" in str(module).lower() for module in modules)
            if not has_search:
                raise RuntimeError(
                    "RediSearch module not found. "
                    "Please use Redis Stack or Redis with RediSearch module."
                )
        except Exception as e:
            raise RuntimeError(f"Failed to verify RediSearch module: {e}")

    def create_collection(
        self,
        name: str,
        dim: int,
        algorithm: str = "HNSW",
        distance_metric: str = "COSINE",
        m: int = 16,
        ef_construction: int = 200,
    ) -> None:
        """
        Create a collection with vector index.

        Args:
            name: Collection name (used as index name)
            dim: Vector dimension
            algorithm: Index algorithm ("HNSW" or "FLAT")
            distance_metric: Distance metric ("COSINE", "L2", "IP")
            m: HNSW M parameter (number of connections)
            ef_construction: HNSW ef_construction parameter
        """
        # Define index schema
        index_name = f"idx:{name}"

        # Vector field configuration
        vector_params = {
            "TYPE": "FLOAT32",
            "DIM": dim,
            "DISTANCE_METRIC": distance_metric,
        }

        if algorithm == "HNSW":
            vector_params["ALGORITHM"] = "HNSW"
            vector_params["M"] = m
            vector_params["EF_CONSTRUCTION"] = ef_construction
        elif algorithm == "FLAT":
            vector_params["ALGORITHM"] = "FLAT"
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        # Create schema with vector field and metadata field
        try:
            self.client.ft(index_name).create_index(
                fields=[
                    VectorField(
                        "vector",
                        algorithm,
                        vector_params
                    ),
                    TextField("metadata")
                ],
                definition=IndexDefinition(
                    prefix=[f"{name}:"],
                    index_type=IndexType.HASH
                )
            )
        except Exception as e:
            if "Index already exists" not in str(e):
                raise RuntimeError(f"Failed to create index: {e}")

    def upsert(
        self,
        name: str,
        _id: int,
        emb: List[float],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Insert or update a vector.

        Args:
            name: Collection name
            _id: Vector ID
            emb: Embedding vector
            meta: Optional metadata
        """
        # Convert vector to bytes
        vector_bytes = np.array(emb, dtype=np.float32).tobytes()

        # Prepare document
        key = f"{name}:{_id}"
        doc = {
            "vector": vector_bytes,
            "metadata": json.dumps(meta) if meta else "{}"
        }

        # Store in Redis
        self.client.hset(key, mapping=doc)

    def query(
        self,
        name: str,
        emb: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        ef_runtime: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Tuple[int, float]]:
        """
        Query for similar vectors using RediSearch.

        Args:
            name: Collection name
            emb: Query embedding
            top_k: Number of results
            filter: Optional metadata filter (basic string matching)
            ef_runtime: HNSW ef_runtime parameter (search accuracy)
            **kwargs: Additional search parameters

        Returns:
            List of (id, distance) tuples
        """
        index_name = f"idx:{name}"

        # Convert query vector to bytes
        query_vector = np.array(emb, dtype=np.float32).tobytes()

        # Build query
        # KNN query: *=>[KNN $K @vector_field $BLOB AS score]
        base_query = f"*=>[KNN {top_k} @vector $vec AS score]"

        # Add metadata filter if provided
        if filter:
            filter_parts = []
            for key, value in filter.items():
                # Simple JSON field matching
                filter_parts.append(f"@metadata:*{key}*{value}*")
            if filter_parts:
                base_query = "(" + " ".join(filter_parts) + ") => " + base_query.split("=>")[1]

        # Create query object
        q = Query(base_query).return_fields("score").sort_by("score").paging(0, top_k)

        # Set HNSW ef_runtime if provided
        params = {"vec": query_vector}
        if ef_runtime is not None:
            q = q.dialect(2)  # Required for PARAMS

        try:
            # Execute search
            results = self.client.ft(index_name).search(q, query_params=params)

            # Parse results
            output = []
            for doc in results.docs:
                # Extract ID from key (format: "collection:id")
                doc_id = int(doc.id.split(":")[-1])
                # Get distance score
                distance = float(doc.score) if hasattr(doc, 'score') else 0.0
                output.append((doc_id, distance))

            return output

        except Exception as e:
            raise RuntimeError(f"Query failed: {e}")

    def delete(self, name: str, ids: List[int]) -> None:
        """
        Delete vectors by IDs.

        Args:
            name: Collection name
            ids: List of vector IDs to delete
        """
        keys = [f"{name}:{_id}" for _id in ids]
        if keys:
            self.client.delete(*keys)

    def drop_collection(self, name: str) -> None:
        """
        Drop a collection and its index.

        Args:
            name: Collection name
        """
        index_name = f"idx:{name}"

        try:
            # Drop index
            self.client.ft(index_name).dropindex(delete_documents=True)
        except Exception as e:
            if "Unknown index name" not in str(e):
                raise RuntimeError(f"Failed to drop index: {e}")

    def get_collection_info(self, name: str) -> Dict[str, Any]:
        """
        Get collection information.

        Args:
            name: Collection name

        Returns:
            Dict with collection info
        """
        index_name = f"idx:{name}"

        try:
            info = self.client.ft(index_name).info()

            # Parse info dict
            info_dict = {}
            for i in range(0, len(info), 2):
                key = info[i].decode() if isinstance(info[i], bytes) else info[i]
                value = info[i + 1]
                info_dict[key] = value

            return {
                "name": name,
                "index_name": index_name,
                "num_docs": info_dict.get("num_docs", 0),
                "info": info_dict
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get collection info: {e}")

    def close(self) -> None:
        """Close connection to Redis."""
        if hasattr(self, 'client'):
            self.client.close()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass


# Convenience function for creating Redis backend
def create_redis_backend(
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
    password: Optional[str] = None,
    **kwargs: Any
) -> RedisBackend:
    """
    Create RedisBackend with explicit parameters.

    Args:
        host: Redis host
        port: Redis port
        db: Redis database number
        password: Optional password
        **kwargs: Additional connection parameters

    Returns:
        RedisBackend instance

    Example:
        ```python
        from vectorwrap.redis_backend import create_redis_backend

        # Local Redis
        db = create_redis_backend()

        # Remote Redis with auth
        db = create_redis_backend(
            host="redis.example.com",
            port=6379,
            password="secret"
        )
        ```
    """
    if password:
        url = f"redis://:{password}@{host}:{port}/{db}"
    else:
        url = f"redis://{host}:{port}/{db}"

    return RedisBackend(url, **kwargs)
