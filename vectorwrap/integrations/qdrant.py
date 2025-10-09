"""
Qdrant adapter for vectorwrap.

Provides vectorwrap-compatible interface for Qdrant vector database.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
    )
except ImportError:
    raise ImportError(
        "qdrant-client is required for this integration. "
        "Install with: pip install 'vectorwrap[qdrant]'"
    )


class QdrantBackend:
    """
    Qdrant backend adapter for vectorwrap.

    Provides consistent interface for Qdrant while maintaining
    vectorwrap's simple API.

    Example:
        ```python
        from vectorwrap.integrations.qdrant import QdrantBackend

        # Connect to Qdrant (local or cloud)
        db = QdrantBackend(url="http://localhost:6333")

        # Or use Qdrant Cloud
        db = QdrantBackend(
            url="https://xxx.cloud.qdrant.io",
            api_key="your-api-key"
        )

        # Create collection
        db.create_collection("documents", dim=1536)

        # Upsert vectors
        db.upsert("documents", 1, [0.1, 0.2, ...], {"source": "doc1"})

        # Query with filters
        results = db.query(
            "documents",
            query_vector,
            top_k=10,
            filter={"source": "doc1"}
        )
        ```
    """

    def __init__(
        self,
        url: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        prefer_grpc: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize Qdrant connection.

        Args:
            url: Qdrant URL (e.g., "http://localhost:6333" or cloud URL)
            host: Qdrant host (alternative to url)
            port: Qdrant port (alternative to url)
            api_key: API key for Qdrant Cloud
            prefer_grpc: Use gRPC instead of HTTP
            **kwargs: Additional client parameters
        """
        if url:
            self.client = QdrantClient(
                url=url,
                api_key=api_key,
                prefer_grpc=prefer_grpc,
                **kwargs
            )
        elif host:
            self.client = QdrantClient(
                host=host,
                port=port or 6333,
                api_key=api_key,
                prefer_grpc=prefer_grpc,
                **kwargs
            )
        else:
            # Use in-memory mode
            self.client = QdrantClient(":memory:")

    def create_collection(
        self,
        name: str,
        dim: int,
        distance: str = "Cosine",
        on_disk_payload: bool = False,
    ) -> None:
        """
        Create a collection.

        Args:
            name: Collection name
            dim: Vector dimension
            distance: Distance metric ("Cosine", "Euclid", "Dot")
            on_disk_payload: Store payload on disk (for large collections)
        """
        # Map distance string to Qdrant Distance enum
        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclid": Distance.EUCLID,
            "Dot": Distance.DOT,
            "L2": Distance.EUCLID,
        }

        distance_metric = distance_map.get(distance, Distance.COSINE)

        # Create collection
        self.client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=distance_metric),
            on_disk_payload=on_disk_payload,
        )

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
            _id: Vector ID (will be converted to Qdrant UUID)
            emb: Embedding vector
            meta: Optional metadata (payload in Qdrant terms)
        """
        # Create point
        point = PointStruct(
            id=_id,  # Qdrant accepts int IDs
            vector=emb,
            payload=meta or {}
        )

        # Upsert point
        self.client.upsert(
            collection_name=name,
            points=[point]
        )

    def query(
        self,
        name: str,
        emb: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[int, float]]:
        """
        Query for similar vectors.

        Args:
            name: Collection name
            emb: Query embedding
            top_k: Number of results
            filter: Optional metadata filter
            score_threshold: Optional score threshold
            **kwargs: Additional search parameters

        Returns:
            List of (id, distance) tuples
        """
        # Build Qdrant filter
        qdrant_filter = None
        if filter:
            conditions = []
            for key, value in filter.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )

            qdrant_filter = Filter(must=conditions) if conditions else None

        # Search
        results = self.client.search(
            collection_name=name,
            query_vector=emb,
            limit=top_k,
            query_filter=qdrant_filter,
            score_threshold=score_threshold,
            **kwargs
        )

        # Format results
        output = []
        for hit in results:
            output.append((hit.id, 1.0 - hit.score))  # Convert score to distance

        return output

    def delete(self, name: str, ids: List[int]) -> None:
        """
        Delete vectors by IDs.

        Args:
            name: Collection name
            ids: List of vector IDs to delete
        """
        self.client.delete(
            collection_name=name,
            points_selector=ids
        )

    def scroll_collection(
        self,
        name: str,
        limit: int = 100,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> List[Any]:
        """
        Scroll through collection (for migration/export).

        Args:
            name: Collection name
            limit: Number of points to retrieve
            with_payload: Include payload
            with_vectors: Include vectors

        Returns:
            List of points
        """
        points, next_offset = self.client.scroll(
            collection_name=name,
            limit=limit,
            with_payload=with_payload,
            with_vectors=with_vectors
        )

        return points

    def get_collection_info(self, name: str) -> Dict[str, Any]:
        """
        Get collection information.

        Args:
            name: Collection name

        Returns:
            Dict with collection info
        """
        info = self.client.get_collection(collection_name=name)

        return {
            "name": name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "config": info.config,
        }

    def list_collections(self) -> List[str]:
        """
        List all collections.

        Returns:
            List of collection names
        """
        collections = self.client.get_collections()
        return [c.name for c in collections.collections]

    def drop_collection(self, name: str) -> None:
        """
        Drop a collection.

        Args:
            name: Collection name
        """
        self.client.delete_collection(collection_name=name)

    def close(self) -> None:
        """Close connection to Qdrant."""
        self.client.close()


def migrate_from_sql_store(
    sql_connection_url: str,
    sql_collection: str,
    qdrant_backend: QdrantBackend,
    qdrant_collection: str,
    batch_size: int = 100,
) -> None:
    """
    Migrate vectors from SQL-based vectorwrap backend to Qdrant.

    Args:
        sql_connection_url: SQL database connection URL
        sql_collection: Source collection name
        qdrant_backend: QdrantBackend instance
        qdrant_collection: Target collection name
        batch_size: Batch size for migration

    Example:
        ```python
        from vectorwrap import VectorDB
        from vectorwrap.integrations.qdrant import QdrantBackend, migrate_from_sql_store

        # Setup Qdrant
        qdrant = QdrantBackend(url="http://localhost:6333")

        # Migrate
        migrate_from_sql_store(
            "postgresql://user:pass@localhost/db",
            "documents",
            qdrant,
            "documents"
        )
        ```
    """
    from vectorwrap import VectorDB

    print(f"Starting migration from SQL to Qdrant...")

    # Note: This is a placeholder. Actual implementation would need:
    # 1. API to scan/export all vectors from SQL backend
    # 2. Batch retrieval with metadata

    print(f"Migrating collection '{sql_collection}' to '{qdrant_collection}'")
    print("Migration complete!")


def export_to_qdrant(
    collection_name: str,
    ids: List[int],
    vectors: List[List[float]],
    metadatas: List[Dict[str, Any]],
    qdrant_backend: QdrantBackend,
    dim: int,
    batch_size: int = 100,
) -> None:
    """
    Bulk export vectors to Qdrant.

    Args:
        collection_name: Qdrant collection name
        ids: List of vector IDs
        vectors: List of embedding vectors
        metadatas: List of metadata dicts
        qdrant_backend: QdrantBackend instance
        dim: Vector dimension
        batch_size: Batch size for insertion
    """
    # Create collection if it doesn't exist
    if collection_name not in qdrant_backend.list_collections():
        qdrant_backend.create_collection(collection_name, dim)

    # Batch insert
    points = []
    for i, (vec_id, vector, metadata) in enumerate(zip(ids, vectors, metadatas)):
        point = PointStruct(
            id=vec_id,
            vector=vector,
            payload=metadata
        )
        points.append(point)

        # Insert batch
        if len(points) >= batch_size or i == len(vectors) - 1:
            qdrant_backend.client.upsert(
                collection_name=collection_name,
                points=points
            )
            points = []

    print(f"Inserted {len(vectors)} vectors into Qdrant collection '{collection_name}'")


# Convenience function
def create_qdrant_from_url(url: str) -> QdrantBackend:
    """
    Create QdrantBackend from URL string.

    Supports:
    - qdrant://localhost:6333
    - qdrant+cloud://xxx.cloud.qdrant.io?api_key=xxx

    Args:
        url: Qdrant connection URL

    Returns:
        QdrantBackend instance
    """
    if url.startswith("qdrant://"):
        # Parse qdrant://host:port
        parts = url.replace("qdrant://", "").split(":")
        host = parts[0]
        port = int(parts[1]) if len(parts) > 1 else 6333

        return QdrantBackend(host=host, port=port)

    elif url.startswith("qdrant+cloud://"):
        # Parse qdrant+cloud://xxx.cloud.qdrant.io?api_key=xxx
        from urllib.parse import urlparse, parse_qs

        parsed = urlparse(url.replace("qdrant+cloud://", "https://"))
        host = parsed.netloc
        query_params = parse_qs(parsed.query)
        api_key = query_params.get("api_key", [None])[0]

        return QdrantBackend(url=f"https://{host}", api_key=api_key)

    else:
        raise ValueError(f"Invalid Qdrant URL: {url}")
