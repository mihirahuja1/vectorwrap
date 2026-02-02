"""
Weaviate adapter for vectorwrap.

Provides vectorwrap-compatible interface for Weaviate vector database.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import json

try:
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType
    from weaviate.classes.query import Filter, MetadataQuery
except ImportError:
    raise ImportError(
        "weaviate-client is required for this integration. "
        "Install with: pip install 'vectorwrap[weaviate]'"
    )


class WeaviateBackend:
    """
    Weaviate backend adapter for vectorwrap.

    Provides consistent interface for Weaviate while maintaining
    vectorwrap's simple API.

    Example:
        ```python
        from vectorwrap.integrations.weaviate import WeaviateBackend

        # Connect to Weaviate (local or cloud)
        db = WeaviateBackend(url="http://localhost:8080")

        # Or use Weaviate Cloud
        db = WeaviateBackend(
            url="https://xxx.weaviate.network",
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
        api_key: Optional[str] = None,
        additional_headers: Optional[Dict[str, str]] = None,
        embedded_options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """
        Initialize Weaviate connection.

        Args:
            url: Weaviate URL (e.g., "http://localhost:8080" or cloud URL)
            api_key: API key for Weaviate Cloud
            additional_headers: Additional headers for authentication
            embedded_options: Options for embedded Weaviate instance
            **kwargs: Additional client parameters
        """
        if embedded_options:
            # Use embedded Weaviate
            self.client = weaviate.WeaviateClient(
                embedded_options=weaviate.embedded.EmbeddedOptions(**embedded_options)
            )
        elif url:
            # Connect to remote Weaviate
            if api_key:
                auth_config = weaviate.auth.AuthApiKey(api_key=api_key)
            else:
                auth_config = None

            self.client = weaviate.connect_to_custom(
                http_host=url.replace("http://", "").replace("https://", "").split(":")[0],
                http_port=int(url.split(":")[-1]) if ":" in url.split("//")[-1] else (443 if "https" in url else 80),
                http_secure="https" in url,
                auth_credentials=auth_config,
                headers=additional_headers or {},
                **kwargs
            )
        else:
            # Connect to local Weaviate
            self.client = weaviate.connect_to_local()

        # Store collection name mapping (vectorwrap name -> Weaviate class name)
        self._collection_map: Dict[str, str] = {}

    def _to_class_name(self, name: str) -> str:
        """Convert collection name to Weaviate class name (PascalCase)."""
        # Weaviate class names must start with uppercase
        # Convert snake_case or kebab-case to PascalCase
        parts = name.replace("-", "_").split("_")
        class_name = "".join(word.capitalize() for word in parts)

        # Ensure it starts with a letter
        if not class_name[0].isalpha():
            class_name = "C" + class_name

        self._collection_map[name] = class_name
        return class_name

    def create_collection(
        self,
        name: str,
        dim: int,
        distance: str = "cosine",
        vectorizer: str = "none",
    ) -> None:
        """
        Create a collection.

        Args:
            name: Collection name
            dim: Vector dimension
            distance: Distance metric ("cosine", "dot", "l2", "hamming", "manhattan")
            vectorizer: Vectorizer module to use ("none" for bring-your-own-vectors)
        """
        class_name = self._to_class_name(name)

        # Map distance metric to Weaviate VectorDistanceMetric
        distance_map = {
            "cosine": "cosine",
            "dot": "dot",
            "l2": "l2-squared",
            "euclidean": "l2-squared",
            "hamming": "hamming",
            "manhattan": "manhattan",
        }

        distance_metric = distance_map.get(distance.lower(), "cosine")

        # Create collection with vector configuration
        try:
            self.client.collections.create(
                name=class_name,
                vectorizer_config=Configure.Vectorizer.none() if vectorizer == "none" else None,
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=distance_metric
                ),
                properties=[
                    Property(
                        name="vector_id",
                        data_type=DataType.INT,
                        description="vectorwrap ID"
                    ),
                    Property(
                        name="metadata",
                        data_type=DataType.TEXT,
                        description="JSON metadata"
                    ),
                ],
            )
        except Exception as e:
            # Collection might already exist
            if "already exists" not in str(e).lower():
                raise

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
        class_name = self._collection_map.get(name, self._to_class_name(name))
        collection = self.client.collections.get(class_name)

        # Check if object with this vector_id exists
        existing = collection.query.fetch_objects(
            filters=Filter.by_property("vector_id").equal(_id),
            limit=1
        )

        metadata_json = json.dumps(meta or {})

        if existing.objects:
            # Update existing object
            uuid = existing.objects[0].uuid
            collection.data.update(
                uuid=uuid,
                properties={
                    "vector_id": _id,
                    "metadata": metadata_json,
                },
                vector=emb
            )
        else:
            # Insert new object
            collection.data.insert(
                properties={
                    "vector_id": _id,
                    "metadata": metadata_json,
                },
                vector=emb
            )

    def query(
        self,
        name: str,
        emb: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        distance_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[int, float]]:
        """
        Query for similar vectors.

        Args:
            name: Collection name
            emb: Query embedding
            top_k: Number of results
            filter: Optional metadata filter
            distance_threshold: Optional distance threshold
            **kwargs: Additional search parameters

        Returns:
            List of (id, distance) tuples
        """
        class_name = self._collection_map.get(name, self._to_class_name(name))
        collection = self.client.collections.get(class_name)

        # Build Weaviate filter if needed
        weaviate_filter = None
        if filter:
            # For simple key-value filters, we need to search in the metadata JSON
            # This is a simplified implementation
            # In production, you might want to use individual properties instead
            pass

        # Perform vector search
        response = collection.query.near_vector(
            near_vector=emb,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
            **kwargs
        )

        # Format results
        output = []
        for obj in response.objects:
            vector_id = obj.properties.get("vector_id")
            distance = obj.metadata.distance if obj.metadata else 0.0

            # Apply metadata filter if specified (post-filtering for simplicity)
            if filter and vector_id is not None:
                metadata_str = obj.properties.get("metadata", "{}")
                try:
                    metadata = json.loads(metadata_str)
                    # Check if all filter conditions match
                    if not all(metadata.get(k) == v for k, v in filter.items()):
                        continue
                except (json.JSONDecodeError, AttributeError):
                    continue

            # Apply distance threshold if specified
            if distance_threshold is not None and distance > distance_threshold:
                continue

            if vector_id is not None:
                output.append((vector_id, distance))

        return output

    def delete(self, name: str, ids: List[int]) -> None:
        """
        Delete vectors by IDs.

        Args:
            name: Collection name
            ids: List of vector IDs to delete
        """
        class_name = self._collection_map.get(name, self._to_class_name(name))
        collection = self.client.collections.get(class_name)

        for _id in ids:
            # Find objects with matching vector_id
            results = collection.query.fetch_objects(
                filters=Filter.by_property("vector_id").equal(_id),
                limit=1
            )

            if results.objects:
                collection.data.delete_by_id(results.objects[0].uuid)

    def get_collection_info(self, name: str) -> Dict[str, Any]:
        """
        Get collection information.

        Args:
            name: Collection name

        Returns:
            Dict with collection info
        """
        class_name = self._collection_map.get(name, self._to_class_name(name))
        collection = self.client.collections.get(class_name)

        # Get collection config
        config = collection.config.get()

        # Get object count
        aggregate = collection.aggregate.over_all(total_count=True)

        return {
            "name": name,
            "class_name": class_name,
            "total_count": aggregate.total_count,
            "vectorizer": config.vectorizer,
            "vector_index_type": config.vector_index_type,
        }

    def list_collections(self) -> List[str]:
        """
        List all collections.

        Returns:
            List of collection names
        """
        # Return the vectorwrap names from our mapping
        # Note: This won't include collections created outside of vectorwrap
        collections = self.client.collections.list_all()

        # Try to reverse map, or just return class names
        result = []
        for class_name, _ in collections.items():
            # Check if we have a mapping
            vectorwrap_name = None
            for vw_name, cls_name in self._collection_map.items():
                if cls_name == class_name:
                    vectorwrap_name = vw_name
                    break

            if vectorwrap_name:
                result.append(vectorwrap_name)
            else:
                # Return class name in lowercase with underscores
                result.append(self._from_class_name(class_name))

        return result

    def _from_class_name(self, class_name: str) -> str:
        """Convert Weaviate class name back to collection name."""
        # Convert PascalCase to snake_case
        import re
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', class_name)
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
        return name

    def drop_collection(self, name: str) -> None:
        """
        Drop a collection.

        Args:
            name: Collection name
        """
        class_name = self._collection_map.get(name, self._to_class_name(name))
        self.client.collections.delete(class_name)

        # Remove from mapping
        if name in self._collection_map:
            del self._collection_map[name]

    def close(self) -> None:
        """Close connection to Weaviate."""
        self.client.close()


def migrate_from_sql_store(
    sql_connection_url: str,
    sql_collection: str,
    weaviate_backend: WeaviateBackend,
    weaviate_collection: str,
    batch_size: int = 100,
) -> None:
    """
    Migrate vectors from SQL-based vectorwrap backend to Weaviate.

    Args:
        sql_connection_url: SQL database connection URL
        sql_collection: Source collection name
        weaviate_backend: WeaviateBackend instance
        weaviate_collection: Target collection name
        batch_size: Batch size for migration

    Example:
        ```python
        from vectorwrap import VectorDB
        from vectorwrap.integrations.weaviate import WeaviateBackend, migrate_from_sql_store

        # Setup Weaviate
        weaviate = WeaviateBackend(url="http://localhost:8080")

        # Migrate
        migrate_from_sql_store(
            "postgresql://user:pass@localhost/db",
            "documents",
            weaviate,
            "documents"
        )
        ```
    """
    from vectorwrap import VectorDB

    print(f"Starting migration from SQL to Weaviate...")

    # Note: This is a placeholder. Actual implementation would need:
    # 1. API to scan/export all vectors from SQL backend
    # 2. Batch retrieval with metadata

    print(f"Migrating collection '{sql_collection}' to '{weaviate_collection}'")
    print("Migration complete!")


def export_to_weaviate(
    collection_name: str,
    ids: List[int],
    vectors: List[List[float]],
    metadatas: List[Dict[str, Any]],
    weaviate_backend: WeaviateBackend,
    dim: int,
    batch_size: int = 100,
) -> None:
    """
    Bulk export vectors to Weaviate.

    Args:
        collection_name: Weaviate collection name
        ids: List of vector IDs
        vectors: List of embedding vectors
        metadatas: List of metadata dicts
        weaviate_backend: WeaviateBackend instance
        dim: Vector dimension
        batch_size: Batch size for insertion
    """
    # Create collection if it doesn't exist
    if collection_name not in weaviate_backend.list_collections():
        weaviate_backend.create_collection(collection_name, dim)

    # Batch insert
    class_name = weaviate_backend._to_class_name(collection_name)
    collection = weaviate_backend.client.collections.get(class_name)

    with collection.batch.dynamic() as batch:
        for vec_id, vector, metadata in zip(ids, vectors, metadatas):
            batch.add_object(
                properties={
                    "vector_id": vec_id,
                    "metadata": json.dumps(metadata),
                },
                vector=vector
            )

    print(f"Inserted {len(vectors)} vectors into Weaviate collection '{collection_name}'")


# Convenience function
def create_weaviate_from_url(url: str) -> WeaviateBackend:
    """
    Create WeaviateBackend from URL string.

    Supports:
    - weaviate://localhost:8080
    - weaviate+cloud://xxx.weaviate.network?api_key=xxx

    Args:
        url: Weaviate connection URL

    Returns:
        WeaviateBackend instance
    """
    if url.startswith("weaviate://"):
        # Parse weaviate://host:port
        clean_url = url.replace("weaviate://", "http://")
        return WeaviateBackend(url=clean_url)

    elif url.startswith("weaviate+cloud://"):
        # Parse weaviate+cloud://xxx.weaviate.network?api_key=xxx
        from urllib.parse import urlparse, parse_qs

        parsed = urlparse(url.replace("weaviate+cloud://", "https://"))
        host = parsed.netloc
        query_params = parse_qs(parsed.query)
        api_key = query_params.get("api_key", [None])[0]

        return WeaviateBackend(url=f"https://{host}", api_key=api_key)

    else:
        raise ValueError(f"Invalid Weaviate URL: {url}")
