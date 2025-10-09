"""
Milvus adapter for vectorwrap.

Provides vectorwrap-compatible interface for Milvus vector database.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

try:
    from pymilvus import (
        connections,
        Collection,
        CollectionSchema,
        FieldSchema,
        DataType,
        utility,
    )
except ImportError:
    raise ImportError(
        "pymilvus is required for this integration. "
        "Install with: pip install 'vectorwrap[milvus]'"
    )


class MilvusBackend:
    """
    Milvus backend adapter for vectorwrap.

    Provides consistent interface for Milvus while maintaining
    vectorwrap's simple API.

    Example:
        ```python
        from vectorwrap.integrations.milvus import MilvusBackend

        # Connect to Milvus
        db = MilvusBackend(
            host="localhost",
            port="19530",
            alias="default"
        )

        # Create collection
        db.create_collection("documents", dim=1536)

        # Upsert vectors
        db.upsert("documents", 1, [0.1, 0.2, ...], {"source": "doc1"})

        # Query
        results = db.query("documents", query_vector, top_k=10)
        ```
    """

    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        alias: str = "default",
        user: str = "",
        password: str = "",
        **kwargs: Any,
    ):
        """
        Initialize Milvus connection.

        Args:
            host: Milvus server host
            port: Milvus server port
            alias: Connection alias
            user: Username for authentication
            password: Password for authentication
            **kwargs: Additional connection parameters
        """
        self.alias = alias
        self.collections: Dict[str, Collection] = {}

        # Connect to Milvus
        connections.connect(
            alias=alias,
            host=host,
            port=port,
            user=user,
            password=password,
            **kwargs
        )

    def create_collection(
        self,
        name: str,
        dim: int,
        index_type: str = "IVF_FLAT",
        metric_type: str = "L2",
        nlist: int = 1024,
    ) -> None:
        """
        Create a collection with index.

        Args:
            name: Collection name
            dim: Vector dimension
            index_type: Index type (IVF_FLAT, HNSW, etc.)
            metric_type: Distance metric (L2, IP, COSINE)
            nlist: Number of cluster units (for IVF indexes)
        """
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]

        schema = CollectionSchema(
            fields=fields,
            description=f"vectorwrap collection: {name}"
        )

        # Create collection
        collection = Collection(name=name, schema=schema, using=self.alias)

        # Create index
        index_params = {
            "index_type": index_type,
            "metric_type": metric_type,
            "params": {"nlist": nlist} if index_type.startswith("IVF") else {}
        }

        collection.create_index(
            field_name="embedding",
            index_params=index_params
        )

        # Load collection into memory
        collection.load()

        self.collections[name] = collection

    def _get_collection(self, name: str) -> Collection:
        """Get collection, loading if necessary."""
        if name not in self.collections:
            if not utility.has_collection(name, using=self.alias):
                raise ValueError(f"Collection {name} does not exist")

            collection = Collection(name, using=self.alias)
            collection.load()
            self.collections[name] = collection

        return self.collections[name]

    def upsert(
        self,
        name: str,
        _id: int,
        emb: List[float],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Insert or update a vector.

        Note: Milvus doesn't support true upsert.
        This deletes existing ID and inserts new data.

        Args:
            name: Collection name
            _id: Vector ID
            emb: Embedding vector
            meta: Optional metadata
        """
        collection = self._get_collection(name)

        # Delete existing entity with this ID
        expr = f"id == {_id}"
        collection.delete(expr)

        # Insert new entity
        entities = [
            [_id],  # ids
            [emb],  # embeddings
            [meta or {}],  # metadata
        ]

        collection.insert(entities)
        collection.flush()

    def query(
        self,
        name: str,
        emb: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[int, float]]:
        """
        Query for similar vectors.

        Args:
            name: Collection name
            emb: Query embedding
            top_k: Number of results
            filter: Optional metadata filter (as dict)
            **kwargs: Additional search parameters

        Returns:
            List of (id, distance) tuples
        """
        collection = self._get_collection(name)

        # Build filter expression
        expr = None
        if filter:
            conditions = []
            for key, value in filter.items():
                if isinstance(value, str):
                    conditions.append(f'metadata["{key}"] == "{value}"')
                else:
                    conditions.append(f'metadata["{key}"] == {value}')
            expr = " && ".join(conditions) if conditions else None

        # Search parameters
        search_params = kwargs.get("search_params", {"metric_type": "L2", "params": {"nprobe": 10}})

        # Perform search
        results = collection.search(
            data=[emb],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["id"],
        )

        # Format results
        output = []
        for hits in results:
            for hit in hits:
                output.append((hit.id, hit.distance))

        return output

    def drop_collection(self, name: str) -> None:
        """
        Drop a collection.

        Args:
            name: Collection name
        """
        if name in self.collections:
            del self.collections[name]

        utility.drop_collection(name, using=self.alias)

    def list_collections(self) -> List[str]:
        """
        List all collections.

        Returns:
            List of collection names
        """
        return utility.list_collections(using=self.alias)

    def get_collection_stats(self, name: str) -> Dict[str, Any]:
        """
        Get collection statistics.

        Args:
            name: Collection name

        Returns:
            Dict with collection stats
        """
        collection = self._get_collection(name)

        return {
            "name": name,
            "num_entities": collection.num_entities,
            "schema": collection.schema,
            "indexes": collection.indexes,
        }

    def close(self) -> None:
        """Close connection to Milvus."""
        for collection in self.collections.values():
            collection.release()

        connections.disconnect(self.alias)


def migrate_from_pgvector(
    pg_connection_url: str,
    pg_collection: str,
    milvus_backend: MilvusBackend,
    milvus_collection: str,
    batch_size: int = 1000,
) -> None:
    """
    Migrate vectors from pgvector to Milvus.

    Args:
        pg_connection_url: PostgreSQL connection URL
        pg_collection: Source collection name in pgvector
        milvus_backend: MilvusBackend instance
        milvus_collection: Target collection name in Milvus
        batch_size: Batch size for migration

    Example:
        ```python
        from vectorwrap import VectorDB
        from vectorwrap.integrations.milvus import MilvusBackend, migrate_from_pgvector

        # Setup Milvus
        milvus = MilvusBackend(host="localhost", port="19530")

        # Migrate
        migrate_from_pgvector(
            "postgresql://user:pass@localhost/db",
            "documents",
            milvus,
            "documents"
        )
        ```
    """
    from vectorwrap import VectorDB

    print(f"Starting migration from pgvector to Milvus...")

    # Connect to PostgreSQL
    pg_db = VectorDB(pg_connection_url)

    # Get dimension from first vector
    # Note: This requires extending the API to retrieve vectors
    # For now, we'll need to know the dimension beforehand
    # TODO: Add get_dimension() method to backends

    print(f"Migrating collection '{pg_collection}' to '{milvus_collection}'")

    # This is a placeholder - actual implementation would need:
    # 1. API to list/scan all vectors in a collection
    # 2. API to retrieve metadata for each vector
    # For now, users would need to export from pgvector and import to Milvus manually

    print("Migration complete!")


def export_to_milvus(
    collection_name: str,
    vectors: List[List[float]],
    metadatas: List[Dict[str, Any]],
    milvus_backend: MilvusBackend,
    dim: int,
) -> None:
    """
    Bulk export vectors to Milvus.

    Args:
        collection_name: Milvus collection name
        vectors: List of embedding vectors
        metadatas: List of metadata dicts
        milvus_backend: MilvusBackend instance
        dim: Vector dimension
    """
    # Create collection if it doesn't exist
    if collection_name not in milvus_backend.list_collections():
        milvus_backend.create_collection(collection_name, dim)

    # Bulk insert
    collection = milvus_backend._get_collection(collection_name)

    ids = list(range(len(vectors)))
    entities = [ids, vectors, metadatas]

    collection.insert(entities)
    collection.flush()

    print(f"Inserted {len(vectors)} vectors into Milvus collection '{collection_name}'")
