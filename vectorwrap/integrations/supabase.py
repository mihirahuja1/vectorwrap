"""
Supabase pgvector helper for vectorwrap.

Simplified interface for using vectorwrap with Supabase's managed PostgreSQL + pgvector.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import os

from vectorwrap import VectorDB
from vectorwrap.pg_backend import PgBackend


class SupabaseVectorStore:
    """
    Convenience wrapper for Supabase pgvector integration.

    Provides easy setup and migration helpers for Supabase's managed PostgreSQL
    with pgvector extension.

    Example:
        ```python
        from vectorwrap.integrations.supabase import SupabaseVectorStore

        # Initialize from Supabase credentials
        store = SupabaseVectorStore.from_supabase_credentials(
            project_url="https://xxx.supabase.co",
            service_key="your-service-key",
            collection_name="documents"
        )

        # Or from environment variables
        store = SupabaseVectorStore.from_env()

        # Create schema
        store.create_collection("documents", dim=1536)

        # Bulk upsert
        store.bulk_upsert(
            "documents",
            vectors=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
            metadatas=[{"source": "doc1"}, {"source": "doc2"}]
        )

        # Query
        results = store.query("documents", query_vector, top_k=10)
        ```
    """

    def __init__(self, db: PgBackend):
        """
        Initialize SupabaseVectorStore.

        Args:
            db: PgBackend instance connected to Supabase
        """
        self.db = db

    @classmethod
    def from_supabase_credentials(
        cls,
        project_url: str,
        service_key: str,
        collection_name: Optional[str] = None,
    ) -> "SupabaseVectorStore":
        """
        Create from Supabase project credentials.

        Args:
            project_url: Supabase project URL (e.g., https://xxx.supabase.co)
            service_key: Supabase service role key
            collection_name: Optional default collection name

        Returns:
            SupabaseVectorStore instance
        """
        # Extract project reference from URL
        # Format: https://<project-ref>.supabase.co
        if "supabase.co" not in project_url:
            raise ValueError("Invalid Supabase project URL")

        project_ref = project_url.split("//")[1].split(".")[0]

        # Build PostgreSQL connection string
        # Supabase uses port 5432 for direct PostgreSQL connections
        connection_url = (
            f"postgresql://postgres:{service_key}@"
            f"db.{project_ref}.supabase.co:5432/postgres"
        )

        db = VectorDB(connection_url)
        return cls(db)

    @classmethod
    def from_env(
        cls,
        url_env: str = "SUPABASE_URL",
        key_env: str = "SUPABASE_SERVICE_KEY",
    ) -> "SupabaseVectorStore":
        """
        Create from environment variables.

        Args:
            url_env: Environment variable name for project URL
            key_env: Environment variable name for service key

        Returns:
            SupabaseVectorStore instance

        Raises:
            ValueError: If environment variables are not set
        """
        project_url = os.getenv(url_env)
        service_key = os.getenv(key_env)

        if not project_url or not service_key:
            raise ValueError(
                f"Environment variables {url_env} and {key_env} must be set"
            )

        return cls.from_supabase_credentials(project_url, service_key)

    def create_collection(
        self,
        name: str,
        dim: int,
        enable_rls: bool = False,
    ) -> None:
        """
        Create a collection with optional Row Level Security.

        Args:
            name: Collection name
            dim: Vector dimension
            enable_rls: Enable Supabase Row Level Security (default: False)
        """
        self.db.create_collection(name, dim)

        if enable_rls:
            # Enable RLS on the table
            try:
                with self.db.conn.cursor() as cur:
                    cur.execute(f"ALTER TABLE {name} ENABLE ROW LEVEL SECURITY")
            except Exception as e:
                print(f"Warning: Could not enable RLS: {e}")

    def upsert(
        self,
        collection: str,
        id: int,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Upsert a single vector.

        Args:
            collection: Collection name
            id: Vector ID
            vector: Embedding vector
            metadata: Optional metadata
        """
        self.db.upsert(collection, id, vector, metadata)

    def bulk_upsert(
        self,
        collection: str,
        vectors: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        start_id: int = 0,
    ) -> None:
        """
        Bulk upsert vectors for better performance.

        Args:
            collection: Collection name
            vectors: List of embedding vectors
            metadatas: Optional list of metadata dicts
            start_id: Starting ID for auto-generated IDs
        """
        metadatas = metadatas or [{} for _ in vectors]

        for i, (vector, metadata) in enumerate(zip(vectors, metadatas)):
            self.db.upsert(collection, start_id + i, vector, metadata)

    def query(
        self,
        collection: str,
        query_vector: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[tuple[int, float]]:
        """
        Query for similar vectors.

        Args:
            collection: Collection name
            query_vector: Query embedding
            top_k: Number of results
            filter: Optional metadata filter
            **kwargs: Additional query arguments

        Returns:
            List of (id, distance) tuples
        """
        return self.db.query(collection, query_vector, top_k, filter, **kwargs)

    def create_rls_policy(
        self,
        collection: str,
        policy_name: str,
        using_clause: str,
    ) -> None:
        """
        Create a Row Level Security policy.

        Args:
            collection: Collection name
            policy_name: Policy name
            using_clause: SQL USING clause for the policy

        Example:
            ```python
            store.create_rls_policy(
                "documents",
                "user_documents",
                "auth.uid() = (metadata->>'user_id')::uuid"
            )
            ```
        """
        with self.db.conn.cursor() as cur:
            cur.execute(
                f"CREATE POLICY {policy_name} ON {collection} "
                f"FOR ALL USING ({using_clause})"
            )

    def get_schema_sql(self, collection: str, dim: int) -> str:
        """
        Generate SQL schema for manual table creation.

        Useful for Supabase SQL Editor or migrations.

        Args:
            collection: Collection name
            dim: Vector dimension

        Returns:
            SQL schema string
        """
        return f"""
-- Create table for vector storage
CREATE TABLE IF NOT EXISTS {collection} (
    id BIGINT PRIMARY KEY,
    emb VECTOR({dim}),
    metadata JSONB
);

-- Create HNSW index for fast similarity search
CREATE INDEX IF NOT EXISTS {collection}_emb_idx
ON {collection} USING hnsw (emb vector_l2_ops);

-- Create GIN index for metadata filtering
CREATE INDEX IF NOT EXISTS {collection}_meta_idx
ON {collection} USING gin (metadata);

-- Optional: Enable Row Level Security
-- ALTER TABLE {collection} ENABLE ROW LEVEL SECURITY;

-- Optional: Create RLS policy (customize as needed)
-- CREATE POLICY user_policy ON {collection}
-- FOR ALL
-- USING (auth.uid() = (metadata->>'user_id')::uuid);
        """.strip()

    def export_to_csv(
        self,
        collection: str,
        output_file: str,
        limit: Optional[int] = None,
    ) -> None:
        """
        Export collection to CSV file.

        Args:
            collection: Collection name
            output_file: Output CSV file path
            limit: Optional limit on number of rows
        """
        limit_clause = f"LIMIT {limit}" if limit else ""

        with self.db.conn.cursor() as cur:
            query = f"""
            COPY (
                SELECT id, emb, metadata::text
                FROM {collection}
                {limit_clause}
            ) TO STDOUT WITH CSV HEADER
            """

            with open(output_file, "w") as f:
                cur.copy_expert(query, f)

    def get_collection_stats(self, collection: str) -> Dict[str, Any]:
        """
        Get statistics about a collection.

        Args:
            collection: Collection name

        Returns:
            Dict with collection statistics
        """
        with self.db.conn.cursor() as cur:
            # Get row count
            cur.execute(f"SELECT COUNT(*) FROM {collection}")
            row_count = cur.fetchone()[0]

            # Get table size
            cur.execute(
                f"SELECT pg_size_pretty(pg_total_relation_size('{collection}'))"
            )
            table_size = cur.fetchone()[0]

            # Get index sizes
            cur.execute(
                f"""
                SELECT indexname, pg_size_pretty(pg_relation_size(indexname::regclass))
                FROM pg_indexes
                WHERE tablename = '{collection}'
                """
            )
            indexes = cur.fetchall()

        return {
            "row_count": row_count,
            "table_size": table_size,
            "indexes": [{"name": idx[0], "size": idx[1]} for idx in indexes],
        }


def migrate_from_pinecone(
    pinecone_index: Any,
    supabase_store: SupabaseVectorStore,
    collection_name: str,
    batch_size: int = 100,
) -> None:
    """
    Migrate vectors from Pinecone to Supabase.

    Args:
        pinecone_index: Pinecone index object
        supabase_store: SupabaseVectorStore instance
        collection_name: Target collection name
        batch_size: Batch size for migration

    Example:
        ```python
        import pinecone
        from vectorwrap.integrations.supabase import (
            SupabaseVectorStore,
            migrate_from_pinecone
        )

        # Setup
        pinecone.init(api_key="...", environment="...")
        index = pinecone.Index("my-index")

        store = SupabaseVectorStore.from_env()

        # Migrate
        migrate_from_pinecone(index, store, "documents")
        ```
    """
    # Fetch all vectors from Pinecone
    # Note: This is a simplified version
    # For large datasets, implement pagination

    stats = pinecone_index.describe_index_stats()
    total_vectors = stats.get("total_vector_count", 0)

    print(f"Migrating {total_vectors} vectors from Pinecone to Supabase...")

    # Fetch and migrate in batches
    for i in range(0, total_vectors, batch_size):
        # Fetch batch from Pinecone
        results = pinecone_index.fetch(ids=[str(j) for j in range(i, min(i + batch_size, total_vectors))])

        vectors = []
        metadatas = []

        for vec_id, vec_data in results.get("vectors", {}).items():
            vectors.append(vec_data["values"])
            metadatas.append(vec_data.get("metadata", {}))

        # Bulk upsert to Supabase
        if vectors:
            supabase_store.bulk_upsert(
                collection_name,
                vectors=vectors,
                metadatas=metadatas,
                start_id=i
            )

        print(f"Migrated {min(i + batch_size, total_vectors)}/{total_vectors} vectors")

    print("Migration complete!")
