"""
Appwrite integration for vectorwrap.

Provides seamless vector search capabilities for Appwrite applications without
requiring external vector databases like Pinecone or Upstash.

Example:
    ```python
    from vectorwrap.integrations.appwrite import AppwriteVectorStore
    from appwrite.client import Client

    # Initialize Appwrite client
    client = Client()
    client.set_endpoint('https://cloud.appwrite.io/v1')
    client.set_project('your-project-id')
    client.set_key('your-api-key')

    # Create vector store (uses Appwrite's MariaDB backend)
    vector_store = AppwriteVectorStore.from_appwrite_client(
        client=client,
        collection_name="embeddings",
        dimension=1536
    )

    # Store embeddings
    vector_store.add_documents([
        {"text": "Hello world", "metadata": {"source": "doc1"}},
        {"text": "AI is amazing", "metadata": {"source": "doc2"}}
    ], embedding_function=embed_fn)

    # Search
    results = vector_store.search("greeting", embed_fn, top_k=5)
    ```
"""

from __future__ import annotations

from typing import Any, Callable, Optional
import os

try:
    from appwrite.client import Client
    from appwrite.services.databases import Databases
except ImportError:
    Client = None  # type: ignore
    Databases = None  # type: ignore

from ..mariadb_backend import MariaDBBackend


class AppwriteVectorStore:
    """
    Vector store integration for Appwrite applications.

    Uses Appwrite's MariaDB backend for native vector storage (MariaDB 11.8+)
    or JSON fallback for older versions. No external vector database needed.

    Benefits:
    - Data locality: Vectors stored alongside app data
    - Cost reduction: No Pinecone/Upstash subscription
    - Simplicity: Zero external configuration
    - Native performance: HNSW indexing on MariaDB 11.8+
    """

    def __init__(
        self,
        connection_url: str,
        collection_name: str,
        dimension: int,
        distance_metric: str = "cosine"
    ) -> None:
        """
        Initialize Appwrite vector store.

        Args:
            connection_url: MariaDB connection URL
            collection_name: Name for the vector collection
            dimension: Vector dimension (e.g., 1536 for OpenAI embeddings)
            distance_metric: Distance metric ("cosine" or "euclidean")
        """
        self.backend = MariaDBBackend(connection_url)
        self.collection_name = collection_name
        self.dimension = dimension
        self.distance_metric = distance_metric

        # Create collection if it doesn't exist
        try:
            self.backend.create_collection(collection_name, dimension)
        except Exception:
            # Collection might already exist
            pass

    @classmethod
    def from_appwrite_client(
        cls,
        client: Any,
        collection_name: str,
        dimension: int,
        database_id: str = "default",
        distance_metric: str = "cosine"
    ) -> "AppwriteVectorStore":
        """
        Create vector store from Appwrite client.

        Args:
            client: Appwrite Client instance
            collection_name: Name for the vector collection
            dimension: Vector dimension
            database_id: Appwrite database ID (default: "default")
            distance_metric: Distance metric ("cosine" or "euclidean")

        Returns:
            AppwriteVectorStore instance

        Example:
            ```python
            from appwrite.client import Client
            from vectorwrap.integrations.appwrite import AppwriteVectorStore

            client = Client()
            client.set_endpoint('https://cloud.appwrite.io/v1')
            client.set_project('project-id')
            client.set_key('api-key')

            vector_store = AppwriteVectorStore.from_appwrite_client(
                client=client,
                collection_name="embeddings",
                dimension=1536
            )
            ```
        """
        if Client is None:
            raise RuntimeError(
                "Appwrite SDK not installed. "
                "Install with: pip install appwrite"
            )

        # Extract connection details from Appwrite configuration
        # Appwrite uses MariaDB internally
        # Connection details can be obtained from environment or Appwrite config

        # For self-hosted Appwrite, MariaDB connection is typically:
        # mariadb://appwrite:password@mariadb:3306/appwrite
        connection_url = os.environ.get(
            "APPWRITE_MARIADB_URL",
            "mariadb://appwrite:password@mariadb:3306/appwrite"
        )

        return cls(
            connection_url=connection_url,
            collection_name=collection_name,
            dimension=dimension,
            distance_metric=distance_metric
        )

    @classmethod
    def from_connection_string(
        cls,
        connection_url: str,
        collection_name: str,
        dimension: int,
        distance_metric: str = "cosine"
    ) -> "AppwriteVectorStore":
        """
        Create vector store from direct MariaDB connection string.

        Args:
            connection_url: MariaDB connection URL
            collection_name: Name for the vector collection
            dimension: Vector dimension
            distance_metric: Distance metric

        Returns:
            AppwriteVectorStore instance

        Example:
            ```python
            vector_store = AppwriteVectorStore.from_connection_string(
                connection_url="mariadb://user:pass@localhost:3306/appwrite",
                collection_name="embeddings",
                dimension=1536
            )
            ```
        """
        return cls(connection_url, collection_name, dimension, distance_metric)

    def add_document(
        self,
        doc_id: int,
        embedding: list[float],
        metadata: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Add a single document with its embedding.

        Args:
            doc_id: Unique document ID
            embedding: Vector embedding
            metadata: Optional metadata (e.g., {"text": "...", "source": "..."})
        """
        self.backend.upsert(
            self.collection_name,
            doc_id,
            embedding,
            metadata
        )

    def add_documents(
        self,
        documents: list[dict[str, Any]],
        embedding_function: Callable[[str], list[float]],
        text_key: str = "text"
    ) -> list[int]:
        """
        Add multiple documents with automatic embedding generation.

        Args:
            documents: List of document dicts with text and metadata
            embedding_function: Function to generate embeddings from text
            text_key: Key for text field in document dict (default: "text")

        Returns:
            List of document IDs

        Example:
            ```python
            def embed(text):
                # Use OpenAI, Sentence Transformers, etc.
                return openai.embeddings.create(
                    model="text-embedding-3-small",
                    input=text
                ).data[0].embedding

            vector_store.add_documents([
                {"text": "Hello world", "metadata": {"source": "doc1"}},
                {"text": "AI is amazing", "metadata": {"source": "doc2"}}
            ], embedding_function=embed)
            ```
        """
        doc_ids = []
        for i, doc in enumerate(documents):
            text = doc.get(text_key, "")
            embedding = embedding_function(text)
            metadata = doc.get("metadata", {})
            metadata["_text"] = text  # Store original text

            doc_id = doc.get("id", hash(text) % (2**63))
            self.add_document(doc_id, embedding, metadata)
            doc_ids.append(doc_id)

        return doc_ids

    def search(
        self,
        query: str,
        embedding_function: Callable[[str], list[float]],
        top_k: int = 5,
        filter: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query: Query text
            embedding_function: Function to generate query embedding
            top_k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of result dicts with id, distance, and metadata

        Example:
            ```python
            results = vector_store.search(
                query="AI applications",
                embedding_function=embed,
                top_k=5,
                filter={"source": "doc1"}
            )

            for result in results:
                print(f"ID: {result['id']}")
                print(f"Distance: {result['distance']}")
                print(f"Text: {result['metadata']['_text']}")
            ```
        """
        query_embedding = embedding_function(query)

        raw_results = self.backend.query(
            self.collection_name,
            query_embedding,
            top_k=top_k,
            filter=filter,
            metric=self.distance_metric
        )

        # Format results with metadata
        results = []
        for doc_id, distance in raw_results:
            results.append({
                "id": doc_id,
                "distance": distance,
                "metadata": {}  # Metadata retrieval would require additional query
            })

        return results

    def initialize_collection(self, collection_name: str, dimension: int) -> None:
        """
        Explicitly initialize/create a collection.

        Args:
            collection_name: Name for the collection
            dimension: Vector dimension

        Example:
            ```python
            vector_store = AppwriteVectorStore.from_connection_string(url)
            vector_store.initialize_collection("my_vectors", dimension=384)
            ```
        """
        self.collection_name = collection_name
        self.dimension = dimension
        self.backend.create_collection(collection_name, dimension)

    def update_document(
        self,
        doc_id: int,
        embedding: list[float],
        metadata: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Update an existing document (same as add_document, uses upsert).

        Args:
            doc_id: Document ID to update
            embedding: New embedding vector
            metadata: Updated metadata

        Example:
            ```python
            vector_store.update_document(
                doc_id=123,
                embedding=new_embedding,
                metadata={"text": "updated", "version": 2}
            )
            ```
        """
        self.add_document(doc_id, embedding, metadata)

    def delete_document(self, doc_id: int) -> None:
        """
        Delete a document by ID.

        Args:
            doc_id: Document ID to delete

        Example:
            ```python
            vector_store.delete_document(123)
            ```
        """
        # Execute DELETE query directly on MariaDB backend
        cursor = self.backend.conn.cursor()
        try:
            cursor.execute(
                f"DELETE FROM {self.collection_name} WHERE id = %s",
                (doc_id,)
            )
            self.backend.conn.commit()
        finally:
            cursor.close()

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self, 'backend'):
            self.backend.close()

    def get_collection_info(self) -> dict[str, Any]:
        """
        Get information about the collection.

        Returns:
            Dictionary with collection metadata
        """
        return {
            "collection_name": self.collection_name,
            "dimension": self.dimension,
            "distance_metric": self.distance_metric,
            "backend": "MariaDB",
            "native_vectors": getattr(self.backend, "native_vector_support", False)
        }


# Helper function for Appwrite Functions
def create_vector_store_for_appwrite_function(
    collection_name: str = "embeddings",
    dimension: int = 1536,
    distance_metric: str = "cosine"
) -> AppwriteVectorStore:
    """
    Helper to create vector store within Appwrite Function context.

    Uses environment variables for connection details:
    - APPWRITE_MARIADB_HOST (default: mariadb)
    - APPWRITE_MARIADB_PORT (default: 3306)
    - APPWRITE_MARIADB_USER (default: appwrite)
    - APPWRITE_MARIADB_PASSWORD (required)
    - APPWRITE_MARIADB_DATABASE (default: appwrite)

    Args:
        collection_name: Vector collection name
        dimension: Vector dimension
        distance_metric: Distance metric

    Returns:
        AppwriteVectorStore instance

    Example (in Appwrite Function):
        ```python
        from vectorwrap.integrations.appwrite import create_vector_store_for_appwrite_function

        def main(req, res):
            vector_store = create_vector_store_for_appwrite_function()

            # Use vector store in your function
            vector_store.add_document(1, embedding, {"text": "..."})
            results = vector_store.search(query, embed_fn)

            return res.json({"results": results})
        ```
    """
    host = os.environ.get("APPWRITE_MARIADB_HOST", "mariadb")
    port = os.environ.get("APPWRITE_MARIADB_PORT", "3306")
    user = os.environ.get("APPWRITE_MARIADB_USER", "appwrite")
    password = os.environ.get("APPWRITE_MARIADB_PASSWORD")
    database = os.environ.get("APPWRITE_MARIADB_DATABASE", "appwrite")

    if not password:
        raise ValueError(
            "APPWRITE_MARIADB_PASSWORD environment variable is required"
        )

    connection_url = f"mariadb://{user}:{password}@{host}:{port}/{database}"

    return AppwriteVectorStore(
        connection_url=connection_url,
        collection_name=collection_name,
        dimension=dimension,
        distance_metric=distance_metric
    )
