"""
LlamaIndex VectorStore wrapper for vectorwrap.

Provides seamless integration with LlamaIndex's data framework.
"""

from __future__ import annotations

from typing import Any, List, Optional

try:
    from llama_index.core.schema import BaseNode, TextNode
    from llama_index.core.vector_stores.types import (
        VectorStore,
        VectorStoreQuery,
        VectorStoreQueryResult,
        VectorStoreQueryMode,
    )
except ImportError:
    raise ImportError(
        "LlamaIndex is required for this integration. "
        "Install with: pip install 'vectorwrap[llamaindex]'"
    )

from vectorwrap import VectorDB, VectorBackend


class VectorwrapVectorStore(VectorStore):
    """
    LlamaIndex VectorStore wrapper for vectorwrap.

    Enables using any vectorwrap backend with LlamaIndex's ServiceContext
    for building RAG applications, agents, and data pipelines.

    Example:
        ```python
        from llama_index.core import VectorStoreIndex, Document, ServiceContext
        from llama_index.embeddings.openai import OpenAIEmbedding
        from vectorwrap.integrations.llamaindex import VectorwrapVectorStore

        # Create vector store
        vector_store = VectorwrapVectorStore(
            connection_url="postgresql://user:pass@localhost/db",
            collection_name="documents",
            dimension=1536
        )

        # Create index
        embed_model = OpenAIEmbedding()
        service_context = ServiceContext.from_defaults(embed_model=embed_model)

        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            service_context=service_context
        )

        # Add documents
        documents = [Document(text="Hello world")]
        index.insert_nodes(documents)

        # Query
        query_engine = index.as_query_engine()
        response = query_engine.query("What is this about?")
        ```
    """

    stores_text: bool = True
    is_embedding_query: bool = True

    def __init__(
        self,
        connection_url: str,
        collection_name: str = "llamaindex_store",
        dimension: int = 1536,
    ):
        """
        Initialize VectorwrapVectorStore.

        Args:
            connection_url: Database connection string
            collection_name: Name of the collection/table
            dimension: Vector embedding dimension
        """
        self.db: VectorBackend = VectorDB(connection_url)
        self.collection_name = collection_name
        self.dimension = dimension

        # Create collection
        try:
            self.db.create_collection(collection_name, dimension)
        except Exception:
            # Collection might already exist
            pass

    @property
    def client(self) -> Any:
        """Return the underlying vectorwrap client."""
        return self.db

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """
        Add nodes to the vector store.

        Args:
            nodes: List of nodes to add
            **add_kwargs: Additional arguments

        Returns:
            List of node IDs
        """
        ids = []

        for node in nodes:
            # Get embedding
            embedding = node.get_embedding()
            if embedding is None:
                raise ValueError(f"Node {node.node_id} has no embedding")

            # Convert node_id to numeric ID
            numeric_id = hash(node.node_id) % (2**63)

            # Prepare metadata
            metadata = {
                "text": node.get_content(metadata_mode="all"),
                "node_id": node.node_id,
                **node.metadata,
            }

            # Add to vector store
            self.db.upsert(
                self.collection_name,
                numeric_id,
                embedding,
                metadata
            )

            ids.append(node.node_id)

        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes by reference document ID.

        Note: Current vectorwrap API doesn't support deletion.

        Args:
            ref_doc_id: Reference document ID
            **delete_kwargs: Additional arguments
        """
        raise NotImplementedError("Delete operation not yet supported in vectorwrap")

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """
        Query the vector store.

        Args:
            query: VectorStoreQuery object
            **kwargs: Additional query arguments

        Returns:
            VectorStoreQueryResult with nodes, similarities, and IDs
        """
        if query.query_embedding is None:
            raise ValueError("query_embedding is required")

        # Build filter from query
        filter_dict = None
        if query.filters is not None:
            # Convert LlamaIndex filters to simple dict
            # TODO: Implement more sophisticated filter conversion
            filter_dict = {}

        # Query vector store
        results = self.db.query(
            self.collection_name,
            query.query_embedding,
            top_k=query.similarity_top_k,
            filter=filter_dict
        )

        # Convert results to VectorStoreQueryResult
        node_ids = []
        similarities = []
        nodes = []

        for doc_id, distance in results:
            # Convert distance to similarity score (inverse)
            similarity = 1.0 / (1.0 + distance) if distance >= 0 else 0.0

            # Create minimal node (full metadata retrieval would require API extension)
            node = TextNode(
                id_=str(doc_id),
                text=f"Node {doc_id}",  # TODO: Retrieve actual text
                metadata={"_id": doc_id}
            )

            node_ids.append(str(doc_id))
            similarities.append(similarity)
            nodes.append(node)

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=node_ids
        )

    def persist(
        self,
        persist_path: str,
        **kwargs: Any,
    ) -> None:
        """
        Persist the vector store.

        For file-based backends (SQLite, DuckDB), data is already persisted.
        For network backends, this is a no-op.

        Args:
            persist_path: Path to persist to (ignored for network backends)
            **kwargs: Additional arguments
        """
        # Data is automatically persisted in SQL backends
        pass

    @classmethod
    def from_persist_path(
        cls,
        persist_path: str,
        **kwargs: Any,
    ) -> "VectorwrapVectorStore":
        """
        Load from persisted path.

        Args:
            persist_path: Path to load from
            **kwargs: Additional arguments

        Returns:
            VectorwrapVectorStore instance
        """
        # For SQLite/DuckDB, use file path as connection URL
        if persist_path.endswith(".db"):
            connection_url = f"sqlite:///{persist_path}"
        elif persist_path.endswith(".duckdb"):
            connection_url = f"duckdb:///{persist_path}"
        else:
            raise ValueError("Unsupported persist path. Use .db or .duckdb extension")

        collection_name = kwargs.get("collection_name", "llamaindex_store")
        dimension = kwargs.get("dimension", 1536)

        return cls(
            connection_url=connection_url,
            collection_name=collection_name,
            dimension=dimension
        )


# Convenience function for LlamaIndex users
def create_vector_store(
    backend: str = "sqlite",
    collection_name: str = "llamaindex_store",
    dimension: int = 1536,
    **kwargs: Any,
) -> VectorwrapVectorStore:
    """
    Create a VectorwrapVectorStore with common presets.

    Args:
        backend: Backend type ("sqlite", "duckdb", "postgres", "mysql", "clickhouse")
        collection_name: Collection name
        dimension: Vector dimension
        **kwargs: Additional connection parameters (host, port, user, password, database)

    Returns:
        VectorwrapVectorStore instance

    Example:
        ```python
        # SQLite in-memory
        store = create_vector_store("sqlite")

        # PostgreSQL
        store = create_vector_store(
            "postgres",
            host="localhost",
            user="user",
            password="pass",
            database="vectordb"
        )
        ```
    """
    if backend == "sqlite":
        db_path = kwargs.get("path", ":memory:")
        connection_url = f"sqlite:///{db_path}"

    elif backend == "duckdb":
        db_path = kwargs.get("path", ":memory:")
        connection_url = f"duckdb:///{db_path}"

    elif backend == "postgres":
        host = kwargs.get("host", "localhost")
        port = kwargs.get("port", 5432)
        user = kwargs.get("user", "postgres")
        password = kwargs.get("password", "")
        database = kwargs.get("database", "postgres")
        connection_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"

    elif backend == "mysql":
        host = kwargs.get("host", "localhost")
        port = kwargs.get("port", 3306)
        user = kwargs.get("user", "root")
        password = kwargs.get("password", "")
        database = kwargs.get("database", "vectordb")
        connection_url = f"mysql://{user}:{password}@{host}:{port}/{database}"

    elif backend == "clickhouse":
        host = kwargs.get("host", "localhost")
        port = kwargs.get("port", 8123)
        user = kwargs.get("user", "default")
        password = kwargs.get("password", "")
        database = kwargs.get("database", "default")
        connection_url = f"clickhouse://{user}:{password}@{host}:{port}/{database}"

    else:
        raise ValueError(f"Unsupported backend: {backend}")

    return VectorwrapVectorStore(
        connection_url=connection_url,
        collection_name=collection_name,
        dimension=dimension
    )
