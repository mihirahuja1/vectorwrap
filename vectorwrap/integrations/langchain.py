"""
LangChain VectorStore adapter for vectorwrap.

Provides seamless integration with LangChain's ecosystem.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional, Tuple
from uuid import uuid4

try:
    from langchain.schema import Document
    from langchain.vectorstores.base import VectorStore
    from langchain.embeddings.base import Embeddings
except ImportError:
    raise ImportError(
        "LangChain is required for this integration. "
        "Install with: pip install 'vectorwrap[langchain]'"
    )

from vectorwrap import VectorDB, VectorBackend


class VectorwrapStore(VectorStore):
    """
    LangChain VectorStore adapter for vectorwrap.

    Enables using any vectorwrap backend (PostgreSQL, MySQL, SQLite, DuckDB, ClickHouse)
    with LangChain's document retrieval and RAG pipelines.

    Example:
        ```python
        from langchain.embeddings import OpenAIEmbeddings
        from vectorwrap.integrations.langchain import VectorwrapStore

        embeddings = OpenAIEmbeddings()
        vectorstore = VectorwrapStore(
            connection_url="postgresql://user:pass@localhost/db",
            collection_name="documents",
            embedding_function=embeddings
        )

        # Add documents
        vectorstore.add_texts(
            texts=["Hello world", "LangChain is great"],
            metadatas=[{"source": "intro"}, {"source": "review"}]
        )

        # Search
        results = vectorstore.similarity_search("greeting", k=5)

        # Use as retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        ```
    """

    def __init__(
        self,
        connection_url: str,
        collection_name: str,
        embedding_function: Embeddings,
        dimension: Optional[int] = None,
    ):
        """
        Initialize VectorwrapStore.

        Args:
            connection_url: Database connection string (e.g., "postgresql://...", "sqlite:///...")
            collection_name: Name of the collection/table
            embedding_function: LangChain embeddings instance
            dimension: Vector dimension (auto-detected if not provided)
        """
        self.db: VectorBackend = VectorDB(connection_url)
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self._dimension = dimension
        self._id_counter = 0

        # Create collection if dimension is provided
        if dimension is not None:
            try:
                self.db.create_collection(collection_name, dimension)
            except Exception:
                # Collection might already exist
                pass

    @property
    def embeddings(self) -> Embeddings:
        """Return the embedding function."""
        return self.embedding_function

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Add texts to the vector store.

        Args:
            texts: Iterable of text strings to add
            metadatas: Optional list of metadata dicts for each text
            **kwargs: Additional arguments (ids can be provided)

        Returns:
            List of IDs for the added texts
        """
        texts_list = list(texts)
        metadatas = metadatas or [{} for _ in texts_list]

        # Generate or use provided IDs
        ids = kwargs.get("ids")
        if ids is None:
            ids = [str(uuid4()) for _ in texts_list]

        # Embed texts
        embeddings = self.embedding_function.embed_documents(texts_list)

        # Auto-detect dimension on first insert
        if self._dimension is None and embeddings:
            self._dimension = len(embeddings[0])
            try:
                self.db.create_collection(self.collection_name, self._dimension)
            except Exception:
                # Collection might already exist
                pass

        # Insert into vector store
        for i, (text, embedding, metadata) in enumerate(zip(texts_list, embeddings, metadatas)):
            # Add text to metadata
            metadata_with_text = {**metadata, "_text": text}

            # Use numeric ID for vectorwrap
            numeric_id = hash(ids[i]) % (2**63)  # Convert to positive int

            self.db.upsert(
                self.collection_name,
                numeric_id,
                embedding,
                metadata_with_text
            )

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Search for similar documents.

        Args:
            query: Query string
            k: Number of results to return
            filter: Optional metadata filter
            **kwargs: Additional search arguments

        Returns:
            List of Documents
        """
        # Embed query
        query_embedding = self.embedding_function.embed_query(query)

        # Search
        results = self.db.query(
            self.collection_name,
            query_embedding,
            top_k=k,
            filter=filter
        )

        # Convert to Documents
        # Note: We can't retrieve metadata directly from current API
        # In production, you'd extend the backend to return metadata
        documents = []
        for doc_id, distance in results:
            # For now, create minimal documents
            # TODO: Extend vectorwrap API to return metadata in query results
            documents.append(
                Document(
                    page_content=f"Document ID: {doc_id}",
                    metadata={"id": doc_id, "distance": distance}
                )
            )

        return documents

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents with relevance scores.

        Args:
            query: Query string
            k: Number of results to return
            filter: Optional metadata filter
            **kwargs: Additional search arguments

        Returns:
            List of (Document, score) tuples
        """
        query_embedding = self.embedding_function.embed_query(query)

        results = self.db.query(
            self.collection_name,
            query_embedding,
            top_k=k,
            filter=filter
        )

        documents_with_scores = []
        for doc_id, distance in results:
            doc = Document(
                page_content=f"Document ID: {doc_id}",
                metadata={"id": doc_id}
            )
            documents_with_scores.append((doc, distance))

        return documents_with_scores

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        connection_url: str = "sqlite:///:memory:",
        collection_name: str = "langchain_store",
        **kwargs: Any,
    ) -> "VectorwrapStore":
        """
        Create a VectorwrapStore from a list of texts.

        Args:
            texts: List of texts
            embedding: Embeddings instance
            metadatas: Optional list of metadata dicts
            connection_url: Database connection URL
            collection_name: Collection name
            **kwargs: Additional arguments

        Returns:
            VectorwrapStore instance
        """
        # Determine dimension from first embedding
        sample_embedding = embedding.embed_query(texts[0] if texts else "test")
        dimension = len(sample_embedding)

        store = cls(
            connection_url=connection_url,
            collection_name=collection_name,
            embedding_function=embedding,
            dimension=dimension
        )

        if texts:
            store.add_texts(texts, metadatas, **kwargs)

        return store

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """
        Delete by IDs.

        Note: Current vectorwrap API doesn't support deletion.
        This would need to be implemented in the backend.

        Args:
            ids: List of IDs to delete
            **kwargs: Additional arguments

        Returns:
            True if successful, None if not supported
        """
        # TODO: Implement delete in vectorwrap backends
        raise NotImplementedError("Delete operation not yet supported in vectorwrap")

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        connection_url: str = "sqlite:///:memory:",
        collection_name: str = "langchain_store",
        **kwargs: Any,
    ) -> "VectorwrapStore":
        """
        Create a VectorwrapStore from a list of Documents.

        Args:
            documents: List of Documents
            embedding: Embeddings instance
            connection_url: Database connection URL
            collection_name: Collection name
            **kwargs: Additional arguments

        Returns:
            VectorwrapStore instance
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            connection_url=connection_url,
            collection_name=collection_name,
            **kwargs
        )
