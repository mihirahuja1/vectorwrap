"""
Example: Using vectorwrap with LangChain.

Demonstrates how to use vectorwrap as a LangChain VectorStore backend.
"""

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from vectorwrap.integrations.langchain import VectorwrapStore


def main():
    # Sample documents
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a high-level programming language.",
        "Machine learning is a subset of artificial intelligence.",
        "Vector databases enable semantic search.",
        "LangChain makes it easy to build LLM applications.",
    ]

    # Initialize embeddings (requires OPENAI_API_KEY environment variable)
    embeddings = OpenAIEmbeddings()

    # Create vector store with SQLite (for local development)
    vectorstore = VectorwrapStore.from_texts(
        texts=documents,
        embedding=embeddings,
        connection_url="sqlite:///./langchain_demo.db",
        collection_name="demo_docs"
    )

    # Search for similar documents
    query = "What is vector search?"
    results = vectorstore.similarity_search(query, k=3)

    print(f"Query: {query}\n")
    print("Top 3 similar documents:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content}")

    # Use as retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    docs = retriever.get_relevant_documents("programming languages")

    print("\n\nRetriever results for 'programming languages':")
    for doc in docs:
        print(f"- {doc.page_content}")

    # Switch to PostgreSQL for production (uncomment to use)
    # vectorstore_prod = VectorwrapStore.from_texts(
    #     texts=documents,
    #     embedding=embeddings,
    #     connection_url="postgresql://user:pass@localhost/vectordb",
    #     collection_name="demo_docs"
    # )


if __name__ == "__main__":
    main()
