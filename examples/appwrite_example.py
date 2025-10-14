"""
Example: Using vectorwrap with Appwrite for AI/Vector capabilities.

Demonstrates how to add vector search to Appwrite applications WITHOUT
requiring external vector databases like Pinecone or Upstash.

Benefits:
- Data locality: Vectors stored in Appwrite's MariaDB backend
- Cost reduction: No external vector DB subscription
- Simplicity: Zero additional configuration
- Native performance: HNSW indexing on MariaDB 11.8+

Setup:
1. Self-hosted Appwrite:
   - Access MariaDB: docker exec -it appwrite-mariadb mysql -u appwrite -p
   - Or use connection string: mariadb://appwrite:password@localhost:3306/appwrite

2. Appwrite Cloud:
   - Contact support for database access (advanced use case)
   - Or use separate MariaDB instance

Requirements:
    pip install vectorwrap appwrite openai
"""

from vectorwrap.integrations.appwrite import AppwriteVectorStore
import random


def mock_embed(text: str) -> list[float]:
    """Mock embedding function (replace with real embeddings)."""
    random.seed(hash(text) % (2**32))
    return [random.random() for _ in range(384)]


def example_basic_usage():
    """Basic vector search example."""
    print("="*60)
    print("EXAMPLE 1: Basic Vector Search with Appwrite")
    print("="*60)

    # Create vector store connected to Appwrite's MariaDB
    vector_store = AppwriteVectorStore.from_connection_string(
        connection_url="mariadb://appwrite:password@localhost:3306/appwrite",
        collection_name="document_embeddings",
        dimension=384
    )

    print("\nVector Store Info:")
    info = vector_store.get_collection_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Add documents with embeddings
    print("\n\nAdding documents...")
    documents = [
        {
            "text": "Appwrite is an open-source backend platform",
            "metadata": {"category": "platform", "topic": "backend"}
        },
        {
            "text": "Vector search enables semantic similarity matching",
            "metadata": {"category": "ai", "topic": "vectors"}
        },
        {
            "text": "MariaDB 11.8 includes native VECTOR support",
            "metadata": {"category": "database", "topic": "vectors"}
        },
        {
            "text": "Build AI applications without external databases",
            "metadata": {"category": "ai", "topic": "architecture"}
        },
    ]

    doc_ids = vector_store.add_documents(
        documents,
        embedding_function=mock_embed
    )
    print(f"Added {len(doc_ids)} documents")

    # Search for similar documents
    print("\n\nSearching for 'AI and semantic search'...")
    results = vector_store.search(
        query="AI and semantic search",
        embedding_function=mock_embed,
        top_k=3
    )

    print(f"\nTop 3 results:")
    for i, result in enumerate(results, 1):
        print(f"\n  {i}. Document ID: {result['id']}")
        print(f"     Distance: {result['distance']:.4f}")


def example_with_filtering():
    """Example with metadata filtering."""
    print("\n\n" + "="*60)
    print("EXAMPLE 2: Vector Search with Metadata Filtering")
    print("="*60)

    vector_store = AppwriteVectorStore.from_connection_string(
        connection_url="mariadb://appwrite:password@localhost:3306/appwrite",
        collection_name="filtered_search",
        dimension=384
    )

    # Add documents with different categories
    documents = [
        {"text": "Python programming tutorial", "metadata": {"language": "python", "type": "tutorial"}},
        {"text": "JavaScript best practices", "metadata": {"language": "javascript", "type": "guide"}},
        {"text": "Python machine learning guide", "metadata": {"language": "python", "type": "guide"}},
        {"text": "JavaScript async patterns", "metadata": {"language": "javascript", "type": "tutorial"}},
    ]

    vector_store.add_documents(documents, embedding_function=mock_embed)
    print("Added documents with metadata")

    # Search with filter
    print("\n\nSearching for 'programming' filtered by language=python...")
    results = vector_store.search(
        query="programming tutorial",
        embedding_function=mock_embed,
        top_k=2,
        filter={"language": "python"}
    )

    print(f"\nPython-only results:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. Document ID: {result['id']}, Distance: {result['distance']:.4f}")


def example_appwrite_function():
    """Example for use within Appwrite Functions."""
    print("\n\n" + "="*60)
    print("EXAMPLE 3: Using in Appwrite Functions")
    print("="*60)

    print("""
In your Appwrite Function code:

```python
# appwrite_function.py
from vectorwrap.integrations.appwrite import create_vector_store_for_appwrite_function
import os

def main(req, res):
    # Automatically connects to Appwrite's MariaDB
    vector_store = create_vector_store_for_appwrite_function(
        collection_name="embeddings",
        dimension=1536
    )

    # Add document
    if req.method == "POST":
        data = req.json()
        embedding = generate_embedding(data['text'])  # Your embedding function

        vector_store.add_document(
            doc_id=data['id'],
            embedding=embedding,
            metadata={"text": data['text'], "user_id": data['user_id']}
        )

        return res.json({"status": "success"})

    # Search
    elif req.method == "GET":
        query = req.query.get("q")
        query_embedding = generate_embedding(query)

        results = vector_store.search(
            query=query,
            embedding_function=generate_embedding,
            top_k=5
        )

        return res.json({"results": results})
```

Environment variables needed:
- APPWRITE_MARIADB_PASSWORD (automatically available in Appwrite Functions)
- Optional: APPWRITE_MARIADB_HOST, APPWRITE_MARIADB_USER, etc.
    """)


def example_openai_embeddings():
    """Example with real OpenAI embeddings."""
    print("\n\n" + "="*60)
    print("EXAMPLE 4: Integration with OpenAI Embeddings")
    print("="*60)

    print("""
Using real embeddings with OpenAI:

```python
from openai import OpenAI
from vectorwrap.integrations.appwrite import AppwriteVectorStore

client = OpenAI(api_key="your-api-key")

def embed(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Create vector store
vector_store = AppwriteVectorStore.from_connection_string(
    connection_url="mariadb://appwrite:password@localhost:3306/appwrite",
    collection_name="openai_embeddings",
    dimension=1536  # text-embedding-3-small dimension
)

# Add documents
documents = [
    {"text": "Appwrite is a secure backend platform", "metadata": {"type": "product"}},
    {"text": "Vector search enables AI applications", "metadata": {"type": "feature"}},
]

vector_store.add_documents(documents, embedding_function=embed)

# Semantic search
results = vector_store.search(
    query="What is Appwrite?",
    embedding_function=embed,
    top_k=5
)

for result in results:
    print(f"Match: {result['metadata']['_text']}")
    print(f"Distance: {result['distance']:.4f}")
```
    """)


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Appwrite + vectorwrap: AI Without External Vector Databases")
    print("="*70)

    try:
        example_basic_usage()
    except Exception as e:
        print(f"\nNote: Could not connect to Appwrite MariaDB: {e}")
        print("This is expected if you don't have Appwrite running locally.")

    example_with_filtering()
    example_appwrite_function()
    example_openai_embeddings()

    print("\n\n" + "="*70)
    print("Benefits of vectorwrap + Appwrite:")
    print("="*70)
    print("  - No Pinecone/Upstash subscription needed")
    print("  - No external services to configure")
    print("  - Data stays in your Appwrite database")
    print("  - Native HNSW indexing (MariaDB 11.8+)")
    print("  - Simple Python API")
    print("  - Works in Appwrite Functions")
    print("="*70)


if __name__ == "__main__":
    main()
