"""
Example: Using vectorwrap with MariaDB.

Demonstrates native VECTOR support (MariaDB 11.8 GA LTS+) with automatic
fallback to JSON storage for older versions.
"""

from vectorwrap import VectorDB
import random


def main():
    # Connect to MariaDB (11.8+ recommended for native VECTOR support)
    db = VectorDB("mariadb://root:password@localhost:3306/vectordb")

    print("Connected to MariaDB")

    # Check if native vector support is available
    if hasattr(db, 'native_vector_support') and db.native_vector_support:
        print("Using MariaDB 11.8+ native VECTOR type with HNSW indexing")
    else:
        print("Using JSON-based vector storage (upgrade to MariaDB 11.8+ for better performance)")

    # Create collection
    collection_name = "mariadb_documents"
    dim = 384  # Typical sentence transformer dimension

    db.create_collection(collection_name, dim=dim)
    print(f"\nCreated collection: {collection_name}")

    # Generate sample vectors (normally these would be from an embedding model)
    print("\nInserting sample vectors...")
    for i in range(10):
        vector = [random.random() for _ in range(dim)]
        metadata = {
            "doc_id": f"doc_{i}",
            "category": "tech" if i % 2 == 0 else "science",
            "title": f"Document {i}"
        }

        db.upsert(collection_name, i, vector, metadata)

    print(f"Inserted 10 vectors")

    # Query for similar vectors
    print("\nQuerying for similar vectors...")
    query_vector = [random.random() for _ in range(dim)]

    # Query with cosine distance (default)
    results = db.query(
        collection_name,
        query_vector,
        top_k=5,
        metric="cosine"
    )

    print(f"\nTop 5 similar vectors (cosine distance):")
    for doc_id, distance in results:
        print(f"  ID: {doc_id}, Distance: {distance:.4f}")

    # Query with filtering
    print("\nQuerying with category filter...")
    results_filtered = db.query(
        collection_name,
        query_vector,
        top_k=3,
        filter={"category": "tech"},
        metric="cosine"
    )

    print(f"\nTop 3 'tech' documents:")
    for doc_id, distance in results_filtered:
        print(f"  ID: {doc_id}, Distance: {distance:.4f}")

    # Query with Euclidean distance
    print("\nQuerying with Euclidean distance...")
    results_l2 = db.query(
        collection_name,
        query_vector,
        top_k=5,
        metric="euclidean"
    )

    print(f"\nTop 5 similar vectors (Euclidean distance):")
    for doc_id, distance in results_l2:
        print(f"  ID: {doc_id}, Distance: {distance:.4f}")

    print("\n" + "="*60)
    print("MariaDB 11.8 GA LTS Features:")
    print("  - Native VECTOR(n) data type")
    print("  - HNSW indexing for fast similarity search")
    print("  - VEC_DISTANCE_EUCLIDEAN and VEC_DISTANCE_COSINE functions")
    print("  - Production-ready performance (similar to pgvector)")
    print("\nFor older versions:")
    print("  - Automatic fallback to JSON storage")
    print("  - Python-based distance calculations")
    print("  - Upgrade to 11.8+ recommended for production workloads")
    print("="*60)


if __name__ == "__main__":
    main()
