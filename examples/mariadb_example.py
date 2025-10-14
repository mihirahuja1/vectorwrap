"""
Example: Using vectorwrap with MariaDB.

Demonstrates JSON-based vector storage in MariaDB (no native vector support).
"""

from vectorwrap import VectorDB
import random


def main():
    # Connect to MariaDB (uses same mysql-connector as MySQL)
    # You can use either mysql:// or mariadb:// URL scheme
    db = VectorDB("mariadb://root:password@localhost:3306/vectordb")

    # Alternative: use mysql:// (works the same way)
    # db = VectorDB("mysql://root:password@localhost:3306/vectordb")

    print("Connected to MariaDB")

    # Create collection
    collection_name = "mariadb_documents"
    dim = 384  # Typical sentence transformer dimension

    db.create_collection(collection_name, dim=dim)
    print(f"Created collection: {collection_name}")

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

    print("\nNote: MariaDB doesn't have native vector support.")
    print("Vectors are stored as JSON and distances are computed in Python.")
    print("For production workloads, consider PostgreSQL + pgvector or MySQL 8.2+")


if __name__ == "__main__":
    main()
