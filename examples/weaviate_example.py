"""
Example: Using vectorwrap with Weaviate.

Demonstrates cloud-native vector search with Weaviate.
"""

from vectorwrap.integrations.weaviate import WeaviateBackend
import random
import time


def main():
    # Connect to Weaviate (local or cloud)
    # Local: WeaviateBackend(url="http://localhost:8080")
    # Cloud: WeaviateBackend(url="https://xxx.weaviate.network", api_key="your-api-key")
    db = WeaviateBackend(url="http://localhost:8080")

    print("Connected to Weaviate")
    print("Note: Requires Weaviate instance running locally or in the cloud\n")

    # Create collection with HNSW index
    collection_name = "weaviate_documents"
    dim = 384

    db.create_collection(collection_name, dim=dim, distance="cosine")
    print(f"Created collection with HNSW index: {collection_name}")

    # Insert vectors
    print("\nInserting 1000 vectors...")
    start_time = time.time()

    for i in range(1000):
        vector = [random.random() for _ in range(dim)]
        metadata = {
            "doc_id": f"doc_{i}",
            "category": ["tech", "science", "business"][i % 3],
            "score": random.randint(1, 100),
            "title": f"Document {i}"
        }

        db.upsert(collection_name, i, vector, metadata)

    elapsed = time.time() - start_time
    print(f"Inserted 1000 vectors in {elapsed:.2f}s ({1000/elapsed:.0f} vectors/sec)")

    # Query for similar vectors
    print("\nQuerying for similar vectors...")
    query_vector = [random.random() for _ in range(dim)]

    start_time = time.time()
    results = db.query(collection_name, query_vector, top_k=10)
    query_time = (time.time() - start_time) * 1000  # Convert to ms

    print(f"Query completed in {query_time:.2f}ms\n")
    print(f"Top 10 similar vectors:")
    for doc_id, distance in results[:5]:  # Show first 5
        print(f"  ID: {doc_id}, Distance: {distance:.4f}")
    if len(results) > 5:
        print(f"  ... and {len(results)-5} more")

    # Query with metadata filter
    print("\n\nQuerying with metadata filter (category='tech')...")
    filter_results = db.query(
        collection_name,
        query_vector,
        top_k=5,
        filter={"category": "tech"}
    )

    print(f"Found {len(filter_results)} results:")
    for doc_id, distance in filter_results:
        print(f"  ID: {doc_id}, Distance: {distance:.4f}")

    # Benchmark queries
    print("\n\nBenchmarking query performance...")
    num_queries = 100
    query_times = []

    for _ in range(num_queries):
        query_vec = [random.random() for _ in range(dim)]
        start = time.time()
        _ = db.query(collection_name, query_vec, top_k=10)
        query_times.append((time.time() - start) * 1000)

    avg_time = sum(query_times) / len(query_times)
    p95_time = sorted(query_times)[int(0.95 * len(query_times))]

    print(f"\nQuery Performance ({num_queries} queries):")
    print(f"  Average latency: {avg_time:.2f}ms")
    print(f"  P95 latency: {p95_time:.2f}ms")
    print(f"  Throughput: {1000/avg_time:.0f} queries/sec")

    # Get collection info
    print("\n\nCollection information:")
    try:
        info = db.get_collection_info(collection_name)
        print(f"  Collection: {info.get('name', 'N/A')}")
        print(f"  Class name: {info.get('class_name', 'N/A')}")
        print(f"  Total count: {info.get('total_count', 'N/A')}")
        print(f"  Vectorizer: {info.get('vectorizer', 'N/A')}")
    except Exception as e:
        print(f"  Could not retrieve info: {e}")

    # List all collections
    print("\n\nAll collections:")
    collections = db.list_collections()
    for coll in collections:
        print(f"  - {coll}")

    # Delete some vectors
    print("\n\nDeleting vectors 0-4...")
    db.delete(collection_name, [0, 1, 2, 3, 4])
    print("Deleted 5 vectors")

    # Clean up
    print("\n\nCleaning up...")
    db.drop_collection(collection_name)
    print(f"Dropped collection: {collection_name}")

    db.close()
    print("Connection closed")

    print("\n\nWeaviate is ideal for:")
    print("  - Production-scale vector search")
    print("  - Cloud-native deployments")
    print("  - GraphQL-based queries")
    print("  - Multi-tenancy applications")
    print("  - Hybrid search (vectors + traditional search)")


if __name__ == "__main__":
    main()
