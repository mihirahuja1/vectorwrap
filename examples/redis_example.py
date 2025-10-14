"""
Example: Using vectorwrap with Redis + RediSearch.

Demonstrates ultra-fast in-memory vector search with Redis.
"""

from vectorwrap import VectorDB
import random
import time


def main():
    # Connect to Redis (requires Redis Stack or Redis with RediSearch module)
    db = VectorDB("redis://localhost:6379/0")

    print("Connected to Redis")
    print("Note: Requires Redis Stack or Redis with RediSearch module\n")

    # Create collection with HNSW index
    collection_name = "redis_documents"
    dim = 384

    db.create_collection(collection_name, dim=dim)
    print(f"Created collection with HNSW index: {collection_name}")

    # Insert vectors (Redis is extremely fast for this)
    print("\nInserting 1000 vectors...")
    start_time = time.time()

    for i in range(1000):
        vector = [random.random() for _ in range(dim)]
        metadata = {
            "doc_id": f"doc_{i}",
            "category": ["tech", "science", "business"][i % 3],
            "score": random.randint(1, 100)
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
    print(f"  ... and {len(results)-5} more")

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
        print(f"  Documents: {info.get('num_docs', 'N/A')}")
        print(f"  Index: {info.get('index_name', 'N/A')}")
    except Exception as e:
        print(f"  Could not retrieve info: {e}")

    print("\n\nRedis is ideal for:")
    print("  - Ultra-low latency requirements (<1ms)")
    print("  - High-throughput applications")
    print("  - Real-time search and recommendations")
    print("  - Session-based vector search")


if __name__ == "__main__":
    main()
