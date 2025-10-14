"""
Example: Real-Time Vector Search with RethinkDB.

WORLD'S FIRST: Combines RethinkDB's real-time changefeeds with vector similarity search.

Use cases:
- Live recommendation systems
- Streaming semantic search
- Real-time content discovery
- Dynamic embeddings that update as data changes

Requirements:
    pip install vectorwrap rethinkdb hnswlib numpy
"""

from vectorwrap import VectorDB
import random
import time
import threading


def mock_embed(text: str, dim: int = 384) -> list[float]:
    """Mock embedding function (replace with real embeddings)."""
    random.seed(hash(text) % (2**32))
    return [random.random() for _ in range(dim)]


def example_basic_usage():
    """Basic vector search with RethinkDB."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Real-Time Vector Search")
    print("=" * 60)

    # Connect to RethinkDB
    db = VectorDB("rethinkdb://localhost:28015/vectordb")
    print("\n✓ Connected to RethinkDB")

    # Create collection
    collection = "live_documents"
    dim = 384

    db.create_collection(collection, dim=dim)
    print(f"✓ Created collection '{collection}' with dimension {dim}")

    # Add documents
    print("\nAdding documents...")
    documents = [
        ("Real-time vector search is powerful", {"category": "tech"}),
        ("RethinkDB enables live queries", {"category": "database"}),
        ("Machine learning with streaming data", {"category": "ai"}),
        ("WebSocket updates for embeddings", {"category": "tech"}),
    ]

    for i, (text, metadata) in enumerate(documents):
        embedding = mock_embed(text, dim)
        db.upsert(collection, f"doc_{i}", embedding, metadata)
        print(f"  ✓ Added: {text[:40]}...")

    # Search
    print("\nSearching for 'real-time technology'...")
    query_vector = mock_embed("real-time technology", dim)
    results = db.query(collection, query_vector, top_k=3)

    print("\nTop 3 results:")
    for doc_id, distance in results:
        print(f"  - {doc_id}: distance={distance:.4f}")


def example_realtime_updates():
    """Demonstrate real-time changefeed updates."""
    print("\n\n" + "=" * 60)
    print("EXAMPLE 2: Real-Time Changefeed Updates")
    print("=" * 60)

    db = VectorDB("rethinkdb://localhost:28015/vectordb")
    collection = "streaming_vectors"
    dim = 128

    db.create_collection(collection, dim=dim)
    print(f"\n✓ Created collection '{collection}'")

    # Subscribe to changes
    print("\n✓ Subscribing to real-time changes...")

    change_count = [0]  # Mutable container for closure

    def on_change(change):
        """Callback for vector updates."""
        change_count[0] += 1
        if change.get("new_val"):
            doc = change["new_val"]
            print(f"  [LIVE UPDATE {change_count[0]}] Vector '{doc['id']}' updated")

    # Start changefeed (in background thread)
    db.subscribe_to_changes(collection, on_change)
    print("  Changefeed active (listening for updates)...")

    # Simulate live updates
    print("\nSimulating live vector updates...")
    for i in range(5):
        time.sleep(1)
        vector = [random.random() for _ in range(dim)]
        db.upsert(collection, f"stream_{i}", vector, {"timestamp": time.time()})
        print(f"  → Inserted vector stream_{i}")

    time.sleep(1)  # Let changefeeds process
    print(f"\n✓ Processed {change_count[0]} live updates")


def example_streaming_recommendations():
    """Real-time recommendation system."""
    print("\n\n" + "=" * 60)
    print("EXAMPLE 3: Live Recommendation Engine")
    print("=" * 60)

    db = VectorDB("rethinkdb://localhost:28015/vectordb")
    collection = "user_preferences"
    dim = 256

    db.create_collection(collection, dim=dim)
    print(f"\n✓ Created recommendation collection")

    # Initial user preferences
    users = [
        ("user_alice", "loves sci-fi movies", {"genre": "sci-fi"}),
        ("user_bob", "enjoys action films", {"genre": "action"}),
        ("user_carol", "prefers documentaries", {"genre": "documentary"}),
    ]

    print("\nAdding user preferences...")
    for user_id, preference, metadata in users:
        vector = mock_embed(preference, dim)
        db.upsert(collection, user_id, vector, metadata)
        print(f"  ✓ {user_id}: {preference}")

    # Find similar users
    print("\nFinding users similar to Alice...")
    alice_vector = mock_embed("loves sci-fi movies", dim)
    similar_users = db.query(collection, alice_vector, top_k=3)

    print("\nSimilar users:")
    for user_id, distance in similar_users:
        if user_id != "user_alice":
            print(f"  - {user_id} (distance: {distance:.4f})")

    # Real-time update: User changes preference
    print("\n\n[LIVE UPDATE] Bob now also likes sci-fi...")
    new_preference = mock_embed("loves sci-fi and action movies", dim)
    db.upsert(collection, "user_bob", new_preference, {"genre": "sci-fi+action"})

    # Re-query
    similar_users = db.query(collection, alice_vector, top_k=3)
    print("\nUpdated similar users:")
    for user_id, distance in similar_users:
        if user_id != "user_alice":
            print(f"  - {user_id} (distance: {distance:.4f})")


def example_performance():
    """Benchmark RethinkDB vector search performance."""
    print("\n\n" + "=" * 60)
    print("EXAMPLE 4: Performance Benchmark")
    print("=" * 60)

    db = VectorDB("rethinkdb://localhost:28015/vectordb")
    collection = "benchmark"
    dim = 384
    num_vectors = 1000

    db.create_collection(collection, dim=dim)
    print(f"\n✓ Created benchmark collection (dim={dim})")

    # Insert vectors
    print(f"\nInserting {num_vectors} vectors...")
    start_time = time.time()

    for i in range(num_vectors):
        vector = [random.random() for _ in range(dim)]
        db.upsert(collection, f"vec_{i}", vector, {"index": i})

    insert_time = time.time() - start_time
    print(f"✓ Inserted {num_vectors} vectors in {insert_time:.2f}s")
    print(f"  → {num_vectors/insert_time:.0f} vectors/sec")

    # Query performance
    print(f"\nBenchmarking queries...")
    query_vector = [random.random() for _ in range(dim)]
    num_queries = 100
    query_times = []

    for _ in range(num_queries):
        start = time.time()
        _ = db.query(collection, query_vector, top_k=10)
        query_times.append((time.time() - start) * 1000)

    avg_time = sum(query_times) / len(query_times)
    p95_time = sorted(query_times)[int(0.95 * len(query_times))]

    print(f"✓ Query performance ({num_queries} queries):")
    print(f"  → Average: {avg_time:.2f}ms")
    print(f"  → P95: {p95_time:.2f}ms")
    print(f"  → Throughput: {1000/avg_time:.0f} queries/sec")


def example_filtered_search():
    """Filtered vector search with metadata."""
    print("\n\n" + "=" * 60)
    print("EXAMPLE 5: Filtered Real-Time Search")
    print("=" * 60)

    db = VectorDB("rethinkdb://localhost:28015/vectordb")
    collection = "filtered_docs"
    dim = 256

    db.create_collection(collection, dim=dim)
    print(f"\n✓ Created collection with filtering support")

    # Add documents with categories
    print("\nAdding categorized documents...")
    docs = [
        ("Python programming basics", {"language": "python", "level": "beginner"}),
        ("Advanced Python patterns", {"language": "python", "level": "advanced"}),
        ("JavaScript fundamentals", {"language": "javascript", "level": "beginner"}),
        ("React hooks tutorial", {"language": "javascript", "level": "intermediate"}),
        ("Rust memory safety", {"language": "rust", "level": "advanced"}),
    ]

    for i, (text, metadata) in enumerate(docs):
        vector = mock_embed(text, dim)
        db.upsert(collection, f"doc_{i}", vector, metadata)
        print(f"  ✓ {text}")

    # Search with filter
    print("\nSearching for 'programming' (Python only)...")
    query = mock_embed("programming tutorial", dim)
    results = db.query(
        collection,
        query,
        top_k=5,
        filter={"language": "python"}
    )

    print("\nPython results:")
    for doc_id, distance in results:
        print(f"  - {doc_id}: distance={distance:.4f}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("RethinkDB + vectorwrap: Real-Time Vector Search")
    print("WORLD'S FIRST: Changefeeds + Similarity Search")
    print("=" * 70)

    print("""
Prerequisites:
  1. RethinkDB running: rethinkdb
  2. Install packages: pip install vectorwrap rethinkdb hnswlib numpy

Quick start:
  docker run -d -p 28015:28015 -p 8080:8080 rethinkdb
""")

    try:
        example_basic_usage()
        example_realtime_updates()
        example_streaming_recommendations()
        example_filtered_search()
        example_performance()

        print("\n\n" + "=" * 70)
        print("Key Features Demonstrated:")
        print("=" * 70)
        print("  ✓ Real-time changefeeds for live vector updates")
        print("  ✓ In-memory HNSW indexing for fast queries")
        print("  ✓ Metadata filtering with JSON flexibility")
        print("  ✓ Streaming recommendations")
        print("  ✓ High-throughput vector operations")
        print("\n" + "=" * 70)
        print("Use Cases:")
        print("=" * 70)
        print("  - Live content recommendations")
        print("  - Real-time semantic search")
        print("  - Streaming embeddings from ML models")
        print("  - Dynamic user preference matching")
        print("  - WebSocket-powered vector updates")
        print("=" * 70)

    except Exception as e:
        print(f"\n\n[ERROR] Could not connect to RethinkDB: {e}")
        print("\nMake sure RethinkDB is running:")
        print("  docker run -d -p 28015:28015 -p 8080:8080 rethinkdb")
        print("\nOr install locally:")
        print("  macOS: brew install rethinkdb")
        print("  Ubuntu: apt install rethinkdb")


if __name__ == "__main__":
    main()
