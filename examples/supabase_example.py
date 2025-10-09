"""
Example: Using vectorwrap with Supabase.

Demonstrates Supabase pgvector integration with RLS and bulk operations.
"""

import os
from vectorwrap.integrations.supabase import SupabaseVectorStore


def main():
    # Setup (requires SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables)
    try:
        store = SupabaseVectorStore.from_env()
    except ValueError:
        print("Please set SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables")
        print("Example:")
        print("  export SUPABASE_URL='https://xxx.supabase.co'")
        print("  export SUPABASE_SERVICE_KEY='your-service-key'")
        return

    # Create collection with Row Level Security
    collection_name = "user_documents"
    store.create_collection(collection_name, dim=384, enable_rls=True)

    print(f"Created collection: {collection_name}")

    # Generate SQL schema for reference
    sql = store.get_schema_sql(collection_name, dim=384)
    print("\nGenerated SQL schema:")
    print(sql)

    # Bulk upsert (more efficient than individual inserts)
    import random
    vectors = [[random.random() for _ in range(384)] for _ in range(100)]
    metadatas = [
        {"user_id": "user-1", "source": f"doc{i}.txt"}
        for i in range(100)
    ]

    store.bulk_upsert(
        collection_name,
        vectors=vectors,
        metadatas=metadatas
    )

    print(f"\nInserted {len(vectors)} vectors")

    # Query
    query_vector = [random.random() for _ in range(384)]
    results = store.query(
        collection_name,
        query_vector,
        top_k=5,
        filter={"user_id": "user-1"}
    )

    print(f"\nFound {len(results)} results")
    for doc_id, distance in results[:3]:
        print(f"  ID: {doc_id}, Distance: {distance:.4f}")

    # Get collection statistics
    stats = store.get_collection_stats(collection_name)
    print(f"\nCollection stats:")
    print(f"  Rows: {stats['row_count']}")
    print(f"  Size: {stats['table_size']}")
    print(f"  Indexes: {len(stats['indexes'])}")


if __name__ == "__main__":
    main()
