"""
Test Appwrite integration with local MariaDB.

This test can be run without a full Appwrite deployment by using a local
MariaDB instance.

Setup:
    docker run -d \
      --name mariadb-test \
      -e MYSQL_ROOT_PASSWORD=test123 \
      -e MYSQL_DATABASE=appwrite_test \
      -p 3306:3306 \
      mariadb:11.8

    # Wait for startup
    sleep 10

Run:
    python tests/test_appwrite_local.py
"""

from vectorwrap.integrations.appwrite import AppwriteVectorStore
import numpy as np
import sys


def test_appwrite_integration():
    """Test all Appwrite integration features."""
    print("=" * 60)
    print("Testing Appwrite Integration with Local MariaDB")
    print("=" * 60)

    # Connection string for local test
    connection_string = "mariadb://root:test123@localhost:3306/appwrite_test"

    try:
        # 1. Create vector store
        print("\n1. Creating AppwriteVectorStore...")
        vector_store = AppwriteVectorStore.from_connection_string(
            connection_url=connection_string,
            collection_name="test_docs",
            dimension=384
        )
        print("   [OK] Created vector store")

        # Check backend info
        info = vector_store.get_collection_info()
        print(f"   [OK] Collection: {info['collection_name']}")
        print(f"   [OK] Dimension: {info['dimension']}")
        print(f"   [OK] Native vectors: {info['native_vectors']}")

        # 2. Initialize collection (explicitly)
        print("\n2. Initializing collection...")
        vector_store.initialize_collection("test_docs", dimension=384)
        print("   [OK] Collection initialized")

        # 3. Add test documents
        print("\n3. Adding test documents...")
        test_data = [
            {
                "id": 1,
                "embedding": np.random.rand(384).tolist(),
                "metadata": {"title": "Introduction to AI", "category": "tech"}
            },
            {
                "id": 2,
                "embedding": np.random.rand(384).tolist(),
                "metadata": {"title": "Machine Learning Basics", "category": "tech"}
            },
            {
                "id": 3,
                "embedding": np.random.rand(384).tolist(),
                "metadata": {"title": "Cooking Recipes", "category": "food"}
            },
            {
                "id": 4,
                "embedding": np.random.rand(384).tolist(),
                "metadata": {"title": "Deep Learning Guide", "category": "tech"}
            }
        ]

        for doc in test_data:
            vector_store.add_document(
                doc["id"],
                doc["embedding"],
                doc["metadata"]
            )
            print(f"   [OK] Added: {doc['metadata']['title']}")

        # 4. Test search
        print("\n4. Testing vector search...")
        query_embedding = np.random.rand(384).tolist()
        results = vector_store.backend.query(
            "test_docs",
            query_embedding,
            top_k=2,
            filter={"category": "tech"}
        )

        print(f"   [OK] Found {len(results)} results (filtered by category=tech)")
        for doc_id, distance in results:
            print(f"      - ID: {doc_id}, Distance: {distance:.4f}")

        # Test without filter
        all_results = vector_store.backend.query(
            "test_docs",
            query_embedding,
            top_k=4
        )
        print(f"   [OK] Found {len(all_results)} results (no filter)")

        # 5. Test update
        print("\n5. Testing document update...")
        vector_store.update_document(
            doc_id=1,
            embedding=np.random.rand(384).tolist(),
            metadata={
                "title": "Advanced AI",
                "category": "tech",
                "updated": True
            }
        )
        print("   [OK] Updated document ID 1")

        # 6. Test delete
        print("\n6. Testing document deletion...")
        vector_store.delete_document(3)
        print("   [OK] Deleted document ID 3")

        # 7. Verify remaining documents
        print("\n7. Verifying remaining documents...")
        remaining = vector_store.backend.query(
            "test_docs",
            query_embedding,
            top_k=10
        )
        print(f"   [OK] Total documents remaining: {len(remaining)}")
        for doc_id, _ in remaining:
            print(f"      - Document ID: {doc_id}")

        assert len(remaining) == 3, f"Expected 3 documents, got {len(remaining)}"
        print("   [OK] Count verified (3 documents)")

        # 8. Test batch add with add_documents
        print("\n8. Testing batch document addition...")

        def mock_embed(text):
            """Mock embedding function."""
            np.random.seed(hash(text) % (2**32))
            return np.random.rand(384).tolist()

        batch_docs = [
            {"text": "Python programming", "metadata": {"category": "code"}},
            {"text": "JavaScript tutorial", "metadata": {"category": "code"}}
        ]

        doc_ids = vector_store.add_documents(
            batch_docs,
            embedding_function=mock_embed
        )
        print(f"   [OK] Added {len(doc_ids)} documents in batch")

        # 9. Test collection info
        print("\n9. Getting collection info...")
        info = vector_store.get_collection_info()
        for key, value in info.items():
            print(f"   [OK] {key}: {value}")

        # 10. Cleanup
        print("\n10. Cleaning up...")
        vector_store.close()
        print("   [OK] Connection closed")

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_connection_methods():
    """Test different connection methods."""
    print("\n" + "=" * 60)
    print("Testing Connection Methods")
    print("=" * 60)

    print("\n1. Testing from_connection_string()...")
    try:
        vector_store = AppwriteVectorStore.from_connection_string(
            connection_url="mariadb://root:test123@localhost:3306/appwrite_test",
            collection_name="connection_test",
            dimension=128
        )
        print("   [OK] from_connection_string() works")
        vector_store.close()
    except Exception as e:
        print(f"   [ERROR] {e}")
        return False

    print("\n2. Testing direct instantiation...")
    try:
        vector_store = AppwriteVectorStore(
            connection_url="mariadb://root:test123@localhost:3306/appwrite_test",
            collection_name="direct_test",
            dimension=128
        )
        print("   [OK] Direct instantiation works")
        vector_store.close()
    except Exception as e:
        print(f"   [ERROR] {e}")
        return False

    print("\n[OK] All connection methods work")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Appwrite Integration Test Suite")
    print("=" * 60)
    print("\nPrerequisites:")
    print("  - MariaDB 11.8+ running on localhost:3306")
    print("  - Database: appwrite_test")
    print("  - User: root, Password: test123")
    print("\nSetup with Docker:")
    print("  docker run -d --name mariadb-test \\")
    print("    -e MYSQL_ROOT_PASSWORD=test123 \\")
    print("    -e MYSQL_DATABASE=appwrite_test \\")
    print("    -p 3306:3306 mariadb:11.8")
    print("=" * 60)

    # Run tests
    success = True

    try:
        if not test_connection_methods():
            success = False

        if not test_appwrite_integration():
            success = False

    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        return 1

    # Return exit code
    if success:
        print("\n✓ All tests completed successfully!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
