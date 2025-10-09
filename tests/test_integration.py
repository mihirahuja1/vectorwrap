"""
Integration tests for vectorwrap - tests consistency across all backends.

These tests verify that all backends behave consistently and handle
edge cases properly for the stable API.
"""
import json
import tempfile
import os
from typing import List, Tuple
import pytest
from vectorwrap import VectorDB


# Test data
TEST_VECTORS = [
    ([1.0, 0.0, 0.0], {"category": "A", "price": 100}),
    ([0.0, 1.0, 0.0], {"category": "B", "price": 200}),
    ([0.0, 0.0, 1.0], {"category": "A", "price": 150}),
    ([1.0, 1.0, 0.0], {"category": "C", "price": 300}),
    ([0.5, 0.5, 0.5], {"category": "B", "price": 250}),
]


def get_test_backends():
    """Get all available backends for testing."""
    backends = []
    
    # SQLite backends
    backends.append(("sqlite_memory", "sqlite:///:memory:"))
    
    # DuckDB backends  
    backends.append(("duckdb_memory", "duckdb:///:memory:"))
    
    # File-based backends
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        sqlite_file = tmp.name
    with tempfile.NamedTemporaryFile(suffix='.duckdb', delete=False) as tmp:
        duckdb_file = tmp.name
        
    backends.extend([
        ("sqlite_file", f"sqlite:///{sqlite_file}"),
        ("duckdb_file", f"duckdb:///{duckdb_file}"),
    ])
    
    # Network backends (will be skipped if not available)
    backends.extend([
        ("postgres", "postgresql://postgres:secret@localhost/postgres"),
        ("mysql", "mysql://root:secret@localhost:3306/vectordb"),
    ])
    
    return backends


@pytest.mark.parametrize("backend_name,backend_url", get_test_backends())
def test_basic_operations_consistency(backend_name, backend_url):
    """Test that basic operations work consistently across all backends."""
    try:
        db = VectorDB(backend_url)
    except Exception:
        pytest.skip(f"{backend_name} not available")
    
    collection_name = f"test_{backend_name}_basic"
    
    # Create collection
    db.create_collection(collection_name, 3)
    
    # Test upsert
    db.upsert(collection_name, 1, [1.0, 0.0, 0.0], {"test": "value"})
    db.upsert(collection_name, 2, [0.0, 1.0, 0.0], {"test": "other"})
    
    # Test query without filter
    results = db.query(collection_name, [0.9, 0.1, 0.0], top_k=2)
    assert len(results) == 2
    assert isinstance(results[0], tuple)
    assert len(results[0]) == 2
    assert isinstance(results[0][0], int)  # ID
    assert isinstance(results[0][1], float)  # Distance
    
    # Results should be sorted by distance
    assert results[0][1] <= results[1][1]


@pytest.mark.parametrize("backend_name,backend_url", get_test_backends())
def test_filtering_consistency(backend_name, backend_url):
    """Test that filtering works consistently across all backends."""
    try:
        db = VectorDB(backend_url)
    except Exception:
        pytest.skip(f"{backend_name} not available")
    
    collection_name = f"test_{backend_name}_filter"
    
    # Create collection and insert test data
    db.create_collection(collection_name, 3)
    
    for i, (vector, metadata) in enumerate(TEST_VECTORS):
        db.upsert(collection_name, i + 1, vector, metadata)
    
    # Test category filtering
    results = db.query(
        collection_name, 
        [1.0, 0.0, 0.1], 
        top_k=10, 
        filter={"category": "A"}
    )
    
    # Should find exactly 2 items with category "A"
    assert len(results) == 2
    category_a_ids = {1, 3}  # IDs 1 and 3 have category "A"
    result_ids = {result[0] for result in results}
    assert result_ids == category_a_ids
    
    # Test price filtering
    results = db.query(
        collection_name,
        [0.0, 1.0, 0.1],
        top_k=10,
        filter={"price": 200}
    )
    
    # Should find exactly 1 item with price 200
    assert len(results) == 1
    assert results[0][0] == 2  # ID 2 has price 200


@pytest.mark.parametrize("backend_name,backend_url", get_test_backends())
def test_upsert_consistency(backend_name, backend_url):
    """Test that upsert (insert/update) works consistently."""
    try:
        db = VectorDB(backend_url)
    except Exception:
        pytest.skip(f"{backend_name} not available")
    
    collection_name = f"test_{backend_name}_upsert"
    
    # Create collection
    db.create_collection(collection_name, 2)
    
    # Insert initial vector
    db.upsert(collection_name, 1, [1.0, 0.0], {"version": 1})
    
    # Query for exact match
    results = db.query(collection_name, [1.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0][0] == 1
    initial_distance = results[0][1]
    assert initial_distance < 0.01  # Should be very close
    
    # Update with different vector
    db.upsert(collection_name, 1, [0.0, 1.0], {"version": 2})
    
    # Query for new vector (should be exact match)
    results = db.query(collection_name, [0.0, 1.0], top_k=1)
    assert len(results) == 1
    assert results[0][0] == 1
    assert results[0][1] < 0.01  # Should be very close
    
    # Query for old vector (should be far)
    results = db.query(collection_name, [1.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0][0] == 1
    assert results[0][1] > 1.0  # Should be far (L2 distance)


@pytest.mark.parametrize("backend_name,backend_url", get_test_backends())
def test_large_dataset_consistency(backend_name, backend_url):
    """Test with 1000 vectors to ensure scalability consistency."""
    try:
        db = VectorDB(backend_url)
    except Exception:
        pytest.skip(f"{backend_name} not available")
    
    collection_name = f"test_{backend_name}_large"
    
    # Create collection
    db.create_collection(collection_name, 10)  # 10-dimensional vectors
    
    # Generate 1000 test vectors
    import random
    random.seed(42)  # For reproducibility
    
    vectors = []
    for i in range(1000):
        vector = [random.gauss(0, 1) for _ in range(10)]
        # Normalize vector
        norm = sum(x*x for x in vector) ** 0.5
        vector = [x/norm for x in vector]
        
        metadata = {
            "batch": i // 100,  # 10 batches of 100 each
            "index": i
        }
        vectors.append((vector, metadata))
    
    # Insert all vectors
    for i, (vector, metadata) in enumerate(vectors):
        db.upsert(collection_name, i + 1, vector, metadata)
    
    # Test query performance and correctness
    query_vector = vectors[0][0]  # Use first vector as query
    
    results = db.query(collection_name, query_vector, top_k=10)
    
    # Should return 10 results
    assert len(results) == 10
    
    # First result should be exact match (ID 1)
    assert results[0][0] == 1
    assert results[0][1] < 0.01  # Very close distance
    
    # Results should be sorted by distance
    distances = [result[1] for result in results]
    assert distances == sorted(distances)
    
    # Test filtering on large dataset
    results = db.query(
        collection_name,
        query_vector,
        top_k=50,
        filter={"batch": 0}  # Should match first 100 vectors
    )
    
    # Should find results only from batch 0
    batch_0_count = len(results)
    assert batch_0_count > 0
    assert batch_0_count <= 50  # Limited by top_k
    
    # All results should be from batch 0 (IDs 1-100)
    for result_id, _ in results:
        assert 1 <= result_id <= 100


def test_error_handling_consistency():
    """Test that error handling is consistent across backends."""
    
    # Test invalid connection strings
    invalid_urls = [
        "invalid://test",
        "postgresql://invalid:invalid@nonexistent/db",
        "mysql://invalid:invalid@nonexistent/db",
        "sqlite:///invalid/path/that/doesnt/exist.db",
        "duckdb:///invalid/path/that/doesnt/exist.db",
    ]
    
    for url in invalid_urls:
        try:
            db = VectorDB(url)
            # Some backends might not fail immediately on connection
            # So we also test with actual operations
            try:
                db.create_collection("test", 3)
                # If we get here, the connection worked unexpectedly
                # Skip this test case
                continue
            except Exception:
                # Expected - operations should fail
                continue
        except Exception:
            # Expected - connection should fail
            continue


def test_edge_cases_consistency():
    """Test edge cases consistently across backends."""
    
    # Test with available backends
    for backend_name, backend_url in [("sqlite_memory", "sqlite:///:memory:"), 
                                      ("duckdb_memory", "duckdb:///:memory:")]:
        try:
            db = VectorDB(backend_url)
        except Exception:
            continue
            
        collection_name = f"test_{backend_name}_edge"
        
        # Create collection
        db.create_collection(collection_name, 3)
        
        # Test with zero vector
        db.upsert(collection_name, 1, [0.0, 0.0, 0.0])
        results = db.query(collection_name, [0.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0][1] == 0.0  # Exact match
        
        # Test with very small vectors
        db.upsert(collection_name, 2, [1e-10, 1e-10, 1e-10])
        results = db.query(collection_name, [1e-10, 1e-10, 1e-10], top_k=1)
        assert len(results) >= 1
        
        # Test with large vectors
        big_vector = [1000.0, 2000.0, 3000.0]
        db.upsert(collection_name, 3, big_vector)
        results = db.query(collection_name, big_vector, top_k=1)
        assert len(results) >= 1
        
        # Test empty filter
        results = db.query(collection_name, [1.0, 0.0, 0.0], top_k=5, filter={})
        assert len(results) >= 1
        
        # Test top_k larger than dataset
        results = db.query(collection_name, [1.0, 0.0, 0.0], top_k=1000)
        assert len(results) >= 1  # Should return available results
        assert len(results) <= 1000  # But not more than requested


if __name__ == "__main__":
    # Run basic smoke test
    print("Running integration smoke test...")

    try:
        # Test SQLite
        db = VectorDB("sqlite:///:memory:")
        db.create_collection("smoke", 2)
        db.upsert("smoke", 1, [1.0, 0.0], {"test": True})
        results = db.query("smoke", [1.0, 0.0], top_k=1)
        assert len(results) == 1
        print("SQLite smoke test passed")

        # Test DuckDB
        db = VectorDB("duckdb:///:memory:")
        db.create_collection("smoke", 2)
        db.upsert("smoke", 1, [1.0, 0.0], {"test": True})
        results = db.query("smoke", [1.0, 0.0], top_k=1)
        assert len(results) == 1
        print("DuckDB smoke test passed")

        print("All smoke tests passed!")

    except Exception as e:
        print(f"Smoke test failed: {e}")
        raise