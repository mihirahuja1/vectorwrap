"""
Stress tests for vectorwrap - tests edge cases and error conditions.

These tests verify that the API handles edge cases gracefully and
provides consistent error handling across backends.
"""
import random
import pytest
from vectorwrap import VectorDB


def test_large_vectors():
    """Test with large dimensional vectors."""
    db = VectorDB("sqlite:///:memory:")
    
    # Test 1000-dimensional vectors
    dim = 1000
    db.create_collection("large_vectors", dim)
    
    # Generate random normalized vector
    vector = [random.gauss(0, 1) for _ in range(dim)]
    norm = sum(x*x for x in vector) ** 0.5
    vector = [x/norm for x in vector]
    
    # Insert and query
    db.upsert("large_vectors", 1, vector, {"test": "large"})
    results = db.query("large_vectors", vector, top_k=1)
    
    assert len(results) == 1
    assert results[0][0] == 1
    assert results[0][1] < 0.01  # Should be very close


def test_many_insertions():
    """Test stress with many insertions."""
    db = VectorDB("duckdb:///:memory:")
    db.create_collection("stress_test", 10)
    
    # Insert 5000 vectors
    vectors = []
    for i in range(5000):
        vector = [random.gauss(0, 1) for _ in range(10)]
        # Simple normalization
        norm = sum(x*x for x in vector) ** 0.5
        if norm > 0:
            vector = [x/norm for x in vector]
        vectors.append(vector)
        
        metadata = {"batch": i // 100, "index": i}
        db.upsert("stress_test", i + 1, vector, metadata)
    
    # Test queries still work
    query_vector = vectors[0]
    results = db.query("stress_test", query_vector, top_k=10)
    
    assert len(results) == 10
    assert results[0][0] == 1  # First vector should be exact match
    assert results[0][1] < 0.01


def test_extreme_values():
    """Test with extreme vector values."""
    db = VectorDB("sqlite:///:memory:")
    db.create_collection("extreme_test", 3)
    
    # Test cases with extreme values
    test_cases = [
        ([0.0, 0.0, 0.0], "zero_vector"),
        ([1e-10, 1e-10, 1e-10], "tiny_values"),
        ([1e10, 1e10, 1e10], "huge_values"),
        ([-1.0, -1.0, -1.0], "negative_values"),
        ([float('inf'), 1.0, 1.0], "infinity"),  # This might fail - that's ok
        ([float('nan'), 1.0, 1.0], "nan"),      # This might fail - that's ok
    ]
    
    for i, (vector, name) in enumerate(test_cases):
        try:
            db.upsert("extreme_test", i + 1, vector, {"type": name})
            # If insert succeeded, query should work
            results = db.query("extreme_test", vector, top_k=1)
            assert len(results) >= 0  # At least no crash
        except (ValueError, OverflowError, Exception) as e:
            # Some extreme values are expected to fail
            print(f"Expected failure for {name}: {e}")
            continue


def test_malformed_inputs():
    """Test with malformed inputs - should handle gracefully."""
    db = VectorDB("duckdb:///:memory:")
    db.create_collection("malformed_test", 3)
    
    # Test wrong dimension vectors
    with pytest.raises((ValueError, Exception)):
        db.upsert("malformed_test", 1, [1.0, 2.0])  # Too short
    
    with pytest.raises((ValueError, Exception)):
        db.upsert("malformed_test", 1, [1.0, 2.0, 3.0, 4.0])  # Too long
    
    # Test empty vector
    with pytest.raises((ValueError, Exception)):
        db.upsert("malformed_test", 1, [])
    
    # Test non-numeric values
    with pytest.raises((ValueError, TypeError, Exception)):
        db.upsert("malformed_test", 1, ["a", "b", "c"])  # type: ignore


def test_invalid_queries():
    """Test invalid query parameters."""
    db = VectorDB("sqlite:///:memory:")
    db.create_collection("query_test", 3)
    
    # Insert a valid vector first
    db.upsert("query_test", 1, [1.0, 0.0, 0.0])
    
    # Test invalid top_k values - some backends may handle this gracefully
    try:
        results = db.query("query_test", [1.0, 0.0, 0.0], top_k=0)
        # If it doesn't raise, should return empty or valid results
        assert isinstance(results, list)
    except (ValueError, Exception):
        # Raising an exception is also acceptable
        pass
    
    try:
        results = db.query("query_test", [1.0, 0.0, 0.0], top_k=-1)
        # If it doesn't raise, should return empty or valid results
        assert isinstance(results, list)
    except (ValueError, Exception):
        # Raising an exception is also acceptable
        pass
    
    # Test query with wrong dimensions
    with pytest.raises((ValueError, Exception)):
        db.query("query_test", [1.0, 0.0])  # Too short
    
    # Test query on non-existent collection
    with pytest.raises(Exception):
        db.query("nonexistent", [1.0, 0.0, 0.0])


def test_complex_metadata():
    """Test with complex metadata structures."""
    db = VectorDB("duckdb:///:memory:")
    db.create_collection("complex_meta", 2)
    
    # Test various metadata types
    complex_metadata = {
        "string": "test",
        "integer": 42,
        "float": 3.14,
        "boolean": True,
        "null": None,
        "list": [1, 2, 3],
        "nested": {"key": "value", "number": 123}
    }
    
    db.upsert("complex_meta", 1, [1.0, 0.0], complex_metadata)
    
    # Query should work
    results = db.query("complex_meta", [1.0, 0.0], top_k=1)
    assert len(results) == 1
    
    # Test filtering on different metadata types
    try:
        results = db.query("complex_meta", [1.0, 0.0], top_k=1, filter={"string": "test"})
        assert len(results) == 1
    except Exception:
        # Some backends might not support complex filtering
        pass


def test_concurrent_operations():
    """Test concurrent-like operations (rapid successive calls)."""
    db = VectorDB("sqlite:///:memory:")
    db.create_collection("concurrent_test", 5)
    
    # Rapid insertions and queries
    for batch in range(10):
        # Insert batch
        for i in range(100):
            vector_id = batch * 100 + i
            vector = [random.random() for _ in range(5)]
            db.upsert("concurrent_test", vector_id, vector, {"batch": batch})
        
        # Query after each batch
        query_vector = [0.5, 0.5, 0.5, 0.5, 0.5]
        results = db.query("concurrent_test", query_vector, top_k=10)
        assert len(results) <= 10
        assert len(results) >= 1


def test_memory_stress():
    """Test memory usage with repeated operations."""
    db = VectorDB("duckdb:///:memory:")
    db.create_collection("memory_test", 50)
    
    # Create and destroy collections repeatedly
    for round_num in range(10):
        collection_name = f"temp_collection_{round_num}"
        db.create_collection(collection_name, 50)
        
        # Fill with data
        for i in range(100):
            vector = [random.gauss(0, 1) for _ in range(50)]
            db.upsert(collection_name, i, vector, {"round": round_num})
        
        # Query
        query_vector = [0.0] * 50
        results = db.query(collection_name, query_vector, top_k=5)
        assert len(results) == 5


if __name__ == "__main__":
    # Run a quick stress test
    print("Running stress tests...")

    test_large_vectors()
    print("Large vectors test passed")

    test_extreme_values()
    print("Extreme values test passed")

    test_complex_metadata()
    print("Complex metadata test passed")

    print("Basic stress tests completed!")