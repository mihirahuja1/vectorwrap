import pytest
from vectorwrap import VectorDB


# PostgreSQL connection string - skip tests if not available
PG_URL = "postgresql://postgres:secret@localhost/postgres"


def test_postgres_connection():
    """Test PostgreSQL connection and basic setup"""
    try:
        db = VectorDB(PG_URL)
        # This will fail if PostgreSQL is not available or pgvector extension is missing
        db.create_collection("connection_test", 2)
    except Exception:
        pytest.skip("PostgreSQL not available or pgvector extension missing")


def test_postgres_basic():
    """Test basic PostgreSQL operations"""
    try:
        db = VectorDB(PG_URL)
    except Exception:
        pytest.skip("PostgreSQL not available")
    
    # Create collection
    db.create_collection("basic_test", 3)
    
    # Insert vector
    db.upsert("basic_test", 7, [0.0, 0.0, 1.0])
    
    # Query vector
    res = db.query("basic_test", [0.0, 0.1, 0.9], 1)
    assert len(res) == 1
    assert res[0][0] == 7  # id
    assert res[0][1] < 0.2  # distance should be small


def test_postgres_multiple_vectors():
    """Test PostgreSQL with multiple vectors"""
    try:
        db = VectorDB(PG_URL)
    except Exception:
        pytest.skip("PostgreSQL not available")
    
    db.create_collection("multi_test", 2)
    
    # Insert multiple vectors
    db.upsert("multi_test", 1, [1.0, 0.0])
    db.upsert("multi_test", 2, [0.0, 1.0])
    db.upsert("multi_test", 3, [1.0, 1.0])
    db.upsert("multi_test", 4, [-1.0, 0.0])
    
    # Query for closest to [0.9, 0.1]
    results = db.query("multi_test", [0.9, 0.1], top_k=2)
    assert len(results) == 2
    assert results[0][0] == 1  # Should be closest to [1.0, 0.0]
    assert results[1][0] == 3  # Second closest should be [1.0, 1.0]
    
    # Verify distances are sorted
    assert results[0][1] <= results[1][1]


def test_postgres_metadata():
    """Test PostgreSQL with metadata (note: current implementation doesn't store metadata)"""
    try:
        db = VectorDB(PG_URL)
    except Exception:
        pytest.skip("PostgreSQL not available")
    
    db.create_collection("meta_test", 3)
    
    # Insert vectors with metadata (metadata is ignored in current implementation)
    db.upsert("meta_test", 1, [1.0, 0.0, 0.0], {"category": "phone", "price": 999})
    db.upsert("meta_test", 2, [0.0, 1.0, 0.0], {"category": "laptop", "price": 1500})
    db.upsert("meta_test", 3, [0.0, 0.0, 1.0], {"category": "phone", "price": 799})
    
    # Query all vectors (filter is ignored in current implementation)
    results = db.query("meta_test", [1.0, 0.0, 0.1], top_k=3)
    assert len(results) == 3
    assert results[0][0] == 1  # Closest to [1.0, 0.0, 0.0]


def test_postgres_upsert():
    """Test PostgreSQL upsert functionality"""
    try:
        db = VectorDB(PG_URL)
    except Exception:
        pytest.skip("PostgreSQL not available")
    
    db.create_collection("upsert_test", 2)
    
    # Insert vector
    db.upsert("upsert_test", 1, [1.0, 2.0])
    results = db.query("upsert_test", [1.0, 2.0], top_k=1)
    assert len(results) == 1
    assert results[0][1] == 0.0  # Exact match
    
    # Update same ID with different vector
    db.upsert("upsert_test", 1, [2.0, 3.0])
    results = db.query("upsert_test", [2.0, 3.0], top_k=1)
    assert len(results) == 1
    assert results[0][0] == 1
    assert results[0][1] == 0.0  # Exact match with updated vector
    
    # Verify old vector is gone
    results = db.query("upsert_test", [1.0, 2.0], top_k=1)
    assert len(results) == 1
    assert results[0][1] > 1.0  # Distance should be large now


def test_postgres_large_vectors():
    """Test PostgreSQL with larger dimensional vectors"""
    try:
        db = VectorDB(PG_URL)
    except Exception:
        pytest.skip("PostgreSQL not available")
    
    # Test with 384-dimensional vectors (common embedding size)
    dim = 384
    db.create_collection("large_test", dim)
    
    # Create normalized random-like vectors
    import random
    random.seed(42)
    
    vector1 = [random.gauss(0, 1) for _ in range(dim)]
    vector2 = [random.gauss(0, 1) for _ in range(dim)]
    
    # Normalize vectors
    import math
    norm1 = math.sqrt(sum(x*x for x in vector1))
    norm2 = math.sqrt(sum(x*x for x in vector2))
    vector1 = [x/norm1 for x in vector1]
    vector2 = [x/norm2 for x in vector2]
    
    # Insert vectors
    db.upsert("large_test", 1, vector1)
    db.upsert("large_test", 2, vector2)
    
    # Query
    results = db.query("large_test", vector1, top_k=2)
    assert len(results) == 2
    assert results[0][0] == 1  # Exact match should be first
    assert results[0][1] < 0.01  # Distance should be very small


def test_postgres_hnsw_index():
    """Test that HNSW index is working (performance test)"""
    try:
        db = VectorDB(PG_URL)
    except Exception:
        pytest.skip("PostgreSQL not available")
    
    # Create collection with HNSW index
    dim = 100
    db.create_collection("hnsw_test", dim)
    
    # Insert many vectors to test index performance
    import random
    import time
    random.seed(42)
    
    num_vectors = 1000
    vectors = []
    
    start_time = time.time()
    for i in range(num_vectors):
        vector = [random.gauss(0, 1) for _ in range(dim)]
        # Simple normalization
        norm = sum(x*x for x in vector) ** 0.5
        vector = [x/norm for x in vector]
        vectors.append(vector)
        db.upsert("hnsw_test", i, vector)
    
    insert_time = time.time() - start_time
    
    # Test query performance
    query_vector = vectors[0]  # Use first vector as query
    
    start_time = time.time()
    results = db.query("hnsw_test", query_vector, top_k=10)
    query_time = time.time() - start_time
    
    assert len(results) == 10
    assert results[0][0] == 0  # Should find exact match first
    assert results[0][1] < 0.01  # Distance should be very small
    
    # Performance assertions (these are loose bounds)
    assert insert_time < 30.0  # Should insert 1000 vectors in under 30 seconds
    assert query_time < 1.0    # Should query in under 1 second
    
    print(f"PostgreSQL HNSW Performance:")
    print(f"  Insert: {num_vectors/insert_time:.1f} vectors/sec")
    print(f"  Query:  {1/query_time:.1f} QPS")


def test_postgres_ensure_k():
    """Test PostgreSQL ensure_k parameter for better recall"""
    try:
        db = VectorDB(PG_URL)
    except Exception:
        pytest.skip("PostgreSQL not available")
    
    db.create_collection("ensure_k_test", 3)
    
    # Insert vectors
    db.upsert("ensure_k_test", 1, [1.0, 0.0, 0.0])
    db.upsert("ensure_k_test", 2, [0.0, 1.0, 0.0])
    db.upsert("ensure_k_test", 3, [0.0, 0.0, 1.0])
    db.upsert("ensure_k_test", 4, [0.5, 0.5, 0.0])
    
    # Query with ensure_k=True for better recall
    results = db.query("ensure_k_test", [0.6, 0.4, 0.0], top_k=3, ensure_k=True)
    assert len(results) == 3
    
    # Query with ensure_k=False (default)
    results_default = db.query("ensure_k_test", [0.6, 0.4, 0.0], top_k=3, ensure_k=False)
    assert len(results_default) == 3
    
    # Both should find results, but ensure_k might have better accuracy
    # (This is mainly a smoke test to ensure the parameter works)