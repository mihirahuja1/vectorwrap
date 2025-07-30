from vectorwrap import VectorDB


def test_sqlite_basic(tmp_path):
    db = VectorDB(f"sqlite:///{tmp_path/'test.db'}")
    db.create_collection("t", 3)
    db.upsert("t", 7, [0.0, 0.0, 1.0])
    res = db.query("t", [0.0, 0.1, 0.9], 1)
    assert res[0][0] == 7


def test_sqlite_filtering(tmp_path):
    """Test SQLite filtering functionality with metadata."""
    db = VectorDB(f"sqlite:///{tmp_path/'filter_test.db'}")
    db.create_collection("products", 3)
    
    # Insert vectors with metadata
    db.upsert("products", 1, [1.0, 0.0, 0.0], {"category": "phone", "price": 999})
    db.upsert("products", 2, [0.0, 1.0, 0.0], {"category": "laptop", "price": 1500})
    db.upsert("products", 3, [0.0, 0.0, 1.0], {"category": "phone", "price": 799})
    db.upsert("products", 4, [1.0, 1.0, 0.0], {"category": "tablet", "price": 699})
    
    # Query without filter
    results = db.query("products", [1.0, 0.0, 0.1], top_k=4)
    assert len(results) == 4
    
    # Query with category filter
    results = db.query("products", [1.0, 0.0, 0.1], top_k=5, filter={"category": "phone"})
    assert len(results) == 2
    phone_ids = {result[0] for result in results}
    assert phone_ids == {1, 3}
    
    # Query with price filter
    results = db.query("products", [0.0, 1.0, 0.1], top_k=5, filter={"price": 1500})
    assert len(results) == 1
    assert results[0][0] == 2
    
    # Query with multiple filters
    results = db.query("products", [1.0, 0.0, 0.1], top_k=5, filter={"category": "phone", "price": 799})
    assert len(results) == 1
    assert results[0][0] == 3


def test_sqlite_memory():
    """Test SQLite in-memory database."""
    db = VectorDB("sqlite:///:memory:")
    db.create_collection("vectors", 2)
    
    # Insert multiple vectors
    db.upsert("vectors", 1, [1.0, 0.0], {"type": "A"})
    db.upsert("vectors", 2, [0.0, 1.0], {"type": "B"})
    db.upsert("vectors", 3, [1.0, 1.0], {"type": "A"})
    
    # Query without filter
    results = db.query("vectors", [0.9, 0.1], top_k=2)
    assert len(results) == 2
    assert results[0][0] == 1  # Should be closest to [1.0, 0.0]
    
    # Query with filter
    results = db.query("vectors", [0.9, 0.1], top_k=2, filter={"type": "A"})
    assert len(results) == 2
    # Both results should have type "A"
    type_a_ids = {result[0] for result in results}
    assert type_a_ids == {1, 3}


def test_sqlite_upsert():
    """Test SQLite upsert functionality."""
    db = VectorDB("sqlite:///:memory:")
    db.create_collection("test", 2)
    
    # Insert
    db.upsert("test", 1, [1.0, 2.0], {"version": 1})
    results = db.query("test", [1.0, 2.0], top_k=1)
    assert len(results) == 1
    assert results[0][1] < 0.01  # Should be close to exact match
    
    # Update same ID
    db.upsert("test", 1, [2.0, 3.0], {"version": 2}) 
    results = db.query("test", [2.0, 3.0], top_k=1)
    assert len(results) == 1
    assert results[0][0] == 1
    assert results[0][1] < 0.01  # Should be close to exact match
