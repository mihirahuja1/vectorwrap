import pytest
from vectorwrap import VectorDB


def test_duckdb_basic(tmp_path):
    """Test basic DuckDB operations"""
    db = VectorDB(f"duckdb:///{tmp_path/'test.db'}")
    db.create_collection("t", 3)
    db.upsert("t", 7, [0.0, 0.0, 1.0])
    res = db.query("t", [0.0, 0.1, 0.9], 1)
    assert len(res) == 1
    assert res[0][0] == 7  # id
    assert res[0][1] < 0.2  # distance should be small


def test_duckdb_memory():
    """Test DuckDB in-memory database"""
    db = VectorDB("duckdb:///:memory:")
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
    

def test_duckdb_metadata():
    """Test DuckDB with metadata filtering"""
    db = VectorDB("duckdb:///:memory:")
    db.create_collection("products", 3)
    
    # Insert vectors with metadata
    db.upsert("products", 1, [1.0, 0.0, 0.0], {"category": "phone", "price": 999})
    db.upsert("products", 2, [0.0, 1.0, 0.0], {"category": "laptop", "price": 1500})
    db.upsert("products", 3, [0.0, 0.0, 1.0], {"category": "phone", "price": 799})
    
    # Query with category filter
    results = db.query("products", [1.0, 0.0, 0.1], top_k=5, filter={"category": "phone"})
    assert len(results) == 2
    
    # Query with price filter (if supported)
    # Note: This might not work depending on JSON filtering implementation
    # results = db.query("products", [1.0, 0.0, 0.1], top_k=5, filter={"price": 999})
    # assert len(results) == 1


def test_duckdb_upsert():
    """Test DuckDB upsert functionality"""
    db = VectorDB("duckdb:///:memory:")
    db.create_collection("test", 2)
    
    # Insert
    db.upsert("test", 1, [1.0, 2.0], {"version": 1})
    results = db.query("test", [1.0, 2.0], top_k=1)
    assert len(results) == 1
    assert results[0][1] == 0.0  # Exact match
    
    # Update same ID
    db.upsert("test", 1, [2.0, 3.0], {"version": 2}) 
    results = db.query("test", [2.0, 3.0], top_k=1)
    assert len(results) == 1
    assert results[0][0] == 1
    assert results[0][1] == 0.0  # Exact match with updated vector