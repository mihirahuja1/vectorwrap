"""Tests for ClickHouse backend."""
import pytest
from vectorwrap import VectorDB


@pytest.fixture
def clickhouse_url():
    """Return ClickHouse connection URL for testing."""
    # This assumes a local ClickHouse instance is running
    # For CI/CD, you might want to use environment variables
    return "clickhouse://default@localhost:8123/default"


@pytest.fixture
def db(clickhouse_url):
    """Create a ClickHouse database connection for testing."""
    try:
        return VectorDB(clickhouse_url)
    except RuntimeError as e:
        pytest.skip(f"ClickHouse not available: {e}")


def test_create_collection(db):
    """Test collection creation."""
    db.create_collection("test_collection", dim=128)
    # No error means success


def test_upsert_and_query(db):
    """Test basic upsert and query operations."""
    db.create_collection("test_vectors", dim=3)

    # Insert some test vectors
    db.upsert("test_vectors", 1, [1.0, 0.0, 0.0], {"label": "x-axis"})
    db.upsert("test_vectors", 2, [0.0, 1.0, 0.0], {"label": "y-axis"})
    db.upsert("test_vectors", 3, [0.0, 0.0, 1.0], {"label": "z-axis"})

    # Query for nearest neighbors
    results = db.query("test_vectors", [1.0, 0.1, 0.0], top_k=2)

    assert len(results) == 2
    assert results[0][0] == 1  # First result should be ID 1
    assert isinstance(results[0][1], float)  # Distance should be a float


def test_query_with_filter(db):
    """Test querying with metadata filters."""
    db.create_collection("test_filtered", dim=3)

    # Insert vectors with different categories
    db.upsert("test_filtered", 1, [1.0, 0.0, 0.0], {"category": "A"})
    db.upsert("test_filtered", 2, [0.9, 0.1, 0.0], {"category": "B"})
    db.upsert("test_filtered", 3, [0.8, 0.2, 0.0], {"category": "A"})

    # Query with filter
    results = db.query(
        "test_filtered",
        [1.0, 0.0, 0.0],
        top_k=5,
        filter={"category": "A"}
    )

    # Should only return vectors with category A
    assert len(results) <= 2
    for result_id, _ in results:
        assert result_id in [1, 3]


def test_upsert_updates_existing(db):
    """Test that upsert updates existing vectors."""
    db.create_collection("test_update", dim=2)

    # Insert initial vector
    db.upsert("test_update", 1, [1.0, 0.0], {"version": "v1"})

    # Update the same ID
    db.upsert("test_update", 1, [0.0, 1.0], {"version": "v2"})

    # Query should return the updated vector
    results = db.query("test_update", [0.0, 1.0], top_k=1)

    assert len(results) == 1
    assert results[0][0] == 1
    assert results[0][1] < 0.1  # Should be very close to the query vector


def test_empty_metadata(db):
    """Test upsert with no metadata."""
    db.create_collection("test_no_meta", dim=2)

    # Insert without metadata
    db.upsert("test_no_meta", 1, [1.0, 0.0])

    # Query should work fine
    results = db.query("test_no_meta", [1.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0][0] == 1
