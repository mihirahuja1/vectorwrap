# Testing Appwrite Integration

This guide explains how to test the vectorwrap Appwrite integration locally.

## Quick Start

### Method 1: Docker Compose (Recommended)

```bash
# Start MariaDB test container
cd tests
docker-compose -f docker-compose.appwrite-test.yml up -d

# Wait for MariaDB to be ready
sleep 10

# Run tests
python test_appwrite_local.py

# Cleanup
docker-compose -f docker-compose.appwrite-test.yml down -v
```

### Method 2: Docker Run

```bash
# Start MariaDB
docker run -d \
  --name mariadb-test \
  -e MYSQL_ROOT_PASSWORD=test123 \
  -e MYSQL_DATABASE=appwrite_test \
  -p 3306:3306 \
  mariadb:11.8

# Wait for startup
sleep 10

# Run tests
python tests/test_appwrite_local.py

# Cleanup
docker stop mariadb-test
docker rm mariadb-test
```

## Test Coverage

The test suite covers:

1. **Connection Methods**
   - `from_connection_string()`
   - Direct instantiation
   - `from_appwrite_client()` (with Appwrite SDK)

2. **Collection Operations**
   - Initialize collection
   - Get collection info
   - Check native vector support

3. **Document Operations**
   - Add single document
   - Add batch documents
   - Update document
   - Delete document

4. **Search Operations**
   - Vector similarity search
   - Filtered search (by metadata)
   - Top-K retrieval

5. **Data Integrity**
   - Verify document count
   - Check updates persist
   - Confirm deletions

## Test Output

Expected output:

```
============================================================
Appwrite Integration Test Suite
============================================================

Prerequisites:
  - MariaDB 11.8+ running on localhost:3306
  - Database: appwrite_test
  - User: root, Password: test123

Setup with Docker:
  docker run -d --name mariadb-test \
    -e MYSQL_ROOT_PASSWORD=test123 \
    -e MYSQL_DATABASE=appwrite_test \
    -p 3306:3306 mariadb:11.8
============================================================

============================================================
Testing Connection Methods
============================================================

1. Testing from_connection_string()...
   [OK] from_connection_string() works

2. Testing direct instantiation...
   [OK] Direct instantiation works

[OK] All connection methods work

============================================================
Testing Appwrite Integration with Local MariaDB
============================================================

1. Creating AppwriteVectorStore...
   [OK] Created vector store
   [OK] Collection: test_docs
   [OK] Dimension: 384
   [OK] Native vectors: True

2. Initializing collection...
   [OK] Collection initialized

3. Adding test documents...
   [OK] Added: Introduction to AI
   [OK] Added: Machine Learning Basics
   [OK] Added: Cooking Recipes
   [OK] Added: Deep Learning Guide

4. Testing vector search...
   [OK] Found 2 results (filtered by category=tech)
      - ID: 1, Distance: 0.3421
      - ID: 2, Distance: 0.4589
   [OK] Found 4 results (no filter)

5. Testing document update...
   [OK] Updated document ID 1

6. Testing document deletion...
   [OK] Deleted document ID 3

7. Verifying remaining documents...
   [OK] Total documents remaining: 3
      - Document ID: 1
      - Document ID: 2
      - Document ID: 4
   [OK] Count verified (3 documents)

8. Testing batch document addition...
   [OK] Added 2 documents in batch

9. Getting collection info...
   [OK] collection_name: test_docs
   [OK] dimension: 384
   [OK] distance_metric: cosine
   [OK] backend: MariaDB
   [OK] native_vectors: True

10. Cleaning up...
   [OK] Connection closed

============================================================
All tests passed! ✓
============================================================

✓ All tests completed successfully!
```

## Testing with Real Appwrite

To test with a real Appwrite instance:

1. **Access Appwrite's MariaDB**:
   ```bash
   # For self-hosted Appwrite
   docker exec -it appwrite-mariadb mysql -u appwrite -p
   ```

2. **Get connection details**:
   - Host: `localhost` (or Appwrite server IP)
   - Port: `3306`
   - Database: `appwrite`
   - User: `appwrite`
   - Password: (from Appwrite env vars)

3. **Update test connection string**:
   ```python
   connection_string = "mariadb://appwrite:password@localhost:3306/appwrite"
   ```

4. **Run tests**:
   ```bash
   python tests/test_appwrite_local.py
   ```

## Testing in Appwrite Functions

Create a test function in Appwrite:

```python
# appwrite_function_test.py
from vectorwrap.integrations.appwrite import create_vector_store_for_appwrite_function
import json

def main(req, res):
    try:
        # Create vector store
        vector_store = create_vector_store_for_appwrite_function(
            collection_name="function_test",
            dimension=384
        )

        # Add test document
        import random
        test_embedding = [random.random() for _ in range(384)]

        vector_store.add_document(
            doc_id=1,
            embedding=test_embedding,
            metadata={"test": "success", "source": "function"}
        )

        # Search
        results = vector_store.backend.query(
            "function_test",
            test_embedding,
            top_k=1
        )

        return res.json({
            "status": "success",
            "results": len(results),
            "message": "Appwrite integration working!"
        })

    except Exception as e:
        return res.json({
            "status": "error",
            "error": str(e)
        }, 500)
```

## Troubleshooting

### MariaDB Connection Errors

```python
# Error: Can't connect to MySQL server
# Solution: Ensure MariaDB is running and accessible
docker ps | grep mariadb
```

### Native Vector Support Not Detected

```python
# If native_vectors shows False, check MariaDB version
# Should be 11.8+
docker exec mariadb-test mysql -V
```

### Permission Errors

```python
# Grant necessary permissions
docker exec -it mariadb-test mysql -u root -ptest123
# In MySQL:
GRANT ALL PRIVILEGES ON appwrite_test.* TO 'appwrite'@'%';
FLUSH PRIVILEGES;
```

## Cleanup

```bash
# Stop and remove test container
docker-compose -f tests/docker-compose.appwrite-test.yml down -v

# Or with docker run
docker stop mariadb-test && docker rm mariadb-test
```

## CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Test Appwrite Integration

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      mariadb:
        image: mariadb:11.8
        env:
          MYSQL_ROOT_PASSWORD: test123
          MYSQL_DATABASE: appwrite_test
        ports:
          - 3306:3306
        options: >-
          --health-cmd="healthcheck.sh --connect --innodb_initialized"
          --health-interval=10s
          --health-timeout=5s
          --health-retries=5

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -e .
          pip install numpy

      - name: Run Appwrite integration tests
        run: python tests/test_appwrite_local.py
```

## Next Steps

- Test with real OpenAI embeddings
- Benchmark performance vs external vector DBs
- Test with large datasets (10K+ vectors)
- Implement metadata filtering edge cases
- Test concurrent access patterns
