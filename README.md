# vectorwrap 0.3.0a2 [![PyPI version](https://img.shields.io/pypi/v/vectorwrap)](https://pypi.org/project/vectorwrap/)

**One API — multiple vector databases**

Switch between PostgreSQL, MySQL, and SQLite vector backends with a single line of code. Perfect for prototyping, testing, and production deployments.

## 🚀 Quick Start

```bash
# Core install (PostgreSQL + MySQL support)
pip install vectorwrap

# Add SQLite support (requires system SQLite with extension support)
pip install "vectorwrap[sqlite]"
```

```python
from vectorwrap import VectorDB

# Your embedding function (use OpenAI, Hugging Face, etc.)
def embed(text: str) -> list[float]:
    # Return your 1536-dim embeddings here
    return [0.1, 0.2, ...] 

# Connect to any supported database
db = VectorDB("postgresql://user:pass@host/db")  # or mysql://... or sqlite:///path.db
db.create_collection("products", dim=1536)

# Insert vectors with metadata
db.upsert("products", 1, embed("Apple iPhone 15 Pro"), {"category": "phone", "price": 999})
db.upsert("products", 2, embed("Samsung Galaxy S24"), {"category": "phone", "price": 899})

# Semantic search with filtering
results = db.query(
    collection="products",
    query_vector=embed("latest smartphone"),
    top_k=5,
    filter={"category": "phone"}
)
print(results)  # → [(1, 0.023), (2, 0.087)]
```

## 🗄️ Supported Backends

| Database | Vector Type | Indexing | Installation | Notes |
|----------|-------------|----------|--------------|-------|
| **PostgreSQL 16+ + pgvector** | `VECTOR(n)` | HNSW | `CREATE EXTENSION vector;` | Production ready |
| **MySQL 8.2+ HeatWave** | `VECTOR(n)` | Automatic | Built-in | Native vector support |
| **MySQL ≤8.0 (legacy)** | JSON arrays | None | Built-in | Slower, Python distance |
| **SQLite + sqlite-vss** | Virtual table | HNSW | `pip install "vectorwrap[sqlite]"` | Great for prototyping |

## 📖 Examples

### Complete Example with OpenAI Embeddings

```python
from openai import OpenAI
from vectorwrap import VectorDB

client = OpenAI()

def embed(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Use any database - just change the connection string!
db = VectorDB("postgresql://user:pass@localhost/vectors")
db.create_collection("documents", dim=1536)

# Add some documents
documents = [
    ("Python is a programming language", {"topic": "programming"}),
    ("Machine learning uses neural networks", {"topic": "ai"}),
    ("Databases store structured data", {"topic": "data"}),
]

for i, (text, metadata) in enumerate(documents):
    db.upsert("documents", i, embed(text), metadata)

# Search for similar content
query = "What is artificial intelligence?"
results = db.query("documents", embed(query), top_k=2)

for doc_id, distance in results:
    print(f"Document {doc_id}: distance={distance:.3f}")
```

### Database-Specific Connection Strings

```python
# PostgreSQL with pgvector
db = VectorDB("postgresql://user:password@localhost:5432/mydb")

# MySQL (8.2+ with native vectors or legacy JSON mode)  
db = VectorDB("mysql://user:password@localhost:3306/mydb")

# SQLite (local file or in-memory)
db = VectorDB("sqlite:///./vectors.db")
db = VectorDB("sqlite:///:memory:")
```

## 🛠️ API Reference

### `VectorDB(connection_string: str)`
Create a vector database connection.

### `create_collection(name: str, dim: int)`
Create a new collection for vectors of dimension `dim`.

### `upsert(collection: str, id: int, vector: list[float], metadata: dict = None)`
Insert or update a vector with optional metadata.

### `query(collection: str, query_vector: list[float], top_k: int = 5, filter: dict = None)`
Find the `top_k` most similar vectors. Returns list of `(id, distance)` tuples.

**Filtering Support:**
- PostgreSQL & MySQL: Native SQL filtering
- SQLite: Adaptive oversampling (fetches more results, then filters)

## 🔧 Installation Notes

### SQLite Setup
SQLite support requires loadable extensions. On some systems you may need:

```bash
# macOS with Homebrew
brew install sqlite
export LDFLAGS="-L$(brew --prefix sqlite)/lib"
export CPPFLAGS="-I$(brew --prefix sqlite)/include"
pip install "vectorwrap[sqlite]"

# Or use system package manager
# Ubuntu: apt install libsqlite3-dev
# CentOS: yum install sqlite-devel
```

### PostgreSQL Setup
```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
```

### MySQL Setup
MySQL 8.2+ has native `VECTOR` type support. For older versions, vectorwrap automatically falls back to JSON storage with Python-based distance calculations.

## 🎯 Use Cases

- **Prototyping**: Start with SQLite, scale to PostgreSQL
- **Testing**: Use SQLite in-memory databases for fast tests  
- **Multi-tenant**: Different customers on different database backends
- **Migration**: Move vector data between database systems seamlessly
- **Hybrid deployments**: PostgreSQL for production, SQLite for edge computing

## 🚧 Roadmap

Coming soon:
- **DuckDB** with `duckdb-vss` extension
- **Redis** with RediSearch
- **Elasticsearch** with dense vector fields
- **Qdrant** and **Weaviate** support
- **Batch operations** for bulk inserts
- **Index configuration** options

## 📝 License

MIT © 2025 Mihir Ahuja

---

**[PyPI Package](https://pypi.org/project/vectorwrap/) • [GitHub Repository](https://github.com/mihirahuja/vectorwrap) • [Report Issues](https://github.com/mihirahuja/vectorwrap/issues)**