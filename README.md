# vectorwrap 0.2.0

**One API, multiple vector databases.**

```python
from vectorwrap import VectorDB, embed   # embed = your own function

db = VectorDB("postgresql://user:pass@host/db")   # or mysql://…
db.create_collection("products", 1536)
db.upsert("products", 1, embed("iPhone 15 Pro"), {"category":"phone"})
hits = db.query("products", embed("latest iPhone"), filter={"category":"phone"})
```

**Backends:**
- **PostgreSQL + pgvector**: Native vector operations with HNSW indexing
- **MySQL 8.0+**: JSON-based vectors with Python distance calculation
- **MySQL HeatWave 8.2+**: Native VECTOR type support (future)

**Features:**
- Unified API across different vector databases
- Euclidean distance search
- Metadata filtering support (filter={"col":"val"})
- Same code → different DB by swapping the connection string