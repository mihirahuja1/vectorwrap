# vectorwrap 0.2.0

**One API, multiple vector databases.**

```python
from vectorwrap import VectorDB, embed   # embed = your own function

db = VectorDB("postgresql://user:pass@host/db")   # or mysql://…
db.create_collection("products", 1536)
db.upsert("products", 1, embed("iPhone 15 Pro"), {"category":"phone"})
hits = db.query("products", embed("latest iPhone"), filter={"category":"phone"})
```

Backends: PostgreSQL + pgvector, MySQL 8.2 / HeatWave Vector Store

Supports metadata filtering (filter={"col":"val"})

Same code → different DB by swapping the connection string.