# vectorwrap 0.2.0 <a href="https://pypi.org/project/vectorwrap/"><img src="https://img.shields.io/pypi/v/vectorwrap"></a>

**One API — multiple vector databases**

```bash
pip install vectorwrap        # python ≥3.11


from openai import OpenAI                  # any embedder works
from vectorwrap import VectorDB

client = OpenAI()                          # example embed() helper
def embed(text: str) -> list[float]:
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding                    # 1536‑dim float list

db = VectorDB("postgresql://user:pw@host/db")     # swap to mysql://…
db.create_collection("products", dim=1536)

db.upsert("products", 1, embed("Apple iPhone 15 Pro"),
          {"category": "phone"})

hits = db.query("products",
                embed("latest iPhone"),
                top_k=5,
                filter={"category": "phone"})
print(hits)   # → [(1, 0.02…)]

```

Supported back‑ends
Engine	Vector type & distance	Indexing	Notes
PostgreSQL 16 + pgvector ≥ 0.8	VECTOR(n) • <=> operator	HNSW (automatic)	CREATE EXTENSION vector;
MySQL 8.2 / HeatWave	VECTOR(n) • DISTANCE()	Automatic	Native since 8.2
MySQL ≤ 8.0 (legacy)	JSON fallback • Python distance	No ANN index	✔ works, but slower

**Backends:**
- **PostgreSQL + pgvector**: Native vector operations with HNSW indexing
- **MySQL 8.0+**: JSON-based vectors with Python distance calculation
- **MySQL HeatWave 8.2+**: Native VECTOR type support (future)

**Features:**
- Unified API across different vector databases
- Euclidean distance search
- Metadata filtering support (filter={"col":"val"})
- Same code → different DB by swapping the connection string
