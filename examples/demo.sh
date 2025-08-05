#!/usr/bin/env bash
python - <<'PY'
from vectorwrap import VectorDB
def embed(x): return [0.1,0.2,0.3]           # stub

db = VectorDB("sqlite:///demo.db")
db.create_collection("docs", 3)
db.upsert("docs", 1, embed("hello"))
print("SQLite →", db.query("docs", embed("hello"), 1))

db = VectorDB("postgresql://postgres:secret@localhost/postgres")
db.create_collection("docs", 3)
db.upsert("docs", 1, embed("hello"))
print("Postgres →", db.query("docs", embed("hello"), 1))
PY
