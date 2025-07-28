from vectorwrap import VectorDB

db = VectorDB("postgresql://postgres:secret@localhost:5432/postgres")
db.create_collection("docs", 3)
db.upsert("docs", 1, [0.1, 0.2, 0.3], {})
print(db.query("docs", [0.1, 0.2, 0.4], 1))
