from vectorwrap import VectorDB


def embed(x):
    return [0.1, 0.2, 0.3]


print("▶  SQLite")
db = VectorDB("sqlite:///demo.db")
db.create_collection("docs", 3)
db.upsert("docs", 1, embed("hello"))
print(db.query("docs", embed("hello"), 1))


print("\n▶  Postgres")
db = VectorDB("postgresql://postgres:secret@localhost/postgres")
db.create_collection("docs", 3)
db.upsert("docs", 1, embed("hello"))
print(db.query("docs", embed("hello"), 1))
