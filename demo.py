# demo.py
from vectorwrap import VectorDB
vec = [0.1,0.2,0.3]

pg = VectorDB("postgresql://postgres:secret@localhost/postgres")
pg.create_collection("prod", 3)
pg.upsert("prod", 1, vec, {"category":"phone"})
print("PG →", pg.query("prod", [0.1,0.2,0.4]))

my = VectorDB("mysql://root:secret@localhost:3306/vectordb")
my.create_collection("prod", 3)
my.upsert("prod", 1, vec, {"category":"phone"})
print("MySQL →", my.query("prod", [0.1,0.2,0.4]))