import pytest, time, os
from vectorwrap import VectorDB

PG_URL = "postgresql://postgres:secret@localhost/postgres"
MY_URL = "mysql://root:secret@localhost:3306/vectordb"
VECTOR = [0.1,0.2,0.3]

@pytest.mark.parametrize("url", [PG_URL, MY_URL])
def test_basic(url):
    db = VectorDB(url)
    db.create_collection("t", 3)
    db.upsert("t", 42, VECTOR, {"category":"c"})
    hits = db.query("t", [0.1,0.2,0.35], 1, filter={"category":"c"})
    assert hits[0][0] == 42