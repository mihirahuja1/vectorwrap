from vectorwrap import VectorDB


def test_sqlite_basic(tmp_path):
    db = VectorDB(f"sqlite:///{tmp_path/'test.db'}")
    db.create_collection("t", 3)
    db.upsert("t", 7, [0.0, 0.0, 1.0])
    res = db.query("t", [0.0, 0.1, 0.9], 1)
    assert res[0][0] == 7
