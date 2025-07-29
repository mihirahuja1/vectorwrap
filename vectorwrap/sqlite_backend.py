# vectorwrap/sqlite_backend.py
try:
    import pysqlite3 as sqlite3
except ImportError:
    import sqlite3

import numpy as np


def _lit(v):
    return "[" + ",".join(map(str, np.asarray(v, dtype=float))) + "]"


class SQLiteBackend:
    """Backend for local prototype databases using sqlite-vss (HNSW)."""

    def __init__(self, url: str):
        # url pattern: sqlite:///absolute/path.db  or  sqlite:///:memory:
        path = url.replace("sqlite:///", "", 1)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        try:
            self.conn.enable_load_extension(True)
        except AttributeError:
            # SQLite was compiled without extension loading support
            pass
        
        # load vss extension (bundled with package)
        try:
            import sqlite_vss
            sqlite_vss.load(conn=self.conn)
        except ImportError:
            raise RuntimeError(
                "sqlite-vss not installed. Install with: pip install 'vectorwrap[sqlite]'"
            )
        except AttributeError:
            # SQLite was compiled without extension loading support
            raise RuntimeError(
                "SQLite was compiled without extension loading support. "
                "Install with: pip install 'vectorwrap[sqlite]'"
            )

    def create_collection(self, name: str, dim: int):
        cur = self.conn.cursor()
        cur.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {name} "
            f"USING vss0(emb({dim}));"
        )
        cur.close()

    def upsert(self, name, _id, emb, meta=None):
        cur = self.conn.cursor()
        # sqlite-vss stores rowid internally
        cur.execute(f"REPLACE INTO {name}(rowid, emb) VALUES (?, ?);", (_id, _lit(emb)))
        cur.close()
        self.conn.commit()

    def query(self, name, emb, top_k=5, filter=None, **_):
        if filter:
            raise NotImplementedError("SQLite backend does not support filters yet")
        cur = self.conn.cursor()
        cur.execute(
            f"SELECT rowid, distance "
            f"FROM {name} WHERE vss_search(emb, ?) LIMIT {top_k};",
            (_lit(emb),),
        )
        rows = cur.fetchall()
        cur.close()
        return rows
