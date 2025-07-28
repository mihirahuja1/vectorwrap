import urllib.parse as up, mysql.connector, numpy as np

def _lit(v): return "[" + ",".join(map(str, np.asarray(v, dtype=float))) + "]"

def _where(flt: dict[str,str]):
    if not flt:
        return "", []
    clauses, vals = [], []
    for col, val in flt.items():
        clauses.append(f"{col} = %s")
        vals.append(val)
    return " WHERE " + " AND ".join(clauses), vals

class MySQLBackend:
    def __init__(self, url: str):
        p = up.urlparse(url)
        self.db = p.path.lstrip("/")
        self.conn = mysql.connector.connect(
            host=p.hostname, port=p.port or 3306,
            user=p.username, password=p.password,
            database=self.db, autocommit=True
        )

    def create_collection(self, name: str, dim: int):
        cur = self.conn.cursor()
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {name}("
            f"id BIGINT PRIMARY KEY, emb VECTOR({dim}));"
        )
        cur.close()

    def upsert(self, name, _id, emb, meta=None):
        cur = self.conn.cursor()
        cur.execute(
            f"REPLACE INTO {name}(id, emb) VALUES (%s, %s)",
            (_id, _lit(emb))
        )
        cur.close()

    def query(self, name, emb, top_k=5, filter=None, **_):
        where_sql, vals = _where(filter or {})
        sql = (f"SELECT id, "
               f"DISTANCE(emb, STRING_TO_VECTOR(%s), 'EUCLIDEAN') AS dist "
               f"FROM {name}{where_sql} ORDER BY dist LIMIT %s")
        cur = self.conn.cursor()
        cur.execute(sql, [_lit(emb)] + vals + [top_k])
        rows = cur.fetchall()
        cur.close()
        return rows