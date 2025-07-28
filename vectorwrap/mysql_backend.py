import urllib.parse as up, mysql.connector, numpy as np
import json

def _euclidean_distance(v1, v2):
    """Calculate Euclidean distance between two vectors"""
    v1, v2 = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float)
    return float(np.sqrt(np.sum((v1 - v2) ** 2)))

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
            f"id BIGINT PRIMARY KEY, "
            f"emb JSON NOT NULL, "
            f"INDEX(id));"
        )
        cur.close()

    def upsert(self, name, _id, emb, meta=None):
        cur = self.conn.cursor()
        emb_json = json.dumps(list(np.asarray(emb, dtype=float)))
        cur.execute(
            f"REPLACE INTO {name}(id, emb) VALUES (%s, %s)",
            (_id, emb_json)
        )
        cur.close()

    def query(self, name, emb, top_k=5, filter=None, **_):
        where_sql, vals = _where(filter or {})
        
        # Fetch all vectors and calculate distances in Python
        # For large datasets, this should be optimized with proper indexing
        cur = self.conn.cursor()
        cur.execute(f"SELECT id, emb FROM {name}{where_sql}", vals)
        rows = cur.fetchall()
        cur.close()
        
        # Calculate distances and sort
        query_vec = np.asarray(emb, dtype=float)
        results = []
        
        for row_id, emb_json in rows:
            stored_vec = json.loads(emb_json)
            distance = _euclidean_distance(query_vec, stored_vec)
            results.append((row_id, distance))
        
        # Sort by distance and return top_k
        results.sort(key=lambda x: x[1])
        return results[:top_k]