import psycopg, numpy as np
from psycopg import sql

def _lit(v: list[float]) -> str:
    return "[" + ",".join(map(str, np.asarray(v, dtype=float))) + "]"

def _where(flt: dict[str, str]):
    if not flt:
        return sql.SQL(""), []
    clauses, vals = [], []
    for col, val in flt.items():
        clauses.append(sql.SQL("{} = %s").format(sql.Identifier(col)))
        vals.append(val)
    return sql.SQL(" WHERE ").join([sql.SQL(""), sql.SQL(" AND ").join(clauses)]), vals

class PgBackend:
    def __init__(self, url: str):
        self.conn = psycopg.connect(url, autocommit=True)

    def create_collection(self, name: str, dim: int):
        with self.conn.cursor() as cur:
            cur.execute(
                sql.SQL("CREATE TABLE IF NOT EXISTS {}("
                        "id BIGINT PRIMARY KEY, "
                        "emb VECTOR({}))").format(sql.Identifier(name), sql.SQL(str(dim))))
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} "
                        "USING hnsw (emb vector_l2_ops)")
                .format(sql.Identifier(f"{name}_emb_idx"), sql.Identifier(name)))

    def upsert(self, name, _id, emb, meta=None):
        with self.conn.cursor() as cur:
            cur.execute(
                sql.SQL("INSERT INTO {}(id, emb) VALUES (%s, %s) "
                        "ON CONFLICT(id) DO UPDATE SET emb = EXCLUDED.emb")
                    .format(sql.Identifier(name)),
                (_id, _lit(emb))
            )

    def query(self, name, emb, top_k=5, filter=None, ensure_k=False):
        if ensure_k:
            with self.conn.cursor() as cur:
                cur.execute("SET hnsw.iterative_scan = 'relaxed_order'")
        where_sql, vals = _where(filter or {})
        with self.conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT id, emb <=> %s::vector AS dist FROM {}{} "
                        "ORDER BY dist LIMIT %s")
                    .format(sql.Identifier(name), where_sql),
                [_lit(emb)] + vals + [top_k]
            )
            return cur.fetchall()
