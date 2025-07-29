from __future__ import annotations

from typing import Any, cast
import psycopg
import numpy as np
from psycopg import sql


def _lit(v: list[float]) -> str:
    """Convert vector to PostgreSQL array literal format."""
    return "[" + ",".join(map(str, np.asarray(v, dtype=float))) + "]"


def _where(flt: dict[str, Any]) -> tuple[sql.SQL, list[Any]]:
    """Build WHERE clause from filter dictionary."""
    if not flt:
        return sql.SQL(""), []
    clauses, vals = [], []
    for col, val in flt.items():
        clauses.append(sql.SQL("{} = %s").format(sql.Identifier(col)))
        vals.append(val)
    where_clause = sql.SQL(" WHERE ").join([sql.SQL(""), sql.SQL(" AND ").join(clauses)])
    return cast(sql.SQL, where_clause), vals


class PgBackend:
    """Backend for PostgreSQL with pgvector extension."""

    def __init__(self, url: str) -> None:
        """Initialize PostgreSQL connection."""
        self.conn = psycopg.connect(url, autocommit=True)

    def create_collection(self, name: str, dim: int) -> None:
        """Create a new collection with HNSW index."""
        with self.conn.cursor() as cur:
            cur.execute(
                sql.SQL("CREATE TABLE IF NOT EXISTS {}("
                        "id BIGINT PRIMARY KEY, "
                        "emb VECTOR({}))").format(sql.Identifier(name), sql.SQL(str(dim))))
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} "
                        "USING hnsw (emb vector_l2_ops)")
                .format(sql.Identifier(f"{name}_emb_idx"), sql.Identifier(name)))

    def upsert(self, name: str, _id: int, emb: list[float], meta: dict[str, Any] | None = None) -> None:
        """Insert or update a vector with optional metadata."""
        with self.conn.cursor() as cur:
            cur.execute(
                sql.SQL("INSERT INTO {}(id, emb) VALUES (%s, %s) "
                        "ON CONFLICT(id) DO UPDATE SET emb = EXCLUDED.emb")
                    .format(sql.Identifier(name)),
                (_id, _lit(emb))
            )

    def query(
        self, 
        name: str, 
        emb: list[float], 
        top_k: int = 5, 
        filter: dict[str, Any] | None = None, 
        ensure_k: bool = False,
        **kwargs: Any
    ) -> list[tuple[int, float]]:
        """Query for similar vectors using pgvector distance operator."""
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
