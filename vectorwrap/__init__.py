from .pg_backend import PgBackend
from .mysql_backend import MySQLBackend

def VectorDB(url: str):
    if url.startswith("postgres"):
        return PgBackend(url)
    if url.startswith("mysql"):
        return MySQLBackend(url)
    raise ValueError("Unsupported URL scheme")
