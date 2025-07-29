# vectorwrap/__init__.py
from .pg_backend import PgBackend
from .mysql_backend import MySQLBackend
from .sqlite_backend import SQLiteBackend


def VectorDB(url: str):
    if url.startswith("postgres"):
        return PgBackend(url)
    if url.startswith("mysql"):
        return MySQLBackend(url)
    if url.startswith("sqlite"):
        return SQLiteBackend(url)
    raise ValueError("Unsupported URL scheme")
