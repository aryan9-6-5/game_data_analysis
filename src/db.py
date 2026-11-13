"""Database helpers: connection from .env, database creation, schema/view loading, CSV load."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import psycopg2
import sqlalchemy
from dotenv import load_dotenv
from psycopg2 import sql

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SQL_DIR = ROOT / "sql"
load_dotenv(ROOT / ".env")

TABLE_ORDER = ["platforms", "builds", "testers", "test_cases", "test_executions", "defects"]


def _params(dbname: str | None = None) -> dict:
    return dict(
        host=os.getenv("PGHOST", "localhost"),
        port=os.getenv("PGPORT", "5432"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD"),
        dbname=dbname or os.getenv("PGDATABASE", "game_qa"),
    )


def connect():
    return psycopg2.connect(**_params())


def ensure_database() -> None:
    name = os.getenv("PGDATABASE", "game_qa")
    conn = psycopg2.connect(**_params("postgres"))
    conn.autocommit = True
    with conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,))
        if not cur.fetchone():
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    conn.close()


def run_sql_file(conn, name: str) -> None:
    with conn.cursor() as cur:
        cur.execute((SQL_DIR / name).read_text(encoding="utf-8"))
    conn.commit()


def copy_csv(conn, table: str) -> int:
    path = RAW / f"{table}.csv"
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    stmt = sql.SQL("COPY {} ({}) FROM STDIN WITH (FORMAT csv, HEADER true)").format(
        sql.Identifier(table), sql.SQL(", ").join(map(sql.Identifier, cols)))
    with conn.cursor() as cur, open(path, encoding="utf-8") as fh:
        cur.copy_expert(stmt.as_string(conn), fh)
        cur.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table)))
        n = cur.fetchone()[0]
    conn.commit()
    return n


def engine():
    p = _params()
    url = sqlalchemy.URL.create("postgresql+psycopg2", username=p["user"], password=p["password"],
                                host=p["host"], port=int(p["port"]), database=p["dbname"])
    return sqlalchemy.create_engine(url)


def query_df(query: str, eng=None) -> pd.DataFrame:
    with (eng or engine()).connect() as c:
        return pd.read_sql_query(sqlalchemy.text(query), c)
