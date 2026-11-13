"""Create the database, load the schema and CSVs, then build the metric views.

Run:  python src/load_db.py
"""
from db import TABLE_ORDER, connect, copy_csv, ensure_database, run_sql_file


def main() -> None:
    ensure_database()
    conn = connect()
    run_sql_file(conn, "01_schema.sql")
    for table in TABLE_ORDER:
        print(f"loaded {table:<16} {copy_csv(conn, table):>7} rows")
    run_sql_file(conn, "02_views.sql")
    print("views created")
    conn.close()


if __name__ == "__main__":
    main()
