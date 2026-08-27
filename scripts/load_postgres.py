"""Load the generated CSVs in ../data into Postgres using COPY.

Usage:
    python scripts/load_postgres.py

Reads connection info from DATABASE_URL env var, defaulting to the
docker-compose credentials (postgres://streammatch:streammatch@localhost:5432/streammatch).
Assumes sql/schema.sql has already been applied (docker-compose does this
automatically via docker-entrypoint-initdb.d on first container start).
"""
import os
import csv
import psycopg2

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://streammatch:streammatch@localhost:5433/streammatch"
)

# Order matters: respect FK dependencies.
TABLES = [
    "categories",
    "streamers",
    "viewers",
    "viewer_category_affinity",
    "streams",
    "watch_events",
    "chat_events",
    "follow_events",
]


def main():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            for table in TABLES:
                path = os.path.join(DATA_DIR, f"{table}.csv")
                with open(path, "r", encoding="utf-8") as f:
                    header = next(csv.reader([f.readline()]))
                    f.seek(0)
                    cur.copy_expert(
                        f"COPY {table} ({', '.join(header)}) FROM STDIN WITH (FORMAT csv, HEADER true, NULL '')",
                        f,
                    )
                print(f"  loaded {table}")
        conn.commit()
        print("Done.")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
