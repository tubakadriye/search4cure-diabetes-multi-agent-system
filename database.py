"""Prepare local PostgreSQL credentials, initialize tables, and check connectivity."""

import argparse
import json
import logging
import os
from pathlib import Path
import secrets

import psycopg
from psycopg.rows import dict_row

PASSWORD_FILE = Path("data/postgres/password.txt")
HEALTH_SQL = """
SELECT current_database() AS database,
       current_user AS db_user,
       current_setting('server_version') AS postgres_version,
       (SELECT extversion FROM pg_extension WHERE extname = 'vector') AS pgvector_version,
       (SELECT count(*) FROM pg_tables WHERE schemaname = 'public') AS public_table_count,
       1 AS check_value;
"""


def create_password_file(path=PASSWORD_FILE):
    """Create a private local password once; never replace an existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(secrets.token_hex(32) + "\n")
    return True


def open_connection(password_path=PASSWORD_FILE):
    """Open a bounded local connection; the caller must close it."""
    password = Path(password_path).read_text(encoding="utf-8").rstrip("\n")
    if not password:
        raise ValueError("The local PostgreSQL password file is empty.")
    port = int(os.environ.get("POSTGRES_PORT", "55432"))
    if not 1 <= port <= 65535:
        raise ValueError("POSTGRES_PORT must be between 1 and 65535.")
    logging.info("Connecting to local PostgreSQL at 127.0.0.1:%s", port)
    return psycopg.connect(host="127.0.0.1", port=port, dbname="search4cure", user="search4cure",
                          password=password, connect_timeout=5, autocommit=True,
                          options="-c statement_timeout=5000", row_factory=dict_row)


def initialize_schema(connection=None):
    """Create document, page-context and text-chunk tables on existing or new volumes."""
    if connection is None:
        with open_connection() as owned_connection:
            initialize_schema(owned_connection)
        return
    schema_dir = Path(__file__).resolve().parent / "sql"
    with connection.transaction():
        for filename in ("002_document_versions.sql", "003_page_contexts.sql", "004_text_chunks.sql"):
            connection.execute((schema_dir / filename).read_text(encoding="utf-8"))


def check_connection(password_path=PASSWORD_FILE):
    """Run a read-only server check and return no credentials."""
    with open_connection(password_path) as connection:
        result = connection.execute(HEALTH_SQL).fetchone()
    if result is None or result["check_value"] != 1:
        raise RuntimeError("The database did not return the expected connection check.")
    if result["pgvector_version"] is None:
        raise RuntimeError("The vector extension is not enabled in this database.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["init-password", "init-schema", "check"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.command == "init-password":
            created = create_password_file()
            print("Local PostgreSQL password file created." if created else
                  "Existing PostgreSQL password file kept unchanged.")
        elif args.command == "init-schema":
            initialize_schema()
            print("Document, page-context and text-chunk tables initialized; existing data kept.")
        else:
            print(json.dumps(check_connection(), indent=2))
    except (OSError, ValueError, RuntimeError, psycopg.Error) as error:
        # Connection options and password values must never be printed.
        parser.exit(1, f"Database setup/check failed ({type(error).__name__}). "
                    "Check Docker, the port, and the private password file for this volume.\n")


if __name__ == "__main__":
    main()
