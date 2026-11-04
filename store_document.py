"""Validate one saved arXiv download and store its version metadata in PostgreSQL."""

import argparse
from contextlib import nullcontext
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import re
from urllib.parse import urlparse

import psycopg
from psycopg.types.json import Jsonb

from database import open_connection

INSERT_SQL = """
INSERT INTO document_versions (
    arxiv_id, version, title, source_url, pdf_url, first_submitted_at,
    version_submitted_at, metadata_retrieved_at, pdf_retrieved_at,
    sha256, size_bytes, local_path, metadata_path, status, download_record
) VALUES (
    %(arxiv_id)s, %(version)s, %(title)s, %(source_url)s, %(pdf_url)s,
    %(first_submitted_at)s, %(version_submitted_at)s, %(metadata_retrieved_at)s,
    %(pdf_retrieved_at)s, %(sha256)s, %(size_bytes)s, %(local_path)s,
    %(metadata_path)s, %(status)s, %(download_record)s
)
ON CONFLICT (arxiv_id, version) DO NOTHING
RETURNING arxiv_id, version, sha256;
"""
DOCUMENT_SQL = """
SELECT arxiv_id, version, title, source_url, pdf_url, first_submitted_at,
       version_submitted_at, metadata_retrieved_at, pdf_retrieved_at,
       sha256, size_bytes, local_path, metadata_path, status, stored_at
FROM document_versions
WHERE arxiv_id = %s AND version = %s;
"""


def _timestamp(value, field, optional=False):
    """Keep unknown source timestamps null and reject ambiguous naive times."""
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid timestamp: {field}")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"Timestamp requires a UTC offset: {field}")
    return parsed


def validate_download(metadata_path):
    """Validate required metadata and stream-check the original PDF fingerprint."""
    path = Path(metadata_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("status") != "downloaded":
        raise ValueError("A successful download record is required.")
    search = record.get("search")
    document = search.get("document") if isinstance(search, dict) else None
    if not isinstance(document, dict):
        raise ValueError("Missing document metadata.")
    values = {key: document.get(key) for key in ("arxiv_id", "version", "title", "source_url")}
    for key in ("arxiv_id", "title", "source_url"):
        if not isinstance(values[key], str) or not values[key].strip():
            raise ValueError(f"Missing required document field: {key}")
    if not re.fullmatch(r"\d{4}\.\d{4,5}|[A-Za-z.-]+/\d{7}", values["arxiv_id"]):
        raise ValueError("Invalid arXiv identifier.")
    if type(values["version"]) is not int or not 1 <= values["version"] <= 2147483647:
        raise ValueError("A positive PostgreSQL integer version is required.")
    source = urlparse(values["source_url"])
    if source.scheme not in ("http", "https") or source.netloc != "arxiv.org":
        raise ValueError("An arxiv.org source URL is required.")
    if source.path not in (f'/abs/{values["arxiv_id"]}',
                           f'/abs/{values["arxiv_id"]}v{values["version"]}'):
        raise ValueError("Source URL does not match the document version.")
    expected_url = f'https://arxiv.org/pdf/{values["arxiv_id"]}v{values["version"]}'
    if record.get("pdf_url") != expected_url:
        raise ValueError("PDF URL does not match the document version.")
    checksum = record.get("sha256")
    if not isinstance(checksum, str) or not re.fullmatch("[0-9a-f]{64}", checksum):
        raise ValueError("A SHA-256 fingerprint is required.")
    if type(record.get("size_bytes")) is not int or record["size_bytes"] <= 0:
        raise ValueError("A positive PDF size is required.")
    if not isinstance(record.get("local_path"), str) or not record["local_path"].strip():
        raise ValueError("A local PDF path is required.")
    for key in ("first_submitted_at", "version_submitted_at"):
        values[key] = _timestamp(document.get(key), key, optional=True)
    values["metadata_retrieved_at"] = _timestamp(search.get("retrieved_at"), "metadata_retrieved_at")
    values["pdf_retrieved_at"] = _timestamp(record.get("pdf_retrieved_at"), "pdf_retrieved_at")
    digest, size = hashlib.sha256(), 0
    with Path(record["local_path"]).open("rb") as stream:
        header = stream.read(5)
        if header != b"%PDF-":
            raise ValueError("The source file lacks a PDF header.")
        digest.update(header)
        size += len(header)
        while block := stream.read(64 * 1024):
            digest.update(block)
            size += len(block)
    if digest.hexdigest() != checksum or size != record["size_bytes"]:
        raise ValueError("The source PDF no longer matches its download record.")
    values.update({key: record[key] for key in ("pdf_url", "sha256", "size_bytes", "local_path", "status")})
    values["metadata_path"] = str(path)
    values["download_record"] = Jsonb(record)
    return values


def store_document(metadata_path, connection=None):
    """Insert once, preserve the first record, and reject changed bytes for a version."""
    values = validate_download(metadata_path)
    context = open_connection() if connection is None else nullcontext(connection)
    with context as database:
        with database.transaction():
            inserted = database.execute(INSERT_SQL, values).fetchone()
            if inserted is None:
                existing = database.execute(DOCUMENT_SQL, (values["arxiv_id"], values["version"])).fetchone()
                if existing is None or existing["sha256"] != values["sha256"]:
                    raise ValueError("Content conflict: this document version has a different stored PDF hash.")
    result = {"status": "inserted" if inserted else "unchanged",
              "arxiv_id": values["arxiv_id"], "version": values["version"], "sha256": values["sha256"]}
    logging.info("Document metadata %s: %sv%s", result["status"], result["arxiv_id"], result["version"])
    return result


def read_document(arxiv_id, version, connection=None):
    """Read a single version using bound SQL parameters; return None if absent."""
    context = open_connection() if connection is None else nullcontext(connection)
    with context as database:
        return database.execute(DOCUMENT_SQL, (arxiv_id, version)).fetchone()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="Successful download JSON, from the project root")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        print(json.dumps(store_document(args.metadata), indent=2))
    except (OSError, ValueError, psycopg.Error) as error:
        # Do not include database error details or connection credentials in CLI output.
        logging.error("Document import failed: metadata=%s error_type=%s", args.metadata, type(error).__name__)
        parser.exit(1, "Import failed. Check the source record, PDF, schema, and local database.\n")


if __name__ == "__main__":
    main()
