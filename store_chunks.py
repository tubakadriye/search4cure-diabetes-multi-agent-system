"""Validate and store text chunks once, linking them to matching SQL page contexts."""

import argparse
from contextlib import nullcontext
import hashlib
import json
import logging
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb
import pymupdf

from chunk_text import _page_windows
from database import open_connection
from extract_images import _file_hash
from render_pages import _identity, _load_links
from store_document import _timestamp

INSERT_CHUNK_SQL = """
INSERT INTO text_chunks (
    chunk_id, page_id, arxiv_id, version, source_pdf_sha256, page_number,
    chunk_number, start_char, end_char, chunk_text, word_count,
    parser_config, chunking_config, source_chunks_path, source_extraction_sha256, processed_at
) VALUES (
    %(chunk_id)s, %(page_id)s, %(arxiv_id)s, %(version)s, %(source_pdf_sha256)s,
    %(page_number)s, %(chunk_number)s, %(start_char)s, %(end_char)s,
    %(chunk_text)s, %(word_count)s, %(parser_config)s, %(chunking_config)s,
    %(source_chunks_path)s, %(source_extraction_sha256)s, %(processed_at)s
)
ON CONFLICT (chunk_id) DO NOTHING
RETURNING chunk_id;
"""
MATCHING_CONTEXTS_SQL = """
SELECT context_id, page_text, text_sha256, chunk_ids,
       page_record -> 'text_parser' AS parser_config,
       page_record -> 'chunking' AS chunking_config
FROM page_contexts
WHERE page_id = %s AND arxiv_id = %s AND version = %s
  AND source_pdf_sha256 = %s AND page_number = %s AND text_status = 'extracted'
ORDER BY context_id;
"""
INSERT_LINK_SQL = """
INSERT INTO page_context_chunks (context_id, chunk_id, page_id)
VALUES (%s, %s, %s)
ON CONFLICT (context_id, chunk_id) DO NOTHING
RETURNING context_id;
"""
READ_CHUNKS_SQL = """
SELECT c.chunk_id, c.page_id, c.arxiv_id, c.version, c.source_pdf_sha256,
       c.page_number, c.chunk_number, c.start_char, c.end_char, c.chunk_text,
       c.word_count, c.parser_config, c.chunking_config, c.processed_at, c.stored_at,
       (SELECT count(*) FROM page_context_chunks AS l WHERE l.chunk_id = c.chunk_id) AS context_count
FROM text_chunks AS c
WHERE c.arxiv_id = %s AND c.version = %s
ORDER BY c.page_number, c.chunk_number, c.chunk_id;
"""
READ_CONTEXTS_SQL = """
SELECT p.context_id, p.arxiv_id, p.version, p.source_pdf_sha256, p.page_number,
       p.page_text, p.png_path, p.png_sha256, p.text_status, p.image_status, p.status,
       c.chunk_id, c.chunk_text, c.start_char, c.end_char
FROM page_context_chunks AS l
JOIN page_contexts AS p ON p.context_id = l.context_id AND p.page_id = l.page_id
JOIN text_chunks AS c ON c.chunk_id = l.chunk_id AND c.page_id = l.page_id
WHERE c.chunk_id = %s
ORDER BY p.stored_at, p.context_id;
"""


def validate_chunks(chunks_path):
    """Verify source snapshots, PDF and every window/ID before any SQL writes."""
    chunks_path = Path(chunks_path)
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    extraction_path = Path(chunks["source_extraction_path"])
    source = json.loads(extraction_path.read_text(encoding="utf-8"))
    source, _, links = _load_links(source["download"], extraction_path, chunks_path)
    download = source["download"]
    if download.get("status") != "downloaded" or _file_hash(download["local_path"]) != download["sha256"]:
        raise ValueError("The original PDF does not match the successful download.")
    with pymupdf.open(download["local_path"]) as pdf:
        if not pdf.is_pdf or pdf.needs_pass or pdf.page_count != source["page_count"]:
            raise ValueError("The PDF and text-extraction page inventory differ.")
    config = chunks["chunking"]
    max_words, overlap = config.get("max_words"), config.get("overlap_words")
    if type(max_words) is not int or max_words < 1 or type(overlap) is not int or not 0 <= overlap < max_words:
        raise ValueError("Invalid word-window limits.")
    expected_config = {
        "method": "page_word_windows_v1", "max_words": max_words, "overlap_words": overlap,
        "word_definition": "non-whitespace runs", "offsets": "Unicode character indices, end-exclusive",
        "cross_page": False,
    }
    if config != expected_config:
        raise ValueError("Unsupported or inconsistent chunking configuration.")
    expected = []
    document = download["search"]["document"]
    for page in source["pages"]:
        for number, window in enumerate(_page_windows(page["text"], max_words, overlap), 1):
            identity = {
                "arxiv_id": document["arxiv_id"], "version": document["version"],
                "pdf_sha256": download["sha256"], "parser": source["parser"], "chunking": config,
                "page_number": page["page_number"], "start_char": window["start_char"],
                "end_char": window["end_char"], "text": window["text"],
            }
            expected.append({
                "chunk_id": _identity(identity), "arxiv_id": document["arxiv_id"], "version": document["version"],
                "source_pdf_sha256": download["sha256"], "page_number": page["page_number"],
                "chunk_number": number, **window,
            })
    if chunks["chunks"] != expected:
        raise ValueError("Chunk content, positions, counts, ordering or IDs do not match the recorded word-window policy.")
    return chunks, source, links


def store_chunks(chunks_path, connection=None):
    """Store a chunk and its context links atomically; preserve other successful chunks."""
    chunks, source, links = validate_chunks(chunks_path)
    document = source["download"]["search"]["document"]
    processed_at = _timestamp(chunks.get("processed_at"), "chunks.processed_at")
    result = {"arxiv_id": document["arxiv_id"], "version": document["version"],
              "inserted_count": 0, "unchanged_count": 0, "linked_count": 0, "failed_count": 0, "chunks": []}
    context = open_connection() if connection is None else nullcontext(connection)
    with context as database:
        for chunk in chunks["chunks"]:
            page_id = _identity({"arxiv_id": chunk["arxiv_id"], "version": chunk["version"],
                                 "source_pdf_sha256": chunk["source_pdf_sha256"], "page_number": chunk["page_number"]})
            page_text = source["pages"][chunk["page_number"] - 1]["text"]
            values = {key: chunk[key] for key in (
                "chunk_id", "arxiv_id", "version", "source_pdf_sha256", "page_number", "chunk_number",
                "start_char", "end_char", "word_count",
            )}
            values.update(page_id=page_id, chunk_text=chunk["text"],
                          parser_config=Jsonb(chunks["parser"]), chunking_config=Jsonb(chunks["chunking"]),
                          source_chunks_path=str(chunks_path), source_extraction_sha256=links["extraction_sha256"],
                          processed_at=processed_at)
            outcome = {"chunk_id": chunk["chunk_id"], "page_number": chunk["page_number"]}
            try:
                with database.transaction():
                    candidates = database.execute(MATCHING_CONTEXTS_SQL, (
                        page_id, chunk["arxiv_id"], chunk["version"], chunk["source_pdf_sha256"], chunk["page_number"],
                    )).fetchall()
                    matching = [row for row in candidates if row["page_text"] == page_text
                                and row["text_sha256"] == hashlib.sha256(page_text.encode("utf-8")).hexdigest()
                                and row["parser_config"] == chunks["parser"]
                                and row["chunking_config"] == chunks["chunking"]
                                and chunk["chunk_id"] in row["chunk_ids"]]
                    if not matching:
                        raise ValueError("Import a matching page context with the exact text, chunk ID and configuration first.")
                    inserted = database.execute(INSERT_CHUNK_SQL, values).fetchone()
                    new_links = 0
                    for row in matching:
                        linked = database.execute(INSERT_LINK_SQL, (row["context_id"], chunk["chunk_id"], page_id)).fetchone()
                        new_links += linked is not None
                outcome.update(status="inserted" if inserted else "unchanged", linked_contexts=len(matching), new_links=new_links)
                result[outcome["status"] + "_count"] += 1
                result["linked_count"] += new_links
            except (ValueError, psycopg.Error) as error:
                message = f"Database write failed ({type(error).__name__})" if isinstance(error, psycopg.Error) else str(error)
                outcome.update(status="failed", error_type=type(error).__name__, error=message)
                result["failed_count"] += 1
                logging.warning("Chunk SQL import failed: page=%s chunk=%s error=%s",
                                chunk["page_number"], chunk["chunk_number"], message)
            result["chunks"].append(outcome)
    result["status"] = "stored" if not result["failed_count"] else "partial" if result["inserted_count"] + result["unchanged_count"] else "failed"
    logging.info("Chunk SQL import: inserted=%s unchanged=%s new_links=%s failed=%s",
                 result["inserted_count"], result["unchanged_count"], result["linked_count"], result["failed_count"])
    return result


def read_chunks(arxiv_id, version, connection=None):
    """Read stored chunks without relevance ranking or configuration selection."""
    context = open_connection() if connection is None else nullcontext(connection)
    with context as database:
        return database.execute(READ_CHUNKS_SQL, (arxiv_id, version)).fetchall()


def read_chunk_contexts(chunk_id, connection=None):
    """Read the linked page contexts; callers must check image readiness for vision use."""
    context = open_connection() if connection is None else nullcontext(connection)
    with context as database:
        return database.execute(READ_CONTEXTS_SQL, (chunk_id,)).fetchall()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", required=True, help="Successful JSON produced by chunk_text")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        result = store_chunks(args.chunks)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "stored" else 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, psycopg.Error) as error:
        parser.exit(1, f"Chunk import failed ({type(error).__name__}). Check source records, schema and matching SQL page contexts.\n")


if __name__ == "__main__":
    raise SystemExit(main())
