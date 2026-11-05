"""Store immutable page-context snapshots in PostgreSQL; keep PNG files on disk."""

import argparse
from contextlib import nullcontext
import hashlib
import json
import logging
from pathlib import Path
import re
import struct

import psycopg
from psycopg.types.json import Jsonb
import pymupdf

from database import open_connection
from extract_images import _file_hash
from render_pages import _identity, _load_links
from store_document import _timestamp

INSERT_PAGE_SQL = """
INSERT INTO page_contexts (
    context_id, page_id, render_id, arxiv_id, version, source_pdf_sha256,
    page_number, page_text, text_sha256, text_status, image_status, status,
    png_path, png_sha256, chunk_ids, page_record, source_manifest_path, processed_at
) VALUES (
    %(context_id)s, %(page_id)s, %(render_id)s, %(arxiv_id)s, %(version)s,
    %(source_pdf_sha256)s, %(page_number)s, %(page_text)s, %(text_sha256)s,
    %(text_status)s, %(image_status)s, %(status)s, %(png_path)s, %(png_sha256)s,
    %(chunk_ids)s, %(page_record)s, %(source_manifest_path)s, %(processed_at)s
)
ON CONFLICT (context_id) DO NOTHING
RETURNING context_id;
"""
READ_PAGES_SQL = """
SELECT context_id, page_id, render_id, arxiv_id, version, source_pdf_sha256,
       page_number, page_text, text_sha256, text_status, image_status, status,
       png_path, png_sha256, chunk_ids, processed_at, stored_at,
       page_record ->> 'text_error' AS text_error,
       page_record ->> 'image_error' AS image_error,
       page_record ->> 'error' AS error
FROM page_contexts
WHERE arxiv_id = %s AND version = %s
ORDER BY page_number, stored_at, context_id;
"""


def _hash_value(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_png(image, rendering):
    """Check the PNG fingerprint, declared dimensions and actual decodability."""
    path = Path(image["local_path"])
    if not _hash_value(image.get("sha256")) or _file_hash(path) != image["sha256"]:
        raise ValueError("PNG fingerprint does not match its page record.")
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError("A PNG with a valid dimension header is required.")
    width, height = struct.unpack(">II", header[16:24])
    limit = rendering.get("max_pixels")
    if type(limit) is not int or limit < 1 or width < 1 or height < 1 or width * height > limit:
        raise ValueError("PNG dimensions exceed the recorded pixel limit.")
    if (width, height) != (image.get("width"), image.get("height")) or path.stat().st_size != image.get("size_bytes"):
        raise ValueError("PNG dimensions or size do not match the page record.")
    try:
        pixmap = pymupdf.Pixmap(str(path))
        if (pixmap.width, pixmap.height) != (width, height):
            raise ValueError("Decoded PNG dimensions differ from its header.")
    except Exception as error:
        raise ValueError("The page PNG cannot be decoded consistently.") from error


def validate_page(page, document, pdf_hash, manifest_path):
    """Build an identity from meaningful content, excluding attempt paths/timestamps."""
    identifier, version = document["arxiv_id"], document["version"]
    number = page["page_number"]
    if type(number) is not int or number < 1:
        raise ValueError("A positive physical page number is required.")
    if (page["arxiv_id"], page["version"], page["source_pdf_sha256"]) != (identifier, version, pdf_hash):
        raise ValueError("Page provenance does not match the document manifest.")
    page_id = _identity({"arxiv_id": identifier, "version": version,
                         "source_pdf_sha256": pdf_hash, "page_number": number})
    rendering, parser = page["rendering"], page["text_parser"]
    if not isinstance(rendering, dict) or not rendering or not isinstance(parser, dict) or not parser:
        raise ValueError("Rendering and text-parser configurations are required.")
    if page["page_id"] != page_id or page["render_id"] != _identity({"page_id": page_id, "rendering": rendering}):
        raise ValueError("Page or rendering identity is inconsistent.")
    text, text_status = page["text"], page["text_status"]
    if text_status == "failed":
        if text is not None or page["text_sha256"] is not None or not page.get("text_error"):
            raise ValueError("Failed text must remain unknown and retain its error.")
    elif text_status in ("empty", "extracted"):
        if not isinstance(text, str) or page["text_sha256"] != hashlib.sha256(text.encode("utf-8")).hexdigest():
            raise ValueError("Page text does not match its fingerprint.")
        if (text_status == "empty") != (not text.strip()) or page.get("text_error") is not None:
            raise ValueError("Text content and status are inconsistent.")
    else:
        raise ValueError("Unknown text status.")
    image = page["image"]
    if page["image_status"] == "rendered":
        if not isinstance(image, dict) or page.get("image_error") is not None:
            raise ValueError("Rendered images require image metadata without an image error.")
        _validate_png(image, rendering)
    elif page["image_status"] != "failed" or image is not None or not page.get("image_error"):
        raise ValueError("Failed images must retain their error and have no saved image.")
    if page["status"] not in ("rendered", "failed") or (page["status"] == "rendered" and image is None):
        raise ValueError("Page status is inconsistent with the image result.")
    chunks = page["chunk_ids"]
    if not isinstance(chunks, list) or any(not _hash_value(value) for value in chunks) or len(set(chunks)) != len(chunks):
        raise ValueError("Chunk IDs must be unique SHA-256 identifiers.")
    semantic = {key: page[key] for key in (
        "page_id", "render_id", "rendering", "text_parser", "text", "text_sha256",
        "text_status", "text_error", "image_status", "image_error", "status", "error",
        "chunk_ids", "chunks_status", "chunking",
    )}
    semantic["image"] = None if image is None else {key: image[key] for key in ("sha256", "width", "height", "size_bytes")}
    return {
        "context_id": _identity(semantic), "page_id": page_id, "render_id": page["render_id"],
        "arxiv_id": identifier, "version": version, "source_pdf_sha256": pdf_hash,
        "page_number": number, "page_text": text, "text_sha256": page["text_sha256"],
        "text_status": text_status, "image_status": page["image_status"], "status": page["status"],
        "png_path": image["local_path"] if image else None,
        "png_sha256": image["sha256"] if image else None,
        "chunk_ids": chunks, "page_record": Jsonb(page), "source_manifest_path": str(manifest_path),
        "processed_at": _timestamp(page.get("processed_at"), "page.processed_at"),
    }


def store_pages(manifest_path, connection=None):
    """Import each page transactionally; preserve successes and report per-page failures."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    download, pages = manifest["download"], manifest["pages"]
    document, pdf_hash = download["search"]["document"], download["sha256"]
    if download.get("status") != "downloaded":
        raise ValueError("A successful download record is required.")
    if not _hash_value(pdf_hash) or _file_hash(download["local_path"]) != pdf_hash:
        raise ValueError("The source PDF does not match its manifest fingerprint.")
    with pymupdf.open(download["local_path"]) as pdf:
        count = pdf.page_count
        if not pdf.is_pdf or pdf.needs_pass or not count:
            raise ValueError("A readable, unencrypted PDF is required.")
    if not isinstance(pages, list) or len(pages) != count or manifest.get("page_count") != count:
        raise ValueError("The manifest must contain one record for every physical page.")
    numbers = [page["page_number"] for page in pages]
    if any(type(number) is not int for number in numbers) or numbers != list(range(1, count + 1)):
        raise ValueError("Manifest pages must be consecutive and unique.")
    source, by_page, links = _load_links(download, manifest.get("extraction_path"), manifest.get("chunks_path"))
    if any(manifest.get(key) != value for key, value in links.items()):
        raise ValueError("Manifest source references do not match the actual extraction/chunk snapshots.")
    result = {"arxiv_id": document["arxiv_id"], "version": document["version"],
              "inserted_count": 0, "unchanged_count": 0, "failed_count": 0, "pages": []}
    context = open_connection() if connection is None else nullcontext(connection)
    with context as database:
        parent = database.execute("SELECT sha256 FROM document_versions WHERE arxiv_id = %s AND version = %s",
                                  (document["arxiv_id"], document["version"])).fetchone()
        if parent is None or parent["sha256"] != pdf_hash:
            raise ValueError("Import the matching document version and PDF hash before its pages.")
        for page in pages:
            row = {"page_number": page["page_number"]}
            try:
                if any(page.get(key) != value for key, value in links.items()):
                    raise ValueError("Page source references do not match its manifest.")
                expected_chunk_status = "linked" if links["chunks_path"] is not None else "not_provided"
                if page.get("chunk_ids") != by_page.get(page["page_number"], []) or page.get("chunks_status") != expected_chunk_status:
                    raise ValueError("Page chunk IDs do not match their source-page assignment.")
                if source is not None:
                    source_page = source["pages"][page["page_number"] - 1]
                    if page.get("text") != source_page["text"] or page.get("text_parser") != source["parser"]:
                        raise ValueError("Page text/parser does not match its extraction snapshot.")
                values = validate_page(page, document, pdf_hash, manifest_path)
                with database.transaction():
                    inserted = database.execute(INSERT_PAGE_SQL, values).fetchone()
                row.update(status="inserted" if inserted else "unchanged", context_id=values["context_id"])
                result[row["status"] + "_count"] += 1
            except (ValueError, KeyError, TypeError, OSError, psycopg.Error) as error:
                message = f"Database write failed ({type(error).__name__})" if isinstance(error, psycopg.Error) else str(error)
                row.update(status="failed", error_type=type(error).__name__, error=message)
                result["failed_count"] += 1
                logging.warning("Page %s SQL import failed: %s", page["page_number"], message)
            result["pages"].append(row)
    result["status"] = "stored" if not result["failed_count"] else "partial" if result["inserted_count"] + result["unchanged_count"] else "failed"
    logging.info("Page SQL import: inserted=%s unchanged=%s failed=%s",
                 result["inserted_count"], result["unchanged_count"], result["failed_count"])
    return result


def read_page_contexts(arxiv_id, version, connection=None):
    """Return all stored variants, including failures; this is not relevance ranking."""
    context = open_connection() if connection is None else nullcontext(connection)
    with context as database:
        return database.execute(READ_PAGES_SQL, (arxiv_id, version)).fetchall()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Document-level JSON produced by render_pages")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        result = store_pages(args.manifest)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "stored" else 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, psycopg.Error) as error:
        parser.exit(1, f"Page import failed ({type(error).__name__}). Check the manifest, assets, schema and parent document.\n")


if __name__ == "__main__":
    raise SystemExit(main())
