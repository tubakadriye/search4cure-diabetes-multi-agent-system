"""Render complete visible PDF pages and link their images, text and chunks."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re

import pymupdf

from download_pdf import save_metadata
from extract_images import DEFAULT_MAX_PIXELS, _file_hash, _save_png


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _load_links(download, extraction_path, chunks_path):
    """Reject mismatched versions, extraction snapshots, page numbers or text slices."""
    document = download["search"]["document"]
    links = {"extraction_path": None, "extraction_sha256": None,
             "chunks_path": None, "chunks_sha256": None, "chunking": None}
    if extraction_path is None:
        if chunks_path is not None:
            raise ValueError("A chunk record requires its source extraction record.")
        return None, {}, links
    extraction_bytes = Path(extraction_path).read_bytes()
    source = json.loads(extraction_bytes)
    if source.get("status") != "extracted":
        raise ValueError("A completed text extraction is required when supplied.")
    source_document = source["download"]["search"]["document"]
    if (source_document["arxiv_id"], source_document["version"], source["download"]["sha256"]) != (
            document["arxiv_id"], document["version"], download["sha256"]):
        raise ValueError("Text extraction and download refer to different document versions or hashes.")
    pages = source["pages"]
    count = source["page_count"]
    if type(count) is not int or count < 1 or not isinstance(pages, list) or len(pages) != count:
        raise ValueError("Invalid extraction page count.")
    numbers = [page["page_number"] for page in pages]
    if any(type(n) is not int for n in numbers) or numbers != list(range(1, count + 1)):
        raise ValueError("Extraction pages must be consecutive physical page numbers.")
    if any(not isinstance(page.get("text"), str) for page in pages):
        raise ValueError("Every extracted page requires a text string.")
    if not isinstance(source.get("parser"), dict) or not source["parser"]:
        raise ValueError("Text parser configuration is required.")
    links.update(extraction_path=str(extraction_path),
                 extraction_sha256=hashlib.sha256(extraction_bytes).hexdigest())
    by_page = {number: [] for number in numbers}
    if chunks_path is not None:
        chunk_bytes = Path(chunks_path).read_bytes()
        chunks = json.loads(chunk_bytes)
        if chunks.get("status") != "chunked" or chunks.get("source_extraction_sha256") != links["extraction_sha256"]:
            raise ValueError("Chunks must refer to this exact successful extraction snapshot.")
        if (chunks["document"]["arxiv_id"], chunks["document"]["version"], chunks["source_pdf_sha256"]) != (
                document["arxiv_id"], document["version"], download["sha256"]):
            raise ValueError("Chunks refer to a different document version or PDF hash.")
        if chunks.get("parser") != source["parser"] or not isinstance(chunks.get("chunking"), dict):
            raise ValueError("Chunk parser/chunking configuration is inconsistent.")
        items = chunks["chunks"]
        if not isinstance(items, list) or not items or chunks.get("chunk_count") != len(items):
            raise ValueError("Invalid chunk count.")
        seen = set()
        for chunk in items:
            number, start, end = chunk["page_number"], chunk["start_char"], chunk["end_char"]
            if type(number) is not int or number not in by_page:
                raise ValueError("A chunk refers to an unknown physical page.")
            text = pages[number - 1]["text"]
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
                raise ValueError("Invalid chunk text offsets.")
            if chunk["text"] != text[start:end]:
                raise ValueError("A chunk does not match its associated page text.")
            if (chunk["arxiv_id"], chunk["version"], chunk["source_pdf_sha256"]) != (
                    document["arxiv_id"], document["version"], download["sha256"]):
                raise ValueError("A chunk has mismatched document provenance.")
            identifier = chunk["chunk_id"]
            if not isinstance(identifier, str) or not re.fullmatch(r"[0-9a-f]{64}", identifier) or identifier in seen:
                raise ValueError("Chunk IDs must be valid and unique.")
            seen.add(identifier)
            by_page[number].append(identifier)
        if any(page["text"].strip() and not by_page[page["page_number"]] for page in pages):
            raise ValueError("A nonempty text page is missing its chunks.")
        links.update(chunks_path=str(chunks_path), chunks_sha256=hashlib.sha256(chunk_bytes).hexdigest(),
                     chunking=chunks["chunking"])
    return source, by_page, links


def _render_page(page, folder, config):
    """Bound the output dimensions before allocating a full-page RGB pixmap."""
    bounds = (page.rect * pymupdf.Matrix(config["dpi"] / 72, config["dpi"] / 72)).irect
    if bounds.width < 1 or bounds.height < 1 or bounds.width * bounds.height > config["max_pixels"]:
        raise ValueError("Page exceeds the configured rendering pixel limit or has invalid dimensions.")
    pixmap = page.get_pixmap(dpi=config["dpi"], colorspace=pymupdf.csRGB, alpha=False, annots=True)
    if pixmap.width * pixmap.height > config["max_pixels"]:
        raise ValueError("Rendered page exceeds the configured pixel limit.")
    data = pixmap.tobytes("png")
    path, checksum = _save_png(folder, data)
    return {"local_path": str(path), "sha256": checksum, "size_bytes": len(data),
            "width": pixmap.width, "height": pixmap.height, "mime_type": "image/png"}


def render_pages(metadata_path, extraction_path=None, chunks_path=None, output_dir=None,
                 dpi=144, max_pixels=DEFAULT_MAX_PIXELS):
    """Save one page record per physical page, including failed render attempts."""
    if type(dpi) is not int or not 1 <= dpi <= 600:
        raise ValueError("dpi must be an integer between 1 and 600.")
    if type(max_pixels) is not int or max_pixels < 1:
        raise ValueError("max_pixels must be a positive integer.")
    metadata_path = Path(metadata_path)
    download = json.loads(metadata_path.read_text(encoding="utf-8"))
    try:
        document = download["search"]["document"]
        identifier, version = document["arxiv_id"], document["version"]
        pdf_path, checksum = Path(download["local_path"]), download["sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError("The download record is missing required metadata.") from error
    if download.get("status") != "downloaded" or not isinstance(identifier, str) or not identifier:
        raise ValueError("A successful, identified download is required.")
    if type(version) is not int or version < 1 or not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ValueError("An explicit version and PDF SHA-256 are required.")
    folder = Path(output_dir) if output_dir is not None else pdf_path.parent / "pages"
    assets, records = folder / "assets", folder / "records"
    assets.mkdir(parents=True, exist_ok=True)
    records.mkdir(parents=True, exist_ok=True)
    config = {"name": "PyMuPDF", "version": pymupdf.VersionBind, "dpi": dpi,
              "max_pixels": max_pixels, "colorspace": "RGB", "alpha": False,
              "annotations": True, "page_box": "complete visible CropBox", "rotation": "honor PDF rotation"}
    text_parser = {"name": "PyMuPDF", "version": pymupdf.VersionBind, "text_mode": "text",
                   "sort": True, "ocr": False, "nul_handling": "replace NUL with U+FFFD"}
    result = {"source_metadata_path": str(metadata_path), "download": download, "rendering": config,
              "status": "failed", "error": None, "page_count": None, "pages": []}
    failure = None
    try:
        logging.info("Verifying PDF hash for full-page rendering: %s", pdf_path)
        if _file_hash(pdf_path) != checksum:
            raise ValueError("The saved PDF does not match the recorded SHA-256.")
        source, by_page, links = _load_links(download, extraction_path, chunks_path)
        if source is not None:
            text_parser = source["parser"]
        result.update(links)
        with pymupdf.open(pdf_path) as pdf:
            if not pdf.is_pdf or pdf.needs_pass:
                raise ValueError("An unencrypted PDF is required.")
            result["page_count"] = pdf.page_count
            result["parser_repaired_pdf"] = pdf.is_repaired
            if pdf.is_repaired:
                logging.warning("The parser repaired the PDF while opening it")
            if source is not None and source["page_count"] != pdf.page_count:
                raise ValueError("Text extraction and actual PDF page counts differ.")
            for number, page in enumerate(pdf, 1):
                page_id = _identity({"arxiv_id": identifier, "version": version,
                                     "source_pdf_sha256": checksum, "page_number": number})
                item = {"page_id": page_id, "render_id": _identity({"page_id": page_id, "rendering": config}),
                        "arxiv_id": identifier, "version": version, "document": document,
                        "source_pdf_sha256": checksum, "source_metadata_path": str(metadata_path),
                        "page_number": number, "rendering": config, "text_parser": text_parser,
                        "page_rect": list(page.rect), "cropbox": list(page.cropbox), "rotation": page.rotation,
                        "text": None, "text_status": "failed", "text_sha256": None, "text_error": None,
                        "image_status": "failed", "image_error": None,
                        "chunk_ids": by_page.get(number, []),
                        "chunks_status": "linked" if chunks_path is not None else "not_provided",
                        **links, "status": "failed", "error": None, "image": None}
                try:
                    if source is None:
                        raw = page.get_text("text", sort=True)
                        text, replacements = raw.replace("\x00", "\ufffd"), raw.count("\x00")
                    else:
                        source_page = source["pages"][number - 1]
                        text, replacements = source_page["text"], source_page.get("nul_replacements")
                    item.update(text=text, text_status="extracted" if text.strip() else "empty",
                                text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                                nul_replacements=replacements)
                except Exception as error:
                    item["text_error"] = f"{type(error).__name__}: {error}"
                    logging.warning("Page %s text extraction failed: %s", number, item["text_error"])
                try:
                    # A text extraction error must not prevent a usable page image.
                    item["image"] = _render_page(page, assets, config)
                    item["image_status"] = "rendered"
                    item["status"] = "rendered"
                except Exception as error:
                    item["image_error"] = f"{type(error).__name__}: {error}"
                    logging.warning("Page %s rendering failed: %s", number, item["image_error"])
                item["error"] = "; ".join(message for message in (item["text_error"], item["image_error"]) if message) or None
                item["processed_at"] = datetime.now(timezone.utc).isoformat()
                try:
                    item["record_path"] = str(save_metadata(records, item))
                except OSError as error:
                    item.update(status="failed", error=f"Page record storage failed: {error}")
                    logging.warning("Page %s record could not be stored: %s", number, type(error).__name__)
                result["pages"].append(item)
                logging.info("Full page %s/%s: %s; text=%s; linked chunks=%s", number, pdf.page_count,
                             item["status"], item["text_status"], len(item["chunk_ids"]))
        successful = sum(item["status"] == "rendered" for item in result["pages"])
        complete = successful == result["page_count"] and all(item["text_status"] != "failed" for item in result["pages"])
        result["status"] = "rendered" if complete else "partial" if successful else "failed"
    except Exception as error:
        failure = error
        result["error"] = f"{type(error).__name__}: {error}"
    result["processed_at"] = datetime.now(timezone.utc).isoformat()
    result["rendered_count"] = sum(item["status"] == "rendered" for item in result["pages"])
    result["failed_count"] = sum(item["status"] == "failed" for item in result["pages"])
    result["text_failed_count"] = sum(item["text_status"] == "failed" for item in result["pages"])
    result["image_failed_count"] = sum(item["image_status"] == "failed" for item in result["pages"])
    path = save_metadata(folder, result)
    logging.info("Full-page attempt record saved: %s", path)
    if failure is not None:
        raise ValueError(f"Page rendering failed; details: {path}: {failure}") from failure
    return {**result, "metadata_path": str(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="Successful download JSON")
    parser.add_argument("--extraction", help="Optional completed text-extraction JSON")
    parser.add_argument("--chunks", help="Optional chunks JSON from exactly that extraction")
    parser.add_argument("--output-dir", help="Defaults to pages beside the PDF")
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        result = render_pages(args.metadata, args.extraction, args.chunks, args.output_dir, args.dpi, args.max_pixels)
        print(json.dumps({key: value for key, value in result.items() if key not in ("download", "pages")}, indent=2))
        return 0 if result["status"] == "rendered" else 1
    except (OSError, ValueError) as error:
        parser.exit(1, f"Full-page rendering failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
