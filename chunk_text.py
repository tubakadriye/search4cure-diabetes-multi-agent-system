"""Split extracted pages into traceable word windows without crossing page boundaries."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re

from download_pdf import save_metadata


def _page_windows(text, max_words, overlap_words):
    """Yield exact text slices; offsets are Unicode character indices, end-exclusive."""
    words = list(re.finditer(r"\S+", text))
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        first_char, last_char = words[start].start(), words[end - 1].end()
        yield {
            "start_char": first_char, "end_char": last_char,
            "text": text[first_char:last_char], "word_count": end - start,
        }
        if end == len(words):
            break
        start = end - overlap_words


def chunk_text(extraction_path, output_dir=None, max_words=100, overlap_words=20):
    """Create stable chunk IDs and save a separate processing-attempt record."""
    if type(max_words) is not int or max_words < 1:
        raise ValueError("max_words must be a positive integer.")
    if type(overlap_words) is not int or not 0 <= overlap_words < max_words:
        raise ValueError("overlap_words must be between zero and max_words - 1.")
    extraction_path = Path(extraction_path)
    source_bytes = extraction_path.read_bytes()
    source = json.loads(source_bytes)
    try:
        document = source["download"]["search"]["document"]
        identifier, version = document["arxiv_id"], document["version"]
        pdf_hash = source["download"]["sha256"]
        pdf_path = Path(source["download"]["local_path"])
        parser_config, pages, page_count = source["parser"], source["pages"], source["page_count"]
    except (KeyError, TypeError) as error:
        raise ValueError("The extraction record is missing required metadata.") from error
    if source.get("status") != "extracted" or not isinstance(pages, list):
        raise ValueError("A completed extraction record is required.")
    if not isinstance(parser_config, dict) or not parser_config:
        raise ValueError("The source parser configuration is required.")
    if not isinstance(identifier, str) or not identifier or type(version) is not int or version < 1:
        raise ValueError("An explicit document identifier and version are required.")
    if not isinstance(pdf_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", pdf_hash):
        raise ValueError("A valid source PDF SHA-256 is required.")

    folder = Path(output_dir) if output_dir is not None else pdf_path.parent / "chunks"
    folder.mkdir(parents=True, exist_ok=True)
    config = {
        "method": "page_word_windows_v1", "max_words": max_words,
        "overlap_words": overlap_words, "word_definition": "non-whitespace runs",
        "offsets": "Unicode character indices, end-exclusive", "cross_page": False,
    }
    result = {
        "source_extraction_path": str(extraction_path),
        "source_extraction_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "document": document, "source_pdf_sha256": pdf_hash,
        "parser": parser_config, "chunking": config,
        "status": "failed", "error": None, "chunks": [], "pages_without_text": [],
    }
    failure = None
    try:
        if type(page_count) is not int or page_count != len(pages) or page_count < 1:
            raise ValueError("The extraction page count is inconsistent.")
        numbers = [page["page_number"] for page in pages]
        if any(type(number) is not int for number in numbers) or numbers != list(range(1, page_count + 1)):
            raise ValueError("Physical page numbers must be consecutive and start at 1.")
        for page in pages:
            page_number, text = page["page_number"], page["text"]
            if not isinstance(text, str):
                raise ValueError("Each page must contain a text string.")
            count = 0
            for count, window in enumerate(_page_windows(text, max_words, overlap_words), start=1):
                # Exclude retrieval timestamps and UUID paths so equivalent reruns retain IDs.
                identity = {
                    "arxiv_id": identifier, "version": version, "pdf_sha256": pdf_hash,
                    "parser": parser_config, "chunking": config, "page_number": page_number,
                    "start_char": window["start_char"], "end_char": window["end_char"],
                    "text": window["text"],
                }
                encoded = json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
                result["chunks"].append({
                    "chunk_id": hashlib.sha256(encoded).hexdigest(),
                    "arxiv_id": identifier, "version": version, "source_pdf_sha256": pdf_hash,
                    "page_number": page_number, "chunk_number": count, **window,
                })
            if count == 0:
                result["pages_without_text"].append(page_number)
            logging.info("Page %s/%s: %s chunks", page_number, page_count, count)
        if not result["chunks"]:
            raise ValueError("No nonempty text chunks could be created.")
        result["status"] = "chunked"
    except Exception as error:
        failure = error
        result["error"] = f"{type(error).__name__}: {error}"
    result["chunk_count"] = len(result["chunks"])
    result["processed_at"] = datetime.now(timezone.utc).isoformat()
    result_path = save_metadata(folder, result)
    logging.info("Chunk record saved: %s", result_path)
    if failure is not None:
        raise ValueError(f"Chunking failed; details: {result_path}: {failure}") from failure
    return {**result, "metadata_path": str(result_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction", required=True, help="JSON record produced by extract_text")
    parser.add_argument("--output-dir", help="Defaults to a chunks subfolder beside the PDF")
    parser.add_argument("--max-words", type=int, default=100)
    parser.add_argument("--overlap-words", type=int, default=20)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        result = chunk_text(args.extraction, args.output_dir, args.max_words, args.overlap_words)
        print(json.dumps({key: value for key, value in result.items() if key != "chunks"},
                         indent=2, ensure_ascii=False))
    except (OSError, ValueError) as error:
        parser.exit(1, f"Text chunking failed: {error}\n")


if __name__ == "__main__":
    main()
