"""Extract page-level text from a downloaded PDF while preserving its provenance."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re

import pymupdf

from download_pdf import CHUNK_BYTES, save_metadata


def extract_text(metadata_path, output_dir=None):
    """Verify the source PDF, extract physical pages, and save an attempt record."""
    metadata_path = Path(metadata_path)
    download = json.loads(metadata_path.read_text(encoding="utf-8"))
    try:
        document = download["search"]["document"]
        identifier, version = document["arxiv_id"], document["version"]
        pdf_path, expected_hash = Path(download["local_path"]), download["sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError("The download record is missing required metadata.") from error
    if download.get("status") != "downloaded" or not isinstance(identifier, str) or not identifier:
        raise ValueError("A successful download record with an arXiv identifier is required.")
    if type(version) is not int or version < 1:
        raise ValueError("An explicit document version is required.")
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError("A valid SHA-256 is required in the download record.")

    folder = Path(output_dir) if output_dir is not None else pdf_path.parent / "extracted"
    folder.mkdir(parents=True, exist_ok=True)
    result = {
        "source_metadata_path": str(metadata_path),
        "download": download,
        "parser": {
            "name": "PyMuPDF", "version": pymupdf.VersionBind,
            "text_mode": "text", "sort": True, "ocr": False,
            "page_numbering": "physical PDF pages, starting at 1",
            "nul_handling": "replace NUL with U+FFFD and record the replacement count",
        },
        "status": "failed", "error": None, "page_count": None, "pages": [],
    }
    failure = None
    try:
        logging.info("Verifying PDF hash: %s", pdf_path)
        digest = hashlib.sha256()
        with pdf_path.open("rb") as stream:
            while block := stream.read(CHUNK_BYTES):
                digest.update(block)
        if digest.hexdigest() != expected_hash:
            raise ValueError("The saved PDF does not match the recorded SHA-256.")

        with pymupdf.open(pdf_path) as pdf:
            if not pdf.is_pdf or pdf.needs_pass:
                raise ValueError("An unencrypted PDF is required.")
            result["parser_repaired_pdf"] = pdf.is_repaired
            if pdf.is_repaired:
                logging.warning("The parser repaired the PDF while opening it")
            result["page_count"] = pdf.page_count
            for index, page in enumerate(pdf):
                raw_text = page.get_text("text", sort=True)
                replacements = raw_text.count("\x00")
                text = raw_text.replace("\x00", "\ufffd")
                result["pages"].append({
                    "page_number": index + 1,
                    "text": text,
                    "status": "extracted" if text.strip() else "empty",
                    "nul_replacements": replacements,
                })
                logging.info("Page %s/%s: %s characters%s", index + 1, pdf.page_count,
                             len(text), " (empty)" if not text.strip() else "")
                if replacements:
                    logging.warning("Page %s: replaced %s NUL characters", index + 1, replacements)
        if not any(page["status"] == "extracted" for page in result["pages"]):
            raise ValueError("No extractable text was found; OCR is not implemented.")
        result["status"] = "extracted"
    except Exception as error:
        # Record the failure before re-raising it; never present partial output as success.
        failure = error
        result["error"] = f"{type(error).__name__}: {error}"

    result["processed_at"] = datetime.now(timezone.utc).isoformat()
    result["empty_pages"] = [page["page_number"] for page in result["pages"] if page["status"] == "empty"]
    result["total_characters"] = sum(len(page["text"]) for page in result["pages"])
    result_path = save_metadata(folder, result)
    logging.info("Extraction record saved: %s", result_path)
    if failure is not None:
        raise ValueError(f"Text extraction failed; details: {result_path}: {failure}") from failure
    return {**result, "metadata_path": str(result_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="JSON record produced by download_pdf")
    parser.add_argument("--output-dir", help="Defaults to an extracted subfolder beside the PDF")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        result = extract_text(args.metadata, args.output_dir)
        summary = {key: value for key, value in result.items() if key not in ("pages", "download")}
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    except (OSError, ValueError) as error:
        parser.exit(1, f"PDF text extraction failed: {error}\n")


if __name__ == "__main__":
    main()
