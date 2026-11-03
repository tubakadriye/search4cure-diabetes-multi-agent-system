"""Search arXiv and download one explicitly versioned PDF with provenance."""

import argparse
from datetime import datetime, timezone
import hashlib
from http.client import IncompleteRead
import json
import logging
from pathlib import Path
import re
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid
import xml.etree.ElementTree as ET

from arxiv_search import USER_AGENT, search_one

CHUNK_BYTES = 64 * 1024
MAX_PDF_BYTES = 100 * 1024 * 1024


def save_metadata(folder, record):
    """Publish a complete JSON record; remove incomplete metadata on failure."""
    destination = folder / (uuid.uuid4().hex + ".json")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=folder,
                                         suffix=".part", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(record, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def download_pdf(search_result, output_dir="data/arxiv"):
    """Stream one pinned version to disk and record its content hash and origin."""
    document = search_result.get("document")
    if not document:
        raise ValueError("No publication was found to download.")
    identifier, version = document["arxiv_id"], document.get("version")
    if not re.fullmatch(r"\d{4}\.\d{4,5}|[A-Za-z.-]+/\d{7}", identifier):
        raise ValueError("Invalid arXiv identifier.")
    if type(version) is not int or version < 1:
        raise ValueError("An explicit arXiv version is required before downloading.")

    url = f"https://arxiv.org/pdf/{identifier}v{version}"
    folder = Path(output_dir) / f"{identifier.replace('/', '_')}v{version}"
    folder.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, 4):
        temporary = None
        try:
            digest, size = hashlib.sha256(), 0
            next_progress = 1024 * 1024
            logging.info("Downloading %s (attempt %s/3)", url, attempt)
            with urlopen(request, timeout=30) as response:
                length = response.headers.get("Content-Length")
                expected_size = int(length) if length is not None else None
                if expected_size is not None and expected_size > MAX_PDF_BYTES:
                    raise ValueError("The PDF exceeds the 100 MiB limit.")
                with tempfile.NamedTemporaryFile(dir=folder, suffix=".part", delete=False) as stream:
                    temporary = Path(stream.name)
                    while block := response.read(CHUNK_BYTES):
                        if size == 0 and not block.startswith(b"%PDF-"):
                            raise ValueError("The response does not start with a PDF header.")
                        size += len(block)
                        if size > MAX_PDF_BYTES:
                            raise ValueError("The PDF exceeds the 100 MiB limit.")
                        stream.write(block)
                        digest.update(block)
                        if size >= next_progress:
                            logging.info("Downloaded %s bytes", size)
                            next_progress = size + 1024 * 1024
            if size == 0:
                raise ValueError("The PDF response is empty.")
            if expected_size is not None and size != expected_size:
                raise IncompleteRead(b"", expected_size - size)

            checksum = digest.hexdigest()
            destination = folder / (checksum + ".pdf")
            temporary.replace(destination)
            logging.info("PDF saved: %s (%s bytes)", destination, size)
            record = {
                "search": search_result,
                "pdf_url": url,
                "pdf_retrieved_at": datetime.now(timezone.utc).isoformat(),
                "sha256": checksum,
                "size_bytes": size,
                "local_path": str(destination),
                "status": "downloaded",
            }
            metadata_path = save_metadata(folder, record)
            logging.info("Metadata saved: %s", metadata_path)
            return {**record, "metadata_path": str(metadata_path)}
        except (URLError, TimeoutError, IncompleteRead) as error:
            logging.warning("PDF request failed: arxiv_id=%s version=%s attempt=%s/3 error=%s",
                            identifier, version, attempt, error)
            if isinstance(error, HTTPError):
                error.close()
                if error.code < 500 or (error.headers and error.headers.get("Retry-After")):
                    raise
            if attempt == 3:
                raise
            time.sleep(3.1)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="all:diabetes", help="arXiv search expression")
    parser.add_argument("--output-dir", default="data/arxiv", help="Local destination directory")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        logging.info("Searching arXiv: %s", args.query)
        result = search_one(args.query)
        if result["document"] is None:
            parser.exit(1, "No publication matched the query.\n")
        document = result["document"]
        logging.info("Found %sv%s: %s", document["arxiv_id"], document["version"], document["title"])
        # Keep the PDF request separate from the metadata request.
        logging.info("Waiting 3.1 seconds before the PDF request")
        time.sleep(3.1)
        print(json.dumps(download_pdf(result, args.output_dir), indent=2, ensure_ascii=False))
    except (OSError, IncompleteRead, ValueError, ET.ParseError) as error:
        parser.exit(1, f"arXiv download failed: {error}\n")


if __name__ == "__main__":
    main()
