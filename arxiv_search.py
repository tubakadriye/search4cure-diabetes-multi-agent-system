"""Fetch metadata for one arXiv publication using only the Python standard library."""

import argparse
from datetime import datetime, timezone
import json
import logging
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

ATOM = "{http://www.w3.org/2005/Atom}"
USER_AGENT = "Search4CureAI/0.1 (+https://github.com/tubakadriye/search4cure-diabetes-multi-agent-system)"


def parse_document(raw_xml):
    """Keep source timestamps as provided; missing values remain unknown."""
    root = ET.fromstring(raw_xml)
    if root.tag != ATOM + "feed":
        raise ValueError("The response is not an arXiv Atom feed.")
    entry = root.find(ATOM + "entry")
    if entry is None:
        return None

    def field(name):
        return (entry.findtext(ATOM + name) or "").strip() or None

    source = field("id") or ""
    match = re.fullmatch(
        r"https?://arxiv.org/abs/(?P<id>\d{4}\.\d{4,5}|[A-Za-z.-]+/\d{7})(?:v(?P<version>[1-9]\d*))?",
        source,
    )
    if not match or not field("title"):
        raise ValueError("The arXiv entry is missing a valid identifier or title.")
    return {
        "arxiv_id": match["id"],
        "version": int(match["version"]) if match["version"] else None,
        "title": " ".join(field("title").split()),
        "source_url": source,
        "first_submitted_at": field("published"),
        "version_submitted_at": field("updated"),
    }


def search_one(query):
    """Request at most one result, with a timeout and at most three attempts."""
    if not query.strip():
        raise ValueError("The search query must not be empty.")
    parameters = urlencode({
        "search_query": query, "start": 0, "max_results": 1,
        "sortBy": "submittedDate", "sortOrder": "descending",
    })
    url = "https://export.arxiv.org/api/query?" + parameters
    request = Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, 4):
        try:
            with urlopen(request, timeout=30) as response:
                raw_xml = response.read(2 * 1024 * 1024 + 1)
            if len(raw_xml) > 2 * 1024 * 1024:
                raise ValueError("The metadata response exceeds the 2 MiB limit.")
            break
        except (URLError, TimeoutError) as error:
            logging.warning("arXiv request failed: attempt=%s/3 error=%s", attempt, error)
            if isinstance(error, HTTPError):
                error.close()
                # Stop rather than retry before a server-specified waiting period.
                if error.code < 500 or (error.headers and error.headers.get("Retry-After")):
                    raise
            if attempt == 3:
                raise
            time.sleep(3.1)
    return {
        "query": query,
        "request_url": url,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "document": parse_document(raw_xml),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="all:diabetes", help="arXiv search expression")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    try:
        print(json.dumps(search_one(args.query), indent=2, ensure_ascii=False))
    except (URLError, TimeoutError, ValueError, ET.ParseError) as error:
        parser.exit(1, f"arXiv search failed: {error}\n")


if __name__ == "__main__":
    main()
