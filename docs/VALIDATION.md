# Validation

Scope: metadata retrieval and one versioned PDF download using the Python
standard library.

## Offline tests

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_arxiv_search.py -v
```

Recorded result on November 3, 2026: **six tests passed**.
All API responses and network failures in these tests are simulated.

The tests cover:

- Document version and separate first-submission and version-submission times.
- Missing metadata remaining unknown.
- Empty feeds and rejection of invalid entries or unexpected response formats.
- A request limited to one publication with a 30-second timeout.
- Connection retries stopping after three attempts, with 3.1-second delays.
- Rate-limit and `Retry-After` errors stopping automatic retries.

## Live source check

On November 3, 2026, a real query for `all:diabetes` returned
[`2611.03601v1`](https://arxiv.org/abs/2611.03601v1), titled
"DiabAgent: Toward Agentic Physiological Control".

The initial request failed certificate verification because the local Python
installation lacked its default CA bundle. A subsequent request using the
existing system CA bundle succeeded with HTTPS verification enabled. Setup
instructions are in the [README](../aws_version/README.md#run).

This source check confirms metadata retrieval for one publication, not search
relevance or historical public availability.

## PDF download tests

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_download_pdf.py -v
```

Recorded result on November 3, 2026: **seven download tests passed**. The six
metadata tests were also rerun and passed. Download tests use synthetic PDF
bytes and real temporary files; the fixtures are not parseable research papers.

The download tests cover:

- Saved bytes, SHA-256, source metadata, requested version, and timeout.
- Missing versions rejected before network access.
- Empty, non-PDF, and oversized responses rejected with temporary files removed.
- A truncated response retried successfully without leftover partial files.
- Connection failures stopping after three attempts.
- HTTP rate limits and server-specified retry periods stopping automatic retries.
- Metadata-write failure preserving the PDF and allowing a successful rerun.

## Live PDF download

A new `all:diabetes` API query was followed by a download from
`https://arxiv.org/pdf/2611.03601v1` on November 3, 2026.

- Saved size: **2,311,543 bytes**.
- SHA-256: `2915ebe342ca6a60ddfd56842d19715d2580449e3e137d5a4708babc645cdb6d`.
- A separate `shasum -a 256` calculation on the saved file matched its JSON record.
- One PDF and one successful retrieval record were saved; no partial files remained.
- Metadata retrieval and PDF retrieval have separate timestamps. Source submission
  timestamps were retained without inferring public availability.

This confirms transport and local storage for one real PDF. Only its header,
size, and hash were checked; full PDF readability and text extraction have not
been tested. Database storage, embeddings, document search, generated answers,
and AWS integration are not implemented yet.

Failures are reported on stderr; only successful downloads receive JSON records.
A PDF and its JSON record are written separately. A metadata-write failure can
therefore leave a saved PDF without a matching JSON record until a rerun succeeds.
Repeated downloads may share the same hash-named PDF and create separate retrieval
records; database-level duplicate prevention is a later step.

## Runtime progress output

The download command now reports search, request attempts, downloaded bytes,
and saved file paths on stderr while running. Its final stdout output remains
JSON. After this change, all 13 metadata and download tests passed. A real run
emitted progress messages and completed successfully for `2611.03601v1`.
