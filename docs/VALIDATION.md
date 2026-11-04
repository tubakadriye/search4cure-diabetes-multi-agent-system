# Validation

Scope: arXiv metadata retrieval, versioned PDF download, page-level text
extraction, and the executable walkthrough notebook.

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
instructions are in the [README](../README.md#run).

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

This download-only check confirmed transport and local storage for one real
PDF. Its header, size, and hash were checked. Text extraction was subsequently
validated separately as described below.

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

## Project-root entry points

After moving the modules to the project root, all 13 offline tests passed.
Both `python3 -m arxiv_search --help` and `python3 -m download_pdf --help`
completed successfully. No additional live API request was needed for this
layout change.

## Text extraction and notebook

Validated on November 4, 2026 with Python 3.13.12 and PyMuPDF 1.28.2.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u -m unittest discover -s tests -p 'test_*.py' -v
.venv/bin/python -u run_notebook.py
```

All **22 offline tests passed**: the previous 13 tests, eight extraction tests,
and one notebook setup regression test covering both project-root and
notebook-directory starts. Generated PDFs test physical page numbering,
provenance, parser configuration, empty pages, completely empty text,
unreadable PDFs, changed source hashes, missing metadata, NUL replacement,
and recovery after a result-storage failure.

The setup regression test initially failed for both starting directories
because `../data/arxiv` pointed outside the project after setup changed the
working directory. With the project-relative destination corrected, it passed.

The [walkthrough](../notebooks/walkthrough.ipynb) was executed from a fresh kernel
through all six code cells: setup, real metadata query, real download, extraction,
page preview, and offline tests. The notebook retains the actual outputs of the
successful run. The runner reports progress as it executes and removes stale
outputs before each new run.

The real source query returned
[`2611.04405v1`](https://arxiv.org/abs/2611.04405v1), titled
"DR-LiteNet: A Lightweight Explainable Hybrid CNN Framework for Imbalanced
Diabetic Retinopathy Grading via Adaptive SMOTE Fusion".

| Check | Observed result |
|---|---|
| Downloaded PDF size | 577,377 bytes |
| SHA-256 | `932ec59803c4c1495ebfbb7dd3b58e17afdbbc24e459e5a48f35f48a54bd3075` |
| Physical PDF pages | 6 |
| Extracted characters, including whitespace | 37,614 |
| Empty pages | 0 |
| NUL replacements | 0 |
| Parser repaired this PDF | No |
| Final notebook test process exit code | 0 |

Each extraction record links to its download record, exact document version,
PDF hash, parser configuration, and physical page numbers. Source submission,
metadata retrieval, PDF retrieval, and processing times remain distinct.
Failures after input-metadata validation receive a failed extraction record;
if result storage itself fails, the error is raised and the original remains.

These checks establish extraction and traceability for one real publication,
not scientific correctness, text-reading-order accuracy, or search relevance.
OCR, image/table interpretation, embeddings, database storage, search, answer
generation, and AWS deployment are still not implemented. The interactive
browser interface has not been separately inspected; execution was validated
with the registered local kernel through the notebook runner.
