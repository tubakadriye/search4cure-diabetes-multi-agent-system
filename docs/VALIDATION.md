# Validation

Scope: arXiv metadata retrieval, versioned PDF download, page-level text
extraction, traceable text chunks, and the executable walkthrough notebook.

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

At the extraction stage, the [walkthrough](../notebooks/walkthrough.ipynb) was
executed from a fresh kernel through six code cells: setup, real metadata query,
real download, extraction, page preview, and offline tests. The later chunking
stage adds another cell, as described below. The runner reports progress and
removes stale outputs before each new run.

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

## Page-local text chunks

Validated on November 4, 2026. All **30 offline tests passed**, including eight
new chunking tests. They cover word limits, overlap, exact Unicode text offsets,
coverage of source words, page boundaries, empty pages, stable IDs across
equivalent reruns, changed versions/hashes/configuration, invalid parameters,
incomplete extraction records, invalid page numbers, and recovery after storage
failure. The command-line `--help` check also completed successfully.

The notebook was then executed from a fresh kernel through all **seven code
cells**, including the new chunking cell and all 30 tests. It began with a new
real `all:diabetes` query and downloaded `2611.04405v1` again. Outputs are saved
in the notebook.

| Check | Observed result |
|---|---|
| Chunk configuration | At most 100 whitespace-separated words, 20-word overlap |
| Source document | `2611.04405v1`, 6 physical pages |
| Total chunks | 50 |
| Chunks per page | 8, 10, 6, 10, 8, 8 |
| Pages without text | 0 |
| Notebook test-process exit code | 0 |

The live-sample checks compare each chunk with its exact source-page character
slice, check its document/version/PDF-hash fields, verify unique IDs and word
limits, and check coverage of all non-whitespace source characters. Offsets
refer to Unicode character indices in the extracted page text, not PDF bytes.

Each chunking record stores the source extraction file path and SHA-256,
document metadata, source PDF hash, parser configuration and chunking policy.
Chunk IDs exclude retrieval timestamps and attempt filenames; offline tests
confirm that equivalent content retains IDs. Separate attempt records are
still created. Database deduplication and embedding reuse are not implemented.

This word-window policy can split sentences and is not an embedding-token
budget. Model token limits must be checked before embeddings are introduced.
The text preview contains fragmented lines; exact source slices establish
attribution to the parser output, not correct PDF reading order or relevance.
No retrieval-quality improvement is claimed for this configuration.

## Local PostgreSQL setup and notebook connection checks

Validated on November 4, 2026 with Python 3.13.12 and Psycopg 3.3.7.
The initial check found no PostgreSQL driver or local password file. After
installing the runtime requirements, these commands completed successfully:

```bash
.venv/bin/python -m database init-password
docker compose config --quiet
docker compose up -d --wait --wait-timeout 60 postgres
.venv/bin/python -u -m database check
S4C_TEST_POSTGRES=0 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u -m unittest discover -s tests -p 'test_*.py' -v
.venv/bin/python -u run_notebook.py
```

| Check | Observed result |
|---|---|
| Separate Compose project | `search4cure-local` |
| Image | `pgvector/pgvector:0.8.7-pg16-bookworm` |
| Database volume created | `search4cure-local_postgres_data` |
| Published address | `127.0.0.1:55432` |
| Container health | Healthy |
| Connected database / role | `search4cure` / `search4cure` |
| PostgreSQL version | `16.15 (Debian 16.15-1.pgdg12+2)` |
| Enabled pgvector version | `0.8.7` |
| Public application tables | 0 |
| Password file permissions | `0600` |
| Git ignores password file and root `.env` | Confirmed |

The password file is `data/postgres/password.txt`. Existing password files are preserved. Docker receives this file
through a Compose secret; the Python connection reads the same local file.
The file content is never printed by the setup or connection functions.

The default offline run discovered 38 tests: **36 passed and two database
integration tests were skipped**, in 0.619 seconds. Six new offline tests cover
private password creation, preservation of an existing file, missing or empty
credentials, connection/statement timeouts, context-manager cleanup, missing
pgvector, and connection failure propagation.

The notebook was executed from a fresh kernel through **nine code cells**,
including executable password setup, Docker startup, and the read-only SQL
check. Its test subprocess sets `S4C_TEST_POSTGRES=1`: **all 38 tests passed**
in 0.610 seconds, with no skips. The two actual database tests verified a
successful connection and an authentication rejection for an incorrect
password. The latter checks the authentication error text, so an unreachable
server alone would not satisfy the test.

The notebook again queried real arXiv metadata, downloaded `2611.04405v1`,
extracted six pages, and produced 50 chunks. It then reused the existing
password file and started the already-running database without replacing it.
Saved notebook outputs have execution counts 1 through 9 and no error outputs.
An additional check confirmed that the password value is absent from the saved
notebook. These are executed notebook checks, not a separate inspection of the
interactive browser interface.

Only database setup and connectivity are implemented. The SQL check reads
server metadata; documents and chunks remain in files. There are no application
tables, SQL document imports, embeddings, or vector searches yet. Extension
initialization runs only when the database volume is first initialized. A new
password file does not change credentials already stored in an existing volume.
The role is a local bootstrap administrator, not a prepared production role.
Backup/restore, AWS deployment, and production access controls are not tested.

Configuration references:
[Docker Compose secrets](https://docs.docker.com/compose/how-tos/use-secrets/),
[pgvector installation and Docker images](https://github.com/pgvector/pgvector),
and [Psycopg connection usage](https://www.psycopg.org/psycopg3/docs/basic/usage.html).

## Store one document version in PostgreSQL

Validated on November 4, 2026. This step adds `document_versions`, an explicit
schema initializer, download-record validation, a parameterized SQL importer,
and a pandas display in the existing notebook. Runtime requirements are
unchanged; pandas 3.1.0 is a notebook-only dependency. The local installation
resolved NumPy 2.5.4. Transitive dependencies are not fully locked.

The offline command discovered 50 tests: **42 passed and eight database tests
were skipped**, in 11.439 seconds. With `S4C_TEST_POSTGRES=1`, **all 50 passed**
in 1.611 seconds. Twelve new tests cover required fields, document/version URL
consistency, unknown source times, timezone-aware retrieval times, changed or
missing PDFs, failed downloads, invalid hashes, source preservation on database
failure, duplicate import, new versions, content conflicts, quoted titles,
repeat schema initialization, SQL rollback and retry.

Document integration tests create an isolated schema inside a transaction with
forced rollback. Their explicitly synthetic download files are not parser
fixtures and are never imported into the application table. A deliberate SQL
constraint failure verified that an unsuccessful insert leaves zero rows and
that retry succeeds after the test constraint is removed.

The complete notebook then ran from a fresh kernel through **ten code cells**:
real arXiv query, download, extraction, chunking, preview, PostgreSQL setup and
check, the new SQL import/display, and tests. It downloaded `2611.04405v1`
again (577,377 bytes, SHA-256
`932ec59803c4c1495ebfbb7dd3b58e17afdbbc24e459e5a48f35f48a54bd3075`),
extracted six pages, and produced 50 chunks. All **50 notebook tests passed**
in 0.705 seconds, without skips; code-cell execution counts are 1 through 10,
with no error outputs. The command-line import `--help` check also succeeded.

| Live import check | Observed result |
|---|---|
| First import of this version | `inserted` |
| Immediate repeated import of the same download JSON | `unchanged` |
| SQL rows for this document version | 1 |
| First import duration | 0.011682 seconds |
| Repeated import duration | 0.011077 seconds |
| Duplicate versions from `sql/quality.sql` | 0 |
| Rows with unknown submission times in this real sample | 0 |
| Remaining synthetic test schemas | 0 |
| Synthetic documents in the application table | 0 |

Durations were measured with `time.perf_counter()` around each `store_document`
call, including file validation, opening a connection and SQL execution. These
are two observations on one local Docker database, not a throughput benchmark
or evidence of a meaningful speed improvement. The checked improvement is
duplicate prevention: repeat import retains one row and the first snapshot.
Embedding reuse is not implemented or measured.

The primary key is `(arxiv_id, version)`. `ON CONFLICT DO NOTHING` avoids
overwriting existing rows; the importer then compares the stored PDF hash and
rejects changed content for that version. Successful writes are transactional.
The stored snapshot contains the first imported download JSON and its path,
source URLs, PDF path/hash/size, source submission times, metadata and PDF
retrieval times, and the database write time. Unknown source times stay `NULL`.

Limits: the table's `downloaded` status describes downloading only. Parser and
chunk configurations remain in local JSON; page text, chunks, image references,
processing errors and import-run records are not yet stored in SQL. Later
retrieval records stay on disk rather than replacing the SQL snapshot. No
historical availability filter, embedding, retrieval-quality evaluation or AWS
deployment is provided by this step. Schema creation preserves an existing
table; it is not a migration mechanism for a differently defined table.

References consulted for this implementation:
[PostgreSQL 16 INSERT / ON CONFLICT](https://www.postgresql.org/docs/16/sql-insert.html),
[Psycopg parameter binding](https://www.psycopg.org/psycopg3/docs/basic/params.html),
[Psycopg transactions](https://www.psycopg.org/psycopg3/docs/basic/transactions.html),
and the [pandas release on PyPI](https://pypi.org/project/pandas/3.1.0/).

## SQL-only commit verification

On November 4, 2026, the notebook was executed separately with the SQL-storage
implementation and without raster-image code or tests. All ten code cells
completed successfully, and all 50 tests passed in 0.701 seconds without skips.
The real arXiv query again returned `2611.04405v1`. Both SQL imports returned
`unchanged`, retaining one row because this version was already stored. The
first-insert measurements above describe the earlier run; the saved notebook
outputs in this SQL-only commit show this subsequent verification run.

## Referenced raster-image extraction

Validated on November 4, 2026 with the existing PyMuPDF 1.28.2 dependency.
No additional package was installed for this step. `extract_images.py` accepts
the successful download JSON, verifies the source PDF hash, and records raster
occurrences by physical page and unrotated page coordinates. PNG assets are
named by their output SHA-256; separate occurrence IDs include document version,
PDF hash, page/position, PDF object reference and parser configuration.

Executed checks:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u -m unittest discover -s tests -p test_extract_images.py -v
.venv/bin/python -u run_notebook.py
.venv/bin/python -m extract_images --help
```

All **12 new offline tests passed** in 0.128 seconds. Generated PDFs cover
reused images on the same page and different pages, exact PNG hashes/dimensions,
stable occurrence IDs and preservation of PNG modification times on repeat,
new document versions, soft transparency masks, text/vector-only pages,
damaged PDFs, changed source hashes, missing metadata, invalid limits and
oversized images. A generated inline-image PDF confirms an explicit unsupported
result. Simulated image-write and manifest-write failures test preservation of
successful assets and successful retry. These fixtures are not production data.

The complete notebook then executed **11 code cells** from a fresh kernel,
starting with a real arXiv query and download. All **62 tests passed** in
0.896 seconds, including the actual PostgreSQL tests; no tests were skipped.
Saved code-cell execution counts are 1 through 11, with no error outputs.

| Real-source image check | Observed result |
|---|---|
| Document version | `2611.04405v1` |
| Physical PDF pages | 6 |
| Raster occurrences / distinct PNG assets | 1 / 1 |
| Physical source page | 3 |
| PDF image object reference (`xref`) | 323 |
| Soft transparency mask object reference | 351 |
| Stored PNG dimensions | 2031 × 865 pixels |
| Reduced notebook preview dimensions | 508 × 217 pixels |
| Failed / unsupported occurrences | 0 / 0 |
| Extraction status | `extracted` |
| Output PNG SHA-256 | `8dfefe9a865277b558960f4492e3be0b5aca33b58278dd953634441925224223` |

The attempt record is
`data/arxiv/2611.04405v1/images/487800d8c0194514a6bbb8ac223c5bad.json`;
the PNG is in its `assets/` subfolder. These generated files remain excluded
from Git. A separate automated check reopened the PNG, verified its dimensions
and hash, matched its object reference and bounding box to the source page,
verified the extracted source-image byte hash, and confirmed a saved PNG
preview in the notebook. It did not assess scientific content or full-page
visual equivalence.

The source PDF in this run still had SHA-256
`932ec59803c4c1495ebfbb7dd3b58e17afdbbc24e459e5a48f35f48a54bd3075`.
The SQL import correctly returned `unchanged` for both calls, because this
version was already stored during the preceding step. Later retrieval and image
processing records remain separate from that first SQL download snapshot.

Status semantics: `no_images` means no raster occurrences were reported, not
that no scientific figures exist. Any failed or unsupported occurrence produces
`partial`, even if none succeeded; a complete source/PDF failure produces a
failed attempt and raises an error. Failure to write the attempt record itself
raises a storage error. Successful assets are preserved. The CLI returns a
nonzero status for partial/failed runs.

Limits: PNG derivatives apply soft masks and convert non-gray/non-RGB color
spaces to RGB. They do not reproduce clipping, page transformations, overlays,
captions or composite figures. Vector drawings and inline/unresolved images
have no fallback rendering. The 25-million-pixel application limit does not
bound every internal parser allocation. A rerun can reuse PNG files but still
decodes images; there is no cross-run parser/embedding cache. Images are not
stored in SQL, interpreted, or embedded with CLIP yet. Scientific relevance
and figure completeness have not been evaluated.

References consulted:
[PyMuPDF image occurrence information](https://pymupdf.readthedocs.io/en/latest/page.html#Page.get_image_info),
[image extraction](https://pymupdf.readthedocs.io/en/latest/document.html#Document.extract_image),
and [image / transparency recipes](https://pymupdf.readthedocs.io/en/latest/recipes-images.html).

## Embedded-image commit verification

Before publishing the embedded-image step on November 4, 2026, its notebook
was executed separately without full-page rendering code or tests. All eleven
code cells completed and all 62 tests passed in 0.900 seconds without skips.
The real arXiv query returned `2611.04405v1`; one raster image on physical
page 3 was extracted successfully. The saved notebook in that commit contains
this separate verification run. Full-page rendering is a subsequent change.
