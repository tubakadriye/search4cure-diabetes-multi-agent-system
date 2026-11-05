# Validation

Scope: arXiv metadata retrieval, versioned PDF download, page-level text
extraction, traceable text chunks, embedded raster-image extraction, full-page
rendering, local PostgreSQL setup and storage of document versions, page contexts,
text chunks and their links, plus the executable walkthrough notebook.

Results are recorded by development stage. Earlier counts and limitations apply
to that stage; later sections document subsequent changes and checks. For the
current data model and diagrams, see the [developer guide](DEVELOPER_GUIDE.md#storage-model).

## Contents

- [Offline tests](#offline-tests)
- [Live source check](#live-source-check)
- [PDF download tests](#pdf-download-tests)
- [Live PDF download](#live-pdf-download)
- [Runtime progress output](#runtime-progress-output)
- [Project-root entry points](#project-root-entry-points)
- [Text extraction and notebook](#text-extraction-and-notebook)
- [Page-local text chunks](#page-local-text-chunks)
- [Local PostgreSQL setup and notebook connection checks](#local-postgresql-setup-and-notebook-connection-checks)
- [Store one document version in PostgreSQL](#store-one-document-version-in-postgresql)
- [SQL-only commit verification](#sql-only-commit-verification)
- [Referenced raster-image extraction](#referenced-raster-image-extraction)
- [Embedded-image commit verification](#embedded-image-commit-verification)
- [Full-page images linked to page text and chunks](#full-page-images-linked-to-page-text-and-chunks)
- [Independent text and page-image error handling](#independent-text-and-page-image-error-handling)
- [Page-context storage in PostgreSQL](#page-context-storage-in-postgresql)
- [Text-chunk storage and page-context links](#text-chunk-storage-and-page-context-links)

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

## Full-page images linked to page text and chunks

Validated on November 4, 2026 with Python 3.13.12 and PyMuPDF 1.28.2.
`render_pages.py` supplements embedded-image extraction without changing it.
No new dependency, model request, or AWS resource was needed.

The initial new test run had 16 passing tests and one error in a test-only
Pixmap crop-constructor call. Replacing that invalid test call with direct
pixel checks in the known caption region resolved it. The rendering
implementation did not require a change for that error.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u -m unittest discover -s tests -p test_render_pages.py -v
.venv/bin/python -u run_notebook.py
.venv/bin/python -m render_pages --help
```

All **17 new tests passed** in 0.471 seconds. Generated PDFs test complete-page
records, exact page-text/chunk links, PNG hashes, vector drawings, caption
pixels, rotation, CropBox dimensions, blank documents, new document versions,
DPI variants, stable page IDs and unchanged PNG modification times on repeat.
They also reject mismatched PDF/extraction snapshots and incorrect chunk
page/text/ID mappings. Corrupt PDFs, changed source hashes, pixel limits,
page-image writes, page-record writes, attempt-record writes and subsequent
retry are covered. The CLI test verifies nonzero exit status for failed pages.
All artificial documents and failures are explicitly synthetic.

The notebook was executed from a fresh kernel through **12 code cells**,
beginning with a real arXiv query and download. The existing single-image
extraction still succeeded. The new full-page cell rendered all physical pages
and displayed a page table, associated text and a reduced preview of page 3.
All **79 tests passed** in 1.482 seconds without skips, including the actual
PostgreSQL integration tests. Saved execution counts are 1 through 12 and
there are no error outputs.

| Real-source page check | Observed result |
|---|---|
| Document version | `2611.04405v1` |
| Rendered / physical pages | 6 / 6 |
| Failed pages | 0 |
| PNG dimensions, all six pages | 1191 × 1684 pixels |
| Rendering | 144 DPI, RGB, no alpha, annotations included |
| Extracted page-text characters | 37,614 |
| Linked text chunks | 50 |
| Chunks per physical page | 8, 10, 6, 10, 8, 8 |
| Combined full-page PNG size | 2,640,064 bytes |
| Reduced notebook page-3 preview | 596 × 842 pixels |

The source PDF SHA-256 remains
`932ec59803c4c1495ebfbb7dd3b58e17afdbbc24e459e5a48f35f48a54bd3075`.
The full-page attempt record is
`data/arxiv/2611.04405v1/pages/d8ba7a9be7df42129ecd920f2e775613.json`.
The page-3 PNG SHA-256 is
`d39cec3d70cdb43fb2f61b92cece1b70dfb64bfb325211fddf4025fb26bac02e`.
Generated files remain excluded from Git.

A separate automated check verified all six saved page records, unique logical
page IDs, PNG hashes and dimensions, exact text equality with each source
extraction page, and all 50 chunk IDs with no missing or duplicate assignments.
It also verified the extraction/chunk snapshot hashes and a saved PNG preview
in the notebook. These checks do not establish scientific correctness or
complete visual legibility.

Each page JSON stores document/version/source metadata, PDF hash, physical page
number, logical page ID, rendering-configuration ID, PNG path/hash/dimensions,
the page text and its hash, text-parser configuration, chunk IDs and exact
source-snapshot references, processing status/errors and processing time.
Unknown source metadata is retained as unknown. Rendering follows the complete
visible PDF CropBox and PDF rotation; hidden content outside that box is not
included. A logical page ID is independent of DPI; a rendering ID also includes
the rendering configuration.

Without a supplied extraction, page text is extracted during rendering, and
even an entirely blank document can produce valid page records. Missing chunk
input is explicitly `not_provided`, rather than a claim that no chunks exist.
If supplied, chunks must match the exact extraction snapshot and the associated
page-text slices. A failed page gets a failed record while other pages continue.
If a page record itself cannot be written, the document-level attempt records
that failure when storage permits. A failed document-level write raises an
error and leaves successfully stored page records/assets available for retry.

Limits: page records are local JSON, not SQL rows. A repeat preserves valid
PNG assets but still renders the pages and creates new attempt records; this
is not a cross-run rendering cache. The application checks pixel dimensions
before allocating its output pixmap, but does not bound every internal parser
allocation. Finite DPI can lose small details; resolution and model input limits
must be evaluated before vision-model use. OCR, page/image retrieval and
image-plus-text answer generation are not implemented. No source-support or
retrieval-quality improvement is claimed.

The [official OpenAI file-input documentation](https://developers.openai.com/api/docs/guides/file-inputs)
describes combining PDF text and page images for vision-capable models. This
informs the page-context design, without adding an OpenAI dependency. The
planned AWS answer step must load selected pages' actual image bytes and
associated text, pass them to a compatible vision model, and cite the
exact version/page. Filenames alone are not image inputs; that request path
and its citation/evidence checks remain to be implemented and tested.
Rendering behavior was checked against the
[PyMuPDF Page.get_pixmap documentation](https://pymupdf.readthedocs.io/en/latest/page.html#Page.get_pixmap).

## Independent text and page-image error handling

Validated on November 4, 2026. An actual exception from `page.get_text()` in
the no-extraction-record path previously skipped the image-rendering call for
that page because both operations shared one exception handler. Empty text
already worked and remains a valid empty result.

Three new regression tests were first run against that implementation. They
produced one assertion failure and two missing-field errors: the direct
regression observed only two saved page images instead of three after a
simulated first-page text exception; separate modality fields/counters did not
yet exist.

Text extraction and image rendering now have independent exception handlers.
Text failure leaves `text` and `text_sha256` null, records `text_error`, and
does not prevent saving the PNG. `image_status` / `image_error` report the image
operation independently. Both errors survive when both operations fail.
An attempt with saved images and a text failure is `partial`, so the CLI
continues to return a nonzero status for incomplete processing.

Counter semantics: `rendered_count` counts successfully rendered/stored page
records, including those with failed text. `failed_count` counts page records
whose rendering or record storage failed. `text_failed_count` and
`image_failed_count` separately count modality failures. Therefore an attempt
can be partial with zero failed image/page records but a nonzero text-failure
count. A record-storage failure does not erase the independent modality fields.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u -m unittest discover -s tests -p test_render_pages.py -v
.venv/bin/python -u run_notebook.py
```

After the correction, all **20 page-rendering tests passed** in 0.620 seconds.
The three added tests verify a text error on one page with continued processing
and successful retry, text errors on every page with all images preserved, and
simultaneous text/image errors without losing either error message. Saved PNG
hashes, persisted error fields and preservation of the PNG modification time
on retry are checked. Existing blank-page tests also still pass.

The full notebook completed **13 code cells**, with all **82 tests passing**
in 1.361 seconds without skips. Its new offline cell generates a temporary,
explicitly synthetic PDF and simulates a text exception. The saved output shows
`text: null`, `image_status: rendered`, an existing PNG and a partial attempt.
The unpatched retry recovers text without rewriting the PNG. This demonstration
is not an observed source failure and is not imported into SQL.

The real `2611.04405v1` download still produced six rendered page records,
50 linked chunks, zero text failures and zero image failures. Its latest page
attempt is `data/arxiv/2611.04405v1/pages/14d873fbb3964d1a97b862b3bcf9838a.json`.
No vision-model behavior or scientific correctness was evaluated in this fix.

## Page-context storage in PostgreSQL

Validated across November 4–5, 2026, with the existing Python,
PyMuPDF, Psycopg and PostgreSQL dependencies. No new package or cloud resource
was required. `initialize_schema()` now applies `003_page_contexts.sql` after
the document table. Existing document data is retained.

`page_contexts` stores page text, PNG references/hashes, chunk IDs, processing
states, source snapshots and configurations. A composite foreign key references
the exact `(arxiv_id, version, sha256)` in `document_versions`; its supporting
unique index is added without changing existing document rows. PNG bytes stay
on disk. Each page has its own SQL transaction, so a rejected page does not
roll back other successful pages.

The first implementation passed seven offline tests and nine PostgreSQL tests.
Its full notebook run passed 98 tests in 2.207 seconds and imported six real
contexts. A final provenance review identified a missing check: a valid-looking
chunk ID from the wrong page could be accepted. A new isolated regression test
reproduced this by storing two rows instead of rejecting the mismatched page.
The importer now reopens and checks the extraction/chunk snapshots, their hashes,
exact per-page chunk assignments, page text and parser configuration. A changed
extraction snapshot is rejected before SQL access.

After that correction, all **18 page-context tests passed** in 0.816 seconds:
eight offline tests and ten actual PostgreSQL tests. They cover stable semantic
IDs despite changed attempt paths/times, incorrect page IDs, text/PNG corruption,
dimension checks, duplicate or incorrectly assigned chunk IDs, incomplete page
inventories, changed extraction snapshots, missing parent documents, repeat
imports, new versions, different rendering configurations, preserved failed-text
snapshots and later recovery, per-page SQL failure and retry, and native SQL
foreign-key/nullability constraints. Generated PDFs and deliberately simulated
failures remain explicitly synthetic; SQL test schemas use forced rollback.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u -m unittest discover -s tests -p test_store_pages.py -v
S4C_TEST_POSTGRES=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u -m unittest discover -s tests -p test_store_pages.py -v
.venv/bin/python -u run_notebook.py
.venv/bin/python -m store_pages --help
```

The final complete notebook run passed **all 100 tests** in 2.007 seconds,
without skips, and executed **14 code cells** without error outputs. It began
with a new real arXiv query and download of `2611.04405v1`, extracted six pages
and 50 chunks, preserved the embedded-image and full-page examples, and read
the stored page contexts back from SQL. All six page texts, PNG hashes and
chunk-ID lists were compared with the newly generated source records.

| Import observation | Initial run | Final verification run |
|---|---:|---:|
| Context rows before import | 0 | 6 |
| Newly inserted contexts | 6 | 0 |
| Unchanged contexts on first call | 0 | 6 |
| Newly inserted contexts on immediate repeat | 0 | 0 |
| Unchanged contexts on immediate repeat | 6 | 6 |
| Context rows after repeat | 6 | 6 |
| First call duration | 0.144062 s | 0.120724 s |
| Repeat call duration | 0.128060 s | 0.113442 s |
| Import failures | 0 | 0 |

Durations use `time.perf_counter()` around `store_pages()`, including source
verification, PNG checking/decoding, connection setup and SQL. They are single
local observations, not evidence of a speed improvement. The confirmed behavior
is duplicate prevention, including a completely new source-processing run with
different retrieval times and JSON filenames. No embedding reuse or model-call
reduction is measured here.

The initial executed notebook is preserved locally at
`data/validation/page-storage-initial.ipynb` (excluded from Git). Its page
manifest is `data/arxiv/2611.04405v1/pages/b4882112d88447dfa5af1db1c6a4b655.json`.
The final manifest is
`data/arxiv/2611.04405v1/pages/fbb88d12a64c452a8a29950060f3d4f4.json`.
Both reference PDF SHA-256
`932ec59803c4c1495ebfbb7dd3b58e17afdbbc24e459e5a48f35f48a54bd3075`.

The final read-only checks found six SQL contexts with 37,614 text characters
and 50 linked chunk IDs. There were zero duplicate context IDs, failed processing
snapshots, multiple variants, or orphaned/hash-mismatched contexts in this real
sample. All six referenced PNGs matched their SQL fingerprints. No synthetic
parent documents or test schemas remained. The synthetic tests separately
exercise failures and multiple variants; the real sample does not establish
how frequently these conditions occur.

Context IDs include meaningful content, configuration, chunk associations and
statuses/errors; attempt paths and timestamps are excluded. Equivalent imports
keep the first stored snapshot. Recovery or a changed configuration produces a
new immutable context, preserving earlier failures. `read_page_contexts()`
returns all variants and does not select evidence or rank relevance. A successful
SQL import means the snapshot was stored, not that its processing succeeded.

Limits at this stage (chunk storage is added below): standalone chunk text/vectors have no SQL table yet. Page records retain
chunk IDs and configuration, but embeddings, retrieval, historical availability
filtering and image-plus-text model requests are unimplemented. Import-validation
failures are reported in logs/results and retained source JSON, not a persistent
SQL import-run audit table. The source snapshot must remain available to repeat
validation; immutable rows retain their first asset paths. Asset existence is
checked during import and this notebook example, not continuously monitored.
Schema creation is not a general migration/repair mechanism. Backup/restore and
AWS deployment have not been tested.

Implementation references:
[PostgreSQL constraints and foreign keys](https://www.postgresql.org/docs/16/ddl-constraints.html)
and [Psycopg JSON adaptation](https://www.psycopg.org/psycopg3/docs/basic/adapt.html#json-adaptation).

## Text-chunk storage and page-context links

Validated on November 5, 2026, using the existing local dependencies and database.
`initialize_schema()` now also applies `004_text_chunks.sql`. It adds
`text_chunks` and `page_context_chunks` without replacing document/page records.
Each chunk and all its matching context links share one transaction. The import
retains first records and supports retry without duplicate chunks or links.

The targeted offline run passed **eight tests** in 0.123 seconds; ten PostgreSQL
tests were explicitly skipped in that run. With the database enabled, **all 18
targeted tests passed** in 1.563 seconds. Tests use generated PDFs, explicitly
synthetic metadata and isolated PostgreSQL schemas with forced rollback.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u -m unittest discover -s tests -p test_store_chunks.py -v
S4C_TEST_POSTGRES=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u -m unittest discover -s tests -p test_store_chunks.py -v
.venv/bin/python -u run_notebook.py
```

The new tests cover exact Unicode text slices; changed PDF/extraction snapshots;
wrong text, IDs, window positions, sequence, word counts and unsupported policies;
duplicate IDs; repeated import; new attempt paths; different rendering resolution;
separate chunking policies and document versions; absent/mismatched page contexts;
failed-image contexts with valid text; native cross-page foreign-key rejection;
and a link failure that rolls back its chunk and all preceding links for that
chunk, followed by a successful retry. Other chunks remain stored.

The complete notebook run passed **all 118 tests in 3.323 seconds**, without skips,
and executed **15 code cells** without error outputs. A new real arXiv request
returned `2611.06569v1`, "One Scale, Four Texts: Gaps, Overlaps and Silent Repairs
in Published Grading Schemes". The versioned PDF contained 18 physical pages;
extraction produced 74,456 characters and 140 chunks with 100-word windows and
20-word overlap. All 18 full-page PNGs rendered at 144 DPI. Embedded extraction
reported `no_images` with zero unsupported or failed occurrences; that does not
establish the absence of vector figures or tables.

| Observation for this document version | First chunk import | Immediate repeat |
|---|---:|---:|
| Newly inserted chunks | 140 | 0 |
| Unchanged chunks | 0 | 140 |
| Newly inserted context links | 140 | 0 |
| Chunk rows after import | 140 | 140 |
| Context links after import | 140 | 140 |
| Import failures | 0 | 0 |
| Duration | 0.409506 s | 0.436360 s |

Before the first call, this version had zero chunk rows and links. Durations use
`time.perf_counter()` around the importer, including source/PDF verification,
connection setup and SQL. These are single local observations. The repeat was
slightly slower; the confirmed improvement is duplicate prevention, not speed.
No embeddings were computed and no generative model was invoked. Embedding reuse
and retrieval/answer quality have not been evaluated.

The notebook compared all 140 SQL chunk texts with the current extraction's exact
page slices and followed an example chunk back to the same document/version,
physical page and verified PNG fingerprint. Source records from this run:

- Download: `data/arxiv/2611.06569v1/d4d078a3eb424843ab6185a8fb0b0981.json`.
- Extraction: `data/arxiv/2611.06569v1/extracted/7cd863584d65472ea2bf143c9eb72536.json`.
- Chunks: `data/arxiv/2611.06569v1/chunks/97dfb8bb3eea46f795ffeec0afbfb59f.json`.
- Pages: `data/arxiv/2611.06569v1/pages/53727ad2f4334cdca555f270a00b1f8c.json`.
- PDF SHA-256: `2f790b9f3d12e90e39791bd304ee122de1dcd24ff12fdb992bd6a9759149a698`.

All twelve read-only statements in `sql/quality.sql` were then executed. Duplicate,
orphan, processing-failure and inconsistent linked-text/configuration checks
returned zero rows. The application database contained two real document
versions, 24 contexts, 140 chunks and 140 links. The earlier six-page document
remained intact; this step did not automatically import its old chunk JSON.
All 24 stored PNG fingerprints matched the actual files, and all 140 SQL texts
matched the original chunk JSON. No synthetic documents or test schemas remained.
The first ad-hoc quality-check invocation split a semicolon inside a SQL comment
and failed with `EMPTY_QUERY`; removing comment lines from that temporary runner
allowed the checks to complete. It made no database changes.

Current limits: validation supports the implemented word-window policy and needs
the original PDF/extraction files. It establishes consistency, not scientific
correctness or relevance. Reads expose all configurations without selecting a
policy or ranking evidence. Image-failed contexts can carry valid text; vision
use must check image readiness and load actual bytes. Embeddings, search,
historical-availability filtering, model requests, persistent import-run audit,
backup/restore verification and AWS deployment remain unimplemented or untested.
