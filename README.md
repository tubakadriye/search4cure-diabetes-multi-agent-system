# Search4Cure.AI

Search arXiv for one publication, download its exact PDF version, extract text
by physical PDF page, create traceable text chunks, and extract referenced
raster images with page provenance.

Complete page PNGs preserve diagrams, captions and surrounding layout together.
Each physical page has a local JSON record linking its image, extracted text,
text-chunk IDs and exact document version.

Metadata search and download use the Python standard library. Text and image
extraction use PyMuPDF. The current environment is tested with Python 3.13.

Local PostgreSQL stores metadata for downloaded document versions. PDFs and
text/chunk results remain local files. The notebook displays SQL results as
a pandas DataFrame, a table in Python memory.

OCR, embeddings, SQL storage of text chunks, vector search, answer generation, and AWS integration
are not implemented yet.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-notebook.txt
.venv/bin/python -m ipykernel install --prefix .venv --name search4cure --display-name "Search4Cure (.venv)"
```

For command-line use without the notebook, install `requirements.txt` instead.

## Local PostgreSQL

Start Docker Desktop. The notebook includes executable cells for these steps,
or run them from the project root:

```bash
.venv/bin/python -m database init-password
docker compose up -d --wait --wait-timeout 60 postgres
.venv/bin/python -m database check
.venv/bin/python -m database init-schema
```

The first command creates `data/postgres/password.txt`, a private password
file excluded from Git. It preserves an existing file and never prints its
contents. No `.env` file is required for this setup; an existing root `.env`
is not used by the Python connection code. Do not replace the password file
while keeping an initialized database volume: creating a new file does not
change the database's existing password.

PostgreSQL is accessible only at `127.0.0.1:55432`. To change the port, set
`POSTGRES_PORT` for both Docker Compose and Python. Compose may also read that
variable from a root `.env`. The image includes pgvector, an extension for
storing vectors. `init-schema` explicitly creates the document table, including
on an existing volume; repeat execution preserves its rows. It does not migrate
an existing table with a different definition.
This local development setup uses a bootstrap database administrator account.

Stop PostgreSQL with `docker compose stop postgres`. Its named volume retains
the database; the password file must also be kept. Backup and restore have not
been tested.

## Notebook

Open [walkthrough.ipynb](notebooks/walkthrough.ipynb), select **Search4Cure (.venv)**,
and run the cells in order. It shows live outputs for metadata, download,
page-level extraction, text chunks, a page preview, PostgreSQL setup and
connection checks, document metadata import, pandas tables, image extraction
with a reduced preview, complete-page rendering with associated text, and the tests.
Docker must be running for the database cells.

You can also open it in JupyterLab:

```bash
.venv/bin/jupyter lab notebooks/walkthrough.ipynb
```

To execute all cells with live terminal output and save their results:

```bash
.venv/bin/python -u run_notebook.py
```

Each run starts with a real arXiv query. Saved notebook outputs are snapshots
of that run. Subsequent development steps extend this same notebook.

## Run

From the repository root:

```bash
.venv/bin/python -m arxiv_search --query 'all:diabetes'
```

To download the publication returned by a new query:

```bash
.venv/bin/python -u -m download_pdf --query 'all:diabetes' --output-dir data/arxiv
```

Progress messages appear on stderr during the run; the final JSON is written
to stdout. `-u` enables unbuffered Python output.

Each version has its own folder containing a PDF named by its content hash and
a JSON record for each successful retrieval. `data/` is excluded from Git.
Downloads require a known version, use 64 KiB chunks, and are limited to 100 MiB.
A PDF header check does not prove that a parser can read the document.

To extract text, use the `metadata_path` returned by the download command:

```bash
.venv/bin/python -u -m extract_text --metadata path/to/download-record.json
```

Extraction verifies the PDF hash and records its parser configuration, physical
page numbers, empty pages, and processing status. Results are stored beside
the PDF in an `extracted/` folder. No OCR or table reconstruction is performed.

To split an extraction into page-local text chunks:

```bash
.venv/bin/python -u -m chunk_text --extraction path/to/extraction-record.json --max-words 100 --overlap-words 20
```

Chunks retain the document version, PDF hash, physical page, and exact text
offsets. Results are stored beside the PDF in a `chunks/` folder. IDs remain
stable for equivalent content and configuration. Word limits do not enforce
an embedding model's token limit; that check belongs to the embedding step.

On macOS, if Python reports `CERTIFICATE_VERIFY_FAILED` and
`/etc/ssl/cert.pem` exists, prefix the command with
`SSL_CERT_FILE=/etc/ssl/cert.pem` to use the system CA bundle.

Run one search at a time and wait at least 3.1 seconds between runs,
following the [arXiv API rules](https://info.arxiv.org/help/api/tou.html).

## Extract raster images

Use the successful download JSON:

```bash
.venv/bin/python -u -m extract_images --metadata path/to/download-record.json
```

The extractor verifies the PDF hash and stores PNG derivatives in `images/assets/`
beside the PDF. A JSON attempt record links each occurrence to document version,
PDF hash, physical page, image object reference and page coordinates. Reused
images share a PNG file; valid existing files are not rewritten. Soft transparency
masks are applied, and non-gray/non-RGB color spaces are converted to RGB.

No raster occurrences produces `no_images`, not a failed document. Unsupported
or failed occurrences produce `partial`; successful assets are retained for
retry. The CLI exits with code 1 for a partial or failed extraction. A failed
manifest write raises an error; existing assets remain available for retry.

The default per-image limit is 25 million pixels (`--max-pixels` overrides it).
This bounds application decoding checks, not all internal parser memory use.
The pipeline processes one source PDF at a time and caches only extracted-asset
metadata across repeated object references.

Raster occurrences are not necessarily complete scientific figures: text,
vector drawings, captions, clipping and page transformations are not reconstructed.
Inline or unresolved images without an extractable object reference are recorded
as unsupported. This extractor does not fall back to page rendering; complete
pages are rendered separately below. Images are not yet
stored in SQL, embedded with CLIP, or interpreted by a model.

## Preserve complete page context

```bash
.venv/bin/python -u -m render_pages --metadata path/to/download-record.json --extraction path/to/extraction-record.json --chunks path/to/chunk-record.json
```

This saves one JSON page record per physical page in `pages/records/`, PNGs in
`pages/assets/`, and a document-level attempt JSON in `pages/`. Records link the
exact page text and chunk IDs after checking document/version/hash consistency
and chunk text offsets. Empty pages are retained. Failed pages remain recorded
without discarding successful pages.

The default is 144 DPI (`--dpi`), RGB, with a 25-million-pixel per-page limit
(`--max-pixels`). Rendering covers the complete visible PDF CropBox, honors
rotation, and includes annotations. Hidden content outside that box is excluded.
Without `--extraction`, text is extracted in the rendering pass; `--chunks`
requires the corresponding extraction record. OCR is not performed.

Text extraction and image rendering handle errors independently. A text error
leaves `text` as `null` and records `text_error`; the PNG can still have
`image_status: rendered`. The attempt is then `partial` (CLI exit code 1).
Empty text is a valid empty result, distinct from an extraction error.

Page records remain local JSON. Search and answer generation are later steps:
retrieve relevant pages, load their actual PNG bytes and associated text into
a vision-capable LLM model, then cite document version and physical page.
No model request is made yet; retrieval quality, image resolution and source
support still need evaluation.

## Store document metadata

After database setup, import the download JSON from the project root:

```bash
.venv/bin/python -u -m store_document --metadata path/to/download-record.json
```

The importer verifies required fields, timezone-aware retrieval times, PDF size
and SHA-256 before writing. A primary key on `(arxiv_id, version)` prevents
duplicates. The same version and hash return `unchanged`; changed bytes for an
existing version raise a conflict without replacing its row. SQL parameters
are bound separately from SQL commands. Writes use a transaction so failed
inserts can be retried.

The table retains the first imported download record for each version. Later
retrievals remain in local JSON files; they do not update this first snapshot.
Unknown source submission times remain `NULL`. Submission times, source
retrieval times and the database storage time have different meanings; none
establish the exact public availability of a version.

The `downloaded` status describes the download, not successful extraction.
Parser/chunk configuration remains in the corresponding JSON results.
Import-run tracking, failed-document records and SQL storage of processing
results are later steps. See [quality queries](sql/quality.sql) for SQL checks.

## Test

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

By default, tests use simulated API responses and generated PDFs; the
tests requiring PostgreSQL are skipped. With local PostgreSQL running, include
actual connection, authentication and document-storage checks:

```bash
S4C_TEST_POSTGRES=1 .venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

The notebook runs this complete suite after starting PostgreSQL.
See [validation results](docs/VALIDATION.md) for the recorded checks.

## Development steps

Each step includes implementation, automated tests, executable notebook examples, and validation notes before the next step starts.
