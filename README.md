# Search4Cure.AI

Search arXiv for one publication, download its exact PDF version, extract text
by physical PDF page, and create traceable text chunks.

Metadata search and download use the Python standard library. Text extraction
uses PyMuPDF. The current environment is tested with Python 3.13.

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
connection checks, document metadata import, a pandas table, and the tests.
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
