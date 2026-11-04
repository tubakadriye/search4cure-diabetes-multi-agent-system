# Search4Cure.AI

Search arXiv for one publication, download its exact PDF version, extract text
by physical PDF page, and create traceable text chunks.

Metadata search and download use the Python standard library. Text extraction
uses PyMuPDF. The current environment is tested with Python 3.13.

OCR, embeddings, database storage, vector search, answer generation, and AWS integration
are not implemented yet.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-notebook.txt
.venv/bin/python -m ipykernel install --prefix .venv --name search4cure --display-name "Search4Cure (.venv)"
```

For command-line use without the notebook, install `requirements.txt` instead.

## Notebook

Open [walkthrough.ipynb](notebooks/walkthrough.ipynb), select **Search4Cure (.venv)**,
and run the cells in order. It shows live outputs for metadata, download,
page-level extraction, text chunks, a page preview, and the offline tests.

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

## Test

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

Tests use simulated API responses and run without network access.
See [validation results](docs/VALIDATION.md) for the recorded checks.

## Development steps

Each step includes implementation, automated tests, executable notebook examples, and validation notes before the next step starts.
