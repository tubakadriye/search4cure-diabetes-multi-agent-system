# Search4Cure: arXiv import

Search arXiv for one publication, inspect its metadata, and optionally download
its exact PDF version with a SHA-256 content hash and a JSON source record.

Requires Python 3. No additional packages or credentials are needed.

PDF text extraction, database storage, vector search, answer generation, and
AWS integration are not implemented in this version yet.

## Run

From the repository root:

```bash
python3 -m aws_version.arxiv_search --query 'all:diabetes'
```

To download the publication returned by a new query:

```bash
python3 -u -m aws_version.download_pdf --query 'all:diabetes' --output-dir data/arxiv
```

Progress messages appear on stderr during the run; the final JSON is written
to stdout. `-u` enables unbuffered Python output.

Each version has its own folder containing a PDF named by its content hash and
a JSON record for each successful retrieval. `data/` is excluded from Git.
Downloads require a known version, use 64 KiB chunks, and are limited to 100 MiB.
A PDF header check does not prove that a parser can read the document.

On macOS, if Python reports `CERTIFICATE_VERIFY_FAILED` and
`/etc/ssl/cert.pem` exists, prefix the command with
`SSL_CERT_FILE=/etc/ssl/cert.pem` to use the system CA bundle.

Run one search at a time and wait at least 3.1 seconds between runs,
following the [arXiv API rules](https://info.arxiv.org/help/api/tou.html).

## Test

```bash
python3 -m unittest discover -s tests -p test_arxiv_search.py
python3 -m unittest discover -s tests -p test_download_pdf.py
```

Tests use simulated API responses and run without network access.
See [validation results](../docs/VALIDATION.md) for the recorded checks.
