# Search4Cure: arXiv search

Search arXiv for one publication and print its metadata as JSON.
This step retrieves metadata only; it does not download PDFs.

Requires Python 3. No additional packages or credentials are needed.

PDF download, storage, vector search, answer generation, and AWS integration
are not implemented in this version yet.

## Run

From the repository root:

```bash
python3 -m aws_version.arxiv_search --query 'all:diabetes'
```

On macOS, if Python reports `CERTIFICATE_VERIFY_FAILED` and
`/etc/ssl/cert.pem` exists, prefix the command with
`SSL_CERT_FILE=/etc/ssl/cert.pem` to use the system CA bundle.

Run one search at a time and wait at least 3.1 seconds between runs,
following the [arXiv API rules](https://info.arxiv.org/help/api/tou.html).

## Test

```bash
python3 -m unittest discover -s tests -p test_arxiv_search.py
```

Tests use simulated API responses and run without network access.
See [validation results](../docs/VALIDATION.md) for the recorded checks.
