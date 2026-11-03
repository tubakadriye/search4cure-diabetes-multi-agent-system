# Validation

Scope: the metadata-only implementation in `aws_version/arxiv_search.py`.

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

This confirms metadata retrieval for one publication. It does not establish
search relevance or validate PDF processing, storage, generated answers, or AWS
integration. Those features are not implemented in this version.
