-- Expected: no rows; the primary key additionally prevents these duplicates.
SELECT arxiv_id, version, count(*) AS row_count
FROM document_versions
GROUP BY arxiv_id, version
HAVING count(*) > 1;

-- Unknown source submission times remain NULL; report them without inventing dates.
SELECT arxiv_id, version, first_submitted_at, version_submitted_at
FROM document_versions
WHERE first_submitted_at IS NULL OR version_submitted_at IS NULL;

-- Current inventory. This stage records successful downloads, not parser status.
SELECT arxiv_id, version, title, status, pdf_retrieved_at, stored_at
FROM document_versions
ORDER BY arxiv_id, version;
