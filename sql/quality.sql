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

-- Expected: no duplicate semantic context IDs; the primary key enforces this.
SELECT context_id, count(*) AS row_count
FROM page_contexts
GROUP BY context_id
HAVING count(*) > 1;

-- Failed processing snapshots remain visible even if a later snapshot succeeds.
SELECT arxiv_id, version, page_number, context_id, text_status, image_status, status,
       page_record ->> 'text_error' AS text_error,
       page_record ->> 'image_error' AS image_error
FROM page_contexts
WHERE text_status = 'failed' OR image_status = 'failed' OR status = 'failed';

-- Multiple variants are intentional, not duplicate imports: inspect their configuration/status.
SELECT arxiv_id, version, page_number, page_id, count(*) AS context_variants
FROM page_contexts
GROUP BY arxiv_id, version, page_number, page_id
HAVING count(*) > 1;

-- Expected: no orphaned or hash-mismatched page contexts; the foreign key enforces this.
SELECT p.context_id
FROM page_contexts AS p
LEFT JOIN document_versions AS d
  ON d.arxiv_id = p.arxiv_id AND d.version = p.version AND d.sha256 = p.source_pdf_sha256
WHERE d.arxiv_id IS NULL;
