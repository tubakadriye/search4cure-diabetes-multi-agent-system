-- This key lets the foreign key enforce the exact parent PDF hash as well as its version.
CREATE UNIQUE INDEX IF NOT EXISTS document_versions_identity_hash
    ON document_versions (arxiv_id, version, sha256);

-- Immutable semantic snapshots: a recovered or differently configured context is a new row.
CREATE TABLE IF NOT EXISTS page_contexts (
    context_id TEXT PRIMARY KEY CHECK (context_id ~ '^[0-9a-f]{64}$'),
    page_id TEXT NOT NULL CHECK (page_id ~ '^[0-9a-f]{64}$'),
    render_id TEXT NOT NULL CHECK (render_id ~ '^[0-9a-f]{64}$'),
    arxiv_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    source_pdf_sha256 TEXT NOT NULL,
    page_number INTEGER NOT NULL CHECK (page_number > 0),
    page_text TEXT,
    text_sha256 TEXT,
    text_status TEXT NOT NULL CHECK (text_status IN ('extracted', 'empty', 'failed')),
    image_status TEXT NOT NULL CHECK (image_status IN ('rendered', 'failed')),
    status TEXT NOT NULL CHECK (status IN ('rendered', 'failed')),
    png_path TEXT,
    png_sha256 TEXT,
    chunk_ids TEXT[] NOT NULL,
    page_record JSONB NOT NULL,
    source_manifest_path TEXT NOT NULL,
    processed_at TIMESTAMPTZ NOT NULL,
    stored_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (arxiv_id, version, source_pdf_sha256)
        REFERENCES document_versions (arxiv_id, version, sha256),
    CHECK (
        (text_status = 'failed' AND page_text IS NULL AND text_sha256 IS NULL)
        OR (text_status IN ('extracted', 'empty') AND page_text IS NOT NULL
            AND text_sha256 IS NOT NULL AND text_sha256 ~ '^[0-9a-f]{64}$')
    ),
    CHECK (
        (image_status = 'failed' AND png_path IS NULL AND png_sha256 IS NULL)
        OR (image_status = 'rendered' AND png_path IS NOT NULL AND btrim(png_path) <> ''
            AND png_sha256 IS NOT NULL AND png_sha256 ~ '^[0-9a-f]{64}$')
    ),
    CHECK (status <> 'rendered' OR image_status = 'rendered')
);

CREATE INDEX IF NOT EXISTS page_contexts_document_page
    ON page_contexts (arxiv_id, version, page_number);
