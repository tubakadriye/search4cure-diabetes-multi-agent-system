-- One row per explicitly identified version. Submission times are not availability times.
CREATE TABLE IF NOT EXISTS document_versions (
    arxiv_id TEXT NOT NULL CHECK (btrim(arxiv_id) <> ''),
    version INTEGER NOT NULL CHECK (version > 0),
    title TEXT NOT NULL CHECK (btrim(title) <> ''),
    source_url TEXT NOT NULL CHECK (btrim(source_url) <> ''),
    pdf_url TEXT NOT NULL CHECK (btrim(pdf_url) <> ''),
    first_submitted_at TIMESTAMPTZ,
    version_submitted_at TIMESTAMPTZ,
    metadata_retrieved_at TIMESTAMPTZ NOT NULL,
    pdf_retrieved_at TIMESTAMPTZ NOT NULL,
    sha256 TEXT NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    size_bytes BIGINT NOT NULL CHECK (size_bytes > 0),
    local_path TEXT NOT NULL CHECK (btrim(local_path) <> ''),
    metadata_path TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status = 'downloaded'),
    download_record JSONB NOT NULL,
    stored_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (arxiv_id, version)
);
