-- The composite keys enforce that linked contexts and chunks belong to the same page.
CREATE UNIQUE INDEX IF NOT EXISTS page_contexts_identity_page
    ON page_contexts (context_id, page_id);

CREATE TABLE IF NOT EXISTS text_chunks (
    chunk_id TEXT PRIMARY KEY CHECK (chunk_id ~ '^[0-9a-f]{64}$'),
    page_id TEXT NOT NULL CHECK (page_id ~ '^[0-9a-f]{64}$'),
    arxiv_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    source_pdf_sha256 TEXT NOT NULL,
    page_number INTEGER NOT NULL CHECK (page_number > 0),
    chunk_number INTEGER NOT NULL CHECK (chunk_number > 0),
    start_char INTEGER NOT NULL CHECK (start_char >= 0),
    end_char INTEGER NOT NULL CHECK (end_char > start_char),
    chunk_text TEXT NOT NULL CHECK (char_length(chunk_text) > 0),
    word_count INTEGER NOT NULL CHECK (word_count > 0),
    parser_config JSONB NOT NULL CHECK (jsonb_typeof(parser_config) = 'object'),
    chunking_config JSONB NOT NULL CHECK (jsonb_typeof(chunking_config) = 'object'),
    source_chunks_path TEXT NOT NULL,
    source_extraction_sha256 TEXT NOT NULL CHECK (source_extraction_sha256 ~ '^[0-9a-f]{64}$'),
    processed_at TIMESTAMPTZ NOT NULL,
    stored_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (arxiv_id, version, source_pdf_sha256)
        REFERENCES document_versions (arxiv_id, version, sha256),
    UNIQUE (chunk_id, page_id),
    CHECK (char_length(chunk_text) = end_char - start_char)
);

CREATE INDEX IF NOT EXISTS text_chunks_document_page
    ON text_chunks (arxiv_id, version, page_number);

CREATE TABLE IF NOT EXISTS page_context_chunks (
    context_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    page_id TEXT NOT NULL,
    stored_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (context_id, chunk_id),
    FOREIGN KEY (context_id, page_id) REFERENCES page_contexts (context_id, page_id),
    FOREIGN KEY (chunk_id, page_id) REFERENCES text_chunks (chunk_id, page_id)
);

CREATE INDEX IF NOT EXISTS page_context_chunks_chunk
    ON page_context_chunks (chunk_id);
