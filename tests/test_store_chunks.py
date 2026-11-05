"""Synthetic source fixtures and rolled-back PostgreSQL chunk/link integration tests."""

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid

import psycopg
from psycopg import sql
import pymupdf

from chunk_text import chunk_text
from database import initialize_schema, open_connection
from extract_text import extract_text
from render_pages import render_pages
from store_chunks import read_chunk_contexts, read_chunks, store_chunks, validate_chunks
from store_document import store_document
from store_pages import store_pages


class ChunkStorageFixture(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.pdf = self.root / "SYNTHETIC.pdf"
        with pymupdf.open() as pdf:
            for number in (1, 2):
                page = pdf.new_page(width=300, height=200)
                page.insert_text((15, 30), f"SYNTHETIC page {number}: café alpha beta gamma delta epsilon zeta", fontsize=8)
            pdf.save(self.pdf)
        self.document = {"arxiv_id": "2401.12345", "version": 1, "title": "SYNTHETIC SQL chunk fixture",
                         "source_url": "https://arxiv.org/abs/2401.12345v1",
                         "first_submitted_at": None, "version_submitted_at": None}
        self.download = {
            "search": {"query": "SYNTHETIC TEST ONLY", "document": self.document,
                       "retrieved_at": "2026-11-04T10:00:00Z"},
            "status": "downloaded", "local_path": str(self.pdf),
            "sha256": hashlib.sha256(self.pdf.read_bytes()).hexdigest(),
            "size_bytes": self.pdf.stat().st_size, "pdf_url": "https://arxiv.org/pdf/2401.12345v1",
            "pdf_retrieved_at": "2026-11-04T10:01:00Z",
        }
        self.metadata = self.root / "SYNTHETIC-download.json"
        self.metadata.write_text(json.dumps(self.download))
        self.extraction = extract_text(self.metadata)
        self.chunks = chunk_text(self.extraction["metadata_path"], max_words=3, overlap_words=1)
        self.chunk_path = Path(self.chunks["metadata_path"])
        self.chunk_record = json.loads(self.chunk_path.read_text())
        self.rendered = render_pages(self.metadata, self.extraction["metadata_path"], self.chunk_path, dpi=72)

    def save_chunks(self, value):
        self.chunk_path.write_text(json.dumps(value))


class ChunkStorageUnitTests(ChunkStorageFixture):
    def test_valid_windows_keep_exact_unicode_slices_and_source_files(self):
        before = self.pdf.read_bytes(), self.chunk_path.read_bytes()
        chunks, source, links = validate_chunks(self.chunk_path)
        self.assertEqual(chunks["chunk_count"], len(self.chunks["chunks"]))
        self.assertEqual(links["extraction_sha256"], chunks["source_extraction_sha256"])
        self.assertTrue(any("café" in c["text"] for c in chunks["chunks"]))
        for chunk in chunks["chunks"]:
            page = source["pages"][chunk["page_number"] - 1]
            self.assertEqual(chunk["text"], page["text"][chunk["start_char"]:chunk["end_char"]])
        self.assertEqual(before, (self.pdf.read_bytes(), self.chunk_path.read_bytes()))

    @patch("store_chunks.open_connection")
    def test_changed_chunk_text_is_rejected_before_sql(self, connect):
        self.chunk_record["chunks"][0]["text"] = "SYNTHETIC wrong text"
        self.save_chunks(self.chunk_record)
        with self.assertRaises(ValueError):
            store_chunks(self.chunk_path)
        connect.assert_not_called()

    @patch("store_chunks.open_connection")
    def test_valid_looking_but_incorrect_chunk_id_is_rejected(self, connect):
        self.chunk_record["chunks"][0]["chunk_id"] = "0" * 64
        self.save_chunks(self.chunk_record)
        with self.assertRaisesRegex(ValueError, "IDs"):
            store_chunks(self.chunk_path)
        connect.assert_not_called()

    def test_wrong_sequence_word_count_or_unsupported_policy_is_rejected(self):
        for key, value in (("chunk_number", 99), ("word_count", 99)):
            record = copy.deepcopy(self.chunk_record)
            record["chunks"][0][key] = value
            self.save_chunks(record)
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_chunks(self.chunk_path)
        record = copy.deepcopy(self.chunk_record)
        record["chunking"]["method"] = "SYNTHETIC_UNSUPPORTED_POLICY"
        self.save_chunks(record)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            validate_chunks(self.chunk_path)

    @patch("store_chunks.open_connection")
    def test_changed_pdf_is_rejected_before_sql(self, connect):
        self.pdf.write_bytes(self.pdf.read_bytes() + b"SYNTHETIC change")
        with self.assertRaisesRegex(ValueError, "original PDF"):
            store_chunks(self.chunk_path)
        connect.assert_not_called()

    @patch("store_chunks.open_connection")
    def test_changed_extraction_snapshot_is_rejected_before_sql(self, connect):
        path = Path(self.extraction["metadata_path"])
        path.write_text(path.read_text() + "\n")
        with self.assertRaises(ValueError):
            store_chunks(self.chunk_path)
        connect.assert_not_called()

    def test_shifted_slice_that_matches_text_but_not_policy_is_rejected(self):
        chunk = self.chunk_record["chunks"][0]
        chunk["start_char"] += 1
        chunk["text"] = chunk["text"][1:]
        self.save_chunks(self.chunk_record)
        with self.assertRaisesRegex(ValueError, "word-window policy"):
            validate_chunks(self.chunk_path)

    def test_duplicate_ids_are_rejected(self):
        self.chunk_record["chunks"][1]["chunk_id"] = self.chunk_record["chunks"][0]["chunk_id"]
        self.save_chunks(self.chunk_record)
        with self.assertRaises(ValueError):
            validate_chunks(self.chunk_path)


@unittest.skipUnless(os.environ.get("S4C_TEST_POSTGRES") == "1",
                     "Set S4C_TEST_POSTGRES=1 for actual PostgreSQL chunk/link tests.")
class ChunkStorageIntegrationTests(ChunkStorageFixture):
    def setUp(self):
        super().setUp()
        self.connection = open_connection()
        self.addCleanup(self.connection.close)
        transaction = self.connection.transaction(force_rollback=True)
        transaction.__enter__()
        self.addCleanup(transaction.__exit__, None, None, None)
        schema = "synthetic_test_" + uuid.uuid4().hex
        self.connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        self.connection.execute(sql.SQL("SET LOCAL search_path TO {}").format(sql.Identifier(schema)))
        initialize_schema(self.connection)
        self.expected_count = len(self.chunk_record["chunks"])

    def prepare_pages(self):
        store_document(self.metadata, self.connection)
        self.assertEqual(store_pages(self.rendered["metadata_path"], self.connection)["status"], "stored")

    def store(self, path=None):
        return store_chunks(path or self.chunk_path, self.connection)

    def counts(self):
        row = self.connection.execute("SELECT (SELECT count(*) FROM text_chunks) AS chunks, (SELECT count(*) FROM page_context_chunks) AS links").fetchone()
        return row["chunks"], row["links"]

    def test_repeat_import_keeps_chunk_and_link_counts(self):
        self.prepare_pages()
        first, repeat = self.store(), self.store()
        self.assertEqual((first["inserted_count"], first["linked_count"]), (self.expected_count, self.expected_count))
        self.assertEqual((repeat["unchanged_count"], repeat["linked_count"]), (self.expected_count, 0))
        self.assertEqual(self.counts(), (self.expected_count, self.expected_count))
        rows = read_chunks("2401.12345", 1, self.connection)
        for row in rows:
            contexts = read_chunk_contexts(row["chunk_id"], self.connection)
            self.assertEqual(len(contexts), 1)
            self.assertEqual(row["chunk_text"], contexts[0]["page_text"][row["start_char"]:row["end_char"]])
            self.assertEqual(row["parser_config"], self.chunk_record["parser"])

    def test_new_attempt_path_keeps_first_chunk_record(self):
        self.prepare_pages()
        self.store()
        before = self.connection.execute("SELECT chunk_id, source_chunks_path, processed_at, stored_at FROM text_chunks ORDER BY chunk_id").fetchall()
        new = chunk_text(self.extraction["metadata_path"], max_words=3, overlap_words=1)
        repeat = self.store(new["metadata_path"])
        self.assertEqual((repeat["unchanged_count"], repeat["linked_count"]), (self.expected_count, 0))
        self.assertEqual(self.connection.execute("SELECT chunk_id, source_chunks_path, processed_at, stored_at FROM text_chunks ORDER BY chunk_id").fetchall(), before)

    def test_new_rendering_links_existing_chunks_without_copying_text(self):
        self.prepare_pages()
        self.store()
        rendered = render_pages(self.metadata, self.extraction["metadata_path"], self.chunk_path, dpi=144)
        store_pages(rendered["metadata_path"], self.connection)
        result = self.store()
        self.assertEqual((result["unchanged_count"], result["linked_count"]), (self.expected_count, self.expected_count))
        self.assertEqual(self.counts(), (self.expected_count, 2 * self.expected_count))
        self.assertEqual(len(read_chunk_contexts(self.chunk_record["chunks"][0]["chunk_id"], self.connection)), 2)

    def test_new_chunking_policy_is_a_separate_variant(self):
        self.prepare_pages()
        self.store()
        other = chunk_text(self.extraction["metadata_path"], max_words=4, overlap_words=1)
        rendered = render_pages(self.metadata, self.extraction["metadata_path"], other["metadata_path"], dpi=72)
        store_pages(rendered["metadata_path"], self.connection)
        result = self.store(other["metadata_path"])
        self.assertEqual(result["inserted_count"], other["chunk_count"])
        self.assertEqual(self.counts()[0], self.expected_count + other["chunk_count"])
        old_contexts = read_chunk_contexts(self.chunk_record["chunks"][0]["chunk_id"], self.connection)
        self.assertEqual(len(old_contexts), 1)

    def test_new_document_version_stays_separate(self):
        self.prepare_pages()
        self.store()
        self.document.update(version=2, source_url="https://arxiv.org/abs/2401.12345v2")
        self.download["pdf_url"] = "https://arxiv.org/pdf/2401.12345v2"
        self.metadata.write_text(json.dumps(self.download))
        extraction = extract_text(self.metadata)
        other = chunk_text(extraction["metadata_path"], max_words=3, overlap_words=1)
        rendered = render_pages(self.metadata, extraction["metadata_path"], other["metadata_path"], dpi=72)
        store_document(self.metadata, self.connection)
        store_pages(rendered["metadata_path"], self.connection)
        self.assertEqual(self.store(other["metadata_path"])["inserted_count"], self.expected_count)
        self.assertEqual(len(read_chunks("2401.12345", 1, self.connection)), self.expected_count)
        self.assertEqual(len(read_chunks("2401.12345", 2, self.connection)), self.expected_count)

    def test_link_failure_rolls_back_chunk_and_all_its_links(self):
        self.prepare_pages()
        rendered = render_pages(self.metadata, self.extraction["metadata_path"], self.chunk_path, dpi=144)
        store_pages(rendered["metadata_path"], self.connection)
        target = self.chunk_record["chunks"][1]
        blocked = self.connection.execute("SELECT max(context_id) AS id FROM page_contexts WHERE page_number = %s", (target["page_number"],)).fetchone()["id"]
        self.connection.execute(sql.SQL("ALTER TABLE page_context_chunks ADD CONSTRAINT synthetic_failure CHECK (NOT (chunk_id = {} AND context_id = {}))").format(sql.Literal(target["chunk_id"]), sql.Literal(blocked)))
        result = self.store()
        self.assertEqual((result["inserted_count"], result["failed_count"]), (self.expected_count - 1, 1))
        self.assertEqual(self.counts(), (self.expected_count - 1, 2 * (self.expected_count - 1)))
        self.assertEqual(read_chunk_contexts(target["chunk_id"], self.connection), [])
        self.connection.execute("ALTER TABLE page_context_chunks DROP CONSTRAINT synthetic_failure")
        retry = self.store()
        self.assertEqual((retry["inserted_count"], retry["unchanged_count"], retry["linked_count"]), (1, self.expected_count - 1, 2))
        self.assertEqual(self.counts(), (self.expected_count, 2 * self.expected_count))

    def test_foreign_keys_reject_cross_page_link(self):
        self.prepare_pages()
        self.store()
        chunk = read_chunks("2401.12345", 1, self.connection)[0]
        wrong = self.connection.execute("SELECT context_id FROM page_contexts WHERE page_number = 2").fetchone()["context_id"]
        with self.assertRaises(psycopg.errors.ForeignKeyViolation):
            with self.connection.transaction():
                self.connection.execute("INSERT INTO page_context_chunks (context_id, chunk_id, page_id) VALUES (%s, %s, %s)",
                                        (wrong, chunk["chunk_id"], chunk["page_id"]))
        self.assertEqual(self.counts(), (self.expected_count, self.expected_count))

    def test_missing_contexts_create_no_orphan_chunks_and_retry_succeeds(self):
        store_document(self.metadata, self.connection)
        result = self.store()
        self.assertEqual((result["status"], result["failed_count"], self.counts()), ("failed", self.expected_count, (0, 0)))
        store_pages(self.rendered["metadata_path"], self.connection)
        self.assertEqual(self.store()["inserted_count"], self.expected_count)

    def test_mismatched_sql_page_text_is_not_used(self):
        self.prepare_pages()
        original = self.extraction["pages"][0]["text"]
        self.connection.execute("UPDATE page_contexts SET page_text = 'SYNTHETIC wrong text' WHERE page_number = 1")
        result = self.store()
        first_page_count = sum(c["page_number"] == 1 for c in self.chunk_record["chunks"])
        self.assertEqual(result["failed_count"], first_page_count)
        self.assertEqual(self.counts()[0], self.expected_count - first_page_count)
        self.connection.execute("UPDATE page_contexts SET page_text = %s WHERE page_number = 1", (original,))
        self.assertEqual(self.store()["inserted_count"], first_page_count)

    def test_text_chunks_can_link_to_context_with_failed_image(self):
        store_document(self.metadata, self.connection)
        failed_images = render_pages(self.metadata, self.extraction["metadata_path"], self.chunk_path, dpi=72, max_pixels=10)
        store_pages(failed_images["metadata_path"], self.connection)
        self.assertEqual(self.store()["inserted_count"], self.expected_count)
        context = read_chunk_contexts(self.chunk_record["chunks"][0]["chunk_id"], self.connection)[0]
        self.assertEqual(context["image_status"], "failed")
        self.assertIsNone(context["png_path"])


if __name__ == "__main__":
    unittest.main()
