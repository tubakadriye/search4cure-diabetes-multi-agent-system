"""Synthetic PDF fixtures and isolated, rolled-back PostgreSQL page-context tests."""

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
from store_document import store_document
from store_pages import read_page_contexts, store_pages, validate_page


class PageContextFixture(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.pdf = self.root / "SYNTHETIC.pdf"
        with pymupdf.open() as pdf:
            for number in (1, 2):
                page = pdf.new_page(width=160, height=200)
                page.insert_text((15, 30), f"SYNTHETIC page {number} for SQL tests", fontsize=8)
            pdf.save(self.pdf)
        self.document = {"arxiv_id": "2401.12345", "version": 1, "title": "SYNTHETIC SQL page fixture",
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
        extraction = extract_text(self.metadata)
        chunks = chunk_text(extraction["metadata_path"], max_words=5, overlap_words=1)
        rendered = render_pages(self.metadata, extraction["metadata_path"], chunks["metadata_path"], dpi=72)
        self.manifest_path = Path(rendered["metadata_path"])
        self.manifest = json.loads(self.manifest_path.read_text())

    def validate(self, page):
        return validate_page(page, self.document, self.download["sha256"], self.manifest_path)


class PageContextUnitTests(PageContextFixture):
    def test_semantic_identity_ignores_attempt_paths_and_times(self):
        first = self.validate(self.manifest["pages"][0])
        second = copy.deepcopy(self.manifest["pages"][0])
        second.update(processed_at="2026-11-04T11:00:00Z", source_metadata_path="SYNTHETIC-other-attempt.json",
                      extraction_path="SYNTHETIC-other-extraction.json", extraction_sha256="f" * 64,
                      record_path="SYNTHETIC-other-page.json")
        self.assertEqual(first["context_id"], self.validate(second)["context_id"])

    def test_inconsistent_page_identity_is_rejected(self):
        page = copy.deepcopy(self.manifest["pages"][0])
        page["page_id"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "identity"):
            self.validate(page)

    def test_changed_text_fingerprint_is_rejected(self):
        page = copy.deepcopy(self.manifest["pages"][0])
        page["text"] += " SYNTHETIC changed text"
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.validate(page)

    def test_changed_png_and_wrong_dimensions_are_rejected(self):
        page = copy.deepcopy(self.manifest["pages"][0])
        page["image"]["width"] += 1
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self.validate(page)
        page = self.manifest["pages"][0]
        Path(page["image"]["local_path"]).write_bytes(b"SYNTHETIC corrupt PNG")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.validate(page)

    def test_duplicate_chunk_ids_are_rejected(self):
        page = copy.deepcopy(self.manifest["pages"][0])
        page["chunk_ids"] += page["chunk_ids"]
        with self.assertRaisesRegex(ValueError, "Chunk IDs"):
            self.validate(page)

    @patch("store_pages.open_connection")
    def test_incomplete_page_inventory_is_rejected_before_sql(self, connect):
        self.manifest["pages"].pop()
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "every physical page"):
            store_pages(self.manifest_path)
        connect.assert_not_called()

    @patch("store_pages.open_connection")
    def test_changed_extraction_snapshot_is_rejected_before_sql(self, connect):
        source = Path(self.manifest["extraction_path"])
        source.write_text(source.read_text() + "\n")
        with self.assertRaises(ValueError):
            store_pages(self.manifest_path)
        connect.assert_not_called()

    def test_failed_text_stays_unknown_but_keeps_usable_image(self):
        with patch("render_pages.pymupdf.Page.get_text", side_effect=RuntimeError("SYNTHETIC text failure")):
            result = render_pages(self.metadata, dpi=72)
        values = self.validate(result["pages"][0])
        self.assertIsNone(values["page_text"])
        self.assertEqual(values["image_status"], "rendered")
        self.assertIsNotNone(values["png_sha256"])
        result["pages"][0]["text"] = ""
        with self.assertRaisesRegex(ValueError, "remain unknown"):
            self.validate(result["pages"][0])


@unittest.skipUnless(os.environ.get("S4C_TEST_POSTGRES") == "1",
                     "Set S4C_TEST_POSTGRES=1 for actual PostgreSQL page-context tests.")
class PageContextIntegrationTests(PageContextFixture):
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

    def register(self):
        store_document(self.metadata, self.connection)

    def store(self, path=None):
        return store_pages(path or self.manifest_path, self.connection)

    def count(self):
        return self.connection.execute("SELECT count(*) AS n FROM page_contexts").fetchone()["n"]

    def test_repeat_import_keeps_two_rows_and_first_snapshot(self):
        self.register()
        first = self.store()
        before = read_page_contexts("2401.12345", 1, self.connection)
        for page in self.manifest["pages"]:
            page["processed_at"] = "2026-11-04T11:00:00Z"
            page["record_path"] = "SYNTHETIC-other-record.json"
        self.manifest_path.write_text(json.dumps(self.manifest))
        repeat = self.store()
        initialize_schema(self.connection)
        self.assertEqual((first["inserted_count"], repeat["unchanged_count"], self.count()), (2, 2, 2))
        self.assertEqual(read_page_contexts("2401.12345", 1, self.connection), before)
        self.assertEqual(before[0]["page_text"], self.manifest["pages"][0]["text"])

    def test_missing_parent_can_be_imported_then_retried(self):
        with self.assertRaisesRegex(ValueError, "matching document"):
            self.store()
        self.assertEqual(self.count(), 0)
        self.register()
        self.assertEqual(self.store()["inserted_count"], 2)

    def test_foreign_key_rejects_wrong_pdf_hash_even_without_python_validation(self):
        self.register()
        self.store()
        with self.assertRaises(psycopg.errors.ForeignKeyViolation):
            with self.connection.transaction():
                self.connection.execute("UPDATE page_contexts SET source_pdf_sha256 = %s", ("0" * 64,))
        self.assertEqual(self.count(), 2)

    def test_text_failure_and_successful_recovery_are_separate_snapshots(self):
        self.register()
        with patch("render_pages.pymupdf.Page.get_text", side_effect=RuntimeError("SYNTHETIC text failure")):
            failed_text = render_pages(self.metadata, dpi=72)
        self.assertEqual(self.store(failed_text["metadata_path"])["inserted_count"], 2)
        rows = read_page_contexts("2401.12345", 1, self.connection)
        self.assertTrue(all(r["page_text"] is None and r["image_status"] == "rendered" and r["text_error"] for r in rows))
        self.assertEqual(self.store()["inserted_count"], 2)
        self.assertEqual(self.count(), 4)
        self.assertEqual(self.store()["unchanged_count"], 2)
        rows = read_page_contexts("2401.12345", 1, self.connection)
        self.assertEqual(sum(r["text_status"] == "failed" for r in rows), 2)
        self.assertEqual(sum(r["text_status"] == "extracted" for r in rows), 2)

    def test_bad_png_does_not_lose_other_pages_and_can_retry(self):
        self.register()
        path = Path(self.manifest["pages"][0]["image"]["local_path"])
        original = path.read_bytes()
        path.write_bytes(b"SYNTHETIC corrupt PNG")
        partial = self.store()
        self.assertEqual((partial["status"], partial["inserted_count"], partial["failed_count"]), ("partial", 1, 1))
        path.write_bytes(original)
        retry = self.store()
        self.assertEqual((retry["inserted_count"], retry["unchanged_count"], self.count()), (1, 1, 2))

    def test_valid_chunk_ids_from_wrong_page_are_rejected(self):
        self.register()
        self.manifest["pages"][0]["chunk_ids"] = self.manifest["pages"][1]["chunk_ids"]
        self.manifest_path.write_text(json.dumps(self.manifest))
        result = self.store()
        self.assertEqual((result["inserted_count"], result["failed_count"], self.count()), (1, 1, 1))
        self.assertIn("chunk", result["pages"][0]["error"].lower())

    def test_sql_failure_on_second_page_preserves_first_and_allows_retry(self):
        self.register()
        before = self.manifest_path.read_bytes(), self.pdf.read_bytes()
        self.connection.execute("ALTER TABLE page_contexts ADD CONSTRAINT synthetic_failure CHECK (page_number <> 2)")
        result = self.store()
        self.assertEqual((result["inserted_count"], result["failed_count"], self.count()), (1, 1, 1))
        self.connection.execute("ALTER TABLE page_contexts DROP CONSTRAINT synthetic_failure")
        retry = self.store()
        self.assertEqual((retry["inserted_count"], retry["unchanged_count"], self.count()), (1, 1, 2))
        self.assertEqual(before, (self.manifest_path.read_bytes(), self.pdf.read_bytes()))

    def test_different_resolution_keeps_a_separate_context(self):
        self.register()
        self.store()
        other = render_pages(self.metadata, dpi=144)
        self.assertEqual(self.store(other["metadata_path"])["inserted_count"], 2)
        self.assertEqual(self.count(), 4)
        rows = read_page_contexts("2401.12345", 1, self.connection)
        self.assertEqual(len({r["page_id"] for r in rows}), 2)
        self.assertEqual(len({r["render_id"] for r in rows}), 4)

    def test_new_document_version_is_kept_separate(self):
        self.register()
        self.store()
        self.document.update(version=2, source_url="https://arxiv.org/abs/2401.12345v2")
        self.download["pdf_url"] = "https://arxiv.org/pdf/2401.12345v2"
        self.metadata.write_text(json.dumps(self.download))
        self.register()
        other = render_pages(self.metadata, dpi=72)
        self.assertEqual(self.store(other["metadata_path"])["inserted_count"], 2)
        self.assertEqual(len(read_page_contexts("2401.12345", 1, self.connection)), 2)
        self.assertEqual(len(read_page_contexts("2401.12345", 2, self.connection)), 2)

    def test_sql_text_nullability_constraint_is_enforced(self):
        self.register()
        self.store()
        with self.assertRaises(psycopg.errors.CheckViolation):
            with self.connection.transaction():
                self.connection.execute("UPDATE page_contexts SET text_status = 'failed'")
        self.assertEqual(self.count(), 2)


if __name__ == "__main__":
    unittest.main()
