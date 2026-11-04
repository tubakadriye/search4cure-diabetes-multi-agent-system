"""Synthetic download fixtures; opt-in SQL tests roll back their isolated test schema."""

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

from database import initialize_schema, open_connection
from store_document import read_document, store_document, validate_download


class DownloadFixture(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.pdf = self.directory / "SYNTHETIC.pdf"
        self.pdf.write_bytes(b"%PDF-SYNTHETIC download fixture; not a parser test")
        self.path = self.directory / "SYNTHETIC-download.json"
        self.record = {
            "status": "downloaded", "local_path": str(self.pdf),
            "sha256": hashlib.sha256(self.pdf.read_bytes()).hexdigest(),
            "size_bytes": self.pdf.stat().st_size,
            "pdf_url": "https://arxiv.org/pdf/2401.12345v1",
            "pdf_retrieved_at": "2026-11-04T10:01:00+00:00",
            "search": {
                "query": "SYNTHETIC_TEST_ONLY",
                "retrieved_at": "2026-11-04T10:00:00+00:00",
                "document": {
                    "arxiv_id": "2401.12345", "version": 1,
                    "title": "SYNTHETIC document", "source_url": "https://arxiv.org/abs/2401.12345v1",
                    "first_submitted_at": "2024-01-01T12:00:00Z",
                    "version_submitted_at": "2024-01-01T12:00:00Z",
                },
            },
        }
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.record), encoding="utf-8")


class DocumentValidationTests(DownloadFixture):
    @patch("store_document.open_connection")
    def test_required_fields_and_version_links_rejected_before_sql(self, connect):
        original = copy.deepcopy(self.record)
        cases = [("title", ""), ("arxiv_id", "invalid"), ("version", True),
                 ("version", 0), ("source_url", "https://arxiv.org/abs/2401.12345v9")]
        for key, value in cases:
            with self.subTest(field=key, value=value):
                self.record = copy.deepcopy(original)
                self.record["search"]["document"][key] = value
                self.save()
                with self.assertRaises(ValueError):
                    store_document(self.path)
        self.record = copy.deepcopy(original)
        self.record["pdf_url"] = "https://arxiv.org/pdf/2401.12345v9"
        self.save()
        with self.assertRaises(ValueError):
            store_document(self.path)
        connect.assert_not_called()

    def test_unknown_source_dates_remain_null(self):
        document = self.record["search"]["document"]
        document.pop("first_submitted_at")
        document["version_submitted_at"] = None
        self.save()
        values = validate_download(self.path)
        self.assertIsNone(values["first_submitted_at"])
        self.assertIsNone(values["version_submitted_at"])
        self.assertIsNotNone(values["metadata_retrieved_at"].utcoffset())

    def test_missing_or_ambiguous_retrieval_time_is_rejected(self):
        for value in (None, "", "2026-11-04T10:00:00", "invalid"):
            with self.subTest(value=value):
                self.record["pdf_retrieved_at"] = value
                self.save()
                with self.assertRaises(ValueError):
                    validate_download(self.path)

    @patch("store_document.open_connection")
    def test_changed_or_missing_pdf_rejected_before_sql(self, connect):
        self.pdf.write_bytes(b"%PDF-SYNTHETIC changed")
        with self.assertRaisesRegex(ValueError, "no longer matches"):
            store_document(self.path)
        self.pdf.unlink()
        with self.assertRaises(FileNotFoundError):
            store_document(self.path)
        connect.assert_not_called()

    def test_failed_download_and_invalid_hash_are_rejected(self):
        self.record["status"] = "failed"
        self.save()
        with self.assertRaises(ValueError):
            validate_download(self.path)
        self.record["status"] = "downloaded"
        self.record["sha256"] = "not-a-hash"
        self.save()
        with self.assertRaises(ValueError):
            validate_download(self.path)

    @patch("store_document.open_connection", side_effect=psycopg.OperationalError("Synthetic database failure"))
    def test_database_failure_preserves_source_files(self, connect):
        before = self.path.read_bytes(), self.pdf.read_bytes()
        with self.assertRaises(psycopg.OperationalError):
            store_document(self.path)
        self.assertEqual(before, (self.path.read_bytes(), self.pdf.read_bytes()))


@unittest.skipUnless(os.environ.get("S4C_TEST_POSTGRES") == "1",
                     "Set S4C_TEST_POSTGRES=1 for actual PostgreSQL document tests.")
class DocumentIntegrationTests(DownloadFixture):
    def setUp(self):
        super().setUp()
        self.connection = open_connection()
        self.addCleanup(self.connection.close)
        transaction = self.connection.transaction(force_rollback=True)
        transaction.__enter__()
        self.addCleanup(transaction.__exit__, None, None, None)
        name = "synthetic_test_" + uuid.uuid4().hex
        self.connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(name)))
        self.connection.execute(sql.SQL("SET LOCAL search_path TO {}").format(sql.Identifier(name)))
        initialize_schema(self.connection)

    def count(self):
        return self.connection.execute("SELECT count(*) AS n FROM document_versions").fetchone()["n"]

    def test_repeat_import_and_schema_setup_preserve_one_original_record(self):
        first = store_document(self.path, self.connection)
        self.record["pdf_retrieved_at"] = "2026-11-04T11:00:00Z"
        self.save()
        second = store_document(self.path, self.connection)
        initialize_schema(self.connection)
        row = read_document("2401.12345", 1, self.connection)
        self.assertEqual((first["status"], second["status"], self.count()), ("inserted", "unchanged", 1))
        self.assertEqual(row["pdf_retrieved_at"].hour, 10)
        self.assertEqual(row["pdf_retrieved_at"].minute, 1)
        self.assertEqual(self.connection.execute("SELECT download_record FROM document_versions").fetchone()["download_record"]["search"]["query"], "SYNTHETIC_TEST_ONLY")

    def test_new_version_is_a_separate_row(self):
        store_document(self.path, self.connection)
        self.record["search"]["document"].update(version=2, source_url="https://arxiv.org/abs/2401.12345v2")
        self.record["pdf_url"] = "https://arxiv.org/pdf/2401.12345v2"
        self.save()
        self.assertEqual(store_document(self.path, self.connection)["status"], "inserted")
        self.assertEqual(self.count(), 2)
        self.assertEqual(read_document("2401.12345", 1, self.connection)["version"], 1)
        self.assertEqual(read_document("2401.12345", 2, self.connection)["version"], 2)

    def test_changed_hash_conflict_preserves_previous_version(self):
        store_document(self.path, self.connection)
        previous = read_document("2401.12345", 1, self.connection)
        self.pdf.write_bytes(b"%PDF-SYNTHETIC different content for the same version")
        self.record.update(sha256=hashlib.sha256(self.pdf.read_bytes()).hexdigest(), size_bytes=self.pdf.stat().st_size)
        self.save()
        with self.assertRaisesRegex(ValueError, "Content conflict"):
            store_document(self.path, self.connection)
        self.assertEqual(self.count(), 1)
        self.assertEqual(read_document("2401.12345", 1, self.connection), previous)

    def test_null_source_times_and_distinct_retrieval_times_survive_sql(self):
        self.record["search"]["document"].update(first_submitted_at=None, version_submitted_at=None)
        self.save()
        store_document(self.path, self.connection)
        row = read_document("2401.12345", 1, self.connection)
        self.assertIsNone(row["first_submitted_at"])
        self.assertIsNone(row["version_submitted_at"])
        self.assertLess(row["metadata_retrieved_at"], row["pdf_retrieved_at"])

    def test_sql_parameters_preserve_quotes_and_prevent_title_execution(self):
        title = "SYNTHETIC O'Brien'); DROP TABLE document_versions; --"
        self.record["search"]["document"]["title"] = title
        self.save()
        store_document(self.path, self.connection)
        self.assertEqual(read_document("2401.12345", 1, self.connection)["title"], title)
        self.assertEqual(self.count(), 1)
        self.assertIsNone(read_document("missing", 1, self.connection))

    def test_failed_sql_insert_rolls_back_and_can_be_retried(self):
        self.connection.execute("ALTER TABLE document_versions ADD CONSTRAINT synthetic_failure CHECK (title <> 'SYNTHETIC document')")
        with self.assertRaises(psycopg.errors.CheckViolation):
            store_document(self.path, self.connection)
        self.assertEqual(self.count(), 0)
        self.connection.execute("ALTER TABLE document_versions DROP CONSTRAINT synthetic_failure")
        self.assertEqual(store_document(self.path, self.connection)["status"], "inserted")
        self.assertEqual(self.count(), 1)


if __name__ == "__main__":
    unittest.main()
