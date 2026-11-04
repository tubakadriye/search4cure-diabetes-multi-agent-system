"""Use generated PDFs to test extraction, provenance, and failure handling offline."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pymupdf

from extract_text import extract_text


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.pdf_path = self.root / "synthetic.pdf"
        self.record_path = self.root / "download.json"

    def make_record(self, texts=("Synthetic first page", "Synthetic second page")):
        with pymupdf.open() as pdf:
            for text in texts:
                page = pdf.new_page()
                if text:
                    page.insert_text((72, 72), text)
            pdf.save(self.pdf_path)
        self.write_record()

    def write_record(self):
        record = {
            "search": {"query": "SYNTHETIC TEST", "retrieved_at": "2024-01-01T00:00:00Z",
                       "document": {"arxiv_id": "2401.12345", "version": 2,
                                    "title": "Synthetic test", "source_url": "https://example.invalid",
                                    "first_submitted_at": None, "version_submitted_at": None}},
            "local_path": str(self.pdf_path), "status": "downloaded",
            "sha256": hashlib.sha256(self.pdf_path.read_bytes()).hexdigest(),
        }
        self.record_path.write_text(json.dumps(record))

    def attempts(self):
        return [json.loads(path.read_text()) for path in (self.root / "extracted").glob("*.json")]

    def test_physical_pages_source_hash_and_parser_configuration(self):
        self.make_record()
        original = self.pdf_path.read_bytes()
        result = extract_text(self.record_path)
        self.assertEqual(result["status"], "extracted")
        self.assertEqual([page["page_number"] for page in result["pages"]], [1, 2])
        self.assertIn("first page", result["pages"][0]["text"])
        self.assertIn("second page", result["pages"][1]["text"])
        self.assertEqual(result["download"]["search"]["document"]["version"], 2)
        self.assertIsNone(result["download"]["search"]["document"]["first_submitted_at"])
        self.assertEqual(result["parser"]["version"], pymupdf.VersionBind)
        self.assertFalse(result["parser"]["ocr"])
        self.assertEqual(self.pdf_path.read_bytes(), original)

    def test_empty_page_retains_its_position(self):
        self.make_record(("", "Synthetic page two"))
        result = extract_text(self.record_path)
        self.assertEqual(result["empty_pages"], [1])
        self.assertEqual(result["pages"][1]["page_number"], 2)
        self.assertIn("page two", result["pages"][1]["text"])

    def test_all_empty_pdf_records_failure_and_can_be_retried(self):
        self.make_record(("",))
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, "No extractable text"):
                extract_text(self.record_path)
        self.assertEqual(len(self.attempts()), 2)
        self.assertTrue(all(attempt["status"] == "failed" for attempt in self.attempts()))
        self.assertTrue(self.pdf_path.exists())

    def test_corrupt_pdf_records_failure(self):
        self.pdf_path.write_bytes(b"%PDF-1.7\nSYNTHETIC broken PDF")
        self.write_record()
        with self.assertRaises(ValueError):
            extract_text(self.record_path)
        self.assertEqual(self.attempts()[0]["status"], "failed")
        self.assertIsNotNone(self.attempts()[0]["error"])

    def test_changed_pdf_is_rejected_before_parsing(self):
        self.make_record()
        self.pdf_path.write_bytes(self.pdf_path.read_bytes() + b"SYNTHETIC change")
        with patch("extract_text.pymupdf.open") as opener:
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                extract_text(self.record_path)
            opener.assert_not_called()

    def test_missing_metadata_is_rejected(self):
        self.record_path.write_text('{}')
        with self.assertRaisesRegex(ValueError, "required metadata"):
            extract_text(self.record_path)

    def test_nul_replacements_are_visible_and_recorded(self):
        self.make_record()
        with patch("extract_text.pymupdf.Page.get_text", return_value="Synthetic\x00text"):
            result = extract_text(self.record_path)
        self.assertEqual(result["pages"][0]["text"], "Synthetic\ufffdtext")
        self.assertEqual(result["pages"][0]["nul_replacements"], 1)

    def test_storage_failure_preserves_source_and_allows_rerun(self):
        self.make_record()
        with patch("extract_text.save_metadata", side_effect=OSError("Synthetic storage failure")):
            with self.assertRaises(OSError):
                extract_text(self.record_path)
        result = extract_text(self.record_path)
        self.assertEqual(result["status"], "extracted")
        self.assertTrue(self.record_path.exists())
        self.assertEqual(len(self.attempts()), 1)


if __name__ == "__main__":
    unittest.main()
