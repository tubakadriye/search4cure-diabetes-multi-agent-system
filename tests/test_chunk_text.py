"""Test chunk boundaries, provenance, stable identities and recovery using synthetic text."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from chunk_text import chunk_text


class ChunkingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source_path = self.root / "synthetic-extraction.json"

    def write_source(self, texts):
        self.source = {
            "status": "extracted", "processed_at": "2024-01-01T00:00:00Z",
            "download": {"local_path": str(self.root / "synthetic.pdf"), "sha256": "a" * 64,
                         "search": {"document": {"arxiv_id": "2401.12345", "version": 2,
                         "title": "Synthetic test", "source_url": "https://example.invalid"}}},
            "parser": {"name": "SYNTHETIC_TEST_PARSER", "version": "test"},
            "page_count": len(texts),
            "pages": [{"page_number": i + 1, "text": text} for i, text in enumerate(texts)],
        }
        self.persist_source()

    def persist_source(self):
        self.source_path.write_text(json.dumps(self.source, ensure_ascii=False), encoding="utf-8")

    def test_word_limit_overlap_and_no_redundant_final_window(self):
        self.write_source(["A B C D E F G H I J"])
        result = chunk_text(self.source_path, max_words=4, overlap_words=1)
        self.assertEqual([chunk["text"] for chunk in result["chunks"]],
                         ["A B C D", "D E F G", "G H I J"])
        self.assertEqual(result["chunk_count"], 3)
        self.assertTrue(all(chunk["word_count"] == 4 for chunk in result["chunks"]))

    def test_unicode_offsets_preserve_exact_text_and_cover_every_word(self):
        text = "  α\tbeta\n gamma  delta epsilon ζeta eta theta  "
        self.write_source([text])
        result = chunk_text(self.source_path, max_words=4, overlap_words=1)
        covered = set()
        for chunk in result["chunks"]:
            start, end = chunk["start_char"], chunk["end_char"]
            self.assertEqual(chunk["text"], text[start:end])
            covered.update(range(start, end))
        self.assertTrue({i for i, char in enumerate(text) if not char.isspace()} <= covered)
        self.assertEqual(result["source_extraction_sha256"],
                         hashlib.sha256(self.source_path.read_bytes()).hexdigest())

    def test_page_boundaries_empty_pages_and_identical_text_on_different_pages(self):
        self.write_source(["Same synthetic text", "", "Same synthetic text"])
        result = chunk_text(self.source_path, max_words=4, overlap_words=1)
        self.assertEqual([chunk["page_number"] for chunk in result["chunks"]], [1, 3])
        self.assertEqual(result["pages_without_text"], [2])
        self.assertNotEqual(result["chunks"][0]["chunk_id"], result["chunks"][1]["chunk_id"])
        self.assertEqual(result["chunks"][0]["arxiv_id"], "2401.12345")
        self.assertEqual(result["chunks"][0]["version"], 2)

    def test_ids_survive_equivalent_reruns_with_new_processing_times(self):
        self.write_source(["Synthetic stable source text"])
        first = chunk_text(self.source_path)
        self.source["processed_at"] = "2024-02-01T00:00:00Z"
        self.persist_source()
        second = chunk_text(self.source_path)
        self.assertNotEqual(first["source_extraction_sha256"], second["source_extraction_sha256"])
        self.assertEqual([c["chunk_id"] for c in first["chunks"]], [c["chunk_id"] for c in second["chunks"]])
        self.assertNotEqual(first["metadata_path"], second["metadata_path"])

    def test_version_hash_and_configuration_changes_produce_new_ids(self):
        self.write_source(["A B C"])
        original = deepcopy(self.source)
        baseline = chunk_text(self.source_path, max_words=4, overlap_words=1)["chunks"][0]["chunk_id"]
        for change in ("version", "hash", "configuration"):
            with self.subTest(change=change):
                self.source = deepcopy(original)
                if change == "version":
                    self.source["download"]["search"]["document"]["version"] = 3
                elif change == "hash":
                    self.source["download"]["sha256"] = "b" * 64
                self.persist_source()
                result = chunk_text(self.source_path, max_words=5 if change == "configuration" else 4,
                                    overlap_words=1)
                self.assertNotEqual(result["chunks"][0]["chunk_id"], baseline)

    def test_invalid_limits_and_incomplete_extraction_are_rejected(self):
        self.write_source(["Synthetic text"])
        for words, overlap in [(0, 0), (4, -1), (4, 4), (True, 0)]:
            with self.subTest(words=words, overlap=overlap), self.assertRaises(ValueError):
                chunk_text(self.source_path, max_words=words, overlap_words=overlap)
        self.source["status"] = "failed"
        self.persist_source()
        with self.assertRaisesRegex(ValueError, "completed extraction"):
            chunk_text(self.source_path)

    def test_invalid_page_numbers_and_all_empty_text_record_failures(self):
        for texts, invalid in [(["A", "B"], "duplicate"), (["A"], "boolean"), (["", "  "], "empty")]:
            with self.subTest(texts=texts):
                self.write_source(texts)
                if invalid == "duplicate":
                    self.source["pages"][1]["page_number"] = 1
                    self.persist_source()
                elif invalid == "boolean":
                    self.source["pages"][0]["page_number"] = True
                    self.persist_source()
                with self.assertRaises(ValueError):
                    chunk_text(self.source_path)
        records = [json.loads(path.read_text()) for path in (self.root / "chunks").glob("*.json")]
        self.assertEqual(len(records), 3)
        self.assertTrue(all(record["status"] == "failed" for record in records))

    def test_storage_failure_keeps_source_and_allows_rerun(self):
        self.write_source(["Synthetic source text"])
        original = self.source_path.read_bytes()
        with patch("chunk_text.save_metadata", side_effect=OSError("Synthetic storage failure")):
            with self.assertRaises(OSError):
                chunk_text(self.source_path)
        result = chunk_text(self.source_path)
        self.assertEqual(result["status"], "chunked")
        self.assertEqual(self.source_path.read_bytes(), original)
        self.assertEqual(len(list((self.root / "chunks").glob("*.json"))), 1)


if __name__ == "__main__":
    unittest.main()
