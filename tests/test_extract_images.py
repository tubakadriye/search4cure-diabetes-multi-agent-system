"""Generated PDFs and explicit simulated failures test image extraction offline."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pymupdf

from extract_images import _save_png, extract_images


class ImageExtractionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.pdf_path = self.root / "SYNTHETIC.pdf"
        self.record_path = self.root / "SYNTHETIC-download.json"

    def image(self, color=(255, 0, 0), alpha=False):
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 6, 4), alpha)
        pixmap.set_rect(pixmap.irect, color)
        return pixmap.tobytes("png")

    def make_pdf(self, transparent=False, two_images=False):
        color = (128, 0, 0, 128) if transparent else (255, 0, 0)
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            xref = page.insert_image(pymupdf.Rect(20, 20, 80, 60), stream=self.image(color, transparent))
            page.insert_image(pymupdf.Rect(100, 20, 160, 60), xref=xref)
            if two_images:
                page.insert_image(pymupdf.Rect(20, 100, 80, 140), stream=self.image((0, 0, 255)))
            page = pdf.new_page()
            page.insert_image(pymupdf.Rect(20, 20, 80, 60), xref=xref)
            pdf.save(self.pdf_path)
        self.write_record()

    def write_record(self, version=1):
        record = {
            "status": "downloaded", "local_path": str(self.pdf_path),
            "sha256": hashlib.sha256(self.pdf_path.read_bytes()).hexdigest(),
            "search": {"query": "SYNTHETIC TEST ONLY", "document": {
                "arxiv_id": "2401.12345", "version": version,
                "title": "Synthetic image fixture", "first_submitted_at": None,
            }},
        }
        self.record_path.write_text(json.dumps(record), encoding="utf-8")

    def attempts(self):
        return [json.loads(path.read_text()) for path in (self.root / "images").glob("*.json")]

    def test_page_occurrences_hashes_and_shared_assets(self):
        self.make_pdf()
        original = self.pdf_path.read_bytes()
        result = extract_images(self.record_path)
        self.assertEqual(result["status"], "extracted")
        self.assertEqual(result["extracted_count"], 3)
        self.assertEqual(result["unique_asset_count"], 1)
        self.assertEqual([item["page_number"] for item in result["images"]], [1, 1, 2])
        self.assertEqual(len({item["image_id"] for item in result["images"]}), 3)
        for item in result["images"]:
            data = Path(item["local_path"]).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), item["sha256"])
            self.assertEqual((item["arxiv_id"], item["version"]), ("2401.12345", 1))
            self.assertEqual(item["source_pdf_sha256"], hashlib.sha256(original).hexdigest())
            self.assertEqual((item["width"], item["height"]), (6, 4))
            self.assertEqual(len(item["bbox"]), 4)
        self.assertNotEqual(result["images"][0]["bbox"], result["images"][1]["bbox"])
        self.assertEqual(result["parser"]["version"], pymupdf.VersionBind)
        self.assertFalse(result["parser"]["page_rendering"])
        self.assertEqual(self.pdf_path.read_bytes(), original)

    def test_equivalent_rerun_retains_ids_and_does_not_rewrite_png(self):
        self.make_pdf()
        first = extract_images(self.record_path)
        asset = Path(first["images"][0]["local_path"])
        before = asset.read_bytes(), asset.stat().st_mtime_ns
        record = json.loads(self.record_path.read_text())
        record["pdf_retrieved_at"] = "2026-11-04T12:00:00Z"
        self.record_path.write_text(json.dumps(record))
        second = extract_images(self.record_path)
        self.assertEqual([i["image_id"] for i in first["images"]], [i["image_id"] for i in second["images"]])
        self.assertEqual(before, (asset.read_bytes(), asset.stat().st_mtime_ns))
        self.assertEqual(len(list(asset.parent.glob("*.png"))), 1)
        self.assertEqual(len(self.attempts()), 2)

    def test_transparency_mask_is_applied(self):
        self.make_pdf(transparent=True)
        result = extract_images(self.record_path)
        self.assertEqual(result["status"], "extracted")
        item = result["images"][0]
        self.assertGreater(item["soft_mask_xref"], 0)
        pixmap = pymupdf.Pixmap(item["local_path"])
        self.assertTrue(item["alpha"])
        self.assertEqual(pixmap.pixel(3, 2)[-1], 128)

    def test_text_and_vector_drawings_do_not_become_raster_images(self):
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_text((72, 72), "SYNTHETIC text")
            page.draw_rect(pymupdf.Rect(20, 100, 100, 180), color=(1, 0, 0))
            pdf.save(self.pdf_path)
        self.write_record()
        result = extract_images(self.record_path)
        self.assertEqual(result["status"], "no_images")
        self.assertEqual(result["extracted_count"], 0)
        self.assertEqual(result["pages"], [{"page_number": 1, "image_count": 0, "extracted_count": 0}])

    def test_corrupt_pdf_is_recorded_as_failed(self):
        self.pdf_path.write_bytes(b"%PDF-1.7 SYNTHETIC broken PDF")
        self.write_record()
        with self.assertRaises(ValueError):
            extract_images(self.record_path)
        self.assertEqual(self.attempts()[0]["status"], "failed")
        self.assertTrue(self.attempts()[0]["error"])

    def test_hash_mismatch_prevents_parsing(self):
        self.make_pdf()
        self.pdf_path.write_bytes(self.pdf_path.read_bytes() + b"SYNTHETIC change")
        with patch("extract_images.pymupdf.open") as opener:
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                extract_images(self.record_path)
            opener.assert_not_called()
        self.assertEqual(self.attempts()[0]["status"], "failed")

    def test_missing_metadata_and_invalid_limits_are_rejected(self):
        self.record_path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "required metadata"):
            extract_images(self.record_path)
        for limit in (0, -1, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                extract_images(self.record_path, max_pixels=limit)

    def test_pixel_limit_is_recorded_without_creating_assets(self):
        self.make_pdf()
        result = extract_images(self.record_path, max_pixels=10)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["failed_count"], 3)
        self.assertEqual(result["extracted_count"], 0)
        self.assertEqual(list((self.root / "images" / "assets").glob("*.png")), [])

    def test_inline_image_is_explicitly_unsupported(self):
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_text((72, 72), "SYNTHETIC inline image fixture")
            content = page.get_contents()[0]
            pdf.update_stream(content, b"q 10 0 0 10 20 20 cm BI /W 1 /H 1 /CS /RGB /BPC 8 ID \xff\x00\x00 EI Q")
            pdf.save(self.pdf_path)
        self.write_record()
        result = extract_images(self.record_path)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["unsupported_count"], 1)
        self.assertEqual(result["images"][0]["xref"], 0)
        self.assertIn("inline or unresolved", result["images"][0]["error"])

    def test_partial_storage_failure_preserves_successes_and_can_retry(self):
        self.make_pdf(two_images=True)
        calls = 0

        def simulated_failure(folder, data):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("SYNTHETIC storage failure")
            return _save_png(folder, data)

        with patch("extract_images._save_png", side_effect=simulated_failure):
            partial = extract_images(self.record_path)
        self.assertEqual((partial["status"], partial["failed_count"], partial["extracted_count"]), ("partial", 1, 3))
        saved = Path(partial["images"][0]["local_path"])
        before = saved.read_bytes(), saved.stat().st_mtime_ns
        retry = extract_images(self.record_path)
        self.assertEqual((retry["status"], retry["unique_asset_count"]), ("extracted", 2))
        self.assertEqual(before, (saved.read_bytes(), saved.stat().st_mtime_ns))
        self.assertEqual(list(saved.parent.glob("*.part")), [])

    def test_manifest_storage_failure_allows_retry(self):
        self.make_pdf()
        with patch("extract_images.save_metadata", side_effect=OSError("SYNTHETIC manifest storage failure")):
            with self.assertRaises(OSError):
                extract_images(self.record_path)
        self.assertTrue(self.pdf_path.exists())
        self.assertEqual(extract_images(self.record_path)["status"], "extracted")

    def test_new_version_changes_occurrence_ids(self):
        self.make_pdf()
        first = extract_images(self.record_path)
        self.write_record(version=2)
        second = extract_images(self.record_path)
        self.assertTrue({i["image_id"] for i in first["images"]}.isdisjoint(
            {i["image_id"] for i in second["images"]}))


if __name__ == "__main__":
    unittest.main()
