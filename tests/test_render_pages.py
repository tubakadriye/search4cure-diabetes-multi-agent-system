"""Generated PDFs exercise full-page context, provenance and recoverable failures."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pymupdf

from chunk_text import chunk_text
from extract_images import extract_images
from extract_text import extract_text
from render_pages import _render_page, render_pages
from download_pdf import save_metadata


class PageRenderingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.pdf = self.root / "SYNTHETIC-pages.pdf"
        self.metadata = self.root / "SYNTHETIC-download.json"
        with pymupdf.open() as document:
            page = document.new_page(width=200, height=300)
            page.insert_text((20, 30), "SYNTHETIC first page context", fontsize=8)
            page.draw_rect(pymupdf.Rect(20, 70, 80, 120), fill=(1, 0, 0), color=(1, 0, 0))
            page.insert_text((20, 140), "SYNTHETIC figure caption", fontsize=8)
            page = document.new_page(width=200, height=300)
            page.insert_text((20, 30), "SYNTHETIC second page", fontsize=8)
            page.set_rotation(90)
            document.new_page(width=200, height=300)
            document.save(self.pdf)
        self.download = {
            "status": "downloaded", "local_path": str(self.pdf),
            "sha256": hashlib.sha256(self.pdf.read_bytes()).hexdigest(),
            "search": {"query": "SYNTHETIC TEST ONLY", "document": {
                "arxiv_id": "2401.12345", "version": 1,
                "title": "Synthetic context", "source_url": "https://arxiv.org/abs/2401.12345v1",
                "first_submitted_at": None, "version_submitted_at": None,
            }},
        }
        self.write_download()
        self.extraction = extract_text(self.metadata)
        self.chunks = chunk_text(self.extraction["metadata_path"], max_words=5, overlap_words=1)

    def write_download(self):
        self.metadata.write_text(json.dumps(self.download))

    def render(self, **kwargs):
        return render_pages(self.metadata, self.extraction["metadata_path"], self.chunks["metadata_path"], dpi=72, **kwargs)

    def attempts(self):
        return [json.loads(path.read_text()) for path in (self.root / "pages").glob("*.json")]

    def test_every_page_links_actual_png_text_chunks_and_source(self):
        original = self.pdf.read_bytes()
        result = self.render()
        self.assertEqual((result["status"], result["page_count"], result["rendered_count"]), ("rendered", 3, 3))
        self.assertEqual([p["page_number"] for p in result["pages"]], [1, 2, 3])
        for page in result["pages"]:
            number = page["page_number"]
            self.assertEqual(page["text"], self.extraction["pages"][number - 1]["text"])
            expected = [c["chunk_id"] for c in self.chunks["chunks"] if c["page_number"] == number]
            self.assertEqual(page["chunk_ids"], expected)
            self.assertEqual((page["arxiv_id"], page["version"]), ("2401.12345", 1))
            self.assertEqual(page["source_pdf_sha256"], self.download["sha256"])
            self.assertIsNone(page["document"]["first_submitted_at"])
            self.assertEqual(page["text_sha256"], hashlib.sha256(page["text"].encode()).hexdigest())
            image = page["image"]
            data = Path(image["local_path"]).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), image["sha256"])
            saved = json.loads(Path(page["record_path"]).read_text())
            self.assertEqual(saved["page_id"], page["page_id"])
            self.assertEqual(saved["text"], page["text"])
        blank = result["pages"][2]
        self.assertEqual((blank["text_status"], blank["text"], blank["chunk_ids"]), ("empty", "", []))
        self.assertEqual(self.pdf.read_bytes(), original)

    def test_vector_drawing_caption_and_rotation_are_rendered(self):
        result = self.render()
        self.assertEqual(extract_images(self.metadata)["status"], "no_images")
        image = pymupdf.Pixmap(result["pages"][0]["image"]["local_path"])
        self.assertEqual((image.width, image.height), (200, 300))
        self.assertEqual(image.pixel(40, 90), (255, 0, 0))
        self.assertEqual(image.pixel(190, 290), (255, 255, 255))
        darkest_caption_pixel = min(min(image.pixel(x, y)) for y in range(130, 145) for x in range(15, 190))
        self.assertLess(darkest_caption_pixel, 100)
        self.assertEqual((result["pages"][1]["image"]["width"], result["pages"][1]["image"]["height"]), (300, 200))

    def test_rerun_keeps_page_ids_and_does_not_rewrite_assets(self):
        first = self.render()
        before = {p["image"]["local_path"]: Path(p["image"]["local_path"]).stat().st_mtime_ns for p in first["pages"]}
        second = self.render()
        self.assertEqual([p["page_id"] for p in first["pages"]], [p["page_id"] for p in second["pages"]])
        self.assertEqual([p["render_id"] for p in first["pages"]], [p["render_id"] for p in second["pages"]])
        self.assertEqual(before, {p: Path(p).stat().st_mtime_ns for p in before})
        self.assertEqual(len(list((self.root / "pages" / "records").glob("*.json"))), 6)

    def test_dpi_changes_render_variant_but_not_logical_page(self):
        first = self.render()
        second = render_pages(self.metadata, dpi=144)
        self.assertEqual(first["pages"][0]["page_id"], second["pages"][0]["page_id"])
        self.assertNotEqual(first["pages"][0]["render_id"], second["pages"][0]["render_id"])
        self.assertEqual((second["pages"][0]["image"]["width"], second["pages"][0]["image"]["height"]), (400, 600))

    def test_new_version_changes_page_ids(self):
        first = self.render()
        self.download["search"]["document"]["version"] = 2
        self.write_download()
        second = render_pages(self.metadata, dpi=72)
        self.assertTrue({p["page_id"] for p in first["pages"]}.isdisjoint({p["page_id"] for p in second["pages"]}))

    def test_blank_document_still_has_a_page_record(self):
        with pymupdf.open() as pdf:
            pdf.new_page(width=100, height=100)
            data = pdf.tobytes()
        self.pdf.write_bytes(data)
        self.download["sha256"] = hashlib.sha256(data).hexdigest()
        self.write_download()
        result = render_pages(self.metadata, dpi=72)
        self.assertEqual(result["status"], "rendered")
        page = result["pages"][0]
        self.assertEqual((page["text_status"], page["text"], page["chunks_status"]), ("empty", "", "not_provided"))
        self.assertTrue(Path(page["record_path"]).is_file())

    def test_text_error_does_not_prevent_image_or_later_pages(self):
        original_get_text = pymupdf.Page.get_text

        def fail_first(page, *args, **kwargs):
            if page.number == 0:
                raise RuntimeError("SYNTHETIC first-page text failure")
            return original_get_text(page, *args, **kwargs)

        with patch("render_pages.pymupdf.Page.get_text", autospec=True, side_effect=fail_first):
            result = render_pages(self.metadata, dpi=72)
        self.assertEqual((result["status"], result["rendered_count"], result["failed_count"]), ("partial", 3, 0))
        self.assertEqual((result["text_failed_count"], result["image_failed_count"]), (1, 0))
        first = result["pages"][0]
        self.assertEqual((first["text_status"], first["image_status"]), ("failed", "rendered"))
        self.assertIsNone(first["text"])
        self.assertIsNone(first["text_sha256"])
        self.assertIn("SYNTHETIC first-page text failure", first["text_error"])
        self.assertIsNone(first["image_error"])
        png = Path(first["image"]["local_path"])
        self.assertEqual(hashlib.sha256(png.read_bytes()).hexdigest(), first["image"]["sha256"])
        saved = json.loads(Path(first["record_path"]).read_text())
        self.assertEqual(saved["text_error"], first["text_error"])
        self.assertEqual(saved["image_status"], "rendered")
        self.assertEqual(result["pages"][1]["text_status"], "extracted")
        self.assertEqual(result["pages"][2]["text_status"], "empty")
        before = png.stat().st_mtime_ns
        retry = render_pages(self.metadata, dpi=72)
        self.assertEqual((retry["status"], retry["text_failed_count"]), ("rendered", 0))
        self.assertEqual(png.stat().st_mtime_ns, before)

    def test_text_error_on_every_page_preserves_all_images(self):
        with patch("render_pages.pymupdf.Page.get_text", side_effect=RuntimeError("SYNTHETIC all-text failure")):
            result = render_pages(self.metadata, dpi=72)
        self.assertEqual((result["status"], result["rendered_count"], result["text_failed_count"]), ("partial", 3, 3))
        self.assertEqual(result["image_failed_count"], 0)
        for page in result["pages"]:
            self.assertIsNone(page["text"])
            self.assertTrue(Path(page["image"]["local_path"]).is_file())
            self.assertEqual(page["image_status"], "rendered")

    def test_text_error_and_image_error_are_both_preserved(self):
        def fail_first_image(page, folder, config):
            if page.number == 0:
                raise OSError("SYNTHETIC first-page image failure")
            return _render_page(page, folder, config)

        with patch("render_pages.pymupdf.Page.get_text", side_effect=RuntimeError("SYNTHETIC text failure")), \
                patch("render_pages._render_page", side_effect=fail_first_image):
            result = render_pages(self.metadata, dpi=72)
        first = result["pages"][0]
        self.assertIn("SYNTHETIC text failure", first["text_error"])
        self.assertIn("SYNTHETIC first-page image failure", first["image_error"])
        self.assertIn("SYNTHETIC text failure", first["error"])
        self.assertIn("SYNTHETIC first-page image failure", first["error"])
        self.assertEqual((result["rendered_count"], result["text_failed_count"], result["image_failed_count"]), (2, 3, 1))

    def test_visible_cropbox_and_rotation_policy(self):
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=200, height=300)
            page.set_cropbox(pymupdf.Rect(20, 40, 180, 260))
            page.set_rotation(90)
            data = pdf.tobytes()
        self.pdf.write_bytes(data)
        self.download["sha256"] = hashlib.sha256(data).hexdigest()
        self.write_download()
        page = render_pages(self.metadata, dpi=72)["pages"][0]
        self.assertEqual(page["cropbox"], [20, 40, 180, 260])
        self.assertEqual((page["image"]["width"], page["image"]["height"]), (220, 160))

    def test_mismatched_document_or_extraction_snapshot_is_rejected(self):
        source_path = Path(self.extraction["metadata_path"])
        original = source_path.read_text()
        changed = json.loads(original)
        changed["download"]["search"]["document"]["version"] = 9
        source_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "different document"):
            self.render()
        source_path.write_text(original + '\n')
        with self.assertRaisesRegex(ValueError, "exact successful extraction"):
            self.render()
        self.assertTrue(all(a["status"] == "failed" for a in self.attempts()))

    def test_wrong_chunk_page_text_or_duplicate_id_is_rejected(self):
        chunk_path = Path(self.chunks["metadata_path"])
        original = json.loads(chunk_path.read_text())
        for case in ("page", "text", "id"):
            changed = copy.deepcopy(original)
            if case == "page":
                changed["chunks"][0]["page_number"] = 99
            elif case == "text":
                changed["chunks"][0]["text"] = "SYNTHETIC wrong text"
            else:
                changed["chunks"][1]["chunk_id"] = changed["chunks"][0]["chunk_id"]
            chunk_path.write_text(json.dumps(changed))
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.render()

    def test_changed_source_hash_prevents_pdf_parsing(self):
        self.pdf.write_bytes(self.pdf.read_bytes() + b"SYNTHETIC change")
        with patch("render_pages.pymupdf.open") as opener:
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                self.render()
            opener.assert_not_called()

    def test_corrupt_pdf_has_failed_attempt_without_invented_page_count(self):
        self.pdf.write_bytes(b"%PDF-1.7 SYNTHETIC broken PDF")
        self.download["sha256"] = hashlib.sha256(self.pdf.read_bytes()).hexdigest()
        self.write_download()
        with self.assertRaises(ValueError):
            render_pages(self.metadata)
        result = self.attempts()[0]
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["page_count"])

    def test_failed_page_write_keeps_other_pages_and_can_retry(self):
        def fail_second(page, folder, config):
            if page.number == 1:
                raise OSError("SYNTHETIC page storage failure")
            return _render_page(page, folder, config)
        with patch("render_pages._render_page", side_effect=fail_second):
            partial = self.render()
        self.assertEqual((partial["status"], partial["rendered_count"], partial["failed_count"]), ("partial", 2, 1))
        self.assertTrue(Path(partial["pages"][1]["record_path"]).exists())
        self.assertIn("SYNTHETIC second", partial["pages"][1]["text"])
        first_asset = Path(partial["pages"][0]["image"]["local_path"])
        before = first_asset.stat().st_mtime_ns
        self.assertEqual(self.render()["status"], "rendered")
        self.assertEqual(before, first_asset.stat().st_mtime_ns)

    def test_page_record_storage_failure_is_visible_in_attempt(self):
        def fail_second_record(folder, record):
            if folder.name == "records" and record["page_number"] == 2:
                raise OSError("SYNTHETIC page record failure")
            return save_metadata(folder, record)
        with patch("render_pages.save_metadata", side_effect=fail_second_record):
            result = self.render()
        self.assertEqual(result["status"], "partial")
        self.assertIn("Page record storage failed", result["pages"][1]["error"])
        self.assertTrue(Path(result["metadata_path"]).is_file())
        self.assertEqual(self.render()["status"], "rendered")

    def test_attempt_storage_failure_preserves_page_records_for_retry(self):
        def fail_attempt(folder, record):
            if folder.name == "pages":
                raise OSError("SYNTHETIC attempt storage failure")
            return save_metadata(folder, record)
        with patch("render_pages.save_metadata", side_effect=fail_attempt), self.assertRaises(OSError):
            self.render()
        self.assertEqual(len(list((self.root / "pages" / "records").glob("*.json"))), 3)
        self.assertEqual(self.render()["status"], "rendered")

    def test_pixel_limit_prevents_allocation_but_records_each_page(self):
        with patch("render_pages.pymupdf.Page.get_pixmap") as render:
            result = self.render(max_pixels=10)
            render.assert_not_called()
        self.assertEqual((result["status"], result["failed_count"], len(result["pages"])), ("failed", 3, 3))

    def test_cli_returns_failure_for_unrendered_pages(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, "-m", "render_pages", "--metadata", str(self.metadata),
                                 "--max-pixels", "10"], cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stdout)["status"], "failed")

    def test_invalid_configuration_and_missing_metadata_are_rejected(self):
        for kwargs in ({"dpi": 0}, {"dpi": True}, {"dpi": 601}, {"max_pixels": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                render_pages(self.metadata, **kwargs)
        self.metadata.write_text("{}")
        with self.assertRaises(ValueError):
            render_pages(self.metadata)


if __name__ == "__main__":
    unittest.main()
