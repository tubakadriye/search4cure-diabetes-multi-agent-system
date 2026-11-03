"""Offline download tests with synthetic PDF bytes and real temporary files."""

from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from aws_version.download_pdf import download_pdf

SYNTHETIC_PDF = b"%PDF-1.7\nSynthetic download fixture, not a parseable research paper.\n%%EOF\n"
SEARCH_RESULT = {
    "query": "synthetic test query",
    "request_url": "https://example.invalid/synthetic-api-response",
    "retrieved_at": "2024-02-02T12:00:00+00:00",
    "document": {
        "arxiv_id": "2401.12345", "version": 2, "title": "Synthetic test publication",
        "source_url": "https://arxiv.org/abs/2401.12345v2",
        "first_submitted_at": "2024-01-01T12:00:00Z",
        "version_submitted_at": None,
    },
}


def response(body=SYNTHETIC_PDF, headers=None):
    stream = BytesIO(body)
    stream.headers = headers if headers is not None else {"Content-Length": str(len(body))}
    return stream


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def files(self, pattern):
        return [path for path in Path(self.directory.name).rglob(pattern) if path.is_file()]

    @patch("aws_version.download_pdf.urlopen")
    def test_saved_bytes_match_hash_and_preserve_provenance(self, opener):
        opener.return_value = response()
        result = download_pdf(SEARCH_RESULT, self.directory.name)
        self.assertEqual(Path(result["local_path"]).read_bytes(), SYNTHETIC_PDF)
        self.assertEqual(result["sha256"], hashlib.sha256(SYNTHETIC_PDF).hexdigest())
        self.assertEqual(result["size_bytes"], len(SYNTHETIC_PDF))
        stored = json.loads(Path(result["metadata_path"]).read_text())
        self.assertEqual(stored["search"], SEARCH_RESULT)
        self.assertIsNone(stored["search"]["document"]["version_submitted_at"])
        self.assertEqual(opener.call_args.args[0].full_url, "https://arxiv.org/pdf/2401.12345v2")
        self.assertEqual(opener.call_args.kwargs["timeout"], 30)
        self.assertEqual(self.files("*.part"), [])

    @patch("aws_version.download_pdf.urlopen")
    def test_missing_version_is_rejected_before_network_access(self, opener):
        metadata = deepcopy(SEARCH_RESULT)
        metadata["document"]["version"] = None
        with self.assertRaisesRegex(ValueError, "explicit arXiv version"):
            download_pdf(metadata, self.directory.name)
        opener.assert_not_called()

    @patch("aws_version.download_pdf.urlopen")
    def test_invalid_empty_and_oversized_responses_leave_no_completed_files(self, opener):
        for body, headers in [(b"<html>synthetic error</html>", {}), (b"", {}),
                              (SYNTHETIC_PDF, {"Content-Length": str(101 * 1024 * 1024)})]:
            with self.subTest(body=body[:20]):
                opener.return_value = response(body, headers)
                with self.assertRaises(ValueError):
                    download_pdf(SEARCH_RESULT, self.directory.name)
                self.assertEqual(self.files("*.*"), [])
        with patch("aws_version.download_pdf.MAX_PDF_BYTES", 10):
            opener.return_value = response(headers={})
            with self.assertRaisesRegex(ValueError, "limit"):
                download_pdf(SEARCH_RESULT, self.directory.name)
            self.assertEqual(self.files("*.*"), [])

    @patch("aws_version.download_pdf.time.sleep")
    @patch("aws_version.download_pdf.urlopen")
    def test_truncated_download_can_retry_without_partial_files(self, opener, sleep):
        opener.side_effect = [response(headers={"Content-Length": str(len(SYNTHETIC_PDF) + 10)}), response()]
        result = download_pdf(SEARCH_RESULT, self.directory.name)
        self.assertEqual(opener.call_count, 2)
        sleep.assert_called_once_with(3.1)
        self.assertEqual(Path(result["local_path"]).read_bytes(), SYNTHETIC_PDF)
        self.assertEqual(len(self.files("*.pdf")), 1)
        self.assertEqual(self.files("*.part"), [])

    @patch("aws_version.download_pdf.time.sleep")
    @patch("aws_version.download_pdf.urlopen", side_effect=URLError("Synthetic connection failure"))
    def test_connection_retries_stop_after_three_attempts(self, opener, sleep):
        with self.assertRaises(URLError):
            download_pdf(SEARCH_RESULT, self.directory.name)
        self.assertEqual(opener.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(self.files("*.*"), [])

    @patch("aws_version.download_pdf.time.sleep")
    @patch("aws_version.download_pdf.urlopen")
    def test_rate_limit_and_retry_after_stop_automatic_retries(self, opener, sleep):
        for code, headers in [(429, {}), (503, {"Retry-After": "60"})]:
            with self.subTest(code=code):
                opener.reset_mock()
                opener.side_effect = HTTPError("https://example.invalid", code, "Synthetic HTTP error", headers, None)
                with self.assertRaises(HTTPError):
                    download_pdf(SEARCH_RESULT, self.directory.name)
                self.assertEqual(opener.call_count, 1)
        sleep.assert_not_called()

    @patch("aws_version.download_pdf.urlopen")
    def test_metadata_storage_failure_keeps_pdf_and_allows_rerun(self, opener):
        opener.side_effect = [response(), response()]
        with patch("aws_version.download_pdf.json.dump", side_effect=OSError("Synthetic storage failure")):
            with self.assertRaises(OSError):
                download_pdf(SEARCH_RESULT, self.directory.name)
        self.assertEqual(len(self.files("*.pdf")), 1)
        self.assertEqual(self.files("*.json"), [])
        self.assertEqual(self.files("*.part"), [])
        result = download_pdf(SEARCH_RESULT, self.directory.name)
        self.assertEqual(result["status"], "downloaded")
        self.assertEqual(len(self.files("*.pdf")), 1)
        self.assertEqual(len(self.files("*.json")), 1)


if __name__ == "__main__":
    unittest.main()
