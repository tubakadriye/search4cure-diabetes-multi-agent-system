"""Test metadata handling with explicitly synthetic responses and no network access."""

import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse

from aws_version.arxiv_search import parse_document, search_one

SYNTHETIC_FEED = b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>http://arxiv.org/abs/2401.12345v2</id><title>Synthetic test publication</title>
<published>2024-01-01T12:00:00Z</published><updated>2024-02-01T12:00:00Z</updated>
</entry></feed>'''


class ArxivSearchTests(unittest.TestCase):
    def test_version_and_distinct_source_timestamps(self):
        document = parse_document(SYNTHETIC_FEED)
        self.assertEqual(document["arxiv_id"], "2401.12345")
        self.assertEqual(document["version"], 2)
        self.assertNotEqual(document["first_submitted_at"], document["version_submitted_at"])

    def test_unknown_metadata_is_not_invented(self):
        feed = SYNTHETIC_FEED.replace(b"v2</id>", b"</id>")
        feed = feed.replace(b"<updated>2024-02-01T12:00:00Z</updated>", b"")
        document = parse_document(feed)
        self.assertIsNone(document["version"])
        self.assertIsNone(document["version_submitted_at"])

    def test_empty_and_invalid_entries(self):
        self.assertIsNone(parse_document(b'<feed xmlns="http://www.w3.org/2005/Atom"/>'))
        with self.assertRaisesRegex(ValueError, "Atom feed"):
            parse_document(b"<html><body>Synthetic service error</body></html>")
        with self.assertRaises(ValueError):
            parse_document(SYNTHETIC_FEED.replace(b"<title>Synthetic test publication</title>", b""))

    @patch("aws_version.arxiv_search.urlopen")
    def test_real_request_shape_limits_results_and_sets_timeout(self, opener):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = SYNTHETIC_FEED
        opener.return_value = response
        result = search_one("all:diabetes")
        request = opener.call_args.args[0]
        parameters = parse_qs(urlparse(request.full_url).query)
        self.assertEqual(parameters["max_results"], ["1"])
        self.assertEqual(parameters["search_query"], ["all:diabetes"])
        self.assertEqual(opener.call_args.kwargs["timeout"], 30)
        self.assertEqual(result["document"]["version"], 2)

    @patch("aws_version.arxiv_search.time.sleep")
    @patch("aws_version.arxiv_search.urlopen", side_effect=URLError("Synthetic connection failure"))
    def test_retries_stop_after_three_attempts(self, opener, sleep):
        with self.assertRaises(URLError):
            search_one("all:diabetes")
        self.assertEqual(opener.call_count, 3)
        self.assertEqual(sleep.call_args_list, [unittest.mock.call(3.1), unittest.mock.call(3.1)])

    @patch("aws_version.arxiv_search.time.sleep")
    @patch("aws_version.arxiv_search.urlopen")
    def test_rate_limit_and_retry_after_stop_automatic_retries(self, opener, sleep):
        for code, headers in [(429, {}), (503, {"Retry-After": "60"})]:
            with self.subTest(code=code):
                opener.reset_mock()
                opener.side_effect = HTTPError("https://example.invalid", code, "Synthetic HTTP error", headers, None)
                with self.assertRaises(HTTPError):
                    search_one("all:diabetes")
                self.assertEqual(opener.call_count, 1)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
