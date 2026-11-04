"""Offline configuration tests and opt-in tests against the actual local PostgreSQL."""

import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

import psycopg

from database import check_connection, create_password_file


class DatabaseUnitTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.password_file = Path(self.directory.name) / "password.txt"

    def test_password_setup_preserves_existing_file(self):
        self.password_file.write_text("SYNTHETIC_EXISTING_PASSWORD\n")
        self.assertFalse(create_password_file(self.password_file))
        self.assertEqual(self.password_file.read_text(), "SYNTHETIC_EXISTING_PASSWORD\n")

    def test_new_password_is_private_and_not_rotated(self):
        self.assertTrue(create_password_file(self.password_file))
        original = self.password_file.read_bytes()
        self.assertTrue(original.strip())
        self.assertEqual(stat.S_IMODE(self.password_file.stat().st_mode), 0o600)
        self.assertFalse(create_password_file(self.password_file))
        self.assertEqual(self.password_file.read_bytes(), original)

    @patch("database.psycopg.connect")
    def test_missing_or_empty_password_prevents_connection(self, connect):
        with self.assertRaises(FileNotFoundError):
            check_connection(self.password_file)
        self.password_file.write_text("")
        with self.assertRaises(ValueError):
            check_connection(self.password_file)
        connect.assert_not_called()

    @patch("database.psycopg.connect")
    def test_connection_timeout_and_result_do_not_expose_password(self, connect):
        self.password_file.write_text("SYNTHETIC_TEST_PASSWORD\n")
        connection = connect.return_value.__enter__.return_value
        connection.execute.return_value.fetchone.return_value = {
            "database": "SYNTHETIC_TEST", "pgvector_version": "synthetic", "check_value": 1,
        }
        result = check_connection(self.password_file)
        self.assertEqual(connect.call_args.kwargs["connect_timeout"], 5)
        self.assertEqual(connect.call_args.kwargs["options"], "-c statement_timeout=5000")
        self.assertNotIn("SYNTHETIC_TEST_PASSWORD", str(result))
        self.assertTrue(connection.execute.call_args.args[0].strip().startswith("SELECT"))
        connect.return_value.__exit__.assert_called_once()

    @patch("database.psycopg.connect")
    def test_missing_vector_extension_is_rejected(self, connect):
        self.password_file.write_text("SYNTHETIC_TEST_PASSWORD\n")
        connect.return_value.__enter__.return_value.execute.return_value.fetchone.return_value = {
            "check_value": 1, "pgvector_version": None,
        }
        with self.assertRaisesRegex(RuntimeError, "vector extension"):
            check_connection(self.password_file)

    @patch("database.psycopg.connect", side_effect=psycopg.OperationalError("Synthetic connection failure"))
    def test_connection_failure_is_not_reported_as_success(self, connect):
        self.password_file.write_text("SYNTHETIC_TEST_PASSWORD\n")
        with self.assertRaises(psycopg.OperationalError):
            check_connection(self.password_file)


@unittest.skipUnless(os.environ.get("S4C_TEST_POSTGRES") == "1",
                     "Set S4C_TEST_POSTGRES=1 to run actual local PostgreSQL integration tests.")
class DatabaseIntegrationTests(unittest.TestCase):
    def test_actual_database_and_pgvector_connection(self):
        result = check_connection()
        self.assertEqual(result["database"], "search4cure")
        self.assertEqual(result["check_value"], 1)
        self.assertIsNotNone(result["pgvector_version"])
        self.assertGreaterEqual(result["public_table_count"], 0)

    def test_actual_database_rejects_incorrect_password(self):
        with tempfile.TemporaryDirectory() as directory:
            wrong_password = Path(directory) / "wrong-password.txt"
            wrong_password.write_text("SYNTHETIC_WRONG_PASSWORD\n")
            with self.assertRaises(psycopg.OperationalError) as failure:
                check_connection(wrong_password)
            self.assertIn("password authentication failed", str(failure.exception))


if __name__ == "__main__":
    unittest.main()
