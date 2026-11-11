"""Synthetic boundary/failure tests; real cached-model checks require explicit opt-in."""

from contextlib import redirect_stdout
from io import StringIO
import json
import math
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from embedding_comparison import (checked_encode, compare_models, load_corpus,
                                  load_questions, main, rank_pages, validate_vectors)
from test_store_chunks import ChunkStorageFixture


SPEC = {"name": "SYNTHETIC", "model_id": "SYNTHETIC_NOT_A_REAL_MODEL", "revision": "0" * 40,
        "dimension": 3, "max_seq_length": 6, "query_prefix": "SYNTHETIC PROMPT "}


def item(identifier, text="synthetic text", page=1):
    return {"id": identifier, "text": text, "arxiv_id": "SYNTHETIC", "version": 1,
            "pdf_sha256": "0" * 64, "page_number": page, "title": "SYNTHETIC test only",
            "source_url": "https://example.invalid/SYNTHETIC", "start_char": 0, "end_char": len(text)}


class SyntheticModel:
    """Word-count tokenizer plus sentinel vectors, never scientific model output."""
    def __init__(self, spec):
        self.max_seq_length = spec["max_seq_length"]
        self.dimension = spec["dimension"]
        self.calls = []

    def tokenizer(self, texts, **kwargs):
        assert kwargs == {"add_special_tokens": True, "padding": False, "truncation": False}
        return {"input_ids": [list(range(len(text.split()) + 2)) for text in texts]}

    def get_embedding_dimension(self):
        return self.dimension

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return [[1.0] + [0.0] * (self.dimension - 1) for text in texts]


class EncodingBoundaryTests(unittest.TestCase):
    def test_exact_limit_is_kept_and_oversize_never_reaches_encoder(self):
        model = SyntheticModel(SPEC)
        rows = [item("boundary", "one two three four"), item("oversize", "one two three four five")]
        result = checked_encode(model, rows, SPEC, "document", 2)
        self.assertEqual([r["id"] for r in result["items"]], ["boundary"])
        self.assertEqual(result["rejected"], [{"id": "oversize", "tokens": 7, "limit": 6, "reason": "token_limit"}])
        self.assertEqual(model.calls[0][0], ["one two three four"])
        self.assertEqual(result["encode_api_calls"], 1)

    def test_query_prefix_counts_toward_limit_and_is_not_added_to_documents(self):
        model = SyntheticModel(SPEC)
        row = item("question", "one two three")
        documents = checked_encode(model, [row], SPEC, "document", 2)
        queries = checked_encode(model, [row], SPEC, "query", 2)
        self.assertEqual(documents["token_counts"][0]["tokens"], 5)
        self.assertEqual(queries["rejected"][0]["tokens"], 7)
        self.assertEqual(len(model.calls), 1)

    def test_prefix_is_applied_once_and_implicit_prompts_are_disabled(self):
        model = SyntheticModel(SPEC)
        result = checked_encode(model, [item("q", "one")], SPEC, "query", 2)
        texts, kwargs = model.calls[0]
        self.assertEqual(texts, ["SYNTHETIC PROMPT one"])
        self.assertEqual(kwargs["prompt"], "")
        self.assertTrue(kwargs["normalize_embeddings"])
        self.assertEqual(kwargs["precision"], "float32")
        self.assertEqual(result["encoded_text_count"], 1)

    def test_empty_and_oversize_only_batches_make_no_encode_call(self):
        model = SyntheticModel(SPEC)
        result = checked_encode(model, [item("empty", " "), item("long", "word " * 10)], SPEC, "document", 2)
        self.assertEqual(result["vectors"], [])
        self.assertEqual(result["encode_api_calls"], 0)
        self.assertEqual({r["reason"] for r in result["rejected"]}, {"empty_text", "token_limit"})
        self.assertEqual(model.calls, [])

    def test_bad_vector_shape_values_and_normalization_are_rejected(self):
        invalid = [[], [[1.0, 0.0]], [[float("nan"), 0.0, 0.0]],
                   [[float("inf"), 0.0, 0.0]], [["1", 0, 0]], [[True, 0, 0]],
                   [[0.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]]
        for vectors in invalid:
            with self.subTest(vectors=vectors), self.assertRaises(ValueError):
                validate_vectors(vectors, 1, 3)
        self.assertEqual(validate_vectors([[1.0, 0.0, 0.0]], 1, 3), [[1.0, 0.0, 0.0]])

    def test_ranked_pages_keep_exact_source_and_best_chunk_without_duplicate_page(self):
        rows = [item("lower", page=1), item("best", page=1), item("second", page=2)]
        result = rank_pages(rows, [[0.8, 0.6, 0], [1, 0, 0], [0.6, 0.8, 0]], [1, 0, 0])
        self.assertEqual([r["id"] for r in result], ["best", "second"])
        self.assertEqual(result[0]["source_url"], rows[1]["source_url"])
        self.assertEqual(result[0]["pdf_sha256"], rows[1]["pdf_sha256"])
        with self.assertRaises(ValueError):
            rank_pages(rows[:1], [[1, 0]], [1, 0, 0])

    @patch("embedding_comparison.load_cpu_model")
    def test_unpinned_and_duplicate_model_specs_fail_before_loading(self, load):
        for specs in ([{**SPEC, "revision": "main"}], [SPEC, SPEC], [{**SPEC, "dimension": True}]):
            with self.subTest(specs=specs), self.assertRaises(ValueError):
                compare_models({}, [], specs)
        load.assert_not_called()

    def test_failed_model_preserves_other_output_and_can_be_retried(self):
        corpus = {"chunks": [item("doc")], "sha256": "SYNTHETIC"}
        questions = [item("q", "one")]
        specs = [SPEC, {**SPEC, "name": "SYNTHETIC_SECOND", "revision": "1" * 40}]
        def failing(spec, *args):
            if spec["name"] == "SYNTHETIC_SECOND":
                raise OSError("SYNTHETIC CACHE FAILURE")
            return SyntheticModel(spec)
        result = compare_models(corpus, questions, specs, model_factory=failing)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(len(result["models"][0]["documents"]["vectors"]), 1)
        self.assertEqual(result["models"][1]["status"], "failed")
        self.assertIsNone(result["quality_metrics"])
        self.assertEqual(result["execution_kind"], "synthetic_test_encoder")
        retry = compare_models(corpus, questions, specs, model_factory=lambda spec, *args: SyntheticModel(spec))
        self.assertEqual(retry["status"], "complete")
        self.assertNotEqual(retry["models"][0]["configuration_id"], retry["models"][1]["configuration_id"])

    def test_loaded_model_dimension_mismatch_is_reported(self):
        def wrong(spec, *args):
            return SyntheticModel({**spec, "dimension": 2})
        result = compare_models({"chunks": [item("d")]}, [item("q")], [SPEC], model_factory=wrong)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["models"][0]["error_type"], "ValueError")


class CorpusAndSplitTests(ChunkStorageFixture):
    def test_equivalent_source_attempts_deduplicate_without_modifying_files(self):
        before = self.chunk_path.read_bytes()
        copy_path = self.root / "SYNTHETIC-equivalent-attempt.json"
        copy_path.write_bytes(before)
        first = load_corpus([self.chunk_path])
        repeated = load_corpus([copy_path, self.chunk_path])
        self.assertEqual(first["sha256"], repeated["sha256"])
        self.assertEqual(repeated["duplicate_chunks_removed"], len(first["chunks"]))
        self.assertEqual(before, self.chunk_path.read_bytes())

    def test_changed_source_chunk_is_rejected(self):
        self.chunk_record["chunks"][0]["text"] += " SYNTHETIC alteration"
        self.save_chunks(self.chunk_record)
        with self.assertRaises(ValueError):
            load_corpus([self.chunk_path])

    def test_question_splits_are_explicit_and_bad_ids_fail(self):
        path = self.root / "SYNTHETIC-questions.json"
        rows = [{"id": "dev", "question": "synthetic query", "split": "development", "label_status": "unreviewed"},
                {"id": "hold", "question": "synthetic heldout", "split": "heldout", "label_status": "unreviewed"}]
        path.write_text(json.dumps({"questions": rows}))
        self.assertEqual([q["id"] for q in load_questions(path)], ["dev"])
        self.assertEqual([q["id"] for q in load_questions(path, "heldout")], ["hold"])
        rows[1]["id"] = "dev"
        path.write_text(json.dumps({"questions": rows}))
        with self.assertRaises(ValueError):
            load_questions(path)

    def test_cli_creates_new_report_directory_and_preserves_source(self):
        questions = self.root / "SYNTHETIC-questions.json"
        questions.write_text(json.dumps({"questions": [{"id": "dev", "question": "synthetic query", "split": "development"}]}))
        models = self.root / "SYNTHETIC-models.json"
        models.write_text(json.dumps([SPEC]))
        output = self.root / "new-reports"
        before = self.chunk_path.read_bytes()
        result = {"status": "complete", "execution_kind": "synthetic_test_encoder", "models": [{"name": "SYNTHETIC", "status": "complete"}]}
        argv = ["embedding_comparison", "--chunks", str(self.chunk_path), "--questions", str(questions),
                "--models", str(models), "--output-dir", str(output)]
        with patch("sys.argv", argv), patch("embedding_comparison.compare_models", return_value=result), redirect_stdout(StringIO()) as stream:
            self.assertEqual(main(), 0)
        saved = json.loads(Path(json.loads(stream.getvalue())["report_path"]).read_text())
        self.assertEqual(saved, result)
        self.assertEqual(self.chunk_path.read_bytes(), before)


@unittest.skipUnless(os.getenv("S4C_TEST_EMBEDDINGS") == "1", "Set S4C_TEST_EMBEDDINGS=1 after caching real model weights.")
class CachedModelTests(unittest.TestCase):
    def test_real_cpu_models_validate_shapes_and_reject_oversize_input_offline(self):
        specs = json.loads(Path("evaluation/models.json").read_text())
        corpus = {"chunks": [item("SYNTHETIC-short", "Diabetic retinopathy grading"),
                             item("SYNTHETIC-long", "diabetes " * 600)]}
        result = compare_models(corpus, [item("SYNTHETIC-query", "retinopathy grading")], specs, offline=True)
        self.assertEqual(result["status"], "partial")
        for outcome in result["models"]:
            self.assertEqual(outcome["status"], "partial")
            self.assertEqual(outcome["documents"]["rejected"][0]["id"], "SYNTHETIC-long")
            vector = outcome["documents"]["vectors"][0]
            self.assertEqual(len(vector), 384)
            self.assertTrue(all(math.isfinite(v) for v in vector))
            self.assertAlmostEqual(sum(v*v for v in vector), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
