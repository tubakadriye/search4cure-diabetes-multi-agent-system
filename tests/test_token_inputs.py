"""Source-coverage tests use synthetic text; real MiniLM inference is opt-in and offline."""

import copy
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from embedding_comparison import compare_models, main
from test_embedding_comparison import SPEC, SyntheticModel, item
from test_store_chunks import ChunkStorageFixture
from token_inputs import prepare_documents, split_spans, token_count

IDENTITY = {"model_id": "SYNTHETIC", "revision": "0" * 40, "tokenizer_sha256": None}


class CharacterTokenizer:
    """Synthetic tokenizer: one token per non-whitespace character plus two specials."""
    def __call__(self, texts, **kwargs):
        assert kwargs == {"add_special_tokens": True, "padding": False, "truncation": False}
        return {"input_ids": [list(range(sum(not c.isspace() for c in text) + 2)) for text in texts]}


class SourceSpanTests(unittest.TestCase):
    def setUp(self):
        self.model = SyntheticModel(SPEC)

    def test_fitting_parent_stays_whole_with_a_separate_stable_input_id(self):
        parent = item("PARENT", "alpha beta")
        result = prepare_documents([parent], self.model.tokenizer, 6, IDENTITY)
        child = result["items"][0]
        self.assertEqual(child["text"], parent["text"])
        self.assertEqual(child["parent_chunk_id"], "PARENT")
        self.assertNotEqual(child["id"], "PARENT")
        self.assertEqual(result["split_parent_count"], 0)

    def test_word_splits_preserve_unicode_whitespace_and_absolute_page_offsets(self):
        text = "  alpha βeta  gamma\nδelta epsilon ζeta eta  "
        parent = {**item("PARENT", text, page=7), "start_char": 20, "end_char": 20 + len(text)}
        before = copy.deepcopy(parent)
        result = prepare_documents([parent], self.model.tokenizer, 5, IDENTITY)
        children = result["items"]
        self.assertEqual(len(children), 3)
        self.assertEqual("".join(c["text"] for c in children), text)
        cursor = parent["start_char"]
        for child in children:
            self.assertEqual(child["start_char"], cursor)
            self.assertEqual(child["text"], text[child["start_char"]-20:child["end_char"]-20])
            self.assertEqual(child["page_number"], 7)
            self.assertEqual(child["pdf_sha256"], parent["pdf_sha256"])
            self.assertLessEqual(token_count(self.model.tokenizer, child["text"]), 5)
            cursor = child["end_char"]
        self.assertEqual(cursor, parent["end_char"])
        self.assertEqual(parent, before)

    def test_single_oversized_word_has_a_checked_character_fallback(self):
        text = "  αβγδεζηθικλμνξοπρστυφχψω  "
        result = prepare_documents([item("LONG-WORD", text)], CharacterTokenizer(), 7, IDENTITY)
        self.assertGreater(result["character_fallback_inputs"], 0)
        self.assertEqual("".join(c["text"] for c in result["items"]), text)
        self.assertTrue(all(c["text"].strip() and c["tokens"] <= 7 for c in result["items"]))

    def test_ids_are_repeatable_but_change_with_budget_or_tokenizer_identity(self):
        parent = item("PARENT", "one two three four five six seven")
        first = prepare_documents([parent], self.model.tokenizer, 6, IDENTITY)
        repeated = prepare_documents([{**parent, "attempt_path": "different-attempt.json"}], self.model.tokenizer, 6, IDENTITY)
        smaller = prepare_documents([parent], self.model.tokenizer, 5, IDENTITY)
        other_tokenizer = prepare_documents([parent], self.model.tokenizer, 6, {**IDENTITY, "tokenizer_sha256": "1" * 64})
        ids = lambda result: {c["id"] for c in result["items"]}
        self.assertEqual(ids(first), ids(repeated))
        self.assertTrue(ids(first).isdisjoint(ids(smaller)))
        self.assertTrue(ids(first).isdisjoint(ids(other_tokenizer)))

    def test_parent_overlap_is_preserved_without_merging_parents_or_pages(self):
        parents = [item("A", "one two three four five", page=1),
                   {**item("B", "four five six seven eight", page=2), "version": 2}]
        result = prepare_documents(parents, self.model.tokenizer, 5, IDENTITY)
        for parent in parents:
            children = [c for c in result["items"] if c["parent_chunk_id"] == parent["id"]]
            self.assertEqual("".join(c["text"] for c in children), parent["text"])
            self.assertTrue(all(c["page_number"] == parent["page_number"] and c["version"] == parent["version"] for c in children))

    def test_invalid_offsets_empty_text_and_impossible_budget_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "offsets"):
            prepare_documents([{**item("A"), "end_char": 999}], self.model.tokenizer, 6, IDENTITY)
        for text in ("", "   "):
            with self.subTest(text=text), self.assertRaises(ValueError):
                split_spans(text, self.model.tokenizer, 6)
        for limit in (0, True, 2):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                split_spans("word", self.model.tokenizer, limit)

    def test_split_policy_adds_coverage_without_changing_parent_corpus(self):
        corpus = {"chunks": [item("PARENT", "one two three four five six seven")]}
        questions = [item("QUESTION", "one")]
        before = copy.deepcopy(corpus)
        factory = lambda spec, *args: SyntheticModel(spec)
        rejected = compare_models(corpus, questions, [SPEC], model_factory=factory)
        split = compare_models(corpus, questions, [SPEC], model_factory=factory, document_policy="split")
        self.assertEqual(rejected["status"], "partial")
        self.assertEqual(split["status"], "complete")
        model = split["models"][0]
        self.assertEqual(len(model["documents"]["vectors"]), 2)
        self.assertEqual(model["input_preparation"]["parent_chunk_count"], 1)
        self.assertEqual(model["results"][0]["pages"][0]["parent_chunk_id"], "PARENT")
        self.assertNotEqual(model["configuration_id"], rejected["models"][0]["configuration_id"])
        self.assertEqual(corpus, before)
        self.assertIsNone(split["quality_metrics"])

    def test_document_policy_never_splits_an_oversized_question(self):
        result = compare_models({"chunks": [item("D", "one two three four five")]},
                                [item("Q", "one two three")], [SPEC],
                                model_factory=lambda spec, *args: SyntheticModel(spec), document_policy="split")
        outcome = result["models"][0]
        self.assertEqual(outcome["queries"]["rejected"][0]["id"], "Q")
        self.assertEqual(outcome["queries"]["encode_api_calls"], 0)
        self.assertEqual(outcome["status"], "partial")


class PolicyCliTests(ChunkStorageFixture):
    def test_cli_selects_one_model_and_passes_the_split_policy(self):
        registry = self.root / "SYNTHETIC-models.json"
        registry.write_text(json.dumps([SPEC, {**SPEC, "name": "SECOND"}]))
        questions = self.root / "SYNTHETIC-questions.json"
        questions.write_text(json.dumps({"questions": [{"id": "Q", "question": "one", "split": "development"}]}))
        argv = ["embedding_comparison", "--chunks", str(self.chunk_path), "--models", str(registry),
                "--model", "SECOND", "--document-policy", "split", "--questions", str(questions),
                "--output-dir", str(self.root / "reports")]
        result = {"status": "complete", "models": [{"name": "SECOND", "status": "complete"}]}
        with patch("sys.argv", argv), patch("embedding_comparison.compare_models", return_value=result) as compare, redirect_stdout(StringIO()):
            self.assertEqual(main(), 0)
        self.assertEqual([s["name"] for s in compare.call_args.args[2]], ["SECOND"])
        self.assertEqual(compare.call_args.kwargs["document_policy"], "split")

    def test_unknown_model_fails_before_source_or_model_loading(self):
        registry = self.root / "SYNTHETIC-models.json"
        registry.write_text(json.dumps([SPEC]))
        argv = ["embedding_comparison", "--chunks", "SYNTHETIC-MISSING.json", "--models", str(registry), "--model", "UNKNOWN"]
        with patch("sys.argv", argv), patch("embedding_comparison.load_corpus") as load, redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
            main()
        self.assertEqual(error.exception.code, 2)
        load.assert_not_called()


@unittest.skipUnless(os.getenv("S4C_TEST_EMBEDDINGS") == "1", "Set S4C_TEST_EMBEDDINGS=1 for cached real MiniLM checks.")
class CachedSplitTests(unittest.TestCase):
    def test_real_minilm_encodes_all_derived_inputs_without_truncation(self):
        spec = next(s for s in json.loads(Path("evaluation/models.json").read_text()) if s["name"] == "minilm")
        parent = item("SYNTHETIC-PARENT", "diabetes " * 400)
        result = compare_models({"chunks": [parent]}, [item("SYNTHETIC-Q", "diabetes")], [spec],
                                offline=True, document_policy="split")
        self.assertEqual(result["status"], "complete")
        model = result["models"][0]
        self.assertEqual(model["input_preparation"]["split_parent_count"], 1)
        self.assertEqual(model["documents"]["rejected"], [])
        self.assertEqual("".join(c["text"] for c in model["documents"]["items"]), parent["text"])
        self.assertTrue(all(len(v) == 384 for v in model["documents"]["vectors"]))
        self.assertTrue(all(c["tokens"] <= 256 for c in model["documents"]["items"]))


if __name__ == "__main__":
    unittest.main()
