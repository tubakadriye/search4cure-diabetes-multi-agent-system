"""Compare pinned local text encoders on validated chunks; no quality claims or SQL writes."""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import distributions
import json
import logging
import math
import os
from pathlib import Path
import platform
import re
import time

from download_pdf import save_metadata
from store_chunks import validate_chunks
from token_inputs import POLICY, prepare_documents


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_corpus(paths):
    """Freeze validated real source windows, deduplicating equivalent attempts by chunk ID."""
    rows, sources, versions = {}, [], {}
    duplicates = 0
    for path in paths:
        path = Path(path)
        chunks, source, _ = validate_chunks(path)
        document = source["download"]["search"]["document"]
        if not document.get("title", "").strip() or not document.get("source_url", "").strip():
            raise ValueError("Corpus documents require a title and source URL.")
        key = document["arxiv_id"], document["version"]
        pdf_hash = chunks["source_pdf_sha256"]
        if key in versions and versions[key] != pdf_hash:
            raise ValueError("Conflicting PDF hashes for one document version.")
        versions[key] = pdf_hash
        sources.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "arxiv_id": key[0], "version": key[1], "pdf_sha256": pdf_hash})
        for chunk in chunks["chunks"]:
            row = {"id": chunk["chunk_id"], "text": chunk["text"], "arxiv_id": chunk["arxiv_id"],
                   "version": chunk["version"], "pdf_sha256": pdf_hash, "page_number": chunk["page_number"],
                   "start_char": chunk["start_char"], "end_char": chunk["end_char"],
                   "title": document["title"], "source_url": document["source_url"]}
            if row["id"] in rows:
                if rows[row["id"]] != row:
                    raise ValueError("Conflicting metadata for a repeated chunk ID.")
                duplicates += 1
            rows[row["id"]] = row
    if not rows:
        raise ValueError("The comparison requires nonempty validated chunks.")
    ordered = sorted(rows.values(), key=lambda row: row["id"])
    return {"chunks": ordered, "sha256": fingerprint(ordered), "sources": sources,
            "duplicate_chunks_removed": duplicates}


def load_questions(path, split="development"):
    """Select one explicit split; relevance labels are retained but never scored here."""
    if split not in ("development", "heldout"):
        raise ValueError("Unknown question split.")
    all_rows = json.loads(Path(path).read_text(encoding="utf-8"))["questions"]
    selected, seen = [], set()
    for row in all_rows:
        if not isinstance(row.get("id"), str) or not row["id"].strip() or row["id"] in seen:
            raise ValueError("Question IDs must be nonempty and unique.")
        seen.add(row["id"])
        if row.get("split") not in ("development", "heldout"):
            raise ValueError("Every question requires a valid split.")
        if row["split"] == split:
            if not isinstance(row.get("question"), str) or not row["question"].strip():
                raise ValueError("Question text must not be empty.")
            selected.append({**row, "text": row["question"]})
    if not selected:
        raise ValueError("No questions in the selected split.")
    return selected


def validate_specs(specs):
    seen = set()
    for spec in specs:
        if not spec.get("name") or spec["name"] in seen or not spec.get("model_id"):
            raise ValueError("Models require unique names and a model ID.")
        seen.add(spec["name"])
        if not re.fullmatch(r"[0-9a-f]{40}", spec.get("revision", "")):
            raise ValueError("Pin each model to a full immutable revision.")
        if any(type(spec.get(key)) is not int or spec[key] < 1 for key in ("dimension", "max_seq_length")):
            raise ValueError("Dimensions and token limits must be positive integers.")
        if not isinstance(spec.get("query_prefix"), str):
            raise ValueError("Record an explicit query prefix, including an empty one.")
    if not seen:
        raise ValueError("At least one model is required.")


def validate_vectors(vectors, count, dimension):
    """Reject wrong shapes, nonnumeric/nonfinite values and non-unit vectors."""
    vectors = vectors.tolist() if hasattr(vectors, "tolist") else vectors
    if not isinstance(vectors, list) or len(vectors) != count:
        raise ValueError("Embedding row count differs from the accepted input count.")
    for row in vectors:
        if not isinstance(row, list) or len(row) != dimension:
            raise ValueError("Wrong embedding dimension.")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in row):
            raise ValueError("Embeddings must contain finite numeric values.")
        if not math.isclose(math.fsum(v * v for v in row), 1.0, abs_tol=1e-4):
            raise ValueError("Expected normalized nonzero embeddings.")
    return vectors


def checked_encode(model, items, spec, role, batch_size):
    """Count the exact prompted sequence, including special tokens, before encoding."""
    if role not in ("document", "query") or type(batch_size) is not int or batch_size < 1:
        raise ValueError("Invalid encoding role or batch size.")
    texts = [(spec["query_prefix"] if role == "query" else "") + item["text"] for item in items]
    started = time.perf_counter()
    ids = model.tokenizer(texts, add_special_tokens=True, padding=False, truncation=False)["input_ids"]
    if len(ids) != len(items):
        raise ValueError("Tokenizer row count differs from the input count.")
    accepted, rejected, lengths, inputs = [], [], [], []
    for item, text, tokens in zip(items, texts, ids):
        record = {"id": item["id"], "tokens": len(tokens), "limit": spec["max_seq_length"]}
        lengths.append(record)
        if not item["text"].strip() or len(tokens) > spec["max_seq_length"]:
            rejected.append({**record, "reason": "empty_text" if not item["text"].strip() else "token_limit"})
        else:
            accepted.append(item)
            inputs.append(text)
    token_seconds = time.perf_counter() - started
    started = time.perf_counter()
    vectors = model.encode(inputs, batch_size=batch_size, prompt="", normalize_embeddings=True,
                           convert_to_numpy=True, precision="float32", show_progress_bar=False) if inputs else []
    encode_seconds = time.perf_counter() - started
    vectors = validate_vectors(vectors, len(accepted), spec["dimension"])
    return {"items": accepted, "vectors": vectors, "rejected": rejected, "token_counts": lengths,
            "token_seconds": token_seconds, "encode_seconds": encode_seconds,
            "encode_api_calls": int(bool(inputs)), "encoded_text_count": len(inputs)}


def rank_pages(chunks, vectors, query_vector, top_k=5):
    """Exact normalized dot product; retain the best chunk per distinct source page."""
    if len(chunks) != len(vectors) or type(top_k) is not int or top_k < 1:
        raise ValueError("Invalid ranking inputs.")
    pages = {}
    for chunk, vector in zip(chunks, vectors):
        if len(vector) != len(query_vector):
            raise ValueError("Cannot compare different vector dimensions.")
        score = math.fsum(a * b for a, b in zip(vector, query_vector))
        key = chunk["arxiv_id"], chunk["version"], chunk["pdf_sha256"], chunk["page_number"]
        candidate = {**chunk, "similarity": score}
        previous = pages.get(key)
        if previous is None or (-score, chunk["id"]) < (-previous["similarity"], previous["id"]):
            pages[key] = candidate
    return sorted(pages.values(), key=lambda row: (-row["similarity"], row["id"]))[:top_k]


def load_cpu_model(spec, cache, offline, threads):
    """Load one pinned public model; all downloads/caches stay in the project data tree."""
    cache = Path(cache).resolve()
    os.environ.setdefault("HF_HOME", str(cache.parent))
    os.environ.setdefault("HF_XET_CACHE", str(cache.parent / "xet"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(threads)
    return SentenceTransformer(spec["model_id"], revision=spec["revision"], device="cpu",
                               cache_folder=str(cache), token=False, trust_remote_code=False,
                               local_files_only=offline, model_kwargs={"use_safetensors": True})


def compare_models(corpus, questions, specs, *, cache="data/models/hub", offline=False,
                   batch_size=16, threads=2, model_factory=None, document_policy="reject"):
    """Keep model spaces and failures separate; report observations without quality metrics."""
    validate_specs(specs)
    if document_policy not in ("reject", "split"):
        raise ValueError("Unknown document input policy.")
    if type(threads) is not int or threads < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("Thread and batch counts must be positive integers.")
    runtime = dict(sorted((d.metadata["Name"], d.version) for d in distributions()))
    result = {"started_at": datetime.now(timezone.utc).isoformat(), "corpus": corpus, "questions": questions,
              "execution_kind": "synthetic_test_encoder" if model_factory else "local_model_inference",
              "python": platform.python_version(), "platform": platform.platform(), "runtime": runtime,
              "offline": offline, "models": [], "quality_metrics": None,
              "quality_note": "Relevance is unscored; similarity is not accuracy or answer confidence."}
    factory = model_factory or load_cpu_model
    for spec in specs:
        model = None
        configuration = {**spec, "device": "cpu", "precision": "float32", "normalize_embeddings": True,
                         "batch_size": batch_size, "threads": threads, "runtime": runtime,
                         "oversize_policy": "reject_before_encode" if document_policy == "reject" else POLICY,
                         "document_prefix": ""}
        outcome = {"configuration": configuration, "configuration_id": fingerprint(configuration), "name": spec["name"]}
        logging.info("Loading %s at revision %s", spec["name"], spec["revision"])
        try:
            started = time.perf_counter()
            model = factory(spec, cache, offline, threads)
            outcome["load_seconds"] = time.perf_counter() - started
            if model.max_seq_length != spec["max_seq_length"] or model.get_embedding_dimension() != spec["dimension"]:
                raise ValueError("Loaded model dimensions/sequence limit differ from the pinned configuration.")
            backend = getattr(model.tokenizer, "backend_tokenizer", None)
            outcome["tokenizer_sha256"] = hashlib.sha256(backend.to_str().encode()).hexdigest() if backend else None
            inputs = corpus["chunks"]
            if document_policy == "split":
                started = time.perf_counter()
                preparation = prepare_documents(inputs, model.tokenizer, spec["max_seq_length"], {
                    "model_id": spec["model_id"], "revision": spec["revision"],
                    "tokenizer_sha256": outcome["tokenizer_sha256"],
                })
                inputs = preparation["items"]
                outcome["input_preparation_seconds"] = time.perf_counter() - started
                outcome["input_preparation"] = {key: value for key, value in preparation.items() if key != "items"}
            logging.info("Checking and encoding %s document inputs with %s", len(inputs), spec["name"])
            documents = checked_encode(model, inputs, spec, "document", batch_size)
            queries = checked_encode(model, questions, spec, "query", batch_size)
            outcome.update(documents=documents, queries=queries, results=[])
            for question, vector in zip(queries["items"], queries["vectors"]):
                outcome["results"].append({"question_id": question["id"], "pages": rank_pages(
                    documents["items"], documents["vectors"], vector)})
            outcome["status"] = "partial" if documents["rejected"] or queries["rejected"] else "complete"
            logging.info("%s: encoded=%s rejected=%s encode_api_calls=%s", spec["name"],
                         documents["encoded_text_count"] + queries["encoded_text_count"],
                         len(documents["rejected"]) + len(queries["rejected"]),
                         documents["encode_api_calls"] + queries["encode_api_calls"])
        except Exception as error:
            outcome.update(status="failed", error_type=type(error).__name__,
                           error=str(error) if isinstance(error, ValueError) else "Model operation failed; inspect the runtime/cache and rerun.")
            logging.warning("%s failed (%s)", spec["name"], type(error).__name__)
        finally:
            del model
            gc.collect()
        result["models"].append(outcome)
    result["status"] = "complete" if all(m["status"] == "complete" for m in result["models"]) else "partial" if any(m["status"] != "failed" for m in result["models"]) else "failed"
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", action="append", required=True, help="Validated chunk JSON; repeat for another document")
    parser.add_argument("--models", default="evaluation/models.json")
    parser.add_argument("--model", action="append", help="Select a registered model name; repeat to select several")
    parser.add_argument("--document-policy", choices=("reject", "split"), default="reject")
    parser.add_argument("--questions", default="evaluation/questions.json")
    parser.add_argument("--split", choices=("development", "heldout"), default="development")
    parser.add_argument("--cache", default="data/models/hub")
    parser.add_argument("--offline", action="store_true", help="Require previously downloaded model snapshots")
    parser.add_argument("--output-dir", default="data/embedding_comparison/runs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    specs = json.loads(Path(args.models).read_text())
    validate_specs(specs)
    if args.model:
        if set(args.model) - {spec["name"] for spec in specs}:
            parser.error("Unknown model name; inspect the model registry.")
        specs = [spec for spec in specs if spec["name"] in args.model]
    corpus = load_corpus(args.chunks)
    questions = load_questions(args.questions, args.split)
    result = compare_models(corpus, questions, specs, cache=args.cache, offline=args.offline,
                            document_policy=args.document_policy)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = save_metadata(output, result)
    print(json.dumps({"status": result["status"], "corpus_sha256": corpus["sha256"],
                      "chunk_count": len(corpus["chunks"]), "report_path": str(path),
                      "models": [{"name": m["name"], "status": m["status"]} for m in result["models"]]}, indent=2))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
