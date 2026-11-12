"""Prepare model-sized document inputs while preserving every parent source character."""

import hashlib
import json
import re

POLICY = "source_spans_v1"


def token_count(tokenizer, text):
    ids = tokenizer([text], add_special_tokens=True, padding=False, truncation=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError("Tokenizer must return one sequence for one input.")
    return len(ids[0])


def split_spans(text, tokenizer, max_tokens):
    """Prefer word ends; halve a single oversized word until its exact slice fits."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Document inputs require nonempty text.")
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("The token budget must be a positive integer.")
    original_tokens = token_count(tokenizer, text)
    if original_tokens <= max_tokens:
        return [(0, len(text), original_tokens, "whole")]

    spans = []
    start = best_end = 0
    for word in re.finditer(r"\S+\s*", text):
        end = word.end()
        if token_count(tokenizer, text[start:end]) <= max_tokens:
            best_end = end
            continue
        if best_end > start:
            spans.append((start, best_end, token_count(tokenizer, text[start:best_end]), "word"))
            start = best_end
        while token_count(tokenizer, text[start:end]) > max_tokens:
            piece = text[start:end]
            leading = len(piece) - len(piece.lstrip())
            body_length = len(piece) - leading
            cut = len(piece)
            # Token counts need not be monotonic in prefix length. Recheck each
            # candidate rather than assuming binary search finds a maximal span.
            while token_count(tokenizer, piece[:cut]) > max_tokens:
                body_length //= 2
                if body_length < 1:
                    raise ValueError("The token budget cannot encode even one source character.")
                cut = leading + body_length
            spans.append((start, start + cut, token_count(tokenizer, piece[:cut]), "character"))
            start += cut
        best_end = end
    if start < len(text):
        spans.append((start, len(text), token_count(tokenizer, text[start:]), "word"))
    if any(end <= start or count > max_tokens or not text[start:end].strip()
           for start, end, count, boundary in spans):
        raise ValueError("Invalid or oversized derived source span.")
    return spans


def prepare_documents(chunks, tokenizer, max_tokens, identity):
    """Create stable derived IDs; never mutate parent IDs, text, pages or offsets."""
    inputs, parents = [], []
    for chunk in chunks:
        text = chunk["text"]
        if chunk["end_char"] - chunk["start_char"] != len(text):
            raise ValueError("Parent text length does not match its source offsets.")
        spans = split_spans(text, tokenizer, max_tokens)
        cursor = 0
        parent_inputs = []
        for number, (start, end, count, boundary) in enumerate(spans, 1):
            if start != cursor:
                raise ValueError("Derived inputs must cover the parent without gaps or overlap.")
            segment = text[start:end]
            key = {"parent_chunk_id": chunk["id"], "identity": identity, "policy": POLICY,
                   "max_tokens": max_tokens, "start": start, "end": end, "text": segment}
            input_id = hashlib.sha256(json.dumps(key, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            derived = {**chunk, "id": input_id, "parent_chunk_id": chunk["id"], "input_number": number,
                       "parent_start_char": chunk["start_char"], "parent_end_char": chunk["end_char"],
                       "start_char": chunk["start_char"] + start, "end_char": chunk["start_char"] + end,
                       "text": segment, "tokens": count, "boundary": boundary}
            inputs.append(derived)
            parent_inputs.append(input_id)
            cursor = end
        if cursor != len(text):
            raise ValueError("Derived inputs do not cover the complete parent text.")
        parents.append({"parent_chunk_id": chunk["id"], "original_tokens": token_count(tokenizer, text),
                        "input_ids": parent_inputs, "input_count": len(spans)})
    return {"policy": POLICY, "items": inputs, "parents": parents,
            "parent_chunk_count": len(chunks), "input_count": len(inputs),
            "split_parent_count": sum(p["input_count"] > 1 for p in parents),
            "character_fallback_inputs": sum(row["boundary"] == "character" for row in inputs),
            "within_parent_overlap_characters": 0}
