"""Passage documents and the spans Antfly's extractor reports back on graph edges."""
from __future__ import annotations

import re


def character_span(text, span):
    """Convert an extractor span (UTF-8 byte offsets) to exact character offsets."""
    encoded = text.encode("utf-8")
    start, end = span.get("start"), span.get("end")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(encoded):
        raise ValueError("Extractor span is outside its passage")
    try:
        prefix, body = encoded[:start].decode("utf-8"), encoded[start:end].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Extractor span is not on UTF-8 character boundaries") from exc
    if body != span.get("text"):
        raise ValueError("Extractor span does not match its passage text")
    return {"text": body, "start": len(prefix), "end": len(prefix) + len(body)}


def passage_documents(prepared):
    """Only the passage text and its exact source provenance; Antfly derives the rest."""
    return {p["id"]: {**{k: v for k, v in p.items() if k != "id"}, "kind": "passage", "passage_id": p["id"]}
            for p in prepared["passages"]}


def search_request(cfg, text, semantic=True, allowed=None):
    terms = sorted(set(re.findall(r"\w+", text, flags=re.UNICODE)))
    filters = [{"term": "passage", "field": "kind"}]
    if allowed is not None:
        filters.append({"disjuncts": [{"term": key, "field": "passage_id"} for key in sorted(allowed)]})
    request = {"full_text_search": {"disjuncts": [{"match": term, "field": "text"} for term in terms]},
               "filter_query": {"conjuncts": filters}, "limit": cfg["retrieval"]["candidates"]}
    if semantic:
        # Antfly embeds the query itself (RETRIEVAL_QUERY) with the index's model.
        request.update(semantic_search=text, indexes=["vectors"], merge_config={"strategy": "rrf"})
    return request
