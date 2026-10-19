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


def entity_key(name):
    """The resolver's key for a surface form ("#754" -> entity/754, "v0.2.3" -> entity/v0_2_3)."""
    return "entity/" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def explicit_links(cfg, title, text):
    """Links written in the text itself ("#752", ".../pull/754"), read with a pattern, not a model.

    Returns {entity key: (type, display name, label)}. Keys match the resolver's, so a
    link lands on the same node as an extracted mention. Earlier configured links win,
    so a passage that describes #752 does not also reference it.
    """
    found = {}
    for link in cfg.get("explicit_links") or []:
        for match in re.finditer(link["pattern"], title if link["field"] == "title" else text, re.M):
            name = link["name"].format(*match.groups())
            found.setdefault(entity_key(name), (link["type"], name, link["label"]))
    return found


def passage_documents(cfg, prepared):
    """Passage text and exact source provenance (plus its timestamp and explicit links, if any).

    Antfly derives everything else. Explicit link targets get a small node document
    (same key and shape as a resolver-promoted entity) so it can be searched.
    """
    docs, targets = {}, {}
    for p in prepared["passages"]:
        source = prepared["sources"][p["source_id"]]
        doc = {**{k: v for k, v in p.items() if k != "id"}, "kind": "passage", "passage_id": p["id"]}
        if source.get("timestamp"):
            doc["timestamp"] = source["timestamp"]
        if cfg.get("explicit_links"):
            links = explicit_links(cfg, source["title"], p["text"])
            doc["links"] = [{"type": kind, "target": key} for key, (kind, _, _) in sorted(links.items())]
            for key, (_, name, label) in links.items():
                targets.setdefault(key, {"kind": "link_target", "canonical_name": name, "entity_type": label, "aliases": [name]})
        docs[p["id"]] = doc
    return {**docs, **targets}


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
