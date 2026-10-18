"""Typed surface buckets and directed, cited model assertions; no identity resolution."""
from __future__ import annotations

import unicodedata
from kg.dataset_config import digest


def entity_key(label, surface):
    normalized = " ".join(unicodedata.normalize("NFKC", surface).casefold().split())
    return "entity:" + digest([label, normalized])[:24]


def edges(**values):
    return {"graph": {name: [{"target": key} for key in sorted(set(keys))]
                      for name, keys in values.items() if keys}}


def anchored(text, span):
    start, end = span["start"], span["end"]
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text) or text[start:end] != span["text"]:
        raise ValueError("Extraction span does not match source prose")
    return start, end


def graph_documents(cfg, prepared, analyses):
    docs = {}
    for sid, source in prepared["sources"].items():
        docs["source:" + digest(sid)[:24]] = {"kind": "source", "source_id": sid, "metadata": source}
    for passage in prepared["passages"]:
        key, text = passage["id"], passage["text"]
        source = "source:" + digest(passage["source_id"])[:24]
        analysis = analyses[key]
        entity_ids = set()
        for span in analysis["entities"]:
            start, end = anchored(text, span)
            if span["score"] < cfg["extraction"]["threshold"]:
                continue
            eid = entity_key(span["label"], span["text"])
            entity_ids.add(eid)
            entity = docs.setdefault(eid, {"kind": "entity", "type": span["label"], "text": span["text"],
                "identity_status": "surface_bucket_not_disambiguated", "aliases": []})
            entity["aliases"] = sorted(set(entity["aliases"]) | {span["text"]})
            mid = "mention:" + digest([key, eid, start, end])[:24]
            docs[mid] = {"kind": "mention", **span, "source_id": passage["source_id"],
                "source_start": passage["source_start"] + start, "source_end": passage["source_start"] + end,
                "_edges": edges(in_passage=[key], refers_to=[eid], from_source=[source])}
        docs[key] = {**{k: v for k, v in passage.items() if k != "id"}, "kind": "passage", "passage_id": key,
                     "analysis": analysis, "entity_ids": sorted(entity_ids),
                     "_edges": edges(from_source=[source], mentions=entity_ids)}
        for assertion in analysis["assertions"]:
            head, tail = assertion["head"], assertion["tail"]
            subject = entity_key(head["label"], head["text"])
            obj = entity_key(tail["label"], tail["text"])
            rid = "assertion:" + digest([key, assertion])[:24]
            docs[rid] = {"kind": "assertion", **assertion, "subject_id": subject, "object_id": obj,
                "source_id": passage["source_id"], "source_start": passage["source_start"],
                "text": text,
                "source_end": passage["source_end"],
                "_edges": edges(subject=[subject], object=[obj], supported_by=[key], from_source=[source])}
    targets = {edge["target"] for row in docs.values() for group in row.get("_edges", {}).get("graph", {}).values() for edge in group}
    if targets - docs.keys():
        raise ValueError("Dangling evidence graph edge")
    return docs


def search_request(cfg, text, vector=None, allowed=None):
    import re
    terms = sorted(set(re.findall(r"\w+", text, flags=re.UNICODE)))
    filters = [{"term": "passage", "field": "kind"}]
    if allowed is not None:
        filters.append({"disjuncts": [{"term": key, "field": "passage_id"} for key in sorted(allowed)]})
    request = {"full_text_search": {"disjuncts": [{"match": term, "field": "text"} for term in terms]},
               "filter_query": {"conjuncts": filters}, "limit": cfg["retrieval"]["candidates"]}
    if vector is not None:
        request.update(embeddings={"vectors": vector}, indexes=["vectors"], merge_config={"strategy": "rrf"})
    return request


def projection(query_analysis, results):
    """Bounded deterministic projection preserving distinct occurrence links."""
    nodes, links = {}, []
    for owner, analysis in [("query", query_analysis)] + [(r["passage_id"], r["analysis"]) for r in results]:
        nodes[owner] = {"id": owner, "kind": "query" if owner == "query" else "passage", "text": owner}
        for index, entity in enumerate(analysis["entities"]):
            eid = entity_key(entity["label"], entity["text"])
            nodes.setdefault(eid, {"id": eid, "kind": "entity", **entity,
                                  "identity_status": "surface_bucket_not_disambiguated"})
            links.append({"id": owner + ":mention:" + str(index), "source": owner, "target": eid,
                          "predicate": "mentions", "record": entity})
        for index, assertion in enumerate(analysis["assertions"]):
            head, tail = assertion["head"], assertion["tail"]
            links.append({"id": owner + ":assertion:" + str(index),
                          "source": entity_key(head["label"], head["text"]),
                          "target": entity_key(tail["label"], tail["text"]),
                          "predicate": assertion["predicate"], "record": assertion, "evidence": owner})
    selected = list(nodes.values())[:50]
    ids = {n["id"] for n in selected}
    eligible = [link for link in links if link["source"] in ids and link["target"] in ids]
    return {"nodes": selected, "edges": eligible[:100],
            "omitted_nodes": len(nodes) - len(selected), "omitted_edges": len(links) - min(100, len(eligible))}
