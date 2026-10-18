"""Query-only entity focus and cited indexed relationships; no identity resolution."""
from __future__ import annotations

import re
import time
import unicodedata

from kg import dataset
from kg.antfly import hits
from kg.dataset_config import digest
from kg.dataset_inputs import read_manifest
from kg.evidence_graph import anchored, entity_key, search_request

ENTITY_LIMIT = 2000
ASSERTION_LIMIT = 500
NODE_LIMIT = 50
EDGE_LIMIT = 100


class StaleSearch(ValueError):
    """Navigation must refer to the currently qualified build."""


def normalize_name(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def entity_record(key, row):
    return {"id": key, "text": row["text"], "type": row["type"]}


def match_candidates(text, rows):
    """Literal whole-name phrases only; prefer a containing name, never an alias.

    Rows are public {id,text,type} entities. Distinct types and disjoint names
    remain ambiguous. A surname alone is not a match for an indexed full name.
    """
    question = normalize_name(text)
    matched = {}
    for row in rows:
        name = normalize_name(row["text"])
        if name and re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", question):
            matched[row["id"]] = row
    values = list(matched.values())
    return sorted((row for row in values if not any(
        normalize_name(row["text"]) != normalize_name(other["text"])
        and re.search(r"(?<!\w)" + re.escape(normalize_name(row["text"])) + r"(?!\w)",
                      normalize_name(other["text"]))
        for other in values)), key=lambda row: (normalize_name(row["text"]), row["type"], row["id"]))


def interpret_intent(cfg, text, focus=None, navigate=False):
    """Recognize explicit complete question frames, not bag-of-keyword intent.

    Specific parent wording is retained as a label, never a new indexed predicate.
    Unsupported questions deliberately return no relationship filter to execute.
    """
    if navigate:
        return {"status": "all", "label": "All indexed relationships", "predicate": None, "direction": "both"}
    unknown = {"status": "unrecognized", "label": "Relationship question not recognized", "predicate": None, "direction": "both"}
    if focus is None:
        return unknown
    question = normalize_name(text).replace("’", "'").rstrip(" ?.!")
    name = normalize_name(focus["text"]).replace("’", "'")
    question = re.sub(r"(?<!\w)" + re.escape(name) + r"(?!\w)", "ENTITY", question)
    if question == "ENTITY" or re.fullmatch(r"(?:show|list)(?: all)? (?:relationships|relations)(?: for| of)? ENTITY", question):
        return {"status": "all", "label": "All indexed relationships", "predicate": None, "direction": "both"}
    frames = [
        (r"who (?:is|was) (?:ENTITY's (?:father|dad)|the (?:father|dad) of ENTITY)", "parent_of", "incoming", "father"),
        (r"who (?:is|was) (?:ENTITY's (?:mother|mom)|the (?:mother|mom) of ENTITY)", "parent_of", "incoming", "mother"),
        (r"who (?:are|were) (?:ENTITY's parents|the parents of ENTITY)", "parent_of", "incoming", "parents"),
        (r"who (?:is|was) (?:ENTITY's parent|the parent of ENTITY)", "parent_of", "incoming", "parents"),
        (r"who (?:are|were) (?:ENTITY's (?:children|sons|daughters)|the (?:children|sons|daughters) of ENTITY)", "parent_of", "outgoing", "children"),
        (r"(?:what|which) house (?:does|did) ENTITY belong to", "member_of", "outgoing", "house membership"),
        (r"(?:who|which characters) (?:belongs?|belonged) to ENTITY", "member_of", "incoming", "members"),
        (r"(?:what|which) (?:house|houses|kingdom|kingdoms|city|cities) (?:does|did) ENTITY rule", "rules", "outgoing", "rules"),
        (r"who (?:rules|ruled) ENTITY", "rules", "incoming", "rulers"),
        (r"where (?:is|was) ENTITY(?: located)?", "located_in", "outgoing", "location"),
        (r"(?:what|which) (?:restaurants|houses|cities) (?:are|were) (?:located )?in ENTITY", "located_in", "incoming", "located here"),
        (r"what (?:does|did) ENTITY serve", "serves", "outgoing", "menu items"),
        (r"which restaurants serve ENTITY", "serves", "incoming", "restaurants serving this"),
        (r"(?:what|which) ingredients (?:does ENTITY contain|are in ENTITY)", "contains_ingredient", "outgoing", "ingredients"),
        (r"(?:what|which) (?:pizzas|menu items) contain ENTITY", "contains_ingredient", "incoming", "items containing this"),
        (r"(?:what|which) dietary options does ENTITY (?:have|offer)", "has_dietary_option", "outgoing", "dietary options"),
        (r"(?:what|which) (?:pizzas|restaurants|menu items) (?:have|offer) (?:a |an )?ENTITY(?: option)?", "has_dietary_option", "incoming", "offering this dietary option"),
        (r"(?:what|which) (?:pizza )?style (?:is|does) ENTITY(?: have)?", "has_style", "outgoing", "pizza style"),
        (r"(?:what|which) (?:pizzas|restaurants|menu items) (?:have|offer) (?:the )?ENTITY style", "has_style", "incoming", "offering this style"),
        (r"what is ENTITY part of", "part_of", "outgoing", "part of"),
        (r"(?:what|which components) (?:is|are) part of ENTITY", "part_of", "incoming", "components"),
        (r"what (?:does ENTITY affect|is affected by ENTITY)", "affects", "outgoing", "affected components"),
        (r"what affects ENTITY", "affects", "incoming", "affecting problems"),
        (r"what (?:mitigates|reduces|resolves) ENTITY", "mitigates", "incoming", "mitigations"),
        (r"what does ENTITY (?:mitigate|reduce|resolve)", "mitigates", "outgoing", "mitigated problems"),
        (r"what (?:was|is) (?:the )?suspected cause of ENTITY", "suspected_cause_of", "incoming", "suspected causes"),
        (r"what (?:was|is) (?:the )?determined cause of ENTITY", "determined_cause_of", "incoming", "determined causes"),
    ]
    for pattern, predicate, direction, label in frames:
        if predicate in cfg["extraction"]["relations"] and re.fullmatch(pattern, question):
            # Sons/daughters require child-specific wording beyond generic parenthood.
            if re.search(r"\b(?:sons|daughters)\b", question):
                return unknown
            return {"status": "focused", "label": label, "predicate": predicate, "direction": direction}
    return unknown


def _total(response, count):
    total = response["responses"][0].get("hits", {}).get("total", {})
    if isinstance(total, int):
        return max(total, count), True
    return max(total.get("value", count), count), total.get("relation") in ("exact", "eq")


def _verified_passage(cfg, sources, key, row):
    source = sources[row["source_id"]]
    start, end = row["source_start"], row["source_end"]
    if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(source["text"])
            or row["text"] != source["text"][start:end]):
        raise ValueError("Retrieved evidence does not match exact raw offsets")
    return {**row, "passage_id": key, "citation": dataset.citation(source, row)}


def _specific_parent(assertion, label):
    """Conservative explicit wording at the actual ordered endpoint occurrences."""
    if label not in ("father", "mother"):
        return True
    text, head, tail = assertion["text"], assertion["head"], assertion["tail"]
    hs, he = anchored(text, head)
    ts, te = anchored(text, tail)
    # An explicit-looking connector inside a denial or hypothesis cannot
    # establish the more specific relationship. Err toward no edge.
    begin, end = min(hs, ts), max(he, te)
    sentence_start = max(text.rfind(mark, 0, begin) for mark in (".", "!", "?", "\n")) + 1
    stops = [index for mark in (".", "!", "?", "\n") if (index := text.find(mark, end)) >= 0]
    sentence = normalize_name(text[sentence_start:min(stops) if stops else len(text)])
    if stops and text[min(stops)] == "?":
        return False
    if re.search(r"\b(?:not|never|no|denied|denies|deny|claimed|claims|claim|rumou?r|supposed|alleged|"
                 r"might|may|could|would|whether|if|possibly|believed|said|suspected|incorrect|false)\b|n't\b", sentence):
        return False
    if he <= ts:
        connector = normalize_name(text[he:ts]).replace("’", "'")
        direct = re.fullmatch(r",?\s*(?:(?:is|was) )?(?:the |a )?" + label + r" of\s*", connector)
        possessive = (connector in ("is", "was") and
                      re.match(r"^'s " + label + r"\b",
                               normalize_name(text[te:]).replace("’", "'")))
        return bool(direct or possessive)
    if te <= hs:
        connector = normalize_name(text[te:hs]).replace("’", "'")
        return bool(re.fullmatch(r"'s " + label + r"(?: (?:is|was))?\s*,?", connector))
    return False


def _relationships(cfg, af, manifest, focus, intent, sources):
    empty = {"nodes": [focus] if focus else [], "edges": [], "omitted_nodes": 0,
             "omitted_edges": 0, "omitted_assertions": 0, "scan_complete": True}
    if focus is None or intent["status"] == "unrecognized":
        return empty, []
    endpoints = [{"term": focus["id"], "field": field} for field in (
        ("object_id",) if intent["direction"] == "incoming" else
        ("subject_id",) if intent["direction"] == "outgoing" else ("subject_id", "object_id"))]
    filters = [{"term": "assertion", "field": "kind"}, {"disjuncts": endpoints}]
    if intent["predicate"]:
        filters.append({"term": intent["predicate"], "field": "predicate"})
    response = af.query(manifest["table"], {"filter_query": {"conjuncts": filters}, "limit": ASSERTION_LIMIT})
    assertions = hits(response)
    total, exact = _total(response, len(assertions))
    omitted = max(0, total - len(assertions))
    notes = []
    if omitted or not exact:
        notes.append(f"Relationship scan bounded at {ASSERTION_LIMIT} assertions; "
                     + (f"{omitted} assertions omitted." if exact else "additional assertions may be omitted."))
    nodes, triples, passages = {focus["id"]: focus}, {}, {}
    specificity_excluded = 0
    for assertion in assertions:
        # Never trust a query hit as an assertion with arbitrary endpoints.
        if assertion.get("kind") != "assertion":
            raise ValueError("Indexed relationship query returned a non-assertion")
        subject, obj = assertion["subject_id"], assertion["object_id"]
        predicate = assertion["predicate"]
        if (predicate not in cfg["extraction"]["relations"]
                or subject != entity_key(assertion["head"]["label"], assertion["head"]["text"])
                or obj != entity_key(assertion["tail"]["label"], assertion["tail"]["text"])):
            raise ValueError("Indexed relationship endpoints differ from their typed source mentions")
        if ((intent["direction"] == "incoming" and obj != focus["id"])
                or (intent["direction"] == "outgoing" and subject != focus["id"])
                or focus["id"] not in (subject, obj)
                or (intent["predicate"] and predicate != intent["predicate"])):
            raise ValueError("Indexed relationship does not match the requested focus/filter")
        # The canonical key resolves the actual indexed passage, not a best text match.
        _verified_passage(cfg, sources, assertion["_id"], assertion)
        source = sources[assertion["source_id"]]
        key = "passage:" + digest([assertion["source_id"], source["sha256"],
                                  assertion["source_start"], assertion["source_end"], assertion["text"]])[:24]
        if key not in passages:
            passage = af.get(manifest["table"], key)
            if passage is None or passage.get("kind") != "passage":
                raise ValueError("Indexed relationship has no supporting indexed passage")
            passages[key] = _verified_passage(cfg, sources, key, passage)
        passage = passages[key]
        stored = {k: v for k, v in assertion.items() if k in (
            "head", "tail", "predicate", "score", "status", "decision_index")}
        if (passage["text"] != assertion["text"] or passage["source_id"] != assertion["source_id"]
                or passage["source_start"] != assertion["source_start"] or passage["source_end"] != assertion["source_end"]
                or stored not in passage["analysis"]["assertions"]):
            raise ValueError("Indexed relationship is not supported by the stored passage analysis")
        anchored(passage["text"], assertion["head"])
        anchored(passage["text"], assertion["tail"])
        if not _specific_parent(assertion, intent["label"]):
            specificity_excluded += 1
            continue
        for eid in (subject, obj):
            if eid not in nodes:
                entity = af.get(manifest["table"], eid)
                if entity is None or entity.get("kind") != "entity":
                    raise ValueError("Indexed relationship entity is missing")
                nodes[eid] = entity_record(eid, entity)
        triple = (subject, predicate, obj)
        edge = triples.setdefault(triple, {"id": "relationship:" + digest(triple)[:24],
            "source": subject, "target": obj, "predicate": predicate,
            "label": predicate.replace("_", " "), "evidence": []})
        if key not in {row["passage_id"] for row in edge["evidence"]}:
            edge["evidence"].append(passage)
    if intent["label"] in ("father", "mother"):
        notes.append(f"Only explicit {intent['label']}-wording at both source mentions qualifies; "
                     f"generic parent_of does not establish {intent['label']} ({specificity_excluded} assertions excluded).")
    # Retain complete edges, not orphan nodes, when imposing display bounds.
    selected_nodes, selected_edges = {focus["id"]: focus}, []
    for triple in sorted(triples):
        edge = triples[triple]
        new_ids = {edge["source"], edge["target"]} - selected_nodes.keys()
        if len(selected_edges) >= EDGE_LIMIT or len(selected_nodes) + len(new_ids) > NODE_LIMIT:
            continue
        selected_nodes.update((eid, nodes[eid]) for eid in sorted(new_ids))
        selected_edges.append(edge)
    graph = {"nodes": list(selected_nodes.values()), "edges": selected_edges,
             "omitted_nodes": len(nodes) - len(selected_nodes), "omitted_edges": len(triples) - len(selected_edges),
             "omitted_assertions": omitted, "scan_complete": exact and not omitted}
    if graph["omitted_nodes"] or graph["omitted_edges"]:
        notes.append(f"Display omits {graph['omitted_nodes']} nodes and {graph['omitted_edges']} relationships.")
    return graph, notes


def focused_search(cfg, text, models=None, focus_id=None, build_id=None, navigate=False):
    """Real hybrid question search; explicit indexed recenter is BM25/inference-free."""
    started = time.perf_counter()
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Query must not be empty")
    manifest = dataset.load_build(cfg)
    navigating = navigate
    if navigating and focus_id is None:
        raise ValueError("Navigation requires an indexed focus_id and build_id")
    if focus_id is not None and build_id != manifest["build_id"]:
        raise StaleSearch("Build changed; search again before selecting an entity")
    if build_id is not None and focus_id is None:
        raise ValueError("build_id is only accepted with focus_id")
    af = dataset.database(cfg)
    try:
        sources = {row["id"]: row for row in read_manifest(cfg)}
        focus = None
        focus_lookup_started = time.perf_counter()
        if focus_id is not None:
            entity = af.get(manifest["table"], focus_id)
            if (entity is None or entity.get("kind") != "entity"
                    or focus_id != entity_key(entity["type"], entity["text"])):
                raise ValueError("Focus must be an indexed entity in the selected build")
            focus = entity_record(focus_id, entity)
            if not navigating and not match_candidates(text, [focus]):
                raise ValueError("Selected focus is not an exact indexed name in this question")
        focus_lookup_ms = (time.perf_counter() - focus_lookup_started) * 1000
        if navigating:
            text = focus["text"]
            retrieval_started = time.perf_counter()
            ranking = hits(af.query(manifest["table"], search_request(cfg, text)))
            results = [_verified_passage(cfg, sources, row["_id"],
                       {"score": row.get("_score"), **{k: v for k, v in row.items() if not k.startswith("_")}})
                       for row in ranking[:cfg["retrieval"]["evidence"]]]
            packet = {"build_id": manifest["build_id"], "partial": manifest["selection"]["partial"],
                      "query": {"text": text, "analysis": None}, "results": results,
                      "timings": {name: {"ms": 0, "cache_hits": 0, "model_calls": 0, "skipped": True}
                                  for name in ("extraction", "relations", "embedding")}}
            packet["timings"]["retrieval"] = {"ms": round((time.perf_counter() - retrieval_started) * 1000, 2)}
            packet["timings"]["focus"] = {"ms": round(focus_lookup_ms, 2)}
            candidates, omitted_candidates, candidate_notes = [], 0, []
            candidates_complete = True
            suggestions, omitted_suggestions = [], 0
        else:
            packet = dataset.query(cfg, text, mode="hybrid", models=models)
            if packet["build_id"] != manifest["build_id"]:
                raise StaleSearch("Build changed during search; search again")
            packet["results"] = [_verified_passage(cfg, sources, row["passage_id"], row)
                                 for row in packet["results"]]
            candidate_started = time.perf_counter()
            terms = sorted(set(re.findall(r"\w+", text, flags=re.UNICODE)))
            response = af.query(manifest["table"], {
                "full_text_search": {"disjuncts": [{"match": term, "field": "text"} for term in terms]},
                "filter_query": {"term": "entity", "field": "kind"}, "limit": ENTITY_LIMIT})
            indexed = hits(response)
            total, exact = _total(response, len(indexed))
            omitted_candidates = max(0, total - len(indexed))
            candidates_complete = exact and not omitted_candidates
            rows = {row["_id"]: entity_record(row["_id"], row) for row in indexed}
            # Exact typed lookup supplements a bounded name scan; it does not merge names.
            for span in packet["query"]["analysis"]["entities"]:
                if span["score"] < cfg["extraction"]["threshold"]:
                    continue
                key = entity_key(span["label"], span["text"])
                if key not in rows:
                    entity = af.get(manifest["table"], key)
                    if entity is not None and entity.get("kind") == "entity":
                        rows[key] = entity_record(key, entity)
            candidates = match_candidates(text, rows.values())
            # These are lexical Explore choices, never aliases or resolved
            # candidates. Clicking one must explicitly navigate to its name.
            suggestions = [entity_record(row["_id"], row) for row in indexed[:12]] if not candidates else []
            omitted_suggestions = (max(0, len(indexed) - len(suggestions)) + omitted_candidates
                                   if not candidates else 0)
            # A bounded scan cannot prove an apparently unique name is unambiguous.
            if focus is None:
                focus = candidates[0] if len(candidates) == 1 and exact and not omitted_candidates else None
            candidate_notes = []
            if suggestions:
                candidate_notes.append("Lexical name suggestions are exploration choices, not resolved identities.")
                if omitted_suggestions:
                    candidate_notes.append(f"{omitted_suggestions} additional lexical suggestions omitted.")
            if omitted_candidates or not exact:
                candidate_notes.append(f"Relevant-name scan bounded at {ENTITY_LIMIT} names; "
                    + (f"{omitted_candidates} names omitted. Choose an offered candidate explicitly." if exact
                       else "additional names may be omitted. Choose an offered candidate explicitly."))
            packet["timings"]["focus"] = {"ms": round(
                (time.perf_counter() - candidate_started) * 1000 + focus_lookup_ms, 2)}
        intent_started = time.perf_counter()
        intent = interpret_intent(cfg, text, focus, navigate=navigating)
        packet["timings"]["intent"] = {"ms": round((time.perf_counter() - intent_started) * 1000, 2),
                                       "model_calls": 0, "method": "explicit_question_frames"}
        relationship_started = time.perf_counter()
        graph, notes = _relationships(cfg, af, manifest, focus, intent, sources)
        packet["timings"]["relationships"] = {"ms": round((time.perf_counter() - relationship_started) * 1000, 2)}
        # These diagnostic substages are included in retrieval, not extra stages
        # to sum again when displaying the end-to-end latency breakdown.
        packet["timings"]["retrieval"]["ms"] = round(
            packet["timings"]["retrieval"]["ms"] + packet["timings"]["focus"]["ms"]
            + packet["timings"]["relationships"]["ms"], 2)
        packet["timings"]["retrieval"]["includes"] = ["focus", "relationships"]
        if focus is None:
            status = ("Several indexed names match; choose an entity to explore its relationships." if len(candidates) > 1
                      else "Choose the candidate to explore; the bounded name scan cannot establish uniqueness." if candidates
                      else "No exact indexed entity name matched. Use an indexed name; aliases and identities are not resolved.")
        elif intent["status"] == "unrecognized":
            status = "The entity matched, but this relationship question is not recognized; no relationship answer is asserted. Search the entity name to explore all relationships."
        elif not graph["edges"]:
            status = "No indexed relationships match this focus and question. The independently ranked text result is not relationship evidence."
        else:
            status = f"{len(graph['edges'])} sourced model-predicted relationships; not independently verified."
        if packet["partial"]:
            notes.append("This is a partial build; missing relationships do not establish absence in the full corpus.")
        packet.update(top_result=packet["results"][0] if packet["results"] else None,
                      focus=focus, focus_candidates=candidates if focus is None else [],
                      focus_candidates_omitted=omitted_candidates,
                      focus_candidates_complete=candidates_complete, intent=intent,
                      focus_suggestions=suggestions if focus is None else [],
                      focus_suggestions_omitted=omitted_suggestions,
                      relationship_status=" ".join([status, *candidate_notes, *notes]), graph=graph)
        packet["timings"]["total_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return packet
    finally:
        af.http.close()
