"""Query-only entity focus and cited indexed relationships over Antfly's resolved entity graph."""
from __future__ import annotations

import re
import time
import unicodedata

from kg import dataset
from kg.antfly import hits
from kg.dataset_config import digest
from kg.dataset_inputs import read_manifest
from kg.evidence_graph import character_span, search_request
from kg.schema import MENTIONS

ENTITY_LIMIT = 2000
NEIGHBOR_LIMIT = 500
FACT_LIMIT = 10000  # traversal's maximum result count
MENTION_LIMIT = 5000
NODE_LIMIT = 50
EDGE_LIMIT = 100


class StaleSearch(ValueError):
    """Navigation must refer to the currently qualified build."""


def normalize_name(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def entity_record(key, row):
    """Public fields of a resolver-promoted entity document, with layout whitespace collapsed."""
    return {"id": key, "text": " ".join(row["canonical_name"].split()), "type": row["entity_type"]}


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
    question = re.sub(r"\b(?:the|a|an) ENTITY", "ENTITY", question)  # "what affects the oxygen tank"
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


def _mentioning_passages(af, manifest, key):
    nodes = af.traverse(manifest["table"], "graph", [key], edge_types=[MENTIONS], direction="in",
                        max_depth=1, limit=MENTION_LIMIT)
    return {node["key"] for node in nodes if node["key"].startswith("passage:")}


def _facts(cfg, af, manifest, focus, intent):
    """Every extracted fact incident on the focus entity, one per owning passage.

    Without node deduplication, traversal returns every reaching path, so parallel
    facts between the same two entities (from different passages) stay separate.
    """
    predicates = [intent["predicate"]] if intent["predicate"] else list(cfg["extraction"]["relations"])
    directions = {"incoming": ["in"], "outgoing": ["out"], "both": ["out", "in"]}[intent["direction"]]
    facts, truncated = [], False
    for direction in directions:
        nodes = af.traverse(manifest["table"], "graph", [focus["id"]], edge_types=predicates, direction=direction,
                            max_depth=1, limit=FACT_LIMIT)
        truncated |= len(nodes) >= FACT_LIMIT
        for node in nodes:
            for edge in node.get("path_edges") or []:
                if edge["from"]["key"] == edge["to"]["key"]:
                    continue  # self-loops are extraction noise
                if edge["type"] not in predicates or not edge.get("owner_document"):
                    raise ValueError("Indexed relationship does not match the requested focus/filter")
                stored = (edge["from"], edge["to"]) if edge["direction"] == "out" else (edge["to"], edge["from"])
                facts.append({"source": stored[0]["key"], "target": stored[1]["key"], "predicate": edge["type"],
                              "passage_id": edge["owner_document"], "score": edge.get("weight"),
                              "head": edge["metadata"]["source"], "tail": edge["metadata"]["target"]})
    return facts, truncated


def _relationships(cfg, af, manifest, focus, intent, sources):
    if focus is None:
        return {"nodes": [], "edges": [], "omitted_nodes": 0, "omitted_edges": 0, "scan_complete": True}, []
    entities = {focus["id"]: af.get(manifest["table"], focus["id"])}
    summary = {"aliases": sorted({" ".join(a.split()) for a in entities[focus["id"]].get("aliases") or [focus["text"]]}),
               "passages": len(_mentioning_passages(af, manifest, focus["id"])), "predicates": {}}
    if intent["status"] == "unrecognized":
        return {"nodes": [focus], "edges": [], "omitted_nodes": 0, "omitted_edges": 0,
                "scan_complete": True, "focus_summary": summary}, []
    facts, truncated = _facts(cfg, af, manifest, focus, intent)
    notes = []
    if truncated:
        notes.append(f"Relationship scan bounded at {FACT_LIMIT} facts; more may exist.")
    passages, wanted = {}, sorted({fact["passage_id"] for fact in facts})
    for offset in range(0, len(wanted), 100):
        batch = wanted[offset:offset + 100]
        for row in hits(af.query(manifest["table"], {"filter_query": {"disjuncts": [
                {"term": key, "field": "passage_id"} for key in batch]}, "limit": len(batch)})):
            passages[row["_id"]] = _verified_passage(cfg, sources, row["_id"],
                {k: v for k, v in row.items() if not k.startswith("_")})
    if set(wanted) - passages.keys():
        raise ValueError("A relationship's owning passage is missing from the index")
    # One connection per entity pair. Each passage's extracted fact is a vote; the
    # majority predicate and direction label the connection. Single-passage labels
    # are noisy, but the pair and its majority label are usually right.
    edges, unanchored = {}, 0
    for fact in facts:
        passage = passages[fact["passage_id"]]
        try:
            head = {**character_span(passage["text"], fact["head"]), "label": fact["head"].get("label")}
            tail = {**character_span(passage["text"], fact["tail"]), "label": fact["tail"].get("label")}
        except ValueError:
            unanchored += 1
            continue
        pair = tuple(sorted((fact["source"], fact["target"])))
        edge = edges.setdefault(pair, {"id": "relationship:" + digest(pair)[:24], "evidence": [], "facts": []})
        edge["facts"].append({"passage_id": fact["passage_id"], "head": head, "tail": tail, "score": fact["score"],
                              "predicate": fact["predicate"], "source": fact["source"], "target": fact["target"]})
        if fact["passage_id"] not in {row["passage_id"] for row in edge["evidence"]}:
            edge["evidence"].append(passage)
    for edge in edges.values():
        # Most confident extractions first; evidence follows that order.
        edge["facts"].sort(key=lambda fact: -(fact["score"] or 0))
        order = {key: index for index, key in reversed(list(enumerate(f["passage_id"] for f in edge["facts"])))}
        edge["evidence"].sort(key=lambda row: order[row["passage_id"]])
        votes = {}
        for fact in edge["facts"]:
            key = (fact["predicate"], fact["source"], fact["target"])
            votes[key] = votes.get(key, 0) + 1
        ranked = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
        (predicate, source, target), _ = ranked[0]
        edge.update(source=source, target=target, predicate=predicate, label=predicate.replace("_", " "),
                    support=len(edge["facts"]), votes=[{"predicate": p, "source": s, "target": t, "count": n}
                                                       for (p, s, t), n in ranked])
        for key in (edge["source"], edge["target"]):
            if key not in entities:
                entities[key] = af.get(manifest["table"], key)
                if entities[key] is None:
                    raise ValueError("Indexed relationship entity is missing")
    if unanchored:
        notes.append(f"{unanchored} extracted facts were skipped because their spans do not match the passage text.")
    if intent["label"] in ("father", "mother"):
        notes.append("parent_of does not distinguish a father from a mother; these are all extracted parents.")
    nodes = {key: entity_record(key, row) for key, row in entities.items()}
    # Retain complete edges, not orphan nodes, when imposing display bounds;
    # the best-supported relationships survive the bound.
    selected_nodes, selected_edges = {focus["id"]: focus}, []
    for pair in sorted(edges, key=lambda p: (-edges[p]["support"], p)):
        edge = edges[pair]
        new_ids = {edge["source"], edge["target"]} - selected_nodes.keys()
        if len(selected_edges) >= EDGE_LIMIT or len(selected_nodes) + len(new_ids) > NODE_LIMIT:
            continue
        selected_nodes.update((eid, nodes[eid]) for eid in sorted(new_ids))
        selected_edges.append(edge)
    for edge in edges.values():
        summary["predicates"].setdefault(edge["predicate"], {"relationships": 0, "facts": 0})["relationships"] += 1
        for fact in edge["facts"]:
            summary["predicates"].setdefault(fact["predicate"], {"relationships": 0, "facts": 0})["facts"] += 1
    summary["predicates"] = dict(sorted(summary["predicates"].items()))
    graph = {"nodes": list(selected_nodes.values()), "edges": selected_edges,
             "omitted_nodes": len(nodes) - len(selected_nodes), "omitted_edges": len(edges) - len(selected_edges),
             "scan_complete": not truncated, "focus_summary": summary}
    if graph["omitted_nodes"] or graph["omitted_edges"]:
        notes.append(f"Display omits {graph['omitted_nodes']} nodes and {graph['omitted_edges']} relationships.")
    return graph, notes


def name_groups(cfg, af, manifest, names, limit=12):
    """Every entity node sharing a word with these names, unmerged, with its footprint."""
    terms = sorted({term for name in names for term in re.findall(r"\w+", name, flags=re.UNICODE)})
    if not terms:
        return []
    rows = hits(af.query(manifest["table"], {
        "full_text_search": {"disjuncts": [{"match": term, "field": "aliases"} for term in terms]}, "limit": limit}))
    groups = []
    for row in rows:
        neighbors = af.traverse(manifest["table"], "graph", [row["_id"]], edge_types=list(cfg["extraction"]["relations"]),
                                direction="both", max_depth=1, limit=NEIGHBOR_LIMIT)
        groups.append({**entity_record(row["_id"], row), "aliases": sorted({" ".join(a.split()) for a in row.get("aliases") or []}),
                       "passages": len(_mentioning_passages(af, manifest, row["_id"])),
                       "neighbors": len({node["key"] for node in neighbors} - {row["_id"]})})
    return sorted(groups, key=lambda group: (-group["passages"], group["text"]))


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
            if entity is None or "canonical_name" not in entity:
                raise ValueError("Focus must be an indexed entity in the selected build")
            focus = entity_record(focus_id, entity)
            if not navigating and not match_candidates(text, [focus]):
                raise ValueError("Selected focus is not an exact indexed name in this question")
        focus_lookup_ms = (time.perf_counter() - focus_lookup_started) * 1000
        if navigating:
            text = focus["text"]
            retrieval_started = time.perf_counter()
            ranking = hits(af.query(manifest["table"], search_request(cfg, text, semantic=False)))
            results = [_verified_passage(cfg, sources, row["_id"],
                       {"score": row.get("_score"), **{k: v for k, v in row.items() if not k.startswith("_")}})
                       for row in ranking[:cfg["retrieval"]["evidence"]]]
            packet = {"build_id": manifest["build_id"], "partial": manifest["selection"]["partial"],
                      "query": {"text": text, "entities": []}, "results": results,
                      "timings": {"extraction": {"ms": 0, "skipped": True}}}
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
                "full_text_search": {"disjuncts": [{"match": term, "field": "aliases"} for term in terms]},
                "limit": ENTITY_LIMIT})
            indexed = hits(response)
            total, exact = _total(response, len(indexed))
            omitted_candidates = max(0, total - len(indexed))
            candidates_complete = exact and not omitted_candidates
            rows = {row["_id"]: entity_record(row["_id"], row) for row in indexed}
            # Exact alias lookup supplements a bounded name scan; it does not merge names.
            spans = [span["text"] for span in packet["query"]["entities"]]
            for row in dataset.entities_by_surface(af, manifest, spans):
                rows.setdefault(row["_id"], entity_record(row["_id"], row))
            candidates = match_candidates(text, rows.values())
            # These are lexical Explore choices, never aliases or resolved
            # candidates. Clicking one must explicitly navigate to its name.
            suggestions = [entity_record(row["_id"], row) for row in indexed[:12]] if not candidates else []
            omitted_suggestions = (max(0, len(indexed) - len(suggestions)) + omitted_candidates
                                   if not candidates else 0)
            # A bounded scan cannot prove an apparently unique name is unambiguous.
            if focus is None and exact and not omitted_candidates:
                # Several names can appear in one question ("father", "Tyrion Lannister");
                # prefer the only one the question's relationship frame is about.
                about = [row for row in candidates if interpret_intent(cfg, text, row)["status"] == "focused"]
                focus = candidates[0] if len(candidates) == 1 else about[0] if len(about) == 1 else None
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
        groups_started = time.perf_counter()
        groups = name_groups(cfg, af, manifest, [focus["text"]] if focus else
                             [row["text"] for row in candidates + suggestions] or [text])
        packet["timings"]["name_groups"] = {"ms": round((time.perf_counter() - groups_started) * 1000, 2)}
        packet.update(top_result=packet["results"][0] if packet["results"] else None, name_groups=groups,
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
