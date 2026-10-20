"""Query-only entity focus and cited indexed relationships over Antfly's resolved entity graph."""
from __future__ import annotations

import math
import re
import time
import unicodedata

from kg import dataset
from kg.antfly import hits
from kg.dataset_config import digest
from kg.dataset_inputs import read_manifest
from kg.evidence_graph import character_span, entity_key, search_request
from kg.schema import LINKS_METRIC, MENTIONS, MENTIONS_METRIC

ENTITY_LIMIT = 2000
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


def interpret_intent(cfg, text, focus=None):
    """Recognize explicit complete question frames, not bag-of-keyword intent.

    Specific parent wording is retained as a label, never a new indexed predicate.
    A bare name (or "show relationships for NAME") is "all": that node's own page.
    """
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


def _fetch_passages(cfg, af, manifest, sources, keys):
    passages, wanted = {}, sorted(set(keys))
    for offset in range(0, len(wanted), 100):
        batch = wanted[offset:offset + 100]
        for row in hits(af.query(manifest["table"], {"filter_query": {"disjuncts": [
                {"term": key, "field": "passage_id"} for key in batch]}, "limit": len(batch)})):
            passages[row["_id"]] = _verified_passage(cfg, sources, row["_id"],
                {k: v for k, v in row.items() if not k.startswith("_")})
    if set(wanted) - passages.keys():
        raise ValueError("A relationship's owning passage is missing from the index")
    return passages


def _link_types(cfg):
    """Explicit links read from a title identify what a passage is about; links read from text reference."""
    links = cfg.get("explicit_links") or []
    return ({link["type"] for link in links if link["field"] == "title"}, {link["type"] for link in links if link["field"] == "text"})


def _link_span(cfg, text, key):
    for link in cfg.get("explicit_links") or []:
        for match in re.finditer(link["pattern"], text, re.M):
            if entity_key(link["name"].format(*match.groups())) == key:
                return {"text": match.group(0), "start": match.start(), "end": match.end(), "label": link["label"]}
    return None


def _link_facts(cfg, af, manifest, focus):
    """Document-level links: a passage about X that references the focus means X -> references -> focus."""
    identity, references = _link_types(cfg)
    if not identity or not references:
        return []
    facts = []
    for node in af.traverse(manifest["table"], "graph", [focus["id"]], edge_types=sorted(identity | references),
                            direction="in", max_depth=1, limit=FACT_LIMIT):
        passage, how = node["key"], node["path_edges"][0]["type"]
        other = references if how in identity else identity
        for linked in af.traverse(manifest["table"], "graph", [passage], edge_types=sorted(other),
                                  direction="out", max_depth=1, limit=FACT_LIMIT):
            kind = linked["path_edges"][0]["type"]
            source, target = (focus["id"], linked["key"]) if how in identity else (linked["key"], focus["id"])
            facts.append({"source": source, "target": target, "predicate": kind if how in identity else how,
                          "passage_id": passage, "score": None, "explicit": True})
    return facts


def _timeline(cfg, af, manifest, focus, sources):
    """Every document that mentions, describes or references the focus, oldest first."""
    identity, references = _link_types(cfg)
    nodes = af.traverse(manifest["table"], "graph", [focus["id"]], edge_types=[MENTIONS, *sorted(identity | references)],
                        direction="in", max_depth=1, limit=MENTION_LIMIT)
    how = {}
    for node in nodes:
        how.setdefault(node["key"], set()).add(node["path_edges"][0]["type"])
    passages = _fetch_passages(cfg, af, manifest, sources, how)
    # A timeline of documents, so only for datasets whose documents are dated (passage
    # times inside one transcript are shown on each passage instead).
    if not any(sources[p["source_id"]].get("timestamp") for p in passages.values()):
        return []
    entries = {}
    for key, passage in sorted(passages.items(), key=lambda kv: kv[1]["source_start"]):
        entry = entries.setdefault(passage["source_id"], {
            "source_id": passage["source_id"], "title": passage["citation"].get("title"),
            "timestamp": passage.get("timestamp"), "how": set(), "passage": passage})
        entry["how"] |= how[key]
    timeline = sorted(entries.values(), key=lambda e: e["timestamp"] or "")
    for entry in timeline:
        entry["how"] = sorted(entry["how"])
    if timeline:
        timeline[-1]["latest"] = True
    return timeline


def _relationships(cfg, af, manifest, focus, intent, sources, among=None):
    """The focus's relationships, one connection per pair; `among` keeps only pairs inside that node set."""
    if focus is None:
        return {"nodes": [], "edges": [], "omitted_nodes": 0, "omitted_edges": 0}, []
    entities = {focus["id"]: af.get(manifest["table"], focus["id"])}
    summary = {"aliases": sorted({" ".join(a.split()) for a in entities[focus["id"]].get("aliases") or [focus["text"]]}),
               "passages": len(_mentioning_passages(af, manifest, focus["id"]))}
    facts, truncated = _facts(cfg, af, manifest, focus, intent)
    if intent["status"] == "all":
        facts += _link_facts(cfg, af, manifest, focus)
    if among is not None:
        facts = [fact for fact in facts if fact["source"] in among and fact["target"] in among]
    notes = []
    if truncated:
        notes.append(f"Relationship scan bounded at {FACT_LIMIT} facts; more may exist.")
    passages = _fetch_passages(cfg, af, manifest, sources, (fact["passage_id"] for fact in facts))
    # One connection per entity pair. Each passage's extracted fact is a vote; the
    # majority predicate and direction label the connection. Single-passage labels
    # are noisy, but the pair and its majority label are usually right.
    edges, unanchored, linked = {}, 0, set()
    for fact in facts:
        passage = passages[fact["passage_id"]]
        if fact.get("explicit"):
            # An explicit link counts once per document, not once per overlapping passage.
            document_link = (passage["source_id"], fact["source"], fact["predicate"], fact["target"])
            if document_link in linked:
                continue
            linked.add(document_link)
            head = _link_span(cfg, passage["text"], fact["source"])
            tail = _link_span(cfg, passage["text"], fact["target"])
        else:
            try:
                head = {**character_span(passage["text"], fact["head"]), "label": fact["head"].get("label")}
                tail = {**character_span(passage["text"], fact["tail"]), "label": fact["tail"].get("label")}
            except ValueError:
                unanchored += 1
                continue
        pair = tuple(sorted((fact["source"], fact["target"])))
        edge = edges.setdefault(pair, {"id": "relationship:" + digest(pair)[:24], "evidence": [], "facts": []})
        edge["facts"].append({"passage_id": fact["passage_id"], "head": head, "tail": tail, "score": fact["score"],
                              "predicate": fact["predicate"], "source": fact["source"], "target": fact["target"],
                              "explicit": bool(fact.get("explicit"))})
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
    graph = {"nodes": list(selected_nodes.values()), "edges": selected_edges,
             "omitted_nodes": len(nodes) - len(selected_nodes), "omitted_edges": len(edges) - len(selected_edges),
             "focus_summary": summary}
    return graph, notes


def _sources_of(af, manifest, keys):
    """Each passage key's source id."""
    found, wanted = {}, sorted(set(keys))
    for offset in range(0, len(wanted), 500):
        batch = wanted[offset:offset + 500]
        found.update((row["_id"], row["source_id"]) for row in hits(af.query(manifest["table"], {
            "filter_query": {"ids": batch}, "limit": len(batch), "fields": ["source_id"]})))
    return found


def _graph_among(cfg, af, manifest, sources, focuses, shown):
    """The relationships among the `shown` nodes, read from each of `focuses`; nodes keep `shown`'s order."""
    among, nodes, edges = {row["id"] for row in shown}, {}, {}
    for row in focuses:
        part, _ = _relationships(cfg, af, manifest, row, interpret_intent(cfg, row["text"], row), sources, among)
        nodes.update((node["id"], node) for node in part["nodes"])
        edges.update((edge["id"], edge) for edge in part["edges"] if edge["id"] not in edges)
    return {"nodes": [nodes[row["id"]] for row in shown if row["id"] in nodes] if edges else [],
            "edges": list(edges.values()), "omitted_nodes": 0, "omitted_edges": 0}


def _contains(name, phrase):
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", name) is not None


_METRICS = {}


def _node_metrics(af, manifest):
    """Each node's relationship and passage-mention counts, read once per build from Antfly's graph metrics."""
    if manifest["build_id"] not in _METRICS:
        _METRICS[manifest["build_id"]] = {name: dict(af.graph_metric(manifest["table"], "graph", name, 10000))
                                          for name in (LINKS_METRIC, MENTIONS_METRIC)}
    return _METRICS[manifest["build_id"]]


def _group(key, row, metrics, recognized):
    """A node button: its names and its footprint (passage mentions, extracted relationships)."""
    return {**entity_record(key, row), "aliases": sorted({" ".join(a.split()) for a in row.get("aliases") or []}),
            "recognized": recognized, "passages": round(metrics[MENTIONS_METRIC].get(key, 0)),
            "links": round(metrics[LINKS_METRIC].get(key, 0))}


def name_groups(cfg, af, manifest, names, recognized=(), limit=12):
    """Every entity node sharing a word with these names, unmerged, with its footprint.

    `recognized` nodes (named in the question) are always included and come first.
    `close` marks a node whose name or alias contains, or is contained in, one of
    the names ("Tyrion" and "Lord Tyrion Lannister" for "Tyrion Lannister"); the
    rest only share a word ("Jason Lannister").
    """
    phrases = [normalize_name(name) for name in names]
    terms = sorted({term for name in names for term in re.findall(r"\w+", name, flags=re.UNICODE)})
    if not terms:
        return []
    rows = hits(af.query(manifest["table"], {
        "full_text_search": {"disjuncts": [{"match": term, "field": "aliases"} for term in terms]}, "limit": limit}))
    found = {row["_id"] for row in rows}
    rows += [{"_id": key, **af.get(manifest["table"], key)} for key in recognized if key not in found]
    metrics = _node_metrics(af, manifest)
    groups = []
    for row in rows:
        group = _group(row["_id"], row, metrics, row["_id"] in recognized)
        surfaces = {normalize_name(a) for a in [row["canonical_name"], *group["aliases"]]}
        group["close"] = any(_contains(p, a) or _contains(a, p) for p in phrases for a in surfaces if a)
        groups.append(group)
    order = {key: index for index, key in enumerate(recognized)}
    return sorted(groups, key=lambda group: (order.get(group["id"], len(order)), not group["close"], -group["passages"], group["text"]))


def overview(cfg, per_type=24, top_k=2000):
    """Landing view, read live from the current build: entity types and the most-linked entities.

    Types are the ones GLiNER was given; counts and the ranking come from the graph
    (a terms aggregation and Antfly's own degree metric), so a rebuild changes them.
    """
    manifest = dataset.load_build(cfg)
    af = dataset.database(cfg)
    try:
        ranked = [(key, score) for key, score in af.graph_metric(manifest["table"], "graph", LINKS_METRIC, top_k)
                  if key.startswith("entity/")]
        rows = {}
        for offset in range(0, len(ranked), 500):
            batch = [key for key, _ in ranked[offset:offset + 500]]
            rows.update((row["_id"], row) for row in hits(af.query(manifest["table"], {
                "filter_query": {"ids": batch}, "limit": len(batch), "fields": ["canonical_name", "entity_type"]})))
        counts = {bucket["key"]: bucket["doc_count"] for bucket in af.query(manifest["table"], {
            "limit": 0, "aggregations": {"types": {"type": "terms", "field": "entity_type", "size": 100}}})
            ["responses"][0]["aggregations"]["types"]["buckets"]}
    finally:
        af.http.close()
    entities = [{**entity_record(key, rows[key]), "links": score} for key, score in ranked if key in rows]
    chosen, per = [], {}
    for entity in entities:
        per[entity["type"]] = per.get(entity["type"], 0) + 1
        if len(chosen) < per_type or per[entity["type"]] <= per_type:
            chosen.append(entity)
    return {"build_id": manifest["build_id"], "counts": manifest["counts"],
            "types": [{"type": t, "entities": counts.get(t, 0)} for t in cfg["extraction"]["entities"]],
            "top": chosen}


def focused_search(cfg, text, models=None, focus_id=None, build_id=None, navigate=False):
    """Real hybrid question search; explicit indexed recenter is BM25/inference-free."""
    started = time.perf_counter()
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Query must not be empty")
    manifest = dataset.load_build(cfg)
    if navigate and focus_id is None:
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
            if not navigate and not match_candidates(text, [focus]):
                raise ValueError("Selected focus is not an exact indexed name in this question")
        focus_lookup_ms = (time.perf_counter() - focus_lookup_started) * 1000
        if navigate:
            text = focus["text"]  # a node page is the same however you reach it
            packet = {"build_id": manifest["build_id"], "partial": manifest["selection"]["partial"],
                      "query": {"text": text, "entities": []}, "results": [],
                      "timings": {"extraction": {"ms": 0, "skipped": True}, "retrieval": {"ms": 0}}}
        else:
            packet = dataset.query(cfg, text, mode="hybrid", models=models)
            if packet["build_id"] != manifest["build_id"]:
                raise StaleSearch("Build changed during search; search again")
            packet["results"] = [_verified_passage(cfg, sources, row["passage_id"], row)
                                 for row in packet["results"]]
        # Nodes named in the question (whole names only), for the query page's Recognized row.
        candidate_started = time.perf_counter()
        terms = sorted(set(re.findall(r"\w+", text, flags=re.UNICODE)))
        response = af.query(manifest["table"], {
            "full_text_search": {"disjuncts": [{"match": term, "field": "aliases"} for term in terms]},
            "limit": ENTITY_LIMIT})
        indexed = hits(response)
        total, exact = _total(response, len(indexed))
        rows = {row["_id"]: entity_record(row["_id"], row) for row in indexed}
        # Exact alias lookup supplements a bounded name scan; it does not merge names.
        spans = [span["text"] for span in packet["query"]["entities"]]
        for row in dataset.entities_by_surface(af, manifest, spans):
            rows.setdefault(row["_id"], entity_record(row["_id"], row))
        candidates = match_candidates(text, rows.values())
        # A bounded scan cannot prove an apparently unique name is unambiguous.
        if focus is None and exact and total <= len(indexed):
            # Focus only a bare name (its node page) or a recognized question frame about one
            # node ("Who is Tyrion Lannister's father?"); anything else stays a query page.
            about = [row for row in candidates if interpret_intent(cfg, text, row)["status"] != "unrecognized"]
            focus = about[0] if len(about) == 1 else None
        packet["timings"]["focus"] = {"ms": round((time.perf_counter() - candidate_started) * 1000 + focus_lookup_ms, 2)}
        intent = interpret_intent(cfg, text, focus)
        query_page = focus is None or intent["status"] == "focused"
        relationship_started = time.perf_counter()
        graph, notes = _relationships(cfg, af, manifest, focus, intent, sources)
        packet["timings"]["relationships"] = {"ms": round((time.perf_counter() - relationship_started) * 1000, 2)}
        # These diagnostic substages are included in retrieval, not extra stages
        # to sum again when displaying the end-to-end latency breakdown.
        packet["timings"]["retrieval"]["ms"] = round(
            packet["timings"]["retrieval"]["ms"] + packet["timings"]["focus"]["ms"]
            + packet["timings"]["relationships"]["ms"], 2)
        packet["timings"]["retrieval"]["includes"] = ["focus", "relationships"]
        if total > len(indexed) or not exact:
            notes.append(f"Name scan bounded at {ENTITY_LIMIT} names; more may match.")
        if packet["partial"]:
            notes.append("This is a partial build; missing relationships do not establish absence in the full corpus.")
        # Recognized nodes in the order the question names them.
        question = normalize_name(text)
        named = sorted(candidates, key=lambda row: question.find(normalize_name(row["text"])))
        if focus is None and len(named) > 1:
            # A query page's graph: the relationships among the nodes the question names.
            graph = _graph_among(cfg, af, manifest, sources, named[:3], named[:4])
        groups_started = time.perf_counter()
        if query_page:
            names = [row["text"] for row in candidates] + ([focus["text"]] if focus else [])
            groups = name_groups(cfg, af, manifest, names or [text], recognized=[row["id"] for row in named])
        else:
            groups = name_groups(cfg, af, manifest, [focus["text"]])
        packet["timings"]["name_groups"] = {"ms": round((time.perf_counter() - groups_started) * 1000, 2)}
        packet["timeline"] = _timeline(cfg, af, manifest, focus, sources) if focus else []
        if focus:
            # Only passages that mention the node, ranked by the same hybrid search as the
            # question (Antfly embeds the text). On a node's own page, passages that support
            # more of its relationships come first and search breaks ties.
            passages_started = time.perf_counter()
            mentioning = _mentioning_passages(af, manifest, focus["id"])
            if cfg.get("feed"):
                # A feed dataset: the node's record passages go on the right; its messages are marked in the feed.
                origin = _sources_of(af, manifest, mentioning)
                packet["feed_mentions"] = sorted(key for key in mentioning if origin.get(key) == cfg["feed"]["messages"])
                mentioning = {key for key in mentioning if origin.get(key) == cfg["feed"]["record"]}
            ranked = []
            if mentioning:
                request = {**search_request(cfg, text, allowed=mentioning), "limit": len(mentioning), "fields": ["passage_id"]}
                ranked = [row["_id"] for row in hits(af.query(manifest["table"], request))]
            if not query_page:
                support = {}
                for edge in graph["edges"]:
                    for fact in edge["facts"]:
                        support[fact["passage_id"]] = support.get(fact["passage_id"], 0) + 1
                ranked.sort(key=lambda key: -support.get(key, 0))  # stable: search order breaks ties
            ranked = ranked[:cfg["retrieval"]["evidence"]]
            passages = _fetch_passages(cfg, af, manifest, sources, ranked)
            packet["results"] = [passages[key] for key in ranked]
            packet["timings"]["retrieval"]["ms"] = round(packet["timings"]["retrieval"]["ms"]
                                                         + (time.perf_counter() - passages_started) * 1000, 2)
        packet.update(name_groups=groups, focus=focus, intent=intent, notes=notes, graph=graph)
        packet["timings"]["total_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return packet
    finally:
        af.http.close()


GET = re.compile(r"(?<![\d:])(\d{2,3}):([0-5]\d):([0-5]\d)(?![\d:])")
_RECORD_TIMES = {}


def _seconds(hours, minutes, seconds):
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


def _record_times(af, manifest, record):
    """Mission times (GET) each record passage quotes, read once per build: the report's own timeline."""
    if manifest["build_id"] not in _RECORD_TIMES:
        rows = hits(af.query(manifest["table"], {"filter_query": {"term": record, "field": "source_id"},
                                                "limit": 5000, "fields": ["text"]}))
        _RECORD_TIMES[manifest["build_id"]] = {
            row["_id"]: [_seconds(*parts) for parts in GET.findall(row["text"])] for row in rows}
    return _RECORD_TIMES[manifest["build_id"]]


def feed_messages(cfg):
    """The feed source's passages in source order, for the dashboard's message column."""
    manifest = dataset.load_build(cfg)
    af = dataset.database(cfg)
    try:
        rows = hits(af.query(manifest["table"], {"filter_query": {"conjuncts": [
            {"term": "passage", "field": "kind"}, {"term": cfg["feed"]["messages"], "field": "source_id"}]},
            "limit": 5000, "fields": ["text", "timestamp", "context", "source_start"]}))
    finally:
        af.http.close()
    rows.sort(key=lambda row: row["source_start"])
    return {"build_id": manifest["build_id"], "messages": [
        {"id": row["_id"], "text": row["text"], "timestamp": row.get("timestamp"),
         **{key: (row.get("context") or {}).get(key) for key in ("met", "met_end", "speaker", "speakers")}} for row in rows]}


def message_view(cfg, passage_id, build_id):
    """One feed message: the nodes it mentions, their relationships, and the record passages sharing them.

    Record passages that quote a mission time inside the message's span come first: the
    report's account of that moment. Then passages mentioning the message's nodes (rarer
    nodes weigh more) plus the relationships among them they state. Where names don't
    line up ("O2 tank" vs "oxygen tank no. 2"), similar wording fills the remaining slots.
    """
    started = time.perf_counter()
    manifest = dataset.load_build(cfg)
    if build_id != manifest["build_id"]:
        raise StaleSearch("Build changed; reload the page")
    feed, table, limit = cfg["feed"], manifest["table"], cfg["retrieval"]["evidence"]
    af = dataset.database(cfg)
    try:
        sources = {row["id"]: row for row in read_manifest(cfg)}
        message = _fetch_passages(cfg, af, manifest, sources, [passage_id])[passage_id]
        if message["source_id"] != feed["messages"]:
            raise ValueError("Not a feed message")
        keys = sorted({node["key"] for node in af.traverse(table, "graph", [passage_id], edge_types=[MENTIONS],
                                                            direction="out", max_depth=1, limit=MENTION_LIMIT)
                       if node["key"].startswith("entity/")})
        rows = {row["_id"]: row for row in hits(af.query(table, {"filter_query": {"ids": keys}, "limit": len(keys)}))} if keys else {}
        metrics, text = _node_metrics(af, manifest), normalize_name(message["text"])
        groups, position = [], {}
        for key in keys:
            group = {**_group(key, rows[key], metrics, True), "close": True}
            names = [normalize_name(a) for a in [rows[key]["canonical_name"], *group["aliases"]]]
            position[key] = min((text.find(name) for name in names if name in text), default=len(text))
            groups.append(group)
        groups.sort(key=lambda group: (position[group["id"]], group["text"]))
        # Relationships among the message's nodes, across the whole corpus.
        weights = {key: math.log(manifest["counts"]["passages"] / max(1, metrics[MENTIONS_METRIC].get(key, 1))) for key in keys}
        graph = _graph_among(cfg, af, manifest, sources, sorted(groups, key=lambda group: -weights[group["id"]])[:8], groups)
        shared = {}
        for key in keys:
            for passage in _mentioning_passages(af, manifest, key):
                shared.setdefault(passage, set()).add(key)
        origin = _sources_of(af, manifest, shared)
        scores = {passage: sum(weights[key] for key in found) for passage, found in shared.items()
                  if origin.get(passage) == feed["record"]}
        for edge in graph["edges"]:
            for fact in edge["facts"]:
                if fact["passage_id"] in scores:
                    scores[fact["passage_id"]] += 1
        # Mission times are a join key the report shares with the transcript.
        met, timed = message["context"].get("met"), set()
        if met:
            begin, end = (_seconds(*t.split(":")) for t in (met, message["context"].get("met_end") or met))
            timed = {key for key, times in _record_times(af, manifest, feed["record"]).items()
                     if any(begin - 30 <= t <= end + 30 for t in times)}
        for key in timed:
            scores[key] = scores.get(key, 0) + 100
        ranked = sorted(scores, key=lambda key: (-scores[key], key))[:limit]
        if len(ranked) < limit:
            request = search_request(cfg, message["text"])
            request["filter_query"]["conjuncts"].append({"term": feed["record"], "field": "source_id"})
            ranked += [row["_id"] for row in hits(af.query(table, request)) if row["_id"] not in ranked][:limit - len(ranked)]
        passages = _fetch_passages(cfg, af, manifest, sources, ranked)
        results = [{**passages[key], "shared": sorted(shared.get(key, ())),
                    "match": "time" if key in timed else "nodes" if key in scores else "text"} for key in ranked]
    finally:
        af.http.close()
    elapsed = round((time.perf_counter() - started) * 1000, 2)
    return {"build_id": manifest["build_id"], "message": message, "focus": None, "intent": {"status": "all"},
            "name_groups": groups, "graph": graph, "results": results, "notes": [], "timeline": [],
            "timings": {"retrieval": {"ms": elapsed}, "total_ms": elapsed}}
