"""Deterministic evidence, model-boundary and publication regressions; no model downloads."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from kg.antfly import Antfly
from kg import dataset
from kg.dataset_decisions import RESERVED_LABELS, decide_relationship, relationship_labels
from kg.dataset_inputs import prepare_inputs, read_manifest, transcript_turns, windows
from kg.evidence_graph import character_span
from kg.search import interpret_intent, match_candidates, _relationships
from kg.workshop_native import DEFAULT_MODELS, DEFAULT_REVISIONS, NativeInferenceError, NativeModels


def config(root):
    """Resolved runtime fixture, independent of collected corpora and local services."""
    root = Path(root)
    return {
        "version": 2, "dataset": "fixture", "title": "Boundary fixture",
        "config_path": str(root / "config.yaml"), "corpus": str(root / "metadata.jsonl"),
        "output": str(root / "generated"), "dimensions": 768,
        "models": dict(DEFAULT_MODELS), "revisions": dict(DEFAULT_REVISIONS),
        "antfly": {"target": "v0.3.0-rc.1", "addr": "127.0.0.1:8796", "health_port": 4296,
                   "binary_env": "ANTFLY_BIN", "data_dir": str(root / "antfly"),
                   "inference_url": "http://fixture"},
        "passages": {"max_bytes": 80, "max_words": 8, "overlap_words": 2},
        "extraction": {
            "entities": ["character", "house", "city"], "threshold": .5,
            "relations": {
                "parent_of": {"description": "HEAD is a parent of TAIL.",
                              "head_types": ["character"], "tail_types": ["character"]},
                "knows": {"description": "HEAD knows TAIL.",
                          "head_types": ["character"], "tail_types": ["character"]},
                "member_of": {"description": "HEAD belongs to house TAIL.",
                              "head_types": ["character"], "tail_types": ["house"]},
            },
        },
        "relation_decision": {"instruction": "Choose the directed relationship explicitly supported by the text."},
        "retrieval": {"candidates": 40, "graph_candidates": 100, "evidence": 5, "text_reserve": 1},
        "collection": {"sources": [{"id": "s1", "path": "raw/source.md"}]},
    }


def source_fixture(cfg, text="😀 Ned saw Árya. Snow covered the old city beside the river."):
    root = Path(cfg["config_path"]).parent
    path = root / "raw/source.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    row = {"id": "s1", "path": "raw/source.md", "title": "Source fixture", "url": "https://example.org/source",
           "media_type": "text/markdown", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "fetched_at": "2026-10-17T00:00:00Z", "license": "CC0-1.0", "attribution": "Fixture author",
           "redistribution": "allowed", "source_kind": "encyclopedia_prose"}
    write_manifest(cfg, [row])
    return row, path, text


def write_manifest(cfg, rows):
    Path(cfg["corpus"]).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def mention(text, surface, label="character", start=None, score=.9):
    start = text.index(surface) if start is None else start
    return {"text": surface, "start": start, "end": start + len(surface), "label": label, "score": score}


class DecideModels:
    """Only inference is stubbed."""
    def __init__(self, winner="knows"):
        self.winner, self.calls = winner, []

    def classify(self, text, labels, instruction):
        self.calls.append((text, deepcopy(labels), instruction))
        rest = .1 / (len(labels) - 1)
        return {"label": self.winner, "scores": {label: .9 if label == self.winner else rest for label in labels}}


class DecideBoundaries(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cfg = config(directory.name)
        self.text = "Ned saw Árya. Ned left."
        self.passage = {"text": self.text}

    def test_candidates_are_the_type_compatible_predicates_plus_reserved_outcomes(self):
        self.assertEqual(set(relationship_labels(self.cfg, "character", "character")),
                         {"parent_of", "knows", *RESERVED_LABELS})
        self.assertEqual(set(relationship_labels(self.cfg, "character", "house")), {"member_of", *RESERVED_LABELS})
        self.assertEqual(set(relationship_labels(self.cfg, "city", "house")), set(RESERVED_LABELS))

    def test_decide_reports_every_score_and_the_exact_input(self):
        models = DecideModels("knows")
        result = decide_relationship(self.cfg, self.passage, mention(self.text, "Ned"), mention(self.text, "Árya"), models)
        self.assertEqual(result["label"], "knows")
        self.assertEqual(set(result["scores"]), {"parent_of", "knows", *RESERVED_LABELS})
        self.assertIn("HEAD: Ned [character]", result["input"])
        self.assertEqual(models.calls[0][2], self.cfg["relation_decision"]["instruction"])

    def test_decide_rejects_a_span_that_is_not_in_the_passage(self):
        bad = {**mention(self.text, "Ned"), "start": 1}
        with self.assertRaises(ValueError):
            decide_relationship(self.cfg, self.passage, bad, mention(self.text, "Árya"), DecideModels())


class ExtractorSpans(unittest.TestCase):
    def test_utf8_byte_offsets_become_character_offsets(self):
        text = "😀 Árya saw Ned."
        start = len(text[:text.index("Ned")].encode("utf-8"))
        self.assertEqual(character_span(text, {"text": "Ned", "start": start, "end": start + 3}),
                         {"text": "Ned", "start": text.index("Ned"), "end": text.index("Ned") + 3})

    def test_misaligned_spans_are_rejected_not_guessed(self):
        text = "😀 Árya saw Ned."
        for start, end in ((1, 4), (0, 99), (5, 9)):
            with self.subTest(start=start), self.assertRaises(ValueError):
                character_span(text, {"text": "Árya", "start": start, "end": end})


class InputIntegrity(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cfg = config(directory.name)
        self.row, self.path, self.text = source_fixture(self.cfg)

    def test_unicode_passages_use_exact_raw_character_offsets_and_cover_every_word(self):
        self.cfg["passages"] = {"max_bytes": 32, "max_words": 5, "overlap_words": 1}
        prepared = prepare_inputs(self.cfg)
        for passage in prepared["passages"]:
            self.assertEqual(passage["text"], self.text[passage["source_start"]:passage["source_end"]])
            self.assertLessEqual(len(passage["text"].encode("utf-8")), 32)
        for position, char in enumerate(self.text):
            if not char.isspace():
                self.assertTrue(any(p["source_start"] <= position < p["source_end"] for p in prepared["passages"]), position)
        self.assertEqual(self.path.read_bytes(), self.text.encode("utf-8"))

    def test_an_indivisible_multibyte_token_cannot_silently_exceed_budget(self):
        with self.assertRaises(ValueError):
            list(windows("雪雪雪", max_bytes=8, max_words=3, overlap_words=0))

    def test_unknown_inline_speakers_remain_exact_text_without_timestamp_inference(self):
        text = "055:50:05 CC\nHello. 02 07 50 06 Yes. Still here. 02 07 50 07 [???] uncertain.\n055:50:08 LMP\nOkay."
        turns = transcript_turns(text)
        self.assertEqual([turn["speaker"] for turn in turns], ["CC", "Yes", "[???]", "LMP"])
        for turn in turns:
            self.assertEqual(turn["text"], text[turn["start_char"]:turn["end_char"]])
        for turn in turns[1:3]:
            self.assertEqual(turn["speaker_role"], "unknown")
            self.assertTrue(turn["parse_warnings"])
            self.assertTrue(turn["text"].startswith(turn["speaker"]))
            self.assertNotIn("02 07 50", turn["text"])
        self.assertEqual(turns[0]["speaker_role"], "CC")

    def test_changed_raw_is_rejected_without_mutating_raw_or_metadata(self):
        changed = self.path.read_bytes() + b" Changed."
        self.path.write_bytes(changed)
        metadata = Path(self.cfg["corpus"]).read_bytes()
        for operation in (read_manifest, prepare_inputs):
            with self.subTest(operation=operation.__name__):
                with self.assertRaises(ValueError) as error:
                    operation(self.cfg)
                self.assertIn(str(self.path), str(error.exception))
                self.assertIn("--refresh", str(error.exception))
                self.assertEqual(self.path.read_bytes(), changed)
                self.assertEqual(Path(self.cfg["corpus"]).read_bytes(), metadata)

    def test_missing_raw_is_actionable_and_does_not_recreate_it(self):
        self.path.unlink()
        with self.assertRaises(ValueError) as error:
            read_manifest(self.cfg)
        self.assertIn(str(self.path), str(error.exception))
        self.assertIn("collect", str(error.exception))
        self.assertFalse(self.path.exists())

    def test_relative_and_symlink_escapes_cannot_read_outside_raw(self):
        root = Path(self.cfg["config_path"]).parent
        outside = root / "outside.md"
        outside.write_text("Private source outside raw.", encoding="utf-8")
        link = self.path.parent / "escape.md"
        link.symlink_to(outside)
        for relative in ("raw/../outside.md", "raw/escape.md"):
            with self.subTest(relative=relative):
                row = {**self.row, "path": relative, "sha256": hashlib.sha256(outside.read_bytes()).hexdigest()}
                write_manifest(self.cfg, [row])
                with self.assertRaises(ValueError):
                    read_manifest(self.cfg)
                self.assertEqual(outside.read_text(encoding="utf-8"), "Private source outside raw.")

    def test_changed_but_consistently_refreshed_source_invalidates_build_receipt(self):
        prepared = prepare_inputs(self.cfg)
        receipt = {"build_id": "previous", "config_signature": dataset.build_signature(self.cfg),
                   "corpus_sha256": prepared["corpus_sha256"], "selection": prepared["selection"]}
        dataset.save(Path(self.cfg["output"]) / "build.json", receipt)
        self.assertEqual(dataset.load_build(self.cfg)["build_id"], "previous")
        self.path.write_text(self.text + " A refreshed statement.", encoding="utf-8")
        write_manifest(self.cfg, [{**self.row, "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest()}])
        with self.assertRaises(ValueError):
            dataset.load_build(self.cfg)


class NativeBoundaries(unittest.TestCase):
    def native(self, response, **options):
        client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response)),
                              base_url="http://fixture/ai/v1/")
        with patch("kg.workshop_native.httpx.Client", return_value=client):
            models = NativeModels(dict(DEFAULT_MODELS), base_url="http://fixture", **options)
        self.addCleanup(models.close)
        return models

    def test_native_byte_spans_convert_to_exact_unicode_characters(self):
        text = "😀 Árya in 雪"
        response = {"data": [{"offset_unit": "utf8_bytes", "entities": [
            {"text": "Árya", "start": 5, "end": 10, "label": "character", "score": .9}], "relations": []}]}
        result = self.native(response).extract(text, {"entities": ["character"], "relations": []}, .5)
        self.assertEqual(result["entities"], [mention(text, "Árya")])

    def test_native_malformed_byte_spans_are_not_rounded_or_reanchored(self):
        text = "😀 Árya in 雪"
        for start, end in ((6, 10), (5, 9), (-1, 10), (True, 10), (5, 100)):
            with self.subTest(start=start, end=end):
                response = {"data": [{"offset_unit": "utf8_bytes", "entities": [
                    {"text": "Árya", "start": start, "end": end, "label": "character", "score": .9}], "relations": []}]}
                with self.assertRaises(NativeInferenceError):
                    self.native(response).extract(text, {"entities": ["character"], "relations": []}, .5)

class BuildPublication(unittest.TestCase):
    def test_failed_indexed_passage_probe_preserves_previous_usable_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(directory)
            source_fixture(cfg)
            prepared = prepare_inputs(cfg)
            previous = {"build_id": "previous", "config_signature": dataset.build_signature(cfg),
                        "corpus_sha256": prepared["corpus_sha256"], "selection": prepared["selection"]}
            path = Path(cfg["output"]) / "build.json"
            dataset.save(path, previous)
            before = path.read_bytes()
            probes = []

            def handler(request):
                if request.method == "GET" and request.url.path == "/db/v1/tables":
                    return httpx.Response(200, json=[])
                if request.method == "GET" and "/indexes/" in request.url.path:
                    return httpx.Response(200, json={"status": {"readiness": {"state": "ready", "complete": True}}})
                if request.method == "POST" and request.url.path.endswith("/query"):
                    probes.append(request.url.path)
                    return httpx.Response(200, json={"hits": {"hits": []}})
                if request.method == "POST":
                    return httpx.Response(200, json={})
                raise AssertionError(f"Unexpected database request: {request.method} {request.url}")

            af = Antfly("http://fixture")
            af.http.close()
            af.http = httpx.Client(base_url="http://fixture", transport=httpx.MockTransport(handler))
            self.addCleanup(af.http.close)
            with patch("kg.dataset.database", return_value=af), \
                    patch("kg.dataset.binary_info", return_value={"version": "isolated_fixture"}):
                with self.assertRaises(ValueError):
                    dataset.build(cfg, max_passages=1)
            self.assertTrue(probes, "The build must reach the intentionally rejected passage probe")
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(dataset.load_build(cfg), previous)

    def test_failed_atomic_receipt_replace_preserves_previous_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "build.json"
            dataset.save(path, {"build_id": "previous"})
            before = path.read_bytes()
            with patch.object(Path, "replace", side_effect=OSError("Publication interrupted")):
                with self.assertRaises(OSError):
                    dataset.save(path, {"build_id": "unpublished"})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["build_id"], "previous")


class FocusedSearchBoundaries(unittest.TestCase):
    def test_whole_name_matching_keeps_ambiguity_without_inventing_aliases(self):
        rows = [
            {"id": "short", "text": "Tyrion", "type": "character"},
            {"id": "full", "text": "Tyrion Lannister", "type": "character"},
            {"id": "house", "text": "Lannister", "type": "house"},
        ]
        self.assertEqual([r["id"] for r in match_candidates("Who is TYRION LANNISTER's father?", rows)], ["full"])
        self.assertEqual(match_candidates("the Imp", rows), [])
        self.assertEqual(match_candidates("Tyrionette", rows), [])
        ambiguous = [{"id": "city", "text": "Winterfell", "type": "city"},
                     {"id": "house", "text": "Winterfell", "type": "house"}]
        self.assertEqual({r["id"] for r in match_candidates("Where is Winterfell?", ambiguous)}, {"city", "house"})
        self.assertEqual({r["id"] for r in match_candidates("Tyrion and Winterfell", rows + ambiguous)},
                         {"short", "city", "house"})

    def test_parent_questions_preserve_direction_and_specificity(self):
        cfg = {"extraction": {"relations": {"parent_of": {}}}}
        focus = {"text": "Tyrion Lannister"}
        for question, label, direction in (
            ("Who is Tyrion Lannister's father?", "father", "incoming"),
            ("Who is Tyrion Lannister’s mother?", "mother", "incoming"),
            ("Who are Tyrion Lannister's parents?", "parents", "incoming"),
            ("Who are Tyrion Lannister's children?", "children", "outgoing"),
        ):
            with self.subTest(question=question):
                intent = interpret_intent(cfg, question, focus)
                self.assertEqual((intent["status"], intent["predicate"], intent["direction"], intent["label"]),
                                 ("focused", "parent_of", direction, label))
        self.assertEqual(interpret_intent({"extraction": {"relations": {"affects": {}}}}, "What affects the oxygen tank?",
                                          {"text": "oxygen\n\ntank"})["predicate"], "affects")
        self.assertEqual(interpret_intent(cfg, "Why is Tyrion Lannister sad?", focus)["status"], "unrecognized")
        self.assertEqual(interpret_intent(cfg, "Who are Tyrion Lannister's sons?", focus)["status"], "unrecognized")
        self.assertEqual(interpret_intent({"extraction": {"relations": {}}},
                                        "Who is Tyrion Lannister's father?", focus)["status"], "unrecognized")
        self.assertEqual(interpret_intent(cfg, "Who is Tyrion Lannister's father?", focus, navigate=True)["status"], "all")

    def relationship_fixture(self):
        """Passages plus the facts Antfly's extractor and resolver would materialize."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        cfg = config(directory.name)
        text = "😀 Ned is the father of Árya. Cersei is the mother of Árya."
        _, _, text = source_fixture(cfg, text)
        sources = {row["id"]: row for row in read_manifest(cfg)}
        docs, facts = {}, []
        for start, end, parent in ((0, text.index(" Cersei"), "Ned"), (text.index("Cersei"), len(text), "Cersei")):
            excerpt = text[start:end]
            key = "passage:" + dataset.digest(["s1", sources["s1"]["sha256"], start, end, excerpt])[:24]
            docs[key] = {"text": excerpt, "source_id": "s1", "source_start": start, "source_end": end,
                         "kind": "passage", "passage_id": key}

            def span(surface):  # the extractor reports UTF-8 byte offsets
                begin = len(excerpt[:excerpt.index(surface)].encode("utf-8"))
                return {"text": surface, "label": "character", "start": begin, "end": begin + len(surface.encode("utf-8"))}
            facts.append({"from": "entity/" + parent.lower(), "to": "entity/árya", "type": "parent_of",
                          "owner_document": key, "weight": .9, "metadata": {"source": span(parent), "target": span("Árya")}})
        entities = {"entity/" + name.lower(): {"canonical_name": name, "entity_type": "character", "aliases": [name]}
                    for name in ("Ned", "Cersei", "Árya")}

        class Index:
            def get(self, table, key):
                return entities.get(key) or docs.get(key)

            def traverse(self, table, index, keys, edge_types, direction, **options):
                start, = keys
                if edge_types == ["mentions"]:
                    return [{"key": k} for k, d in docs.items() if entities[start]["canonical_name"] in d["text"]]
                nodes = []
                for f in facts:
                    stored = (f["from"], f["to"]) if direction == "out" else (f["to"], f["from"])
                    if f["type"] in edge_types and stored[0] == start:
                        nodes.append({"key": stored[1], "path_edges": [{**f, "from": {"key": start}, "to": {"key": stored[1]},
                                                                        "direction": direction}]})
                return nodes

            def query(self, table, body):
                wanted = {clause["term"] for clause in body["filter_query"]["disjuncts"]}
                selected = [{"_id": key, "_source": row} for key, row in docs.items() if key in wanted]
                return {"responses": [{"hits": {"hits": selected, "total": {"value": len(selected), "relation": "eq"}}}]}

        focus = {"id": "entity/árya", "text": "Árya", "type": "character"}
        return cfg, Index(), {"table": "isolated"}, focus, sources, docs

    def test_focused_relationships_have_their_own_evidence_and_parent_direction(self):
        cfg, index, manifest, focus, sources, docs = self.relationship_fixture()
        parents = interpret_intent(cfg, "Who are Árya's parents?", focus)
        graph, _ = _relationships(cfg, index, manifest, focus, parents, sources)
        names = {node["id"]: node["text"] for node in graph["nodes"]}
        self.assertEqual({(names[e["source"]], names[e["target"]]) for e in graph["edges"]},
                         {("Ned", "Árya"), ("Cersei", "Árya")})
        for edge in graph["edges"]:
            evidence, = edge["evidence"]
            self.assertEqual(evidence["text"], sources["s1"]["text"][evidence["source_start"]:evidence["source_end"]])
            fact, = edge["facts"]
            self.assertEqual(evidence["text"][fact["tail"]["start"]:fact["tail"]["end"]], "Árya")
        father = interpret_intent(cfg, "Who is Árya's father?", focus)
        graph, notes = _relationships(cfg, index, manifest, focus, father, sources)
        self.assertEqual(len(graph["edges"]), 2)
        self.assertTrue(any("father from a mother" in note for note in notes))
        children = interpret_intent(cfg, "Who are Árya's children?", focus)
        graph, _ = _relationships(cfg, index, manifest, focus, children, sources)
        self.assertEqual(graph["edges"], [])

    def test_a_question_focuses_the_one_name_its_relationship_frame_is_about(self):
        cfg = {"extraction": {"relations": {"parent_of": {}}}}
        rows = [{"id": "f", "text": "father", "type": "house"}, {"id": "t", "text": "Tyrion Lannister", "type": "character"}]
        question = "Who is Tyrion Lannister's father?"
        about = [row["id"] for row in match_candidates(question, rows) if interpret_intent(cfg, question, row)["status"] == "focused"]
        self.assertEqual(about, ["t"])

    def test_parallel_facts_vote_for_one_labelled_connection_per_pair(self):
        cfg, index, manifest, focus, sources, docs = self.relationship_fixture()
        key = next(iter(docs))
        text = docs[key]["text"]

        def span(surface):
            begin = len(text[:text.index(surface)].encode("utf-8"))
            return {"text": surface, "label": "character", "start": begin, "end": begin + len(surface.encode("utf-8"))}
        knows = {"type": "knows", "owner_document": key, "weight": .5, "direction": "out",
                 "from": {"key": "entity/árya"}, "to": {"key": "entity/ned"},
                 "metadata": {"source": span("Árya"), "target": span("Ned")}}
        original = index.traverse

        def traverse(table, idx, keys, edge_types, direction, **options):
            extra = [{"key": "entity/ned", "path_edges": [knows]}] * 2 if direction == "out" and "knows" in edge_types else []
            return original(table, idx, keys, edge_types, direction, **options) + extra
        index.traverse = traverse
        everything = {"status": "all", "label": "all", "predicate": None, "direction": "both"}
        graph, _ = _relationships(cfg, index, manifest, focus, everything, sources)
        ned = next(e for e in graph["edges"] if "entity/ned" in (e["source"], e["target"]))
        self.assertEqual((ned["predicate"], ned["source"], ned["target"], ned["support"]), ("knows", "entity/árya", "entity/ned", 3))
        self.assertEqual([(v["predicate"], v["count"]) for v in ned["votes"]], [("knows", 2), ("parent_of", 1)])

    def test_a_fact_whose_spans_do_not_match_its_passage_is_skipped_and_reported(self):
        cfg, index, manifest, focus, sources, docs = self.relationship_fixture()
        key = next(iter(docs))
        docs[key] = {**docs[key], "text": docs[key]["text"].replace("Ned", "Jon")}
        with patch("kg.search._verified_passage", side_effect=lambda cfg, sources, key, row: row):
            graph, notes = _relationships(cfg, index, manifest, focus,
                                          interpret_intent(cfg, "Who are Árya's parents?", focus), sources)
        self.assertEqual(len(graph["edges"]), 1)
        self.assertTrue(any("skipped" in note for note in notes))


if __name__ == "__main__":
    unittest.main()
