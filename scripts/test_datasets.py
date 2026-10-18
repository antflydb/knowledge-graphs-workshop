"""Deterministic evidence, model-boundary and publication regressions; no model downloads."""
from contextlib import closing
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import httpx

from kg.antfly import Antfly
from kg import dataset
from kg.dataset_decisions import analyze_chunk, differentiate_relations, extract_chunk
from kg.dataset_inputs import prepare_inputs, read_manifest, transcript_turns, windows
from kg.evidence_graph import graph_documents
from kg.search import interpret_intent, match_candidates, _relationships, _specific_parent
from kg.workshop_native import DEFAULT_MODELS, DEFAULT_REVISIONS, NativeInferenceError, NativeModels
from kg.workshop_python import _normalize_extraction


def config(root, track="stable"):
    """Resolved runtime fixture, independent of collected corpora and local services."""
    root = Path(root)
    return {
        "version": 2, "dataset": "fixture", "title": "Boundary fixture", "track": track,
        "config_path": str(root / "config.yaml"), "corpus": str(root / "metadata.jsonl"),
        "output": str(root / "generated"), "dimensions": 768,
        "models": dict(DEFAULT_MODELS), "revisions": dict(DEFAULT_REVISIONS),
        "tracks": {
            "stable": {"provider": "python", "target": "v0.2.5", "addr": "127.0.0.1:8795"},
            "latest": {"provider": "native", "target": "main", "addr": "127.0.0.1:8796",
                       "inference_url": "http://fixture"},
        },
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


def proposal(head, tail, label="parent_of", score=.8):
    return {"head": {k: head[k] for k in ("text", "start", "end")},
            "tail": {k: tail[k] for k in ("text", "start", "end")}, "label": label, "score": score}


class BoundaryModels:
    """Only inference is stubbed; cache, analysis and graph construction are real."""
    def __init__(self, extracted=None, winner="knows"):
        self.extracted = extracted if extracted is not None else {"entities": [], "relations": []}
        self.winner = winner
        self.metadata = {"provider": "isolated_boundary_fixture"}
        self.extractions = 0
        self.classifications = []

    def extract(self, text, schema, threshold):
        self.extractions += 1
        return deepcopy(self.extracted)

    def classify(self, text, labels, instruction):
        self.classifications.append((text, deepcopy(labels), instruction))
        if self.winner not in labels:
            raise AssertionError("Fixture winner is not a candidate")
        remainder = .1 / (len(labels) - 1)
        return {"label": self.winner,
                "scores": {label: .9 if label == self.winner else remainder for label in labels}}

    def embed(self, texts, task_type, dimensions):
        return [[1.] + [0.] * (dimensions - 1) for _ in texts]

    def close(self):
        pass


class RelationshipBoundaries(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cfg = config(self.directory.name)
        self.cache = dataset.Cache(self.cfg)
        self.addCleanup(self.cache.close)
        self.text = "Ned saw Árya. Ned left."
        self.head = mention(self.text, "Ned")
        self.tail = mention(self.text, "Árya")
        self.extracted = {"entities": [self.head, self.tail], "relations": [proposal(self.head, self.tail)]}

    def analyze(self, extracted=None, winner="knows", context=None):
        models = BoundaryModels(self.extracted if extracted is None else extracted, winner)
        result = analyze_chunk(self.cfg, self.text, self.cache, models, context=context)
        return result, models

    def test_decide_can_change_predicate_without_losing_base_proposals(self):
        original = deepcopy(self.extracted)
        original["relations"].append(proposal(self.head, self.tail, "knows", .6))
        analysis, models = self.analyze(original)
        self.assertEqual(analysis["proposals"], original["relations"])
        self.assertEqual(analysis["entities"], original["entities"])
        self.assertEqual(len(models.classifications), 1)
        decision = analysis["decisions"][0]
        self.assertEqual(decision["proposal_indices"], [0, 1])
        self.assertEqual(set(decision["labels"]), {"parent_of", "knows", "no_relation", "unclear"})
        self.assertEqual(set(decision["scores"]), set(decision["labels"]))
        assertion, = analysis["assertions"]
        self.assertEqual((assertion["head"], assertion["predicate"], assertion["tail"]),
                         (self.head, "knows", self.tail))
        self.assertEqual(assertion["status"], "model_classified_not_verified")
        self.assertEqual(assertion["decision_index"], 0)
        self.assertEqual(original["relations"][0]["label"], "parent_of")

    def test_ordered_occurrences_are_distinct_and_sorted_without_all_pairs_expansion(self):
        later = mention(self.text, "Ned", start=self.text.rindex("Ned"))
        extracted = {"entities": [self.head, self.tail, later], "relations": [
            proposal(later, self.tail), proposal(self.tail, self.head), proposal(self.head, self.tail)]}
        analysis, models = self.analyze(extracted)
        self.assertEqual(len(models.classifications), 3)
        self.assertEqual([(d["head"]["start"], d["tail"]["start"]) for d in analysis["decisions"]],
                         [(self.head["start"], self.tail["start"]),
                          (self.tail["start"], self.head["start"]), (later["start"], self.tail["start"])])
        self.assertEqual([(a["head"]["start"], a["tail"]["start"]) for a in analysis["assertions"]],
                         [(d["head"]["start"], d["tail"]["start"]) for d in analysis["decisions"]])

    def test_no_relation_creates_no_semantic_edge_but_preserves_review(self):
        analysis, _ = self.analyze(winner="no_relation")
        self.assertEqual(analysis["assertions"], [])
        self.assertEqual(analysis["proposals"], self.extracted["relations"])
        self.assertEqual(analysis["decisions"][0]["label"], "no_relation")
        docs = graph_documents(self.cfg, self.prepared(), {"passage:one": analysis})
        self.assertEqual([row for row in docs.values() if row["kind"] == "assertion"], [])
        self.assertEqual(docs["passage:one"]["analysis"]["decisions"], analysis["decisions"])

    def test_unclear_is_retained_as_unresolved_not_a_semantic_assertion(self):
        analysis, _ = self.analyze(winner="unclear")
        decision, = analysis["decisions"]
        self.assertEqual(analysis["assertions"], [])
        self.assertEqual(decision["label"], "unclear")
        self.assertTrue(decision["unresolved_reason"])
        self.assertIn("unclear", decision["scores"])
        self.assertIsNotNone(decision["input"])

    def test_untyped_ambiguous_and_below_threshold_endpoints_are_not_invented(self):
        variants = {
            "untyped": [self.head],
            "ambiguous": [self.head, self.tail, {**self.tail, "label": "house"}],
            "below_threshold": [self.head, {**self.tail, "score": .49}],
            "incompatible_types": [{**self.head, "label": "house"}, self.tail],
        }
        for name, entities in variants.items():
            with self.subTest(name=name):
                extracted = {"entities": entities, "relations": deepcopy(self.extracted["relations"])}
                models = BoundaryModels(extracted)
                analysis = differentiate_relations(self.cfg, self.text, extracted, self.cache, models)
                self.assertEqual(models.classifications, [])
                self.assertEqual(analysis["assertions"], [])
                self.assertEqual(analysis["proposals"], extracted["relations"])
                self.assertEqual(analysis["entities"], entities)
                self.assertTrue(analysis["decisions"][0]["unresolved_reason"])

    def test_entities_without_base_proposals_make_zero_relationship_calls(self):
        analysis, models = self.analyze({"entities": [self.head, self.tail], "relations": []})
        self.assertEqual(models.classifications, [])
        self.assertEqual(analysis["decisions"], [])
        self.assertEqual(analysis["assertions"], [])
        self.assertEqual(self.cache.misses["classify"], 0)
        docs = graph_documents(self.cfg, self.prepared(), {"passage:one": analysis})
        self.assertEqual([d for d in docs.values() if d["kind"] == "assertion"], [])

    def test_wrong_character_offsets_fail_instead_of_reanchoring_repeated_name(self):
        malformed = deepcopy(self.extracted)
        malformed["entities"][0]["start"] = 1
        with self.assertRaises(ValueError):
            extract_chunk(self.cfg, self.text, self.cache, BoundaryModels(malformed))

    def test_context_is_part_of_classification_cache_but_not_extraction_cache(self):
        models = BoundaryModels(self.extracted)
        contexts = [{"kind": "mission_transcript", "met": "055:50:05", "speaker": "CC"},
                    {"kind": "mission_transcript", "met": "055:50:06", "speaker": "CC"},
                    {"kind": "mission_transcript", "met": "055:50:05", "speaker": "CDR"},
                    {"kind": "review_board", "printed_label": "5-12"}]
        for context in contexts:
            first = analyze_chunk(self.cfg, self.text, self.cache, models, context=context)
            again = analyze_chunk(self.cfg, self.text, self.cache, models, context=context)
            self.assertEqual(first, again)
            self.assertEqual(first["context"], context)
        self.assertEqual(models.extractions, 1)
        self.assertEqual(len(models.classifications), len(contexts))
        self.assertEqual(self.cache.misses["classify"], len(contexts))
        self.assertEqual(self.cache.hits["classify"], len(contexts))

    def test_changed_candidate_meaning_or_instruction_invalidates_decision_cache(self):
        models = BoundaryModels(self.extracted)
        analyze_chunk(self.cfg, self.text, self.cache, models)
        analyze_chunk(self.cfg, self.text, self.cache, models)
        self.cfg["extraction"]["relations"]["knows"]["description"] = "HEAD explicitly recognizes TAIL."
        analyze_chunk(self.cfg, self.text, self.cache, models)
        self.cfg["relation_decision"]["instruction"] = "Classify only direct statements, not hypotheses."
        analyze_chunk(self.cfg, self.text, self.cache, models)
        self.assertEqual(models.extractions, 1)
        self.assertEqual(len(models.classifications), 3)

    def prepared(self):
        return {"sources": {"s1": {"id": "s1", "title": "Metadata alone is not an entity"}},
                "passages": [{"id": "passage:one", "source_id": "s1", "source_start": 100,
                              "source_end": 100 + len(self.text), "text": self.text}]}

    def test_graph_keeps_direction_distinct_mentions_and_absolute_unicode_offsets(self):
        analysis, _ = self.analyze()
        prepared = self.prepared()
        prepared["passages"].append({**prepared["passages"][0], "id": "passage:two", "source_start": 500,
                                     "source_end": 500 + len(self.text)})
        docs = graph_documents(self.cfg, prepared, {p["id"]: deepcopy(analysis) for p in prepared["passages"]})
        assertions = [d for d in docs.values() if d["kind"] == "assertion"]
        self.assertEqual({d["source_start"] for d in assertions}, {100, 500})
        for row in assertions:
            self.assertEqual(docs[row["subject_id"]]["text"], "Ned")
            self.assertEqual(docs[row["object_id"]]["text"], "Árya")
            self.assertEqual(row["predicate"], "knows")
            passage = docs[row["_edges"]["graph"]["supported_by"][0]["target"]]
            self.assertEqual(row["source_end"], passage["source_end"])
        mentions = [d for d in docs.values() if d["kind"] == "mention" and d["text"] == "Árya"]
        self.assertEqual({(d["source_start"], d["source_end"]) for d in mentions},
                         {(base + self.tail["start"], base + self.tail["end"]) for base in (100, 500)})
        self.assertEqual(len({d["_edges"]["graph"]["refers_to"][0]["target"] for d in mentions}), 1)


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

    def test_capacity_refinement_preserves_every_source_word_and_unique_offsets(self):
        class LimitedModels(BoundaryModels):
            def extract(self, text, schema, threshold):
                if len(text.split()) > 3:
                    raise ValueError("schema, instruction and text require 555 tokens; limit is 512")
                return super().extract(text, schema, threshold)
        prepared = prepare_inputs(self.cfg)
        before = self.path.read_bytes()
        with closing(dataset.Cache(self.cfg)) as cache:
            extracted = dataset.extract_passages(self.cfg, prepared, cache, LimitedModels())
        self.assertEqual(len(extracted), len(prepared["passages"]))
        self.assertTrue(prepared["selection"]["capacity_splits"])
        for passage in prepared["passages"]:
            self.assertEqual(passage["text"], self.text[passage["source_start"]:passage["source_end"]])
            self.assertLessEqual(len(passage["text"].split()), 3)
        for position, char in enumerate(self.text):
            if not char.isspace():
                self.assertTrue(any(p["source_start"] <= position < p["source_end"] for p in prepared["passages"]), position)
        self.assertEqual(self.path.read_bytes(), before)

    def test_relation_capacity_refines_whole_windows_without_losing_source_coverage(self):
        class LimitedModels(BoundaryModels):
            def extract(self, text, schema, threshold):
                words = text.split()
                head = mention(text, words[0])
                tail = mention(text, words[-1], start=text.rindex(words[-1]))
                return {"entities": [head, tail], "relations": [proposal(head, tail)]}

            def classify(self, text, labels, instruction):
                chunk = text.split("Text:\n", 1)[1].split("\n\nHEAD:", 1)[0]
                if len(chunk.split()) > 3:
                    raise ValueError("schema, instruction and text require 533 tokens; limit is 512")
                self.winner = "no_relation"
                return super().classify(text, labels, instruction)
        prepared = prepare_inputs(self.cfg)
        with closing(dataset.Cache(self.cfg)) as cache:
            with patch("kg.dataset.create_models", side_effect=lambda *args, **kwargs: LimitedModels()):
                analyses = dataset.analyze_passages(self.cfg, prepared, cache)
        self.assertEqual(set(analyses), {row["id"] for row in prepared["passages"]})
        self.assertTrue(any(split["stage"] == "relations" for split in prepared["selection"]["capacity_splits"]))
        for row in prepared["passages"]:
            self.assertEqual(row["text"], self.text[row["source_start"]:row["source_end"]])
            self.assertLessEqual(len(row["text"].split()), 3)
            self.assertEqual(analyses[row["id"]]["assertions"], [])
        for position, char in enumerate(self.text):
            if not char.isspace():
                self.assertTrue(any(row["source_start"] <= position < row["source_end"] for row in prepared["passages"]), position)

    def test_capacity_refinement_does_not_split_an_indivisible_source_word(self):
        class LimitedModels(BoundaryModels):
            def extract(self, text, schema, threshold):
                raise ValueError("schema, instruction and text require 555 tokens; limit is 512")
        source_fixture(self.cfg, "indivisible")
        prepared = prepare_inputs(self.cfg)
        with closing(dataset.Cache(self.cfg)) as cache:
            with self.assertRaises(ValueError):
                dataset.extract_passages(self.cfg, prepared, cache, LimitedModels())

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

    def test_python_adapter_does_not_guess_repeated_relation_occurrences(self):
        text = "Arya saw Ned. Arya left."
        raw = {"entities": {}, "relation_extraction": {"parent_of": [{
            "head": {"text": "Ned", "confidence": .9}, "tail": {"text": "Arya", "confidence": .9}}]}}
        normalized = _normalize_extraction(text, raw, {"entities": ["character"], "relations": ["parent_of"]})
        self.assertEqual(normalized["relations"], [])

    def test_native_embedding_cannot_change_the_indexed_artifact_identity(self):
        response = {"model_identity": "b" * 64,
                    "data": [{"index": 0, "embedding": [1.] + [0.] * 127}]}
        models = self.native(response, model_identity="a" * 64)
        with self.assertRaises(NativeInferenceError):
            models.embed(["query"], "RETRIEVAL_QUERY", 128)

    def test_cached_embedding_from_another_artifact_is_rejected_without_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(directory, track="latest")
            models = SimpleNamespace(metadata={"observed": {"embed": {"model_identity": "a" * 64}}})
            args = ["query", "RETRIEVAL_QUERY", 128]
            with closing(dataset.Cache(cfg)) as cache:
                cache.call("embed", args, lambda: [[1.] + [0.] * 127], models)
            with closing(dataset.Cache(cfg, expected_identity="b" * 64)) as cache:
                with self.assertRaises(ValueError):
                    cache.call("embed", args, lambda: self.fail("A cache hit must not invoke a different model"), models)


class BuildPublication(unittest.TestCase):
    def test_failed_indexed_vector_roundtrip_preserves_previous_usable_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(directory)
            source_fixture(cfg)
            prepared = prepare_inputs(cfg)
            previous = {"build_id": "previous", "config_signature": dataset.build_signature(cfg),
                        "corpus_sha256": prepared["corpus_sha256"], "selection": prepared["selection"]}
            path = Path(cfg["output"]) / "build.json"
            dataset.save(path, previous)
            before = path.read_bytes()
            failed_probe = []

            def handler(request):
                if request.method == "GET" and request.url.path == "/db/v1/tables":
                    return httpx.Response(200, json=[])
                if request.method == "POST" and request.url.path.endswith("/query"):
                    failed_probe.append(request.url.path)
                    return httpx.Response(200, json={"hits": {"hits": []}})
                if request.method == "POST":
                    return httpx.Response(200, json={})
                raise AssertionError(f"Unexpected database request: {request.method} {request.url}")

            af = Antfly("http://fixture")
            af.http.close()
            af.http = httpx.Client(base_url="http://fixture", transport=httpx.MockTransport(handler))
            self.addCleanup(af.http.close)
            with patch("kg.dataset.create_models", side_effect=lambda *args, **kwargs: BoundaryModels()), \
                    patch("kg.dataset.database", return_value=af), \
                    patch("kg.dataset.binary_info", return_value={"version": "isolated_fixture"}):
                with self.assertRaises(ValueError):
                    dataset.build(cfg, max_passages=1)
            self.assertTrue(failed_probe, "The build must reach the intentionally rejected indexed-vector probe")
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
        self.assertEqual(interpret_intent(cfg, "Why is Tyrion Lannister sad?", focus)["status"], "unrecognized")
        self.assertEqual(interpret_intent(cfg, "Who are Tyrion Lannister's sons?", focus)["status"], "unrecognized")
        self.assertEqual(interpret_intent({"extraction": {"relations": {}}},
                                        "Who is Tyrion Lannister's father?", focus)["status"], "unrecognized")
        self.assertEqual(interpret_intent(cfg, "Who is Tyrion Lannister's father?", focus, navigate=True)["status"], "all")

    def test_specific_parent_requires_the_actual_ordered_mentions(self):
        for text, expected in (
            ("😀 Ned is the father of Árya.", True),
            ("Árya's father is Ned.", True),
            ("Ned is the parent of Árya.", False),
            ("Ned is the mother of Árya.", False),
            ("Ned is not the father of Árya.", False),
            ("Ned met Árya. Elsewhere Jon's father arrived.", False),
        ):
            with self.subTest(text=text):
                assertion = {"text": text, "head": mention(text, "Ned"), "tail": mention(text, "Árya")}
                self.assertEqual(_specific_parent(assertion, "father"), expected)

    def relationship_fixture(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        cfg = config(directory.name)
        text = "😀 Ned is the father of Árya. Cersei is the mother of Árya."
        _, _, text = source_fixture(cfg, text)
        sources = {row["id"]: row for row in read_manifest(cfg)}
        passages, analyses = [], {}
        with closing(dataset.Cache(cfg)) as cache:
            for start, end, parent in (
                (0, text.index(" Cersei"), "Ned"),
                (text.index("Cersei"), len(text), "Cersei"),
            ):
                excerpt = text[start:end]
                key = "passage:" + dataset.digest(["s1", sources["s1"]["sha256"], start, end, excerpt])[:24]
                head, tail = mention(excerpt, parent), mention(excerpt, "Árya")
                models = BoundaryModels({"entities": [head, tail], "relations": [proposal(head, tail)]},
                                        winner="parent_of")
                analyses[key] = analyze_chunk(cfg, excerpt, cache, models)
                passages.append({"id": key, "source_id": "s1", "source_start": start,
                                 "source_end": end, "text": excerpt})
        docs = graph_documents(cfg, {"sources": sources, "passages": passages}, analyses)

        def matches(row, query):
            if "conjuncts" in query:
                return all(matches(row, clause) for clause in query["conjuncts"])
            if "disjuncts" in query:
                return any(matches(row, clause) for clause in query["disjuncts"])
            return row.get(query["field"]) == query["term"]

        class Index:
            def get(self, table, key):
                return docs.get(key)

            def query(self, table, body):
                selected = [{"_id": key, "_source": row} for key, row in docs.items()
                            if matches(row, body["filter_query"])]
                return {"responses": [{"hits": {"hits": selected[:body["limit"]],
                                                "total": {"value": len(selected), "relation": "eq"}}}]}

        focus_id = next(key for key, row in docs.items() if row.get("kind") == "entity" and row["text"] == "Árya")
        focus = {"id": focus_id, "text": "Árya", "type": "character"}
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
            self.assertIn(evidence["passage_id"], docs)
        father = interpret_intent(cfg, "Who is Árya's father?", focus)
        graph, _ = _relationships(cfg, index, manifest, focus, father, sources)
        edge, = graph["edges"]
        self.assertEqual(names[edge["source"]], "Ned")
        self.assertEqual(edge["predicate"], "parent_of")
        self.assertNotIn("Cersei", edge["evidence"][0]["text"])
        children = interpret_intent(cfg, "Who are Árya's children?", focus)
        graph, _ = _relationships(cfg, index, manifest, focus, children, sources)
        self.assertEqual(graph["edges"], [])

    def test_indexed_assertion_must_belong_to_its_exact_supporting_passage(self):
        cfg, index, manifest, focus, sources, docs = self.relationship_fixture()
        key = next(key for key, row in docs.items() if row.get("kind") == "passage")
        docs[key] = {**docs[key], "analysis": {**docs[key]["analysis"], "assertions": []}}
        intent = interpret_intent(cfg, "Who are Árya's parents?", focus)
        with self.assertRaises(ValueError):
            _relationships(cfg, index, manifest, focus, intent, sources)


if __name__ == "__main__":
    unittest.main()
