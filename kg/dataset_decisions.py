"""Shared proposed-pair differentiation and independent passage experiments.

Offsets are half-open Python character offsets into the supplied text. Analysis
is deliberately limited to context, entities, proposals, decisions and assertions;
callers own stage timing and count classified decisions (including cache hits).
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
import math
import re
import time


RESERVED_LABELS = {
    "no_relation": "No supported directed relationship of the supplied types is expressed by this text.",
    "unclear": "The evidence for this directed pair is insufficient or ambiguous.",
}
ANALYSIS_VERSION = "typed-proposed-pairs-v1"


class DecisionError(ValueError):
    """An experiment failure with its complete, untruncated inspectable input."""

    def __init__(self, message, payload, status_code=422):
        super().__init__(message)
        self.status_code = status_code
        self.payload = {**payload, "error": message}


def validate_instruction(instruction):
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Decision instruction must be a nonempty string")
    return instruction


def validate_labels(labels):
    """Validate without renaming, reordering or replacing user descriptions."""
    if not isinstance(labels, dict) or len(labels) < 2:
        raise ValueError("Decision labels must map at least two unique names to descriptions")
    names = set()
    for name, description in labels.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Decision label names must be nonempty strings")
        if name.strip() in names:
            raise ValueError("Decision label names must be unique, including surrounding whitespace")
        names.add(name.strip())
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Decision label descriptions must be nonempty strings")
    return dict(labels)


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _context(context):
    if context is None:
        return {}
    if not isinstance(context, dict):
        raise ValueError("Analysis context must be a provenance object")
    result = deepcopy({key: value for key, value in context.items() if value is not None})
    _json(result)
    return result


def _span(text, span):
    if not isinstance(span, dict):
        raise ValueError("Extraction endpoint must be an exact source span")
    start, end, surface = span.get("start"), span.get("end"), span.get("text")
    if (type(start) is not int or type(end) is not int
            or not 0 <= start < end <= len(text) or text[start:end] != surface):
        raise ValueError("Extraction span does not match source text at its character offsets")
    return start, end


def _probability(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Model confidence must be a finite probability")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Model confidence must be a finite probability")
    return value


def _extraction(cfg, text, extracted):
    """Check adapter output without repairing spans or losing base proposals."""
    if not isinstance(text, str):
        raise TypeError("Extraction text must be a string")
    if not isinstance(extracted, dict):
        raise ValueError("Extraction must contain entities and relations")
    options = cfg["extraction"]
    if set(options["relations"]).intersection(RESERVED_LABELS):
        raise ValueError("no_relation and unclear are reserved, not semantic predicates")
    entities, proposals = extracted.get("entities"), extracted.get("relations")
    if not isinstance(entities, list) or not isinstance(proposals, list):
        raise ValueError("Extraction entities and relations must be lists")
    for entity in entities:
        _span(text, entity)
        if entity.get("label") not in options["entities"]:
            raise ValueError("Extraction returned an unconfigured entity type")
        _probability(entity.get("score"))
    for proposal in proposals:
        if not isinstance(proposal, dict) or proposal.get("label") not in options["relations"]:
            raise ValueError("Extraction returned an unconfigured relationship predicate")
        _span(text, proposal.get("head"))
        _span(text, proposal.get("tail"))
        _probability(proposal.get("score"))
    return {"entities": deepcopy(entities), "relations": deepcopy(proposals)}


def extract_chunk(cfg, text, cache, models):
    """Extract only; builds can finish this phase before loading Decide."""
    options = cfg["extraction"]
    schema = {"entities": options["entities"], "relations": list(options["relations"])}
    threshold = options["threshold"]
    args = {"version": ANALYSIS_VERSION, "text": text, "schema": schema, "threshold": threshold}
    extracted = cache.call("extract", args, lambda: models.extract(text, schema, threshold), models)
    return _extraction(cfg, text, extracted)


def _classify(text, labels, instruction, cache, models, *, context, endpoints=None):
    executed = False

    def run():
        nonlocal executed
        executed = True
        return models.classify(text, labels, instruction)

    # A list of entries preserves candidate order even if the cache sorts dict keys.
    args = {"version": ANALYSIS_VERSION, "text": text, "labels": list(labels.items()),
            "instruction": instruction, "context": context, "endpoints": endpoints}
    result = cache.call("classify", args, run, models)
    if not isinstance(result, dict) or result.get("label") not in labels:
        raise ValueError("Decide returned an unknown winning label")
    scores = result.get("scores")
    if not isinstance(scores, dict) or set(scores) != set(labels):
        raise ValueError("Decide must return scores for every candidate label")
    for score in scores.values():
        _probability(score)
    if scores[result["label"]] != max(scores.values()):
        raise ValueError("Decide winning label disagrees with its scores")
    return {**result, "scores": {label: scores[label] for label in labels}}, executed


def differentiate_relations(cfg, text, extracted, cache, models, context=None):
    """Classify each proposed ordered occurrence pair once; never expand pairs."""
    extracted = _extraction(cfg, text, extracted)
    context = _context(context)
    entities, proposals = extracted["entities"], extracted["relations"]
    threshold = cfg["extraction"]["threshold"]
    mentions = defaultdict(dict)
    for entity in entities:
        if entity["score"] >= threshold:
            position = (entity["start"], entity["end"])
            previous = mentions[position].get(entity["label"])
            if previous is None or previous["score"] < entity["score"]:
                mentions[position][entity["label"]] = entity
    pairs = defaultdict(list)
    for index, proposal in enumerate(proposals):
        pair = (_span(text, proposal["head"]), _span(text, proposal["tail"]))
        pairs[pair].append(index)
    decisions, assertions = [], []
    instruction = validate_instruction(cfg["relation_decision"]["instruction"])
    for (head_span, tail_span), indices in sorted(pairs.items()):
        head_options, tail_options = mentions[head_span], mentions[tail_span]
        first = proposals[indices[0]]
        head = deepcopy(next(iter(head_options.values()))) if len(head_options) == 1 else deepcopy(first["head"])
        tail = deepcopy(next(iter(tail_options.values()))) if len(tail_options) == 1 else deepcopy(first["tail"])
        decision = {"head": head, "tail": tail, "proposal_indices": indices,
                    "input": None, "labels": {}, "instruction": instruction,
                    "label": None, "scores": {}, "unresolved_reason": None}
        decisions.append(decision)
        unresolved = []
        for role, choices in (("head", head_options), ("tail", tail_options)):
            if not choices:
                unresolved.append(f"{role}_has_no_above_threshold_typed_mention")
            elif len(choices) != 1:
                unresolved.append(f"{role}_has_ambiguous_typed_mentions")
        if unresolved:
            decision["unresolved_reason"] = "; ".join(unresolved)
            continue
        labels = {predicate: spec["description"]
                  for predicate, spec in cfg["extraction"]["relations"].items()
                  if head["label"] in spec["head_types"] and tail["label"] in spec["tail_types"]}
        decision["labels"] = {**labels, **RESERVED_LABELS}
        if not labels:
            decision["unresolved_reason"] = "no_type_compatible_predicate"
            continue
        classifier_input = (f"Context:\n{_json(context)}\nText:\n{text}"
                            f"\n\nHEAD: {head['text']} [{head['label']}]"
                            f"\nTAIL: {tail['text']} [{tail['label']}]")
        decision["input"] = classifier_input
        result, _ = _classify(classifier_input, decision["labels"], instruction, cache, models,
                              context=context, endpoints=[head, tail])
        decision.update(label=result["label"], scores=result["scores"])
        if "method" in result:
            decision["method"] = result["method"]
        if result["label"] == "unclear":
            decision["unresolved_reason"] = "model_selected_unclear"
        elif result["label"] != "no_relation":
            assertions.append({"head": deepcopy(head), "tail": deepcopy(tail),
                               "predicate": result["label"], "score": result["scores"][result["label"]],
                               "status": "model_classified_not_verified", "decision_index": len(decisions) - 1})
    return {"context": context, "entities": entities, "proposals": proposals,
            "decisions": decisions, "assertions": assertions}


def analyze_chunk(cfg, text, cache, models, context=None):
    """The identical extraction/pair-classification pipeline for corpus and query."""
    return differentiate_relations(cfg, text, extract_chunk(cfg, text, cache, models),
                                   cache, models, context=context)


def _experimental_graph(analysis):
    """All graph members, without rejected proposals or duplicated raw scores."""
    entities = "; ".join(f"{entity['text']} [{entity['label']}]" for entity in analysis["entities"])
    triples = "; ".join(
        f"{assertion['head']['text']} [{assertion['head']['label']}] /{assertion['predicate']}/ "
        f"{assertion['tail']['text']} [{assertion['tail']['label']}]"
        for assertion in analysis["assertions"])
    return f"Model predictions, not verified.\nEntities: {entities}\nDirected triples: {triples}"


def experiment_input(query_text, query_analysis, passage_text, passage_analysis, context):
    """Serialize complete text and model graphs, never truncate to fit Decide."""
    if not isinstance(query_text, str) or not isinstance(passage_text, str):
        raise ValueError("Query and passage text must be strings")
    return (f"Query text:\n{query_text}\n\nQuery graph:\n{_experimental_graph(query_analysis)}"
            f"\n\nPassage text:\n{passage_text}\n\nPassage graph:\n{_experimental_graph(passage_analysis)}"
            f"\n\nSource context:\n{_json(_context(context))}")


def _overflow(exc):
    if getattr(exc, "status_code", None) == 413:
        return True
    return bool(re.search(r"(?:token|request|input|body|text).*(?:limit|too large|too long|exceed|overflow)"
                          r"|(?:limit|too large|too long|exceed|overflow).*(?:token|request|input|body|text)",
                          str(exc), re.IGNORECASE | re.DOTALL))


def _passage_context(passage):
    context = _context(passage.get("context", passage["analysis"]["context"]))
    citation = deepcopy(passage.get("citation"))
    # Titles are provenance, not entities or additional extracted facts.
    if isinstance(citation, dict):
        for key in ("source_kind", "title", "met", "speaker", "printed_label"):
            target = "kind" if key == "source_kind" else key
            if target not in context and citation.get(key) is not None:
                context[target] = citation[key]
    return context


def experiment_payload(cfg, query_text, query_analysis, passage, instruction=None, labels=None):
    """Prepare the exact preview before constructing a possibly unavailable provider."""
    defaults = cfg["query_decision"]
    instruction = defaults["instruction"] if instruction is None else instruction
    labels = defaults["labels"] if labels is None else labels
    text = experiment_input(query_text, query_analysis, passage["text"], passage["analysis"],
                            _passage_context(passage))
    return {"passage_id": passage.get("passage_id", passage.get("id")), "input": text,
            "instruction": instruction, "labels": deepcopy(labels), "citation": deepcopy(passage.get("citation"))}


def decide_passage(cfg, query_text, query_analysis, passage, cache, models, instruction=None, labels=None):
    """Score one actual retrieved passage; callers resolve IDs and stale builds.

    DecisionError.payload always includes the serialized input, supplied labels,
    instruction and citation. Token/request overflow is HTTP 422; model availability
    errors preserve their details and never invoke another provider.
    """
    started = time.perf_counter()
    payload = experiment_payload(cfg, query_text, query_analysis, passage, instruction, labels)
    text, instruction, labels = payload["input"], payload["instruction"], payload["labels"]
    context = _passage_context(passage)
    classify_started = time.perf_counter()
    try:
        instruction = validate_instruction(instruction)
        labels = validate_labels(labels)
        result, executed = _classify(text, labels, instruction, cache, models, context=context)
    except Exception as exc:
        payload["timings"] = {"classification_ms": round((time.perf_counter() - classify_started) * 1000, 3),
                              "total_ms": round((time.perf_counter() - started) * 1000, 3)}
        if _overflow(exc):
            message = (f"{exc}. Shorten the instruction/labels or select a shorter query/passage; "
                       "the complete input is preserved and was not truncated.")
            raise DecisionError(message, payload, 422) from exc
        status = 422 if isinstance(exc, (ValueError, TypeError)) else 503
        # Native inference errors carry their real HTTP status/details in their text.
        if getattr(exc, "status_code", None) is not None:
            status = exc.status_code
        raise DecisionError(str(exc), payload, status) from exc
    return {**payload, **result,
            "timings": {"classification_ms": round((time.perf_counter() - classify_started) * 1000, 3),
                        "total_ms": round((time.perf_counter() - started) * 1000, 3),
                        "cache_hit": not executed}}
