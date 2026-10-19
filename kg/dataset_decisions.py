"""On-demand GLiNER2.5-Decide over one extracted relationship, and query entity extraction.

The graph itself is built by Antfly's extractor at ingest; nothing here runs at build
time. Offsets are half-open Python character offsets into the supplied text.
"""
from __future__ import annotations

import math

RESERVED_LABELS = {
    "no_relation": "No supported directed relationship of the supplied types is expressed by this text.",
    "unclear": "The evidence for this directed pair is insufficient or ambiguous.",
}


class DecisionError(ValueError):
    """A Decide failure that still returns its complete, inspectable input."""

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


def _span(text, span):
    if not isinstance(span, dict):
        raise ValueError("Relationship endpoint must be an exact source span")
    start, end, surface = span.get("start"), span.get("end"), span.get("text")
    if (type(start) is not int or type(end) is not int
            or not 0 <= start < end <= len(text) or text[start:end] != surface):
        raise ValueError("Endpoint span does not match source text at its character offsets")
    return start, end


def relationship_labels(cfg, head_label, tail_label):
    """Every predicate whose declared endpoint types admit this pair, plus the reserved outcomes."""
    labels = {predicate: spec["description"] for predicate, spec in cfg["extraction"]["relations"].items()
              if head_label in spec["head_types"] and tail_label in spec["tail_types"]}
    return {**labels, **RESERVED_LABELS}


def decide_relationship(cfg, passage, head, tail, models, instruction=None, labels=None):
    """Ask Decide which relationship, if any, this passage states between two exact mentions."""
    text = passage["text"]
    for role, span in (("head", head), ("tail", tail)):
        _span(text, span)
        if span.get("label") not in cfg["extraction"]["entities"]:
            raise ValueError(f"{role} needs a configured entity label")
    instruction = validate_instruction(cfg["relation_decision"]["instruction"] if instruction is None else instruction)
    labels = validate_labels(relationship_labels(cfg, head["label"], tail["label"]) if labels is None else labels)
    classifier_input = (f"Text:\n{text}\n\nHEAD: {head['text']} [{head['label']}]"
                        f"\nTAIL: {tail['text']} [{tail['label']}]")
    payload = {"input": classifier_input, "instruction": instruction, "labels": labels}
    try:
        result = models.classify(classifier_input, labels, instruction)
    except Exception as exc:
        raise DecisionError(str(exc), payload, getattr(exc, "status_code", None) or 503) from exc
    scores = result.get("scores") if isinstance(result, dict) else None
    if not isinstance(scores, dict) or set(scores) != set(labels):
        raise DecisionError("Decide must return scores for every candidate label", payload)
    if any(isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s) or not 0 <= s <= 1
           for s in scores.values()):
        raise DecisionError("Decide returned a non-probability score", payload)
    return {**payload, "label": max(labels, key=lambda label: scores[label]),
            "scores": {label: scores[label] for label in labels}}


def query_entities(cfg, text, models):
    """Entity mentions in a question, used to find matching name groups and passages.

    Entities only: adding relation types to the GLiNER schema suppresses entity
    output on short text.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Query must not be empty")
    extracted = models.extract(text, {"entities": cfg["extraction"]["entities"], "relations": []},
                               cfg["extraction"]["threshold"])
    for entity in extracted["entities"]:
        _span(text, entity)
    return extracted["entities"]
