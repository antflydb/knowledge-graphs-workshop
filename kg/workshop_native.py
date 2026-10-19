"""Client for Antfly's native inference API, used for on-demand calls from the page.

Antfly itself runs extraction and embedding at ingest; this client only extracts
entities from a question and runs GLiNER2.5-Decide on one relationship at a time.
Native extraction uses UTF-8 byte offsets; this adapter converts them to half-open
Python character offsets. Decide's 512-token limit includes the label schema; the
server rejects overflow and this adapter never truncates.
"""

from __future__ import annotations

import json
import math
import time
from copy import deepcopy
from typing import Any

import httpx


DEFAULT_REVISIONS = {
    "extract": "72ac19b486cd4557424c8d61114e7530c243e9b0",
    "embed": "914f7f89142e33e77833254d9c9b90c3cef7303b",
    "decide": "7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6",
}
DEFAULT_MODELS = {
    "extract": "fastino/gliner2.5-base-v1",
    "embed": "google/embeddinggemma-2",
    "decide": "fastino/GLiNER2.5-Decide",
}


BUSY_RETRIES = 20


class NativeInferenceError(RuntimeError):
    """Native HTTP or response-contract failure, retaining server error details."""

    def __init__(self, message: str, *, status_code: int | None = None,
                 details: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.details = details


def _probability(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NativeInferenceError("Native inference returned a nonnumeric score")
    score = float(value)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise NativeInferenceError("Native inference returned a score outside [0, 1]")
    return score


def _span(text: str, raw: dict, unit: str, byte_offsets: dict[int, int]) -> dict:
    surface, start, end = raw.get("text"), raw.get("start"), raw.get("end")
    if (not isinstance(surface, str) or not surface
            or type(start) is not int or type(end) is not int):
        raise NativeInferenceError("Native inference returned a missing source span")
    if unit == "utf8_bytes":
        if start not in byte_offsets or end not in byte_offsets:
            raise NativeInferenceError("Native span is not on UTF-8 codepoint boundaries")
        start, end = byte_offsets[start], byte_offsets[end]
    elif unit != "unicode_codepoints":
        raise NativeInferenceError(f"Unsupported native offset unit: {unit!r}")
    if not 0 <= start < end <= len(text) or text[start:end] != surface:
        raise NativeInferenceError("Native span does not match the immutable input text")
    return {"text": surface, "start": start, "end": end}


class NativeModels:
    """Shared workshop contract backed exclusively by real /ai/v1 HTTP calls.

    Options: base_url (origin or /ai/v1 URL), timeout (seconds), revisions
    (per-role declarations) and headers. close() releases the HTTP client,
    not the server's models.
    """

    def __init__(self, models: dict, **options: Any) -> None:
        base_url = options.pop("base_url", "http://127.0.0.1:8090/ai/v1").rstrip("/")
        timeout = options.pop("timeout", 300.0)
        revisions = options.pop("revisions", None)
        headers = options.pop("headers", None)
        if options:
            raise TypeError(f"Unknown NativeModels options: {', '.join(sorted(options))}")
        if not base_url.endswith("/ai/v1"):
            base_url += "/ai/v1"
        self.models = {**DEFAULT_MODELS, **models}
        for role in DEFAULT_MODELS:
            if not isinstance(self.models[role], str) or not self.models[role]:
                raise ValueError(f"Missing native model for {role}")
        # Do not attribute a default checkpoint's revision to an override model.
        declared = {role: revision for role, revision in DEFAULT_REVISIONS.items()
                    if self.models[role] == DEFAULT_MODELS[role]}
        self.revisions = {**declared, **(revisions or {})}
        self._metadata: dict[str, Any] = {
            "provider": "antfly_native",
            "base_url": base_url, "configured_models": dict(self.models),
            "configured_revisions": dict(self.revisions),
            "revision_verification": "server deployment responsibility; not attested by extraction API",
            "observed": {},
        }
        self._client = httpx.Client(base_url=base_url + "/", timeout=timeout,
                                    headers=headers, trust_env=False)

    @property
    def metadata(self) -> dict:
        """Configured provenance and server-observed metadata, without vectors/text."""
        return deepcopy(self._metadata)

    def _post(self, endpoint: str, payload: dict, role: str) -> dict:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if endpoint == "extract" and len(body) > 16 * 1024 * 1024:
            raise ValueError("Native extraction exceeds the 16 MiB request limit")
        for attempt in range(BUSY_RETRIES + 1):
            response = self._client.post(endpoint, content=body,
                                         headers={"Content-Type": "application/json"})
            # The server admits a bounded number of requests; wait when it says so.
            if response.status_code != 503 or attempt == BUSY_RETRIES:
                break
            try:
                busy = response.json()
            except ValueError:
                break
            if not busy.get("retryable"):
                break
            time.sleep(min(busy.get("retry_after_ms", 500), 2000) / 1000)
        try:
            result = response.json()
        except ValueError as exc:
            raise NativeInferenceError(
                f"Native {endpoint} returned HTTP {response.status_code} with invalid JSON: "
                f"{response.text[:1000]}", status_code=response.status_code,
            ) from exc
        if response.is_error:
            raise NativeInferenceError(
                f"Native {endpoint} failed (HTTP {response.status_code}): {result}",
                status_code=response.status_code, details=result,
            )
        if not isinstance(result, dict) or "error" in result:
            raise NativeInferenceError(f"Invalid native {endpoint} response", details=result)
        self._metadata["observed"][role] = {
            key: deepcopy(result[key]) for key in
            ("model", "schema_version", "usage") if key in result
        }
        return result

    @staticmethod
    def _row(result: dict) -> dict:
        rows = result.get("data")
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
            raise NativeInferenceError("Expected exactly one native extraction result", details=result)
        return rows[0]

    @staticmethod
    def _text(text: str) -> None:
        if not isinstance(text, str) or not text:
            raise ValueError("Inference text must be a nonempty string")
        if len(text.encode("utf-8")) > 1024 * 1024:
            raise ValueError("Native extraction exceeds the 1 MiB text limit")

    def extract(self, text: str, schema: dict, threshold: float) -> dict:
        self._text(text)
        if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Extraction threshold must be finite and in [0, 1]")
        entity_labels = schema.get("entities", [])
        relation_labels = schema.get("relations", [])
        if (not isinstance(entity_labels, list) or not isinstance(relation_labels, list)
                or any(not isinstance(label, str) or not label
                       for label in entity_labels + relation_labels)):
            raise ValueError("Workshop schema requires entity and relation label lists")
        result = self._post("extract", {
            "model": self.models["extract"], "schema_version": 2,
            "inputs": [{"content": text}],
            "schema": {"entities": entity_labels,
                       "relations": [{"type": label} for label in relation_labels]},
            "options": {"threshold": threshold, "include_confidence": True,
                        "include_spans": True, "offset_unit": "utf8_bytes"},
        }, "extract")
        row = self._row(result)
        unit = row.get("offset_unit")
        byte_offsets = {0: 0}
        offset = 0
        for index, char in enumerate(text, 1):
            offset += len(char.encode("utf-8"))
            byte_offsets[offset] = index
        entities = []
        raw_entities = row.get("entities", [])
        if not isinstance(raw_entities, list):
            raise NativeInferenceError("Native entities must be a list")
        for entity in raw_entities:
            if not isinstance(entity, dict) or entity.get("label") not in entity_labels:
                raise NativeInferenceError("Native inference returned an undeclared entity label")
            entities.append({**_span(text, entity, unit, byte_offsets),
                             "label": entity["label"], "score": _probability(entity.get("score"))})

        def endpoint(raw: Any) -> dict | None:
            if not isinstance(raw, dict):
                return None
            index = raw.get("entity_index")
            indexed = None
            if index is not None:
                if type(index) is not int or not 0 <= index < len(entities):
                    return None
                indexed = entities[index]
                if "label" in raw and raw["label"] != indexed["label"]:
                    return None
            if "start" in raw or "end" in raw:
                try:
                    anchored = _span(text, raw, unit, byte_offsets)
                except NativeInferenceError:
                    return None
                if indexed and any(anchored[key] != indexed[key] for key in anchored):
                    return None
                return anchored
            if indexed is not None and ("text" not in raw or raw["text"] == indexed["text"]):
                return {key: indexed[key] for key in ("text", "start", "end")}
            # Never guess a repeated surface's occurrence or invent entity identity.
            return None

        relations = []
        rejected = 0
        raw_relations = row.get("relations", [])
        if not isinstance(raw_relations, list):
            raise NativeInferenceError("Native relations must be a list")
        for relation in raw_relations:
            if not isinstance(relation, dict) or relation.get("type") not in relation_labels:
                raise NativeInferenceError("Native inference returned an undeclared relation label")
            head, tail = endpoint(relation.get("source")), endpoint(relation.get("target"))
            if head is None or tail is None:
                rejected += 1
                continue
            relations.append({"label": relation["type"], "head": head, "tail": tail,
                              "score": _probability(relation.get("score"))})
        self._metadata["observed"]["extract"]["rejected_unanchored_relations"] = rejected
        return {"entities": entities, "relations": relations}

    def classify(self, text: str, labels: dict[str, str], instruction: str) -> dict:
        self._text(text)
        if (not isinstance(labels, dict) or len(labels) < 2
                or any(not isinstance(label, str) or not label or not isinstance(description, str)
                       for label, description in labels.items())
                or not isinstance(instruction, str)):
            raise ValueError("Classification requires at least two labels/descriptions and a text instruction")
        result = self._post("extract", {
            "model": self.models["decide"], "schema_version": 2,
            "inputs": [{"content": text}],
            "schema": {"classifications": [{
                "name": "classification", "labels": list(labels),
                "label_definitions": {label: {"description": description}
                                      for label, description in labels.items()},
                "prompt": instruction, "multi_label": False, "top_k": len(labels),
            }]},
            "options": {"include_confidence": True},
        }, "decide")
        rows = self._row(result).get("classifications")
        if not isinstance(rows, list):
            raise NativeInferenceError("Native Decide response lacks ordinary classifications")
        scores = {}
        for row in rows:
            if (not isinstance(row, dict) or row.get("name") != "classification"
                    or row.get("label") not in labels or row["label"] in scores):
                raise NativeInferenceError("Native Decide returned unexpected/duplicate classification labels")
            scores[row["label"]] = _probability(row.get("score"))
        if set(scores) != set(labels):
            raise NativeInferenceError("Native Decide response omits requested label scores")
        if not math.isclose(sum(scores.values()), 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise NativeInferenceError("Native single-label classification did not return softmax scores")
        return {"label": max(labels, key=lambda label: scores[label]),
                "scores": {label: scores[label] for label in labels},
                "method": "gliner2_single_label_softmax"}

    def close(self) -> None:
        self._client.close()
