"""Version-two dataset configuration, validated before any model or table is opened."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import yaml


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


MODEL_PINS = {
    "extract": ("fastino/gliner2.5-base-v1", "72ac19b486cd4557424c8d61114e7530c243e9b0"),
    "embed": ("google/embeddinggemma-2", "914f7f89142e33e77833254d9c9b90c3cef7303b"),
    "decide": ("fastino/GLiNER2.5-Decide", "7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6"),
}


def _mapping(value, allowed, required, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    unknown, missing = set(value) - set(allowed), set(required) - set(value)
    if unknown or missing:
        raise ValueError(f"Invalid {name} keys: unknown={sorted(unknown)}, missing={sorted(missing)}")


def _strings(value, name):
    if not isinstance(value, list) or not value or any(not isinstance(v, str) or not v.strip() for v in value) or len(set(value)) != len(value):
        raise ValueError(f"{name} must contain unique nonempty strings")


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


@dataclass
class DatasetConfig:
    path: Path
    data: dict

    def resolve(self, value):
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.path.parent / path).resolve()

    @property
    def output(self):
        return Path(self.data["output"])


def load_dataset(path):
    path = Path(path).expanduser().resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"version", "dataset", "title", "corpus", "output", "passages", "extraction", "relation_decision", "models", "revisions", "dimensions", "retrieval", "antfly", "collection"}
    _mapping(data, required, required, "config")
    if data["version"] != 2:
        raise ValueError("Require dataset config version: 2")
    if not isinstance(data["dataset"], str) or not re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", data["dataset"]):
        raise ValueError("dataset must be a lowercase hyphen-separated slug")
    for key in ("title", "corpus", "output"):
        _text(data[key], key)
    p = data["passages"]
    _mapping(p, {"max_bytes", "max_words", "overlap_words"}, {"max_bytes", "max_words", "overlap_words"}, "passages")
    for key in ("max_bytes", "max_words"):
        _positive(p[key], f"passages.{key}")
    if type(p["overlap_words"]) is not int or not 0 <= p["overlap_words"] < p["max_words"]:
        raise ValueError("passages.overlap_words must be >= 0 and less than max_words")
    ex = data["extraction"]
    _mapping(ex, {"entities", "relations", "threshold"}, {"entities", "relations", "threshold"}, "extraction")
    _strings(ex["entities"], "extraction.entities")
    if any(not re.fullmatch(r"[a-z][a-z0-9_]*", t) for t in ex["entities"]):
        raise ValueError("Entity types must be lowercase identifiers")
    if type(ex["threshold"]) not in (int, float) or not 0 <= ex["threshold"] <= 1:
        raise ValueError("extraction.threshold must be within [0, 1]")
    if not isinstance(ex["relations"], dict) or not ex["relations"]:
        raise ValueError("extraction.relations must be a nonempty mapping")
    for name, relation in ex["relations"].items():
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in {"no_relation", "unclear"}:
            raise ValueError(f"Invalid or reserved predicate: {name}")
        _mapping(relation, {"description", "head_types", "tail_types"}, {"description", "head_types", "tail_types"}, f"relation {name}")
        _text(relation["description"], f"{name}.description")
        for endpoint in ("head_types", "tail_types"):
            _strings(relation[endpoint], f"{name}.{endpoint}")
            if set(relation[endpoint]) - set(ex["entities"]):
                raise ValueError(f"Unknown endpoint types in {name}.{endpoint}")
    _mapping(data["relation_decision"], {"instruction"}, {"instruction"}, "relation_decision")
    _text(data["relation_decision"]["instruction"], "relation_decision.instruction")
    for group, index in (("models", 0), ("revisions", 1)):
        _mapping(data[group], MODEL_PINS, MODEL_PINS, group)
        if any(data[group][role] != pin[index] for role, pin in MODEL_PINS.items()):
            raise ValueError(f"{group} must retain the qualified immutable model pins")
    if data["dimensions"] != 768 or type(data["dimensions"]) is not int:
        raise ValueError("dimensions must be 768 for the pinned embedding model")
    r = data["retrieval"]
    _mapping(r, {"candidates", "graph_candidates", "evidence", "text_reserve"}, {"candidates", "graph_candidates", "evidence", "text_reserve"}, "retrieval")
    for key in ("candidates", "graph_candidates", "evidence"):
        _positive(r[key], f"retrieval.{key}")
    if type(r["text_reserve"]) is not int or not 0 <= r["text_reserve"] < r["evidence"] <= r["candidates"]:
        raise ValueError("Require 0 <= text_reserve < evidence <= candidates")
    runtime = data["antfly"]
    fields = {"target", "addr", "health_port", "binary_env", "data_dir", "inference_url"}
    _mapping(runtime, fields, fields, "antfly")
    for key in fields - {"health_port"}:
        _text(runtime[key], f"antfly.{key}")
    if type(runtime["health_port"]) is not int or not 1 <= runtime["health_port"] <= 65535:
        raise ValueError("Invalid antfly.health_port")
    if (runtime["target"], runtime["binary_env"]) != ("v0.3.0-rc.1", "ANTFLY_BIN"):
        raise ValueError("antfly must target the v0.3.0-rc.1 release through ANTFLY_BIN")
    for key, value in (("addr", runtime["addr"]), ("inference_url", runtime["inference_url"].removeprefix("http://"))):
        if not re.fullmatch(r"127\.0\.0\.1:\d+", value) or not 1 <= int(value.split(":")[1]) <= 65535:
            raise ValueError(f"antfly.{key} must be a loopback host:port")
    collection = data["collection"]
    _mapping(collection, {"kind", "sources", "user_agent", "request_interval_seconds", "exclude_sections", "transcript_window"}, {"kind", "sources", "user_agent", "request_interval_seconds"}, "collection")
    if collection["kind"] not in {"wikipedia", "restaurant_pages", "apollo13"}:
        raise ValueError("Unknown collection.kind")
    _text(collection["user_agent"], "collection.user_agent")
    if type(collection["request_interval_seconds"]) not in (int, float) or not 0 < collection["request_interval_seconds"] <= 60:
        raise ValueError("Collection request interval must be within (0, 60]")
    if collection["kind"] == "wikipedia":
        _strings(collection.get("exclude_sections"), "collection.exclude_sections")
    if collection["kind"] == "apollo13":
        window = collection.get("transcript_window")
        _mapping(window, {"start", "end", "next", "encoding", "source_sha256", "sha256"}, {"start", "end", "next", "encoding", "source_sha256", "sha256"}, "transcript_window")
        if (window["start"], window["end"], window["next"], window["encoding"]) != ("055:50:05", "094:33:21", "094:33:31", "windows-1252"):
            raise ValueError("Apollo transcript must retain the approved continuous window and encoding")
    if not isinstance(collection["sources"], list) or not collection["sources"]:
        raise ValueError("collection.sources must be a nonempty fixed inventory")
    ids, paths = set(), set()
    for source in collection["sources"]:
        _mapping(source, {"id", "path", "title", "url", "download_url", "format", "restaurant_id", "roles", "page_id", "revision_id", "revision_url", "source_sha256", "sha256"}, {"id", "path", "title", "url", "format"}, "collection source")
        for key in ("id", "path", "title", "url", "format"):
            _text(source[key], f"source.{key}")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9:._-]*", source["id"]):
            raise ValueError(f"Source ID must be a safe nonempty identifier: {source['id']}")
        kind = collection["kind"]
        if kind == "wikipedia":
            if source["format"] != "wikipedia" or not {"page_id", "revision_id", "revision_url"} <= set(source):
                raise ValueError("Wikipedia sources require wikipedia format and fixed page/revision identity")
            for key in ("page_id", "revision_id"):
                _positive(source[key], f"source.{key}")
        elif kind == "restaurant_pages":
            if source["format"] not in {"html", "pdf"} or not {"restaurant_id", "roles"} <= set(source):
                raise ValueError("Restaurant sources require html/pdf format, restaurant_id and roles")
            if not isinstance(source["restaurant_id"], str) or not re.fullmatch(r"[a-z][a-z0-9-]*", source["restaurant_id"]):
                raise ValueError("restaurant_id must be a lowercase slug")
            _strings(source["roles"], "source.roles")
            if set(source["roles"]) - {"menu", "about"}:
                raise ValueError("Restaurant source roles must be menu and/or about")
        elif source["format"] not in {"transcript", "text"} or "download_url" not in source:
            raise ValueError("Apollo sources require transcript/text format and publisher download_url")
        for key in ("sha256", "source_sha256"):
            if key in source and (not isinstance(source[key], str) or not re.fullmatch(r"[0-9a-f]{64}", source[key])):
                raise ValueError(f"Invalid source {key}")
        if source["id"] in ids or source["path"] in paths:
            raise ValueError("Duplicate collection source id/path")
        ids.add(source["id"])
        paths.add(source["path"])
        raw = (path.parent / source["path"]).resolve()
        if Path(source["path"]).is_absolute() or not raw.is_relative_to((path.parent / "raw").resolve()) or raw.suffix != ".md":
            raise ValueError(f"Source path must be Markdown under raw/: {source['path']}")
        for key in ("url", "download_url", "revision_url"):
            if key in source and (not isinstance(source[key], str) or not re.match(r"https://[^/]+/", source[key])):
                raise ValueError(f"Invalid HTTPS source URL: {source[key]}")
        if source["format"] not in {"wikipedia", "html", "pdf", "transcript", "text"}:
            raise ValueError(f"Unknown source format: {source['format']}")
        if "roles" in source:
            _strings(source["roles"], "source.roles")
    cfg = DatasetConfig(path, data)
    data["config_path"] = str(path)
    data["corpus"] = str(cfg.resolve(data["corpus"]))
    data["output"] = str(cfg.resolve(data["output"]))
    runtime["data_dir"] = str(cfg.resolve(runtime["data_dir"]))
    return cfg
