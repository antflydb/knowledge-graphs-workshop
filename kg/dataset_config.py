"""Dataset configuration, validated before any model or table is opened.

A dataset is `datasets/<slug>/`: `config.yaml` holds what a person tunes (GLiNER's
vocabulary, the Decide instruction, passage size), `collection.yaml` what `make
collect` fetches (optional: without it, every `raw/*.md` file is a local source), and
the repo's `workshop.yaml` the shared local Antfly runtime.
Everything else (model pins, retrieval sizes, paths) is fixed here.
"""
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


ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL = {"candidates": 40, "graph_candidates": 100, "evidence": 25, "text_reserve": 1}
TUNABLE = {"title", "passages", "extraction", "relation_decision", "explicit_links", "transcript_epoch", "feed", "dashboard"}


def _local_collection(root):
    """Every Markdown file under raw/ as a bundled source, titled by its front-matter title or first heading."""
    sources = []
    for file in sorted((root / "raw").rglob("*.md")):
        relative, text = file.relative_to(root).as_posix(), file.read_text(encoding="utf-8")
        front = re.match(r"---\n(.*?)\n---\n", text, re.S)
        titled = re.search(r"(?m)^title:\s*['\"]?(.+?)['\"]?\s*$", front.group(1)) if front else None
        heading = titled.group(1) if titled else next((line[2:].strip() for line in text.splitlines() if line.startswith("# ")), "")
        sources.append({"id": root.name + ":" + re.sub(r"[^A-Za-z0-9._-]+", "-", relative[len("raw/"):-len(".md")]),
                        "path": relative, "title": heading or file.stem.replace("_", " ").replace("-", " "), "format": "text"})
    return {"kind": "bundled", "user_agent": "local files", "request_interval_seconds": 1, "sources": sources}


def dashboard_datasets():
    """Every datasets/<slug>/config.yaml the dashboard shows, by slug (read fresh each call)."""
    return {path.parent.name: path for path in sorted((ROOT / "datasets").glob("*/config.yaml"))
            if load_dataset(path).data.get("dashboard", True)}


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


def _explicit_links(links):
    """Optional links written in the text itself (e.g. "#752"), read with a pattern rather than a model."""
    if links is None:
        return
    if not isinstance(links, list) or not links:
        raise ValueError("explicit_links must be a nonempty list")
    for link in links:
        _mapping(link, {"type", "field", "pattern", "name", "label"}, {"type", "field", "pattern", "name", "label"}, "explicit link")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", link["type"]) or link["field"] not in {"text", "title"}:
            raise ValueError("explicit link type must be an identifier and field text or title")
        try:
            if re.compile(link["pattern"], re.M).groups < 1:
                raise ValueError("explicit link pattern needs a capture group")
        except re.error as exc:
            raise ValueError(f"Invalid explicit link pattern: {exc}") from exc
        for key in ("name", "label"):
            _text(link[key], f"explicit link {key}")


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
    _mapping(data, TUNABLE, {"title", "passages", "extraction", "relation_decision"}, "config")
    runtime = yaml.safe_load((ROOT / "workshop.yaml").read_text(encoding="utf-8"))
    _mapping(runtime, {"antfly"}, {"antfly"}, "workshop.yaml")
    runtime["antfly"]["data_dir"] = str((ROOT / runtime["antfly"]["data_dir"]).resolve())
    collection_file = path.parent / "collection.yaml"
    data = {"version": 2, "dataset": path.parent.name, "corpus": "metadata.jsonl", "output": f"../../data/builds/{path.parent.name}",
            "models": {role: pin[0] for role, pin in MODEL_PINS.items()}, "revisions": {role: pin[1] for role, pin in MODEL_PINS.items()},
            "dimensions": 768, "retrieval": dict(RETRIEVAL), "antfly": runtime["antfly"],
            "collection": yaml.safe_load(collection_file.read_text(encoding="utf-8")) if collection_file.exists() else _local_collection(path.parent),
            **data}
    if type(data.get("dashboard", True)) is not bool:
        raise ValueError("dashboard must be true or false")
    if "transcript_epoch" in data and not (isinstance(data["transcript_epoch"], str)
                                           and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", data["transcript_epoch"])):
        raise ValueError("transcript_epoch must be a UTC time like 1970-04-11T19:13:00Z")
    if "feed" in data:
        _mapping(data["feed"], {"messages", "record"}, {"messages", "record"}, "feed")
        for key in ("messages", "record"):
            _text(data["feed"][key], f"feed.{key}")
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
    _explicit_links(data.get("explicit_links"))
    collection = data["collection"]
    _mapping(collection, {"kind", "sources", "user_agent", "request_interval_seconds", "exclude_sections", "transcript_window"}, {"kind", "sources", "user_agent", "request_interval_seconds"}, "collection")
    if collection["kind"] not in {"wikipedia", "restaurant_pages", "apollo13", "bundled"}:
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
        # Local (bundled) files need no URL; fetched sources do.
        required = {"id", "path", "title", "format"} | (set() if collection["kind"] == "bundled" else {"url"})
        _mapping(source, {"id", "path", "title", "url", "download_url", "format", "restaurant_id", "roles", "page_id", "revision_id", "revision_url", "source_sha256", "sha256"}, required, "collection source")
        for key in required:
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
        elif kind == "bundled":
            if source["format"] != "text":
                raise ValueError("Bundled sources are checked-in Markdown text")
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
