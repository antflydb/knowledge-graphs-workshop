"""Manifest-backed core datasets: collection, generation and native graph retrieval."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import time

import httpx
from kg.antfly import Antfly, hits
from kg.dataset_config import digest, load_dataset
from kg.dataset_inputs import prepare_inputs, read_manifest, windows
from kg.dataset_decisions import extract_chunk, differentiate_relations, decide_passage, _overflow
from kg.evidence_graph import anchored, entity_key, graph_documents, search_request, projection
from kg.schema import table_schema

PIPELINE = 2


def settings(cfg):
    return cfg["tracks"][cfg["track"]]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def create_models(cfg, model_identity=None):
    if settings(cfg)["provider"] == "python":
        try:
            import importlib.util
            if importlib.util.find_spec("gliner2") is None:
                raise ImportError("Missing stable inference dependencies; use uv run --extra workshop python -m kg.dataset ...")
            from kg.workshop_python import PythonModels
            return PythonModels(cfg["models"], device=os.environ.get("WORKSHOP_DEVICE", "cpu"), revisions=cfg["revisions"])
        except ImportError as error:
            raise ImportError("Stable inference requires uv run --extra workshop python -m kg.dataset ...") from error
    from kg.workshop_native import NativeModels
    return NativeModels(cfg["models"], base_url=settings(cfg)["inference_url"], model_identity=model_identity, revisions=cfg["revisions"])


@contextmanager
def provider(cfg, model_identity=None):
    model = create_models(cfg, model_identity)
    try:
        yield model
    finally:
        model.close()


def embedding_identity(metadata):
    return metadata.get("observed", {}).get("embed", {}).get("model_identity")


class Cache:
    """One generated SQLite artifact per track; no source/extraction/vector JSONL copies."""
    def __init__(self, cfg, expected_identity=None):
        root = Path(cfg["output"])
        root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(root / "model-cache.sqlite")
        self.db.execute("CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        provider_path = Path(__file__).with_name("workshop_" + ("python" if settings(cfg)["provider"] == "python" else "native") + ".py")
        lock_path = Path(__file__).parent.parent / "uv.lock"
        runtime = [hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   hashlib.sha256(Path(__file__).with_name("dataset_decisions.py").read_bytes()).hexdigest(), hashlib.sha256(provider_path.read_bytes()).hexdigest(), hashlib.sha256(lock_path.read_bytes()).hexdigest()]
        self.salt = [PIPELINE, settings(cfg), cfg["models"], cfg["revisions"], runtime]
        self.provenance, self.expected_identity = {}, expected_identity
        self.hits, self.misses = Counter(), Counter()

    def call(self, operation, args, run, model):
        key = digest([self.salt, operation, args])
        found = self.db.execute("SELECT value FROM calls WHERE key=?", (key,)).fetchone()
        if found:
            self.hits[operation] += 1
            entry = json.loads(found[0])
        else:
            self.misses[operation] += 1
            entry = {"result": run(), "provenance": model.metadata}
            self.db.execute("INSERT INTO calls VALUES (?, ?)", (key, json.dumps(entry, allow_nan=False)))
            self.db.commit()
        metadata = entry["provenance"]
        if operation == "embed":
            previous = embedding_identity(self.provenance.get(operation, {})) or self.expected_identity
            current = embedding_identity(metadata)
            if previous is not None and current != previous:
                raise ValueError("Embedding artifact identity differs from the indexed vector space")
        self.provenance[operation] = metadata
        return entry["result"]

    def close(self):
        self.db.close()


def vectors_checked(vectors, count, dimensions):
    if len(vectors) != count or any(len(v) != dimensions or any(not math.isfinite(x) for x in v) or sum(x*x for x in v) == 0 for v in vectors):
        raise ValueError("Invalid embedding count, dimension or finite/nonzero vector")
    return vectors



def native_binary(configured):
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return str(path)
        raise ValueError("Configured antfly.binary does not exist")
    resolved = shutil.which("antfly")
    if not resolved:
        raise ValueError("Install Antfly CLI or set antfly.binary to a released native binary")
    # Resolve npm's native package without invoking its Node server wrapper.
    actual = Path(resolved).resolve()
    if actual.suffix in {".js", ".mjs", ".cjs"} or b"node" in actual.open("rb").read(100):
        import platform
        target = f"cli-{'darwin' if platform.system() == 'Darwin' else 'linux'}-{'arm64' if platform.machine() in {'arm64', 'aarch64'} else 'x64'}"
        for parent in actual.parents:
            candidates = list(parent.glob(f"node_modules/@antfly/{target}/bin/antfly"))
            if candidates:
                return str(candidates[0])
            if parent == Path("/"):
                break
        raise ValueError("Set antfly.binary to the native binary so stopping db does not orphan the server")
    return str(actual)



def binary_info(cfg):
    track = settings(cfg)
    configured = os.environ.get(track["binary_env"])
    if cfg["track"] == "latest" and not configured:
        raise ValueError("Set ANTFLY_LATEST_BIN to the binary built from the pinned main snapshot; never fall back to stable")
    path = native_binary(configured)
    version = subprocess.run([path, "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout.strip()
    if cfg["track"] == "stable" and not re.search(r"\b0\.2\.5\b", version):
        raise ValueError("Stable workshop requires Antfly v0.2.5")
    if cfg["track"] == "latest" and re.search(r"\b0\.2\.\d+\b", version):
        raise ValueError("Latest workshop requires a main/v0.3 build, not a v0.2 release executable")
    return {"path": path, "version": version, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "target_source_commit": track.get("source_commit"), "binary_commit_attested": False}




def database(cfg):
    return Antfly("http://" + settings(cfg)["addr"], timeout=300)


def build_signature(cfg):
    files = ["dataset.py", "dataset_inputs.py", "dataset_decisions.py", "evidence_graph.py", "schema.py",
             "workshop_python.py", "workshop_native.py"]
    code = [hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in files]
    lock = hashlib.sha256((Path(__file__).parent.parent / "uv.lock").read_bytes()).hexdigest()
    return digest([PIPELINE, {key: cfg[key] for key in ("dataset", "extraction", "relation_decision", "models", "revisions", "dimensions", "passages")},
                   settings(cfg), code, lock])


def prepare(cfg, max_passages=None):
    if max_passages is not None and (type(max_passages) is not int or max_passages < 1):
        raise ValueError("--max-passages must be positive")
    return prepare_inputs(cfg, max_passages=max_passages)


def split_passage(cfg, sources, row, error, stage):
    """Split a whole-word window, preserving exact source occurrence identity."""
    count = len(re.findall(r"\S+", row["text"]))
    if count < 2:
        raise ValueError(f"One source word cannot fit {stage} capacity at {row['source_id']}:{row['source_start']}:{row['source_end']}: {error}") from error
    max_words = (count + 1) // 2
    limits = {**cfg["passages"], "max_words": max_words,
              "overlap_words": min(cfg["passages"]["overlap_words"], max_words - 1)}
    children = []
    for begin, end in windows(row["text"], **limits):
        text = row["text"][begin:end]
        start, stop = row["source_start"] + begin, row["source_start"] + end
        key = "passage:" + digest([row["source_id"], sources[row["source_id"]]["sha256"], start, stop, text])[:24]
        children.append({**row, "id": key, "text": text, "source_start": start, "source_end": stop})
    split = {"passage_id": row["id"], "source_id": row["source_id"], "stage": stage,
             "source_start": row["source_start"], "source_end": row["source_end"],
             "reason": str(error), "children": [child["id"] for child in children]}
    return children, split


def extract_passages(cfg, prepared, cache, model):
    """Refine capacity-rejected windows without dropping or rewriting source words."""
    from collections import deque
    pending = deque(prepared["passages"])
    selected, extracted, splits = [], {}, []
    maximum = prepared["selection"]["max_passages"]
    while pending and (maximum is None or len(selected) < maximum):
        row = pending.popleft()
        if row["id"] in extracted:
            continue
        try:
            result = extract_chunk(cfg, row["text"], cache, model)
        except (ValueError, RuntimeError) as error:
            if not _overflow(error):
                raise
            children, split = split_passage(cfg, prepared["sources"], row, error, "extraction")
            splits.append(split)
            pending.extendleft(reversed(children))
            continue
        selected.append(row)
        extracted[row["id"]] = result
        if len(selected) % 50 == 0:
            print(f"Extracted {len(selected)} passages ({len(pending)} windows remaining)", flush=True)
    prepared["passages"] = selected
    prepared["selection"] = {**prepared["selection"], "passages": len(selected), "capacity_splits": splits}
    return extracted


def analyze_passages(cfg, prepared, cache):
    """Run separate model phases, refining only windows rejected by capacity."""
    remaining = prepared["passages"]
    analyses, accepted, splits = {}, {}, []
    while remaining:
        batch = {"sources": prepared["sources"], "passages": remaining,
                 "selection": {"max_passages": None}}
        with provider(cfg) as model:
            extracted = extract_passages(cfg, batch, cache, model)
        splits.extend(batch["selection"]["capacity_splits"])
        remaining = []
        with provider(cfg) as model:
            for index, row in enumerate(batch["passages"], 1):
                if row["id"] in analyses:
                    continue
                try:
                    analysis = differentiate_relations(cfg, row["text"], extracted[row["id"]], cache, model, row.get("context"))
                except (ValueError, RuntimeError) as error:
                    if not _overflow(error):
                        raise
                    children, split = split_passage(cfg, prepared["sources"], row, error, "relations")
                    splits.append(split)
                    remaining.extend(children)
                    continue
                analyses[row["id"]] = analysis
                accepted[row["id"]] = row
                if index % 50 == 0:
                    print(f"Differentiated {index}/{len(batch['passages'])}", flush=True)
        remaining = [row for row in remaining if row["id"] not in analyses]
    source_order = {sid: index for index, sid in enumerate(prepared["sources"])}
    rows = sorted(accepted.values(), key=lambda row: (source_order[row["source_id"]], row["source_start"], row["source_end"]))
    maximum = prepared["selection"]["max_passages"]
    if maximum is not None:
        rows = rows[:maximum]
    prepared["passages"] = rows
    prepared["selection"] = {**prepared["selection"], "passages": len(rows), "capacity_splits": splits}
    return {row["id"]: analyses[row["id"]] for row in rows}


def build(cfg, max_passages=None):
    prepared = prepare(cfg, max_passages)
    binary = binary_info(cfg)
    cache = Cache(cfg)
    try:
        analyses = analyze_passages(cfg, prepared, cache)
        docs = graph_documents(cfg, prepared, analyses)
        with provider(cfg) as model:
            for row in prepared["passages"]:
                args = [row["text"], "RETRIEVAL_DOCUMENT", cfg["dimensions"]]
                vector = cache.call("embed", args, lambda row=row: model.embed([row["text"]], "RETRIEVAL_DOCUMENT", cfg["dimensions"]), model)
                docs[row["id"]]["_embeddings"] = {"vectors": vectors_checked(vector, 1, cfg["dimensions"])[0]}
        provenance = dict(cache.provenance)
        calls = {"actual": cache.misses["classify"], "cached": cache.hits["classify"],
                 "pairs": sum(len(a["decisions"]) for a in analyses.values())}
    finally:
        cache.close()
    signature = build_signature(cfg)
    content_hash = digest([signature, prepared["corpus_sha256"], prepared["selection"], docs])
    table = "kg_" + cfg["dataset"].replace("-", "_") + "_" + cfg["track"] + "_" + content_hash[:16]
    output = Path(cfg["output"]) / "build.json"
    old = json.loads(output.read_text()) if output.exists() else {}
    af = database(cfg)
    try:
        if table in af.list_tables():
            if old.get("content_hash") != content_hash or old.get("url") != str(af.http.base_url):
                raise ValueError("Existing table has no matching completed build receipt; use a fresh database")
        else:
            af.create_table(table, table_schema(cfg["dimensions"]))
            items = list(docs.items())
            for offset in range(0, len(items), 100):
                af.batch(table, dict(items[offset:offset + 100]))
        probe = prepared["passages"][0]
        observed = hits(af.query(table, {"embeddings": {"vectors": docs[probe['id']]["_embeddings"]["vectors"]},
                      "indexes": ["vectors"], "filter_query": {"term": "passage", "field": "kind"}, "limit": 5}))
        if probe["id"] not in {h["_id"] for h in observed} or af.get(table, probe["id"]) is None:
            raise ValueError("Indexed vector round-trip failed; previous receipt is unchanged")
        manifest = {"pipeline": PIPELINE, "dataset": cfg["dataset"], "track": cfg["track"], "target": settings(cfg)["target"],
            "binary": binary, "source_commit": settings(cfg).get("source_commit"),
            "distributable": cfg["track"] == "stable" and all(s["redistribution"] == "allowed" for s in prepared["sources"].values()),
            "config_signature": signature, "corpus_sha256": prepared["corpus_sha256"], "selection": prepared["selection"],
            "table": table, "build_id": content_hash, "url": str(af.http.base_url), "content_hash": content_hash,
            "counts": dict(Counter(d["kind"] for d in docs.values())), "relationship_calls": calls,
            "analysis_counts": {key: sum(len(a[key]) for a in analyses.values()) for key in ("entities", "proposals", "decisions", "assertions")},
            "models": cfg["models"], "revisions": cfg["revisions"], "model_provenance": provenance,
            "verification": {"indexed_passage_vector_roundtrip": True}}
        save(output, manifest)
        return manifest
    finally:
        af.http.close()


def load_build(cfg):
    path = Path(cfg["output"]) / "build.json"
    if not path.exists():
        raise ValueError(f"No qualified build for {cfg['dataset']}/{cfg['track']}; run python -m kg.dataset {cfg['config_path']} --track {cfg['track']} build")
    manifest = json.loads(path.read_text())
    if manifest["config_signature"] != build_signature(cfg):
        raise ValueError("Build settings/code changed; rebuild selected track")
    prepared = prepare(cfg, manifest["selection"]["max_passages"])
    if manifest["corpus_sha256"] != prepared["corpus_sha256"]:
        raise ValueError("Corpus changed; rebuild before using evidence offsets")
    return manifest


def citation(source, passage):
    keys = ("id", "title", "url", "download_url", "revision_id", "revision_url", "page_id", "license", "attribution", "source_kind")
    return {**{key: source[key] for key in keys if key in source},
            "source_start": passage["source_start"], "source_end": passage["source_end"],
            "context": passage.get("context", {})}


def query(cfg, text, mode="graph", models=None):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Query must not be empty")
    if mode not in {"text", "hybrid", "graph"}:
        raise ValueError("Unknown query mode")
    start = time.perf_counter()
    manifest = load_build(cfg)
    identity = embedding_identity(manifest["model_provenance"]["embed"])
    owned = models is None
    model = models if models is not None else create_models(cfg, identity)
    cache, af = Cache(cfg, identity), database(cfg)
    timings = {}
    def timed(name, operation):
        before = time.perf_counter()
        hit, miss = sum(cache.hits.values()), sum(cache.misses.values())
        value = operation()
        timings[name] = {"ms": round((time.perf_counter() - before) * 1000, 2),
                         "cache_hits": sum(cache.hits.values()) - hit, "model_calls": sum(cache.misses.values()) - miss}
        return value
    try:
        extracted = timed("extraction", lambda: extract_chunk(cfg, text, cache, model))
        analysis = timed("relations", lambda: differentiate_relations(cfg, text, extracted, cache, model, {"kind": "query"}))
        vector = None
        if mode != "text":
            args = [text, "RETRIEVAL_QUERY", cfg["dimensions"]]
            vector = timed("embedding", lambda: vectors_checked(cache.call("embed", args,
                lambda: model.embed([text], "RETRIEVAL_QUERY", cfg["dimensions"]), model), 1, cfg["dimensions"])[0])
        else:
            timings["embedding"] = {"ms": 0, "cache_hits": 0, "model_calls": 0}
        before = time.perf_counter()
        entities = []
        if mode == "graph":
            for span in analysis["entities"]:
                eid = entity_key(span["label"], span["text"])
                if span["score"] >= cfg["extraction"]["threshold"] and af.get(manifest["table"], eid) is not None:
                    entities.append(eid)
        flat = hits(af.query(manifest["table"], search_request(cfg, text, vector)))
        scoped = []
        if entities:
            response = af.query(manifest["table"], {"limit": 0, "graph_queries": {"evidence": {"index": "graph", "traverse": {
                "start": {"keys": sorted(set(entities))}, "edge_types": ["mentions", "subject", "object", "supported_by"],
                "direction": "both", "max_depth": 2, "limit": cfg["retrieval"]["graph_candidates"], "include_documents": True}}}})
            nodes = response["responses"][0]["graph_results"]["evidence"]["nodes"]
            allowed = {node["key"] for node in nodes if node["key"].startswith("passage:")}
            if allowed:
                scoped = hits(af.query(manifest["table"], search_request(cfg, text, vector, allowed)))
        selected, seen = [], set()
        def add(ranking, limit):
            for row in ranking:
                if len(selected) >= limit:
                    break
                if row["_id"] not in seen:
                    selected.append({"passage_id": row["_id"], "score": row.get("_score"),
                                     **{k: v for k, v in row.items() if not k.startswith("_")}})
                    seen.add(row["_id"])
        budget = cfg["retrieval"]["evidence"]
        if scoped:
            add(scoped, budget - cfg["retrieval"]["text_reserve"])
        add(flat, budget)
        sources = {row["id"]: row for row in read_manifest(cfg)}
        for row in selected:
            source = sources[row["source_id"]]
            if row["text"] != source["text"][row["source_start"]:row["source_end"]]:
                raise ValueError("Retrieved evidence does not match exact raw offsets")
            row["citation"] = citation(source, row)
        timings["retrieval"] = {"ms": round((time.perf_counter() - before) * 1000, 2)}
        timings["total_ms"] = round((time.perf_counter() - start) * 1000, 2)
        return {"build_id": manifest["build_id"], "partial": manifest["selection"]["partial"],
                "query": {"text": text, "analysis": analysis}, "results": selected,
                "graph": projection(analysis, selected), "timings": timings}
    finally:
        cache.close()
        af.http.close()
        if owned:
            model.close()


def decide(cfg, text, instruction=None, labels=None, results=1, models=None):
    if results not in (1, 2, 3):
        raise ValueError("results must be 1, 2 or 3")
    packet = query(cfg, text, models=models)
    manifest = load_build(cfg)
    owned = models is None
    model = models if models is not None else create_models(cfg, embedding_identity(manifest["model_provenance"]["embed"]))
    cache = Cache(cfg)
    try:
        packet["experiments"] = [decide_passage(cfg, text, packet["query"]["analysis"], row, cache, model,
                                 instruction=instruction, labels=labels) for row in packet["results"][:results]]
        return packet
    finally:
        cache.close()
        if owned:
            model.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--track", choices=("stable", "latest"), default="stable")
    sub = parser.add_subparsers(dest="command", required=True)
    collector = sub.add_parser("collect")
    collector.add_argument("--refresh", action="store_true")
    for name in ("validate", "prepare", "doctor", "db"):
        sub.add_parser(name)
    builder = sub.add_parser("build")
    builder.add_argument("--max-passages", type=int)
    search = sub.add_parser("query")
    search.add_argument("text")
    search.add_argument("--mode", choices=("text", "hybrid", "graph"), default="graph")
    experiment = sub.add_parser("decide")
    experiment.add_argument("text")
    experiment.add_argument("--instruction")
    experiment.add_argument("--labels", type=json.loads)
    experiment.add_argument("--results", type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()
    try:
        cfg = load_dataset(args.config, args.track).data
        if args.command == "collect":
            from kg.collect import collect
            result = collect(cfg, refresh=args.refresh)
        elif args.command in ("validate", "prepare"):
            prepared = prepare(cfg)
            result = {"valid": True, "sources": len(prepared["sources"]), "passages": len(prepared["passages"]),
                      "corpus_sha256": prepared["corpus_sha256"], "selection": prepared["selection"]}
        elif args.command == "doctor":
            result = {"track": args.track, "binary": binary_info(cfg), "models": cfg["models"],
                      "revisions": cfg["revisions"], "provider": settings(cfg)["provider"],
                      "qualification": "Binary receipt is not model/runtime qualification"}
        elif args.command == "db":
            info = binary_info(cfg)
            path = Path(settings(cfg)["database"])
            path.parent.mkdir(parents=True, exist_ok=True)
            print(json.dumps(info), flush=True)
            raise SystemExit(subprocess.call([info["path"], "lite", "serve", str(path), "--addr", settings(cfg)["addr"],
                                               "--health-port", str(settings(cfg)["health_port"])]))
        elif args.command == "build":
            result = build(cfg, args.max_passages)
        elif args.command == "query":
            result = query(cfg, args.text, args.mode)
        else:
            result = decide(cfg, args.text, args.instruction, args.labels, args.results)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    except (ValueError, KeyError, FileNotFoundError, RuntimeError, ImportError, httpx.HTTPError) as error:
        if hasattr(error, "payload"):
            print(json.dumps(error.payload, indent=2, ensure_ascii=False))
        parser.exit(1, f"dataset: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130)


if __name__ == "__main__":
    main()
