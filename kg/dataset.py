"""Manifest-backed core datasets: collection, Antfly-native graph builds and retrieval."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import httpx
from kg.antfly import Antfly, hits
from kg.dataset_config import digest, load_dataset
from kg.dataset_inputs import prepare_inputs, read_manifest
from kg.dataset_decisions import query_entities
from kg.evidence_graph import passage_documents, search_request
from kg.schema import MENTIONS, passage_table

PIPELINE = 4
READY_TIMEOUT_SECONDS = 3600


def settings(cfg):
    return cfg["antfly"]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def create_models(cfg):
    """Client for on-demand inference (query entities, interactive Decide), never the build."""
    from kg.workshop_native import NativeModels
    return NativeModels(cfg["models"], base_url=settings(cfg)["inference_url"], revisions=cfg["revisions"])


def native_binary(configured):
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return str(path)
        raise ValueError(f"ANTFLY_BIN does not exist: {path}")
    resolved = shutil.which("antfly")
    if not resolved:
        raise ValueError("Install the Antfly CLI or set ANTFLY_BIN to a v0.3 native binary")
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
        raise ValueError("Set ANTFLY_BIN to the native binary so stopping db does not orphan the server")
    return str(actual)


def binary_info(cfg):
    runtime = settings(cfg)
    path = native_binary(os.environ.get(runtime["binary_env"]))
    version = subprocess.run([path, "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout.strip()
    # Artifact-backed graph indexes, extractor enrichments and resolvers first shipped in v0.3.
    if not re.search(r"\b0\.(?:[3-9]|\d{2,})\.\d+", version):
        raise ValueError(f"The workshop requires Antfly {runtime['target']} or later; {path} reports {version!r}. "
                         f"Set {runtime['binary_env']} to a v0.3 binary.")
    return {"path": path, "version": version, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def database(cfg):
    return Antfly("http://" + settings(cfg)["addr"], timeout=300)


def entities_by_surface(af, manifest, texts, limit=100):
    """Resolver-promoted entities with one of these exact surface forms as an alias."""
    texts = sorted({text for text in texts if text.strip()})
    if not texts:
        return []
    return hits(af.query(manifest["table"], {
        "filter_query": {"disjuncts": [{"term": text, "field": "aliases"} for text in texts]}, "limit": limit}))


def build_signature(cfg):
    """What the indexed graph depends on: passages, extraction schema, models and table layout."""
    files = ["dataset.py", "dataset_inputs.py", "evidence_graph.py", "schema.py"]
    code = [hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in files]
    return digest([PIPELINE, {key: cfg[key] for key in ("dataset", "extraction", "models", "revisions", "dimensions", "passages")},
                   cfg.get("explicit_links"), settings(cfg), code])


def prepare(cfg, max_passages=None):
    if max_passages is not None and (type(max_passages) is not int or max_passages < 1):
        raise ValueError("--max-passages must be positive")
    return prepare_inputs(cfg, max_passages=max_passages)


def wait_until_complete(af, table, indexes, timeout=READY_TIMEOUT_SECONDS, progress=print):
    """Poll index readiness; `complete` is Antfly's convergence authority, `failed` is terminal."""
    started = time.monotonic()
    while True:
        statuses = {name: af.index_status(table, name) for name in indexes}
        failed = {name: s["readiness"].get("pending_reasons") for name, s in statuses.items()
                  if s["readiness"].get("state") == "failed"}
        if failed:
            raise RuntimeError(f"Antfly index failed while converging: {failed}")
        if all(s["readiness"].get("complete") for s in statuses.values()):
            return statuses
        elapsed = time.monotonic() - started
        if elapsed > timeout:
            pending = {name: s["readiness"].get("pending_reasons") for name, s in statuses.items()}
            raise RuntimeError(f"Indexes did not complete within {timeout}s: {pending}")
        if progress:
            progress(f"Antfly indexing +{elapsed:.0f}s: " + ", ".join(
                f"{name} {s['readiness'].get('state')}" + (f" ({s['edge_count']} edges)" if s.get("edge_count") else "")
                for name, s in statuses.items()), flush=True)
        time.sleep(5)


def build(cfg, max_passages=None):
    """Write plain passages; Antfly extracts, embeds, resolves entities and links the graph."""
    prepared = prepare(cfg, max_passages)
    binary = binary_info(cfg)
    docs = passage_documents(cfg, prepared)
    probe = prepared["passages"][0]
    signature = build_signature(cfg)
    content_hash = digest([signature, prepared["corpus_sha256"], prepared["selection"], docs])
    table = "kg_" + cfg["dataset"].replace("-", "_") + "_" + content_hash[:16]
    output = Path(cfg["output"]) / "build.json"
    old = json.loads(output.read_text()) if output.exists() else {}
    af = database(cfg)
    started = time.monotonic()
    try:
        existing = af.list_tables()
        if table in existing:
            if old.get("content_hash") != content_hash or old.get("url") != str(af.http.base_url):
                raise ValueError("Existing table has no matching completed build receipt; use a fresh database")
        else:
            af.create_table(table, passage_table(cfg, table))
            items = list(docs.items())
            for offset in range(0, len(items), 100):
                af.batch(table, dict(items[offset:offset + 100]), sync_level="write")
            print(f"Wrote {len(prepared['passages'])} passages ({len(items)} documents) to {table}", flush=True)
        statuses = wait_until_complete(af, table, ("vectors", "graph"))
        observed = hits(af.query(table, search_request(cfg, probe["text"])))
        if probe["id"] not in {h["_id"] for h in observed} or af.get(table, probe["id"]) is None:
            raise ValueError("Indexed passage probe failed; previous receipt is unchanged")
        manifest = {"pipeline": PIPELINE, "dataset": cfg["dataset"], "target": settings(cfg)["target"], "binary": binary,
            "distributable": all(s["redistribution"] == "allowed" for s in prepared["sources"].values()),
            "config_signature": signature, "corpus_sha256": prepared["corpus_sha256"], "selection": prepared["selection"],
            "table": table, "build_id": content_hash,
            "url": str(af.http.base_url), "content_hash": content_hash,
            "counts": {"passages": len(prepared["passages"]), "entities": af.count(table) - len(prepared["passages"]),
                       "edges": statuses["graph"].get("edge_count")},
            "seconds": round(time.monotonic() - started, 1),
            "models": cfg["models"], "revisions": cfg["revisions"],
            "verification": {"indexes_complete": True, "semantic_probe": True}}
        save(output, manifest)
        return manifest
    finally:
        af.http.close()


def load_build(cfg):
    path = Path(cfg["output"]) / "build.json"
    if not path.exists():
        raise ValueError(f"No qualified build for {cfg['dataset']}; run python -m kg.dataset {cfg['config_path']} build")
    manifest = json.loads(path.read_text())
    if manifest.get("config_signature") != build_signature(cfg):
        raise ValueError("Build settings/code changed; rebuild this dataset")
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
    owned = models is None
    model = models if models is not None else create_models(cfg)
    af = database(cfg)
    timings = {}
    try:
        before = time.perf_counter()
        entities = query_entities(cfg, text, model)
        timings["extraction"] = {"ms": round((time.perf_counter() - before) * 1000, 2)}
        before = time.perf_counter()
        semantic = mode != "text"
        keys = set()
        if mode == "graph":
            keys = {row["_id"] for row in entities_by_surface(af, manifest, [span["text"] for span in entities])}
        flat = hits(af.query(manifest["table"], search_request(cfg, text, semantic)))
        scoped = []
        if keys:
            # The resolver links each passage to the entities it mentions.
            nodes = af.traverse(manifest["table"], "graph", keys, edge_types=[MENTIONS], direction="in",
                                max_depth=1, limit=cfg["retrieval"]["graph_candidates"])
            allowed = {node["key"] for node in nodes if node["key"].startswith("passage:")}
            if allowed:
                scoped = hits(af.query(manifest["table"], search_request(cfg, text, semantic, allowed)))
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
                "query": {"text": text, "entities": entities}, "results": selected, "timings": timings}
    finally:
        af.http.close()
        if owned:
            model.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    collector = sub.add_parser("collect")
    collector.add_argument("--refresh", action="store_true")
    for name in ("validate", "prepare", "doctor", "inference", "db"):
        sub.add_parser(name)
    builder = sub.add_parser("build")
    builder.add_argument("--max-passages", type=int)
    search = sub.add_parser("query")
    search.add_argument("text")
    search.add_argument("--mode", choices=("text", "hybrid", "graph"), default="graph")
    args = parser.parse_args()
    try:
        cfg = load_dataset(args.config).data
        if args.command == "collect":
            from kg.collect import collect
            result = collect(cfg, refresh=args.refresh)
        elif args.command in ("validate", "prepare"):
            prepared = prepare(cfg)
            result = {"valid": True, "sources": len(prepared["sources"]), "passages": len(prepared["passages"]),
                      "corpus_sha256": prepared["corpus_sha256"], "selection": prepared["selection"]}
        elif args.command == "doctor":
            result = {"binary": binary_info(cfg), "models": cfg["models"], "revisions": cfg["revisions"],
                      "qualification": "Binary receipt is not model/runtime qualification"}
        elif args.command == "inference":
            info = binary_info(cfg)
            host, port = settings(cfg)["inference_url"].removeprefix("http://").rsplit(":", 1)
            print(json.dumps(info), flush=True)
            raise SystemExit(subprocess.call([info["path"], "inference", "run", "--host", host, "--port", port]))
        elif args.command == "db":
            info = binary_info(cfg)
            data_dir = Path(settings(cfg)["data_dir"])
            data_dir.mkdir(parents=True, exist_ok=True)
            print(json.dumps(info), flush=True)
            host, port = settings(cfg)["addr"].rsplit(":", 1)
            # The default storage engine: artifact-backed graph and embedding workers
            # fail permanently under bulk load on rc.1's Lite engine.
            raise SystemExit(subprocess.call([info["path"], "standalone", "--data-dir", str(data_dir), "--host", host,
                                              "--port", port, "--health-port", str(settings(cfg)["health_port"])]))
        elif args.command == "build":
            result = build(cfg, args.max_passages)
        else:
            result = query(cfg, args.text, args.mode)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    except (ValueError, KeyError, FileNotFoundError, RuntimeError, ImportError, httpx.HTTPError) as error:
        parser.exit(1, f"dataset: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130)


if __name__ == "__main__":
    main()
