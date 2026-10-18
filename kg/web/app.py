"""Single-worker, loopback-only dataset graph and Decide workshop surface."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, StrictStr

from kg import dataset
from kg.antfly import AntflyError
from kg.dataset_config import digest, load_dataset
from kg.dataset_decisions import DecisionError, analyze_chunk, decide_passage, experiment_payload
from kg.dataset_inputs import read_manifest
from kg.search import StaleSearch, focused_search

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "asoiaf": ROOT / "datasets/asoiaf/config.yaml",
    "pizza-portland": ROOT / "datasets/pizza-portland/config.yaml",
    "apollo13": ROOT / "datasets/apollo13/config.yaml",
}
DatasetName = Literal["asoiaf", "pizza-portland", "apollo13"]
TrackName = Literal["stable", "latest"]
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: DatasetName
    track: TrackName = "stable"
    text: StrictStr
    focus_id: StrictStr | None = None
    build_id: StrictStr | None = None
    navigate: bool = False


class DecideRequest(SearchRequest):
    build_id: StrictStr
    passage_id: StrictStr
    instruction: Any = None
    labels: Any = None


class ViewerState:
    """One request at a time; never hold multiple resident model providers."""

    def __init__(self):
        self.lock = RLock()
        self.models = None
        self.model_key = None

    def close(self):
        with self.lock:
            models, self.models = self.models, None
            self.model_key = None
            if models is not None:
                models.close()

    def provider(self, cfg, manifest):
        identity = dataset.embedding_identity(manifest["model_provenance"]["embed"])
        key = digest([cfg["track"], cfg, identity])
        if key != self.model_key:
            self.close()
            self.models = dataset.create_models(cfg, identity)
            self.model_key = key
        return self.models


state = ViewerState()


@asynccontextmanager
async def lifespan(app):
    try:
        yield
    finally:
        state.close()


app = FastAPI(title="Core dataset graph search", lifespan=lifespan)


def config(name, track):
    if name not in DATASETS or track not in ("stable", "latest"):
        raise HTTPException(422, "Select asoiaf, pizza-portland or apollo13 and stable or latest")
    return load_dataset(DATASETS[name], track).data


def failure(exc, *, stale=False):
    if isinstance(exc, HTTPException):
        raise exc
    if isinstance(exc, StaleSearch):
        return JSONResponse({"error": str(exc)}, status_code=409)
    if isinstance(exc, DecisionError):
        return JSONResponse(exc.payload, status_code=exc.status_code)
    if isinstance(exc, (httpx.HTTPError, AntflyError, ImportError, RuntimeError, OSError)):
        return JSONResponse({"error": str(exc)}, status_code=503)
    if isinstance(exc, (ValueError, KeyError)):
        return JSONResponse({"error": str(exc)}, status_code=409 if stale else 422)
    raise exc


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    defaults = {}
    try:
        for name in DATASETS:
            cfg = config(name, "stable")
            defaults[name] = {"title": cfg["title"], "query_decision": cfg["query_decision"]}
    except Exception as exc:
        return failure(exc)
    return templates.TemplateResponse(request=request, name="index.html", context={"defaults": defaults})


@app.post("/api/search")
def search(body: SearchRequest):
    with state.lock:
        try:
            cfg = config(body.dataset, body.track)
            try:
                manifest = dataset.load_build(cfg)
            except (ValueError, KeyError) as exc:
                return failure(exc, stale=body.focus_id is not None)
            # A recenter uses the index alone; do not even construct a provider.
            models = None if body.navigate else state.provider(cfg, manifest)
            packet = focused_search(cfg, body.text, models=models,
                                    focus_id=body.focus_id, build_id=body.build_id,
                                    navigate=body.navigate)
            return packet
        except Exception as exc:
            return failure(exc)


@app.post("/api/decide")
def decide(body: DecideRequest):
    with state.lock:
        cache = None
        preview = None
        try:
            cfg = config(body.dataset, body.track)
            # Validate the current corpus/build before trusting a prior query or offsets.
            try:
                manifest = dataset.load_build(cfg)
            except (ValueError, KeyError) as exc:
                return failure(exc, stale=True)
            if manifest["build_id"] != body.build_id:
                raise HTTPException(409, "Build changed; search again before running Decide")
            if not body.text.strip():
                raise ValueError("Query must not be empty")
            af = dataset.database(cfg)
            try:
                passage = af.get(manifest["table"], body.passage_id)
            finally:
                af.http.close()
            if passage is None or passage.get("kind") != "passage":
                raise HTTPException(404, "Indexed passage no longer exists; search again")
            passage = {**passage, "passage_id": body.passage_id}
            sources = {row["id"]: row for row in read_manifest(cfg)}
            source = sources[passage["source_id"]]
            start, end = passage["source_start"], passage["source_end"]
            if (type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(source["text"])
                    or passage["text"] != source["text"][start:end]):
                raise HTTPException(409, "Indexed evidence differs from raw source; rebuild and search again")
            passage["citation"] = dataset.citation(source, passage)
            identity = dataset.embedding_identity(manifest["model_provenance"]["embed"])
            cache = dataset.Cache(cfg, identity)
            # Normal searches replay their cached real analysis. Navigation and
            # locally restored Back packets compute it lazily here, never on a
            # graph click. No server-side last-result/session restriction.
            models = state.provider(cfg, manifest)
            query_analysis = analyze_chunk(cfg, body.text, cache, models, {"kind": "query"})
            preview = experiment_payload(cfg, body.text, query_analysis, passage,
                                         instruction=body.instruction, labels=body.labels)
            return decide_passage(cfg, body.text, query_analysis, passage, cache,
                                  models, instruction=body.instruction, labels=body.labels)
        except Exception as exc:
            response = failure(exc)
            if preview is not None and not isinstance(exc, DecisionError):
                return JSONResponse({**preview, "error": str(exc)}, status_code=response.status_code)
            return response
        finally:
            if cache is not None:
                cache.close()


@app.get("/raw/{dataset_name}/{source_id:path}", response_class=PlainTextResponse)
def raw(dataset_name: str, source_id: str):
    try:
        cfg = config(dataset_name, "stable")
        # read_manifest validates registered paths, hashes and confinement to raw/.
        source = next((row for row in read_manifest(cfg) if row["id"] == source_id), None)
        if source is None:
            raise HTTPException(404, "Source is not registered in this dataset manifest")
        return PlainTextResponse(source["text"], headers={"X-Content-Type-Options": "nosniff"})
    except Exception as exc:
        return failure(exc)
