"""Single-worker, loopback-only workshop surface: graph search and the text-to-graph playground."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, StrictStr

from kg import dataset
from kg.antfly import AntflyError
from kg.dataset_config import digest, load_dataset
from kg.dataset_decisions import DecisionError, decide_relationship
from kg.dataset_inputs import read_manifest
from kg.search import StaleSearch, focused_search

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "asoiaf": ROOT / "datasets/asoiaf/config.yaml",
    "pizza-portland": ROOT / "datasets/pizza-portland/config.yaml",
    "apollo13": ROOT / "datasets/apollo13/config.yaml",
}
DatasetName = Literal["asoiaf", "pizza-portland", "apollo13"]
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
# Prebuilt from playground/ (`make playground`); committed so nobody needs Node to run it.
PLAYGROUND = Path(__file__).parent / "static" / "playground"


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: DatasetName
    text: StrictStr
    focus_id: StrictStr | None = None
    build_id: StrictStr | None = None
    navigate: bool = False


class Span(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: StrictStr
    start: int
    end: int
    label: StrictStr


class DecideRequest(BaseModel):
    """Re-classify one extracted relationship: an exact head and tail in one passage."""
    model_config = ConfigDict(extra="forbid")
    dataset: DatasetName
    build_id: StrictStr
    passage_id: StrictStr
    head: Span
    tail: Span
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

    def provider(self, cfg):
        key = digest(cfg)
        if key != self.model_key:
            self.close()
            self.models = dataset.create_models(cfg)
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
app.mount("/playground", StaticFiles(directory=PLAYGROUND, html=True), name="playground")


def config(name):
    if name not in DATASETS:
        raise HTTPException(422, "Select asoiaf, pizza-portland or apollo13")
    return load_dataset(DATASETS[name]).data


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
            cfg = config(name)
            defaults[name] = {"title": cfg["title"], "instruction": cfg["relation_decision"]["instruction"]}
    except Exception as exc:
        return failure(exc)
    return templates.TemplateResponse(request=request, name="index.html", context={"defaults": defaults})


@app.post("/api/search")
def search(body: SearchRequest):
    with state.lock:
        try:
            cfg = config(body.dataset)
            try:
                dataset.load_build(cfg)
            except (ValueError, KeyError) as exc:
                return failure(exc, stale=body.focus_id is not None)
            # A recenter uses the index alone; do not even construct a provider.
            models = None if body.navigate else state.provider(cfg)
            packet = focused_search(cfg, body.text, models=models,
                                    focus_id=body.focus_id, build_id=body.build_id,
                                    navigate=body.navigate)
            return packet
        except Exception as exc:
            return failure(exc)


@app.post("/api/decide")
def decide(body: DecideRequest):
    """On-demand GLiNER2.5-Decide for one relationship; nothing is written back to the graph."""
    with state.lock:
        try:
            cfg = config(body.dataset)
            try:
                manifest = dataset.load_build(cfg)
            except (ValueError, KeyError) as exc:
                return failure(exc, stale=True)
            if manifest["build_id"] != body.build_id:
                raise HTTPException(409, "Build changed; search again before running Decide")
            af = dataset.database(cfg)
            try:
                passage = af.get(manifest["table"], body.passage_id)
            finally:
                af.http.close()
            if passage is None or passage.get("kind") != "passage":
                raise HTTPException(404, "Indexed passage no longer exists; search again")
            return decide_relationship(cfg, passage, body.head.model_dump(), body.tail.model_dump(),
                                       state.provider(cfg), instruction=body.instruction, labels=body.labels)
        except Exception as exc:
            return failure(exc)


@app.api_route("/ai/{path:path}", methods=["GET", "POST"])
async def inference(path: str, request: Request):
    """Forward the playground's inference calls to Antfly, keeping the page same-origin."""
    base = load_dataset(DATASETS["asoiaf"]).data["antfly"]["inference_url"]
    async with httpx.AsyncClient(base_url=base, timeout=300) as client:
        try:
            upstream = await client.request(request.method, f"/ai/{path}", content=await request.body(),
                                            headers={"content-type": request.headers.get("content-type", "application/json")})
        except httpx.HTTPError as exc:
            return JSONResponse({"error": "INFERENCE_UNREACHABLE", "message": str(exc)}, status_code=502)
    return Response(upstream.content, status_code=upstream.status_code,
                    media_type=upstream.headers.get("content-type"))


@app.get("/raw/{dataset_name}/{source_id:path}", response_class=PlainTextResponse)
def raw(dataset_name: str, source_id: str):
    try:
        cfg = config(dataset_name)
        # read_manifest validates registered paths, hashes and confinement to raw/.
        source = next((row for row in read_manifest(cfg) if row["id"] == source_id), None)
        if source is None:
            raise HTTPException(404, "Source is not registered in this dataset manifest")
        return PlainTextResponse(source["text"], headers={"X-Content-Type-Options": "nosniff"})
    except Exception as exc:
        return failure(exc)
