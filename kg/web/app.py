"""Single-worker, loopback-only workshop surface: graph search and the text-to-graph playground."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from threading import RLock
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, StrictStr

from kg import dataset
from kg.antfly import AntflyError
from kg.dataset_config import digest, load_dataset
from kg.dataset_decisions import RESERVED_LABELS, DecisionError, decide, relationship_labels
from kg.dataset_inputs import read_manifest
from kg.search import StaleSearch, feed_messages, focused_search, message_view, overview

ROOT = Path(__file__).resolve().parents[2]
# Every datasets/<slug>/config.yaml; adding or deleting a dataset directory is all it takes.
# `dashboard: false` in a config keeps a dataset buildable from the CLI but off the page.
DATASETS = {path.parent.name: path for path in sorted((ROOT / "datasets").glob("*/config.yaml"))
            if load_dataset(path).data.get("dashboard", True)}
LABELS = {"asoiaf": "ASoIaF", "pizza-portland": "Portland pizza", "apollo13": "Apollo 13",
          "antfly-prs": "Antfly PRs: edge-weight fix"}
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
# Prebuilt from playground/ (`make playground`); committed so nobody needs Node to run it.
PLAYGROUND = Path(__file__).parent / "static" / "playground"


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: StrictStr
    text: StrictStr
    focus_id: StrictStr | None = None
    build_id: StrictStr | None = None
    navigate: bool = False


class MessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: StrictStr
    passage_id: StrictStr
    build_id: StrictStr


class DecideRequest(BaseModel):
    """Exactly what Decide receives: any input text, an instruction and labels."""
    model_config = ConfigDict(extra="forbid")
    dataset: StrictStr
    input: StrictStr
    instruction: Any
    labels: Any


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
        raise HTTPException(422, f"Select one of: {', '.join(DATASETS)}")
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
            types, relations = cfg["extraction"]["entities"], cfg["extraction"]["relations"]
            # Decide's pre-filled boxes: the instruction, and the labels for each (HEAD, TAIL) type pair.
            defaults[name] = {"title": cfg["title"], "label": LABELS.get(name, cfg["title"]),
                              "decide_model": cfg["models"]["decide"],
                              "instruction": cfg["relation_decision"]["instruction"],
                              "labels": {f"{h}|{t}": relationship_labels(cfg, h, t) for h in types for t in types},
                              "all_labels": {**{p: spec["description"] for p, spec in relations.items()}, **RESERVED_LABELS}}
            if cfg.get("feed"):
                titles = {row["id"]: row["title"] for row in read_manifest(cfg)}
                defaults[name]["feed"] = {role: titles[source] for role, source in cfg["feed"].items()}
    except Exception as exc:
        return failure(exc)
    return templates.TemplateResponse(request=request, name="index.html", context={"defaults": defaults})


@app.get("/api/overview")
def landing(dataset: str):
    try:
        return overview(config(dataset))
    except Exception as exc:
        return failure(exc)


@app.get("/api/feed")
def feed(dataset: str):
    try:
        return feed_messages(config(dataset))
    except Exception as exc:
        return failure(exc)


@app.post("/api/message")
def message(body: MessageRequest):
    try:
        return message_view(config(body.dataset), body.passage_id, body.build_id)
    except Exception as exc:
        return failure(exc, stale=True)


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
def decision(body: DecideRequest):
    """On-demand GLiNER2.5-Decide over whatever the page sends; nothing is written back."""
    with state.lock:
        try:
            return decide(state.provider(config(body.dataset)), body.input, body.instruction, body.labels)
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
