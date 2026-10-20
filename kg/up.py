"""`make up`: everything the dashboard needs, in one terminal; Ctrl-C stops it all.

Pulls GLiNER and EmbeddingGemma (a quick check once cached; DECIDE=1 adds the optional
Decide model, or `make decide`), downloads the prebuilt database
if there is none (BUILD=1 builds it here instead), starts native inference and Antfly
in the background (logs in data/logs/), rebuilds any dataset whose config or text
changed, then serves the dashboard. Servers already running on the configured ports
are reused.
"""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import httpx

from kg.dataset import binary_info, settings
from kg.dataset_config import MODEL_PINS, ROOT, dashboard_datasets, load_dataset

PULL_TASKS = {"extract": [], "embed": ["--tasks", "embed"], "decide": ["--tasks", "extract"]}


def pull(*roles):
    """Pull pinned models; already-cached ones are only verified."""
    binary = binary_info(load_dataset(next(iter(dashboard_datasets().values()))).data)["path"]
    for role in roles:
        name, revision = MODEL_PINS[role]
        run(binary, "inference", "pull", f"{name}:safetensors@{revision}", *PULL_TASKS[role])


def run(*args):
    subprocess.run(args, check=True, cwd=ROOT)


def reachable(url, timeout=3):
    try:
        return httpx.get(url, timeout=timeout).status_code < 500
    except httpx.HTTPError:
        return False


def wait(url, name, seconds=180):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if reachable(url, timeout=5):
            return
        time.sleep(1)
    raise SystemExit(f"{name} did not come up at {url}; see data/logs/")


def main():
    # Closing the terminal or `kill` should stop the servers too, like Ctrl-C.
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: sys.exit(0))
    datasets = dashboard_datasets()
    config = str(next(iter(datasets.values())))
    runtime = settings(load_dataset(config).data)
    # Building and searching need only these two; Decide is for the dashboard's Decide drawer.
    # rc.1: pulling (even re-verifying) under a running inference server breaks it, so a
    # server that is already up keeps the models it has.
    if reachable(runtime["inference_url"] + "/ai/v1/models"):
        print("Inference is already running; using its models.", flush=True)
    else:
        pull("extract", "embed", *(["decide"] if os.environ.get("DECIDE") else []))
    if not Path(runtime["data_dir"]).exists() and not os.environ.get("BUILD"):
        run("make", "fetch-db")
    logs = ROOT / "data" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    children = []
    try:
        for command in ("inference", "db"):
            with open(logs / f"{command}.log", "a") as log:
                # Own process group, so stopping it also stops the antfly process it starts.
                children.append(subprocess.Popen([sys.executable, "-m", "kg.dataset", config, command], cwd=ROOT,
                                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        wait(runtime["inference_url"] + "/ai/v1/models", "Inference")
        wait("http://" + runtime["addr"] + "/db/v1/tables", "Antfly")
        # A current build is a quick check; a changed config or text rebuilds that dataset.
        for path in datasets.values():
            run(sys.executable, "-m", "kg.dataset", str(path), "collect")
            run(sys.executable, "-m", "kg.dataset", str(path), "build")
        port = os.environ.get("WEB_PORT") or "8000"
        print(f"\nDashboard: http://127.0.0.1:{port}  (Ctrl-C stops everything)\n", flush=True)
        run(sys.executable, "-m", "uvicorn", "kg.web.app:app", "--host", "127.0.0.1", "--port", port, "--workers", "1")
    except KeyboardInterrupt:
        pass
    finally:
        for child in children:
            if child.poll() is None:  # one that already exited (e.g. its port was taken) has nothing to stop
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    child.terminate()
        for child in children:
            child.wait(timeout=30)


if __name__ == "__main__":
    main()
