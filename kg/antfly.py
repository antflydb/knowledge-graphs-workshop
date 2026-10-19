"""Thin HTTP client for an Antfly v0.3 server (`antfly standalone`).

Deliberately raw JSON so the request shapes are visible. The official
`antfly-sdk` can replace this later.
"""

from __future__ import annotations

from urllib.parse import quote

import httpx


class AntflyError(RuntimeError):
    pass


class Antfly:
    def __init__(self, base_url: str = "http://127.0.0.1:8080", timeout: float = 300.0):
        self.http = httpx.Client(base_url=base_url, timeout=timeout)

    def _call(self, method: str, path: str, **kw) -> dict:
        r = self.http.request(method, path, **kw)
        if r.status_code >= 400:
            raise AntflyError(f"{method} {path} -> {r.status_code}: {r.text[:2000]}")
        return r.json() if r.content else {}

    # --- tables -------------------------------------------------------------

    def list_tables(self) -> list[str]:
        out = self._call("GET", "/db/v1/tables")
        items = out if isinstance(out, list) else out.get("tables", [])
        return [t["name"] if isinstance(t, dict) else t for t in items]

    def create_table(self, table: str, spec: dict) -> dict:
        return self._call("POST", f"/db/v1/tables/{table}", json=spec)

    def index_status(self, table: str, index: str) -> dict:
        """The index's runtime status; `readiness.complete` means every source has converged."""
        return self._call("GET", f"/db/v1/tables/{table}/indexes/{index}")["status"]

    def count(self, table: str) -> int:
        total = self.query(table, {"limit": 0})["responses"][0]["hits"]["total"]
        return total["value"] if isinstance(total, dict) else total

    # --- documents ----------------------------------------------------------

    def batch(self, table: str, inserts: dict[str, dict], sync_level: str = "full_index") -> dict:
        return self._call(
            "POST",
            f"/db/v1/tables/{table}/batch",
            json={"inserts": inserts, "sync_level": sync_level},
        )

    def get(self, table: str, key: str) -> dict | None:
        r = self.http.get(f"/db/v1/tables/{table}/documents/{quote(key, safe='')}")
        if r.status_code == 404:
            return None
        if r.status_code >= 400:
            raise AntflyError(f"GET {key} -> {r.status_code}: {r.text[:500]}")
        return r.json()

    def query(self, table: str, body: dict) -> dict:
        return self._call("POST", f"/db/v1/tables/{table}/query", json=body)

    def graph_metric(self, table: str, index: str, metric: str, top_k: int) -> list[tuple[str, float]]:
        """Top nodes by a metric Antfly maintains on the graph index (e.g. degree)."""
        body = {"limit": 0, "graph_metric": {"index": index, "metric": metric, "top_k": top_k}}
        result = self.query(table, body)["responses"][0]["graph_metric_results"][metric]
        return [(row["node"], row["score"]) for row in result["scores"]]

    def traverse(self, table: str, index: str, keys: list[str], **options) -> list[dict]:
        """Every path from documents in `table`, one result per reaching path.

        Parallel facts to the same neighbor stay separate; callers that want
        distinct nodes collect keys. Always undeduplicated: on rc.1, a deduplicated
        traversal from a resolver-promoted entity in the same table returns nothing.
        """
        body = {"limit": 0, "graph_queries": {"t": {"index": index, "traverse": {
            "start": {"keys": sorted(set(keys))}, **options, "include_paths": True, "deduplicate_nodes": False}}}}
        nodes = self.query(table, body)["responses"][0]["graph_results"]["t"]["nodes"]
        return [node for node in nodes if node.get("path_edges")]  # drop the start nodes themselves

    # --- inference ----------------------------------------------------------

    def extract_entities(
        self,
        model: str,
        texts: list[tuple[str, str]],
        labels: list[str],
        threshold: float,
    ) -> list[dict]:
        """Zero-shot NER. `texts` is [(id, content)]. Returns per-input dicts in order."""
        out = self._call(
            "POST",
            "/ai/v1/extract",
            json={
                "model": model,
                "inputs": [{"id": i, "content": c} for i, c in texts],
                "schema": {"entities": labels},
                "options": {"include_confidence": True, "include_spans": True, "threshold": threshold},
            },
        )
        return out["data"]


def hits(resp: dict) -> list[dict]:
    """Flatten a query response into a list of {_id, _score, **source}."""
    responses = resp.get("responses", [resp])
    out = []
    for r in responses:
        for h in r.get("hits", {}).get("hits", []):
            out.append({"_id": h.get("_id"), "_score": h.get("_score"), **(h.get("_source") or {})})
    return out
