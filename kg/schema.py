"""One Antfly v0.3 table: Antfly extracts, embeds, resolves and links; we only write passages."""

RELATIONS_ARTIFACT = "relations_v1"
LINKS_ARTIFACT = "links_v1"
MENTIONS = "mentions"
LINKS_METRIC = "links"


def _keyword():
    return {"type": "string", "x-antfly-field": {"type": "keyword"}}


def passage_table(cfg, table):
    """Plain passage documents; every index derives from `text`.

    - `vectors`: Antfly embeds passages (and queries) with EmbeddingGemma 2. Entity
      documents have no `text`, so coverage is `partial`. rc.1 rejects
      `embedder.model_identity` at table creation, so the model is not pinned.
    - `graph`: an extractor enrichment runs GLiNER2.5 on each passage at ingest.
      Each extracted relation becomes one fact edge owned by its passage
      (`edge_id` = passage key). The resolver promotes every mention into an
      `entity/<slug>` document in this same table, with passage -> mentions ->
      entity edges. Same-table entities let one traversal return every fact.
    """
    ex = cfg["extraction"]
    inference = {"provider": "antfly", "api_url": cfg["antfly"]["inference_url"]}
    extractor = {"type": "extractor", "config": {**inference, "model": cfg["models"]["extract"],
        "schema": {"entities": ex["entities"], "relations": [{"type": r} for r in ex["relations"]]},
        "options": {"include_confidence": True, "include_spans": True, "long_document": {"mode": "window"}}}}
    properties = {name: _keyword() for name in ("kind", "source_id", "passage_id")}
    properties["text"] = {"type": "string", "x-antfly-field": {"type": "text"}}
    relations = {"artifact": RELATIONS_ARTIFACT, "path": "$.relations[*]", "format": "extraction_relation",
                 "mention_edge_type": MENTIONS,
                 "edge": {"type": "{{ _item.type }}", "weight": "{{ _item.score }}", "edge_id": "{{ _doc.key }}"}}
    edge_types = [MENTIONS, *ex["relations"]]
    if cfg.get("explicit_links"):
        # A second source: each passage's `links` field (explicit references in its
        # text) projects passage -> type -> target edges next to the extracted ones.
        edge_types += sorted({link["type"] for link in cfg["explicit_links"]})
        sources = {"enrichments": [
                       {"name": RELATIONS_ARTIFACT, "kind": "asset", "field": "text",
                        "content_type": "application/json", "producer": extractor},
                       {"name": LINKS_ARTIFACT, "kind": "asset", "field": "links", "content_type": "application/json"}],
                   "sources": [relations, {"artifact": LINKS_ARTIFACT, "path": "$", "nodes": {"target": "{{ _item.target }}"},
                                           "edge": {"type": "{{ _item.type }}"}}]}
    else:
        sources = {"artifact": {"name": RELATIONS_ARTIFACT, "kind": "asset", "content_type": "application/json",
                                "source": {"type": "field", "value": "text"}, "producer": extractor},
                   "source": relations}
    return {"num_shards": 1,
            "schema": {"default_type": "doc", "document_schemas": {"doc": {"schema": {
                "type": "object", "additionalProperties": True, "properties": properties}}}},
            "indexes": {
                "vectors": {"type": "embeddings", "field": "text", "dimension": cfg["dimensions"],
                            "distance_metric": "cosine", "coverage_policy": "partial",
                            "embedder": {**inference, "model": cfg["models"]["embed"]}},
                "graph": {"type": "graph", **sources,
                          "edge_types": [{"name": name} for name in edge_types],
                          # Antfly maintains each node's relationship count (not passage mentions),
                          # so the dashboard can rank the most-linked entities straight from the graph.
                          "metrics": {LINKS_METRIC: {"kind": "degree",
                                                     "edge_filter": {"types": [t for t in edge_types if t != MENTIONS]}}},
                          "resolvers": [{"name": "entities", "table": table,
                                         "source_artifact": RELATIONS_ARTIFACT,
                                         "resolution_artifact": "entities_resolution_v1",
                                         "key_template": "entity/{{ slug _entity.text }}",
                                         "min_confidence": ex["threshold"], "config_generation": 1}]},
            }}
