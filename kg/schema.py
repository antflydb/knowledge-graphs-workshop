"""One Antfly v0.3 table: Antfly extracts, embeds, resolves and links; we only write passages."""

RELATIONS_ARTIFACT = "relations_v1"
MENTIONS = "mentions"


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
    return {"num_shards": 1,
            "schema": {"default_type": "doc", "document_schemas": {"doc": {"schema": {
                "type": "object", "additionalProperties": True, "properties": properties}}}},
            "indexes": {
                "vectors": {"type": "embeddings", "field": "text", "dimension": cfg["dimensions"],
                            "distance_metric": "cosine", "coverage_policy": "partial",
                            "embedder": {**inference, "model": cfg["models"]["embed"]}},
                "graph": {"type": "graph",
                          "artifact": {"name": RELATIONS_ARTIFACT, "kind": "asset", "content_type": "application/json",
                                       "source": {"type": "field", "value": "text"}, "producer": extractor},
                          "source": {"artifact": RELATIONS_ARTIFACT, "path": "$.relations[*]",
                                     "format": "extraction_relation", "mention_edge_type": MENTIONS,
                                     "edge": {"type": "{{ _item.type }}", "weight": "{{ _item.score }}",
                                              "edge_id": "{{ _doc.key }}"}},
                          "edge_types": [{"name": MENTIONS}] + [{"name": r} for r in ex["relations"]],
                          "resolvers": [{"name": "entities", "table": table,
                                         "source_artifact": RELATIONS_ARTIFACT,
                                         "resolution_artifact": "entities_resolution_v1",
                                         "key_template": "entity/{{ slug _entity.text }}",
                                         "min_confidence": ex["threshold"], "config_generation": 1}]},
            }}
