"""Shared evidence graph and externally supplied vector schema."""

EDGES = ("from_source", "mentions", "in_passage", "refers_to", "subject", "object", "supported_by")


def _keyword():
    return {"type": "string", "x-antfly-field": {"type": "keyword"}}


def table_schema(dimensions):
    properties = {name: _keyword() for name in (
        "kind", "type", "source_id", "passage_id", "predicate", "subject_id", "object_id")}
    properties["entity_ids"] = {"type": "array", "items": _keyword()}
    properties["text"] = {"type": "string", "x-antfly-field": {"type": "text"}}
    return {"schema": {"default_type": "doc", "document_schemas": {"doc": {"schema": {
        "type": "object", "additionalProperties": True, "properties": properties}}}},
        "indexes": {"graph": {"type": "graph", "edge_types": [{"name": name} for name in EDGES]},
                    "vectors": {"type": "embeddings", "external": True,
                                "dimension": dimensions, "distance_metric": "cosine"}}}
