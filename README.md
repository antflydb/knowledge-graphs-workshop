# Knowledge graphs on Antfly

Turn a pile of plain documents into a searchable knowledge graph with no other
metadata. We write passages to Antfly v0.3; Antfly runs GLiNER2.5 on each one at
ingest, embeds it, resolves the entities it mentions and links them into a graph.
A small web page explores it, and a second tab, **Text to graph**, extracts a graph
live from whatever sentences you type.

| Dataset | Sources |
|---|---|
| `asoiaf` | 56 frozen Wikipedia articles (CC BY-SA 4.0, bundled) |
| `apollo13` | NASA air-to-ground transcript and Review Board report (bundled) |
| `pizza-portland` | Menus and about pages for 11 Portland pizzerias (not bundled; see below) |

## Run

Needs `uv`, Python 3.12 and Antfly **v0.3.0-rc.1**. Set `ANTFLY_BIN` if the `antfly`
on your `PATH` is older.

```sh
uv sync
antfly inference pull "fastino/gliner2.5-base-v1:safetensors@72ac19b486cd4557424c8d61114e7530c243e9b0"
antfly inference pull "google/embeddinggemma-2:safetensors@914f7f89142e33e77833254d9c9b90c3cef7303b" --tasks embed
antfly inference pull "fastino/GLiNER2.5-Decide:safetensors@7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6" --tasks extract

make inference    # terminal 1: native inference on :8090
make db           # terminal 2: Antfly on :8796
make build        # ~5 min for ASoIaF on an Apple-silicon Mac; DATASET=apollo13 for the other corpus
make web          # http://127.0.0.1:8000 (graph search and text-to-graph tabs)
```

`make` lists every target. [datasets/README.md](datasets/README.md) covers the CLI,
inputs and generated state. The text-to-graph tab is prebuilt from
[`playground/`](playground/); rebuilding it (`make playground`) is optional and needs pnpm.

## How it's stored

One table per build, and the only documents we write are passages (text plus exact
source offsets). Everything else is declared on the table and done by Antfly
([`kg/schema.py`](kg/schema.py)):

- **Embeddings:** EmbeddingGemma 2 over `text`, for hybrid BM25 + vector search.
- **Extraction:** an `extractor` enrichment runs GLiNER2.5 on each passage with the
  dataset's entity types and relations.
- **Graph:** each extracted relation becomes a fact edge owned by its passage, so
  every edge cites its source.
- **Entities:** a resolver promotes every mention into an `entity/<slug>` document in
  the same table, with passage → `mentions` → entity edges.

ASoIaF: 4,641 passages, 2,891 entities and 33,385 edges.

## What the graph means

- **Connections:** two entities are connected when passages relate them. The page
  shows one connection per pair, labelled by the majority predicate across passages
  (each passage votes), with every passage as evidence.
- **Reliability:** strongly attested pairs are almost always real relationships, and
  their majority label is usually right for family, marriage and places. GLiNER often
  mislabels close associates as family ("Jorah sibling of Daenerys"); the votes and
  evidence make that visible.
- **Identity:** an entity is its slugged surface form. "Tyrion's" merges into "Tyrion",
  but "Tyrion" and "Tyrion Lannister" stay separate. The page lists every matching
  name group so you can see the split.
- **Decide:** "Ask Decide" in the evidence panel re-classifies one relationship with
  GLiNER2.5-Decide on demand. Nothing is written back.

## Sources and licensing

Code is MIT ([LICENSE](LICENSE)). Data, quotations, model weights and Antfly are not
covered; see [NOTICE](NOTICE). Every file's origin, hash and rights are in its
dataset's `metadata.jsonl`.

- **Wikipedia** prose keeps its article/revision attribution under CC BY-SA 4.0.
- **NASA** texts are US Government works; transcript formatting is credited to Heiko
  Kueffen.
- **Pizza** sources have no verified redistribution permission. Run
  `make collect DATASET=pizza-portland` locally; never commit its raw text.

## rc.1 caveats

- **Default storage engine:** `make db` uses it because, on the Lite engine,
  artifact-backed graph and embedding workers fail permanently under bulk load.
- **No model pin:** rc.1 rejects `embedder.model_identity` at table creation, so the
  embedding model isn't pinned.
- **Traversal:** deduplicated traversal from a resolver-promoted entity returns
  nothing, so the client always traverses with paths, undeduplicated.
