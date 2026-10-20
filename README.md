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
| `antfly-prs` | Four Antfly pull requests and two release notes, timestamped (bundled) |
| `pizza-portland` | Menus and about pages for 11 Portland pizzerias (not bundled; see below) |

## Run

```sh
npm install -g @antfly/cli@next    # Antfly v0.3.0-rc.1
git clone https://github.com/antflydb/knowledge-graphs-workshop && cd knowledge-graphs-workshop
make up                            # then open http://127.0.0.1:8000; Ctrl-C stops everything
```

`make up` also needs [`uv`](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
Without npm, install Antfly with
`curl -fsSL https://releases.antfly.io/antfly/latest/install.sh | sh -s -- v0.3.0-rc.1`
(macOS arm64 or Linux), or unpack a binary from the
[v0.3.0-rc.1 release](https://github.com/antflydb/antfly/releases/tag/v0.3.0-rc.1). If an older
`antfly` comes first on your `PATH`, set `ANTFLY_BIN` to the rc.1 binary.

The first run pulls GLiNER2.5 and EmbeddingGemma 2 (about 2 GB) and a prebuilt database
(about 350 MB); after that it starts in seconds. `make up BUILD=1` builds the graph on
your machine instead of downloading it (about 8 minutes on an Apple-silicon Mac). The
Decide drawer needs one more model: stop `make up`, run `make decide` (1.8 GB), start again.

After editing a `datasets/*/config.yaml`, restart `make up` (or run `make build-all`
alongside it): changed datasets rebuild, the rest are reused. Server logs are in
`data/logs/`, and `make` lists the step-by-step targets. [datasets/README.md](datasets/README.md)
covers the CLI, inputs and generated state. Coding agents: see [AGENTS.md](AGENTS.md).
The text-to-graph tab is prebuilt from [`playground/`](playground/); rebuilding it
(`make playground`) is optional and needs pnpm.

## Your own documents

```sh
make new DATASET=my-docs                 # datasets/my-docs/config.yaml from a template
cp ~/notes/*.md datasets/my-docs/raw/     # Markdown (or plain text saved as .md)
make collect build DATASET=my-docs       # record the files, then build the graph
```

Reload the page and pick it from the dataset menu. Then tune `config.yaml` (the entity
types and relations GLiNER looks for are most of it) and `make build DATASET=my-docs`
again. Titles come from each file's front-matter `title:` or first `# ` heading; add
`timestamp` or `url` to a file's row in `metadata.jsonl` and they show up on its
passages.

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
  but "Tyrion" and "Tyrion Lannister" stay separate. The page lists nodes with similar
  names so you can see the split.
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
