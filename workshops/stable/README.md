# Stable: Antfly v0.2.5 and Python inference

Stable identifies the released database target, not model truth or completed
acceptance. All three [datasets](../../datasets/README.md) use the same pipeline;
the examples below select ASoIaF. Run from the repository root.
**Only the full pizza graph is runtime-verified; full ASoIaF/Apollo builds and
complete optional-panel/browser qualification are not completed.**

## Install and prepare

Install Antfly **v0.2.5**, Python **3.12** and `uv`:

```sh
uv sync --python 3.12 --extra workshop
uv run python -m kg.dataset datasets/asoiaf/config.yaml collect
uv run python -m kg.dataset datasets/asoiaf/config.yaml validate
uv run python -m kg.dataset datasets/asoiaf/config.yaml prepare
uv run --extra workshop python -m kg.dataset datasets/asoiaf/config.yaml --track stable doctor
```

Set `ANTFLY_STABLE_BIN=/absolute/path/to/antfly` to select a native executable
explicitly. `doctor` checks the actual binary version and records its SHA256;
it does not prove successful model execution.

The optional `workshop` extra is required for stable build/query/decide and a
stable-capable viewer; source preparation needs only base dependencies. Missing
models must give the install command, not use alternative inference. Keep the
pinned environment/lockfile: plain `gliner2`, not `gliner2[local]`, avoids a
Transformers constraint conflict with EmbeddingGemma 2. The shared
[model table](../README.md#exact-model-pins) records all three immutable revisions.

Public Hugging Face checkpoints download on first use. Normal `HF_TOKEN` and
`HF_HOME` settings apply. CPU/F32 is the default; `WORKSHOP_DEVICE=mps` or `cuda`
requests that device strictly, without silently falling back. One model resident
at a time does not bound transient weight-loading peaks. This 8 GiB host is not
a full-build memory qualification; use an admitted larger host if needed.

## Start the database, then build

Leave this running in a separate terminal:

```sh
uv run python -m kg.dataset datasets/asoiaf/config.yaml --track stable db
```

The database is `data/antfly/stable/graph.aflite`, listening at
`127.0.0.1:8795` with health port `4295`. Other stable datasets use the same
server but distinct content-addressed tables. Do not reuse a latest database.

In the working terminal:

```sh
# Optional bounded iteration; this is explicitly partial:
uv run --extra workshop python -m kg.dataset datasets/asoiaf/config.yaml --track stable build --max-passages 64

# Full selected dataset; no passage limit:
uv run --extra workshop python -m kg.dataset datasets/asoiaf/config.yaml --track stable build
```

Build runs real base extraction, real Decide pair classification and real
EmbeddingGemma 2 passage embeddings in separate phases. Unresolved/ambiguous
endpoints stay unresolved; a no-proposal passage makes zero relationship Decide
calls. Complete analysis retains base proposals, exact decision candidates/inputs
and scores, including rejections. Index/vector checks must pass before atomically
publishing `data/builds/asoiaf/stable/build.json`. Failed publication preserves
the prior receipt; tables are not dropped/recreated. Model-call caches preserve
provenance rather than relabeling simulated results as successful execution.

## Query and optional comparison

```sh
uv run --extra workshop python -m kg.dataset datasets/asoiaf/config.yaml --track stable query \
  "Who is Tyrion Lannister's father?" --mode text
uv run --extra workshop python -m kg.dataset datasets/asoiaf/config.yaml --track stable query \
  "Who is Tyrion Lannister's father?" --mode hybrid
uv run --extra workshop python -m kg.dataset datasets/asoiaf/config.yaml --track stable query \
  "Who is Tyrion Lannister's father?" --mode graph
uv run --extra workshop python -m kg.dataset datasets/asoiaf/config.yaml --track stable decide \
  "Who is Tyrion Lannister's father?" --results 1
```

Every query mode performs the shared extraction/pair-classification analysis.
`text` uses Antfly BM25; `hybrid` fuses BM25 and vectors; default `graph` uses
matching surface buckets and two-hop graph-scoped hybrid retrieval with a text
reserve. Queries never add indexed records. Review exact citations, offsets,
continuity and saved passage analysis, not just retrieval scores.

The optional `decide` comparison is separate from automatic relation decisions.
ASoIaF/pizza defaults use `relevant`, `not_relevant`, `unclear`; Apollo uses
`supports`, `contradicts`, `not_comparable`, `unclear`. Change the instruction
with `--instruction TEXT`, labels with `--labels JSON`, and select up to three
independent passages with `--results 1|2|3`. Input includes complete query and
passage texts, predicted entity/triple graphs and source context. The combined
schema/instruction/input must fit the 512-token/request limit; overflow is an
error, never truncation. Labels/scores are model judgments, not verified answers.

For pizza or Apollo, replace the config path and use respectively
“Which pizzas have a gluten-free option?” or
“How did the crew adapt the carbon dioxide scrubbers?”. Pizza collection is
required locally; its text and containing builds are not cleared for sharing.

## Inspect the actual page

```sh
uv run --extra workshop uvicorn kg.web.app:app --host 127.0.0.1 --port 8000 --workers 1
```

Open the local address (choose another free port if occupied). Search, inspect
query and stored passage analysis, click graph nodes/directional edges, open raw
Markdown citations and exercise the expandable Decide panel. It uses the last
successful query, not unsubmitted search-box edits. Each of up to three serial
comparisons retains its own success/error and editable inputs; overflow remains
visible. The graph is bounded to 50 nodes/100 edges with omitted counts.

All assertions are `model_classified_not_verified`; type+surface buckets are not
identity resolution. Do not infer semantic facts from titles/categories, source
inventory or role metadata. Read [source rights](../../README.md#source-origins-and-bundling)
and [current verification status](../README.md#verification-status) before sharing.
Neither a partial smoke nor unit tests establish full-corpus or model-accuracy
acceptance. See the linked status for exercised paths and unqualified paths;
do not describe the stable track as fully qualified.
