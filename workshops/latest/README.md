# Latest: pinned main and native Antfly inference

> **UNQUALIFIED, UNRELEASED — NOT FOR DISTRIBUTION.**
> This targets source commit `c102cbeac5ae8911ffe6a13105731f381285bfc2`, not an
> available released package. No native-main model/build/retrieval pass is claimed.

The three [dataset configs](../../datasets/README.md) share stable's pipeline and
model revisions, but latest runs extraction, embeddings and Decide **only through
native Antfly inference**. Python is the client/orchestrator; it must not silently
run Torch/Transformers as a fallback. Identical pins do not prove output parity.
Run dataset commands from the repository root.

## Observed prerequisites and blocker

The current host reports Antfly `0.2.5 (zig runtime)`, Zig `0.16.0` and
`8589934592` bytes RAM (8 GiB); `ANTFLY_LATEST_BIN` is unset. The pinned source
requires Zig **0.17.0**. These observations do not qualify native main.
A real separately selected main binary and an admitted build/inference host are
missing prerequisites. Do not substitute stable, a mock endpoint or cached Python
results. See [verification status](../README.md#verification-status).

## 1. Build/select the native snapshot

Use a separate checkout/worktree; never reset a checkout containing user work.
From an existing Antfly source checkout, for example:

```sh
git worktree add --detach ../antfly-workshop-main c102cbeac5ae8911ffe6a13105731f381285bfc2
```

With the required Zig 0.17.0 toolchain installed, in that new worktree:

```sh
make build
./antfly --version
```

The root build copies `zig/zig-out/bin/antfly` to `./antfly`. Respect resource
admission checks; do not bypass them. The pinned tree's Darwin compilation
admission claims include up to 28 GiB for storage and 20 GiB for inference,
not an 8 GiB build recipe. Use a host that admits the selected compilation units.

Back in this repository:

```sh
export ANTFLY_LATEST_BIN=/absolute/path/to/antfly-workshop-main/antfly
uv sync --python 3.12
uv run python -m kg.dataset datasets/asoiaf/config.yaml --track latest doctor
uv run python -m kg.dataset datasets/asoiaf/config.yaml --track latest collect
uv run python -m kg.dataset datasets/asoiaf/config.yaml --track latest validate
uv run python -m kg.dataset datasets/asoiaf/config.yaml --track latest prepare
```

`ANTFLY_LATEST_BIN` is mandatory; v0.2 binaries are rejected. `doctor` retains the
actual binary version and SHA256 but a configured source pin is not commit
attestation or successful model execution. Keep separate databases:
`data/antfly/latest/graph.aflite` must not be opened with stable.

## 2. Install the exact native model artifacts

Use a dedicated native model directory, not a mixture of revisions or quants:

```sh
export WORKSHOP_NATIVE_MODELS="$HOME/.antfly/workshop-main-models"
"$ANTFLY_LATEST_BIN" inference pull \
  hf:fastino/gliner2.5-base-v1:safetensors@72ac19b486cd4557424c8d61114e7530c243e9b0 \
  --tasks extract --models-dir "$WORKSHOP_NATIVE_MODELS"
"$ANTFLY_LATEST_BIN" inference pull \
  hf:google/embeddinggemma-2:safetensors@914f7f89142e33e77833254d9c9b90c3cef7303b \
  --tasks embed --models-dir "$WORKSHOP_NATIVE_MODELS"
"$ANTFLY_LATEST_BIN" inference pull \
  hf:fastino/GLiNER2.5-Decide:safetensors@7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6 \
  --tasks extract --models-dir "$WORKSHOP_NATIVE_MODELS"
```

These intended pinned artifacts still require actual backend/resource
qualification. A model listing is not proof of execution. Never replace a rejected
checkpoint with a similarly named one or mutate metadata to bypass admission.

## 3. Start native inference and the separate database

Inference terminal:

```sh
"$ANTFLY_LATEST_BIN" inference run --host 127.0.0.1 --port 8090 \
  --models-dir "$WORKSHOP_NATIVE_MODELS" --max-loaded-models 1 \
  --host-budget-mb 3072 --backend-budget-mb 6144 --combined-budget-mb 9216
```

These explicit budgets come from the pinned Decide CUDA example, which reports
approximately 1.95 GB host and 4.38 GB backend load peaks; its default 1.5 GB host
budget rejects that checkpoint. **This is not a measured CPU/Metal or 8 GiB
configuration.** Qualify the chosen host/backend rather than disabling admission.
One resident model does not bound loading peaks or concurrent request memory.

Database terminal, in this repository:

```sh
uv run python -m kg.dataset datasets/asoiaf/config.yaml --track latest db
```

Database address: `127.0.0.1:8796`; health: `4296`; database:
`data/antfly/latest/graph.aflite`. Inference is separately configured at
`http://127.0.0.1:8090`. Per-dataset generated state is
`data/builds/<dataset>/latest/`.

## 4. Exercise the real native path

```sh
# Optional bounded iteration, explicitly partial:
uv run python -m kg.dataset datasets/asoiaf/config.yaml --track latest build --max-passages 64

# Actual full dataset:
uv run python -m kg.dataset datasets/asoiaf/config.yaml --track latest build
uv run python -m kg.dataset datasets/asoiaf/config.yaml --track latest query \
  "Who is Tyrion Lannister's father?" --mode graph
uv run python -m kg.dataset datasets/asoiaf/config.yaml --track latest decide \
  "Who is Tyrion Lannister's father?" --results 1
```

Repeat with `pizza-portland` / “Which pizzas have a gluten-free option?” and
`apollo13` / “How did the crew adapt the carbon dioxide scrubbers?”. Full builds
have no passage limit. Failures preserve the prior usable manifest; partial runs
are never full-corpus qualification. Pizza raw text and containing graphs remain
local because redistribution permission is unverified.

The client uses native `/ai/v1/extract` for base entity/relation extraction and
ordinary single-label classification for Decide, not a typed `decisions` head.
`/ai/v1/embed` computes the vectors. Antfly also performs indexing, BM25/vector
fusion and graph traversal. All query modes run the shared transient-query
analysis, with no index writes. The optional `decide` comparison includes complete
query/passage texts and predicted graphs plus source context; it is separate from
automatic predicate differentiation and does not generate a verified answer.

Embedding responses must carry exact `model_identity`; caches/builds retain it
and query vectors pin it. Extraction revisions are deployment requirements, not
digest attestations supplied by the extraction response. Never confuse configured
pins with observed artifact verification.

## Limits and viewer

- Base single-window native rows bound bytes/words/encoded sequence length; the
  configured 550-byte/75-word windows do not by themselves prove every schema fits.
  Schema-prefix tokens count. Preserve actual qualification/overflow failures.
- Decide's 512-token/request budget includes instruction, label descriptions and
  complete serialized text/graphs. No hidden truncation or graph-member omission.
- Native UTF-8 byte spans are checked and converted to Python character offsets;
  malformed boundaries are errors, not silently repaired spans.
- EmbeddingGemma 2's F32 masked-mean retrieval recipe is a separate vector space
  from original EmbeddingGemma; model identity mismatches must fail.

For a native-only page, `uv run uvicorn kg.web.app:app --host 127.0.0.1 --port 8000
--workers 1` needs only base dependencies; install `--extra workshop` if the same
viewer will also serve stable. Use a free port rather than stopping an unrelated
service. The [plain viewer contract](../../README.md#plain-querygraph-page) includes
saved analysis/citations, bounded directional SVG, editable per-result comparisons
and visible 422 overflow / 409 stale-build errors. Unavailable latest must show its
actual actionable failure, never return stable results as if native.

## Release/qualification gate

Keep latest non-distributable until an actual released artifact is available and
its real extraction, pair classification, embedding, full builds, indexed vector
round-trip, cited graph retrieval and optional comparisons pass on all selected
datasets. Record binary SHA, provider/model provenance, counts, timing/cache
behavior and browser observations. Update the source-snapshot banner only after
that evidence exists. Unit tests, version strings and partial/stable receipts are
not substitutes. [Source rights and graph caveats](../../README.md) still apply.
