# Core-dataset runtime tracks

The three [dataset configurations](../datasets/README.md) each contain both runtime
tracks. Use `python -m kg.dataset datasets/<slug>/config.yaml --track TRACK COMMAND`;
there is no separate workshop CLI or shared ASoIaF-only runtime configuration.

| | [Stable](stable/README.md) | [Latest](latest/README.md) |
|---|---|---|
| Database | Released Antfly v0.2.5 | Pinned unreleased main |
| Inference | Python GLiNER2 / Transformers / Torch | Native Antfly inference |
| Client environment | `uv run --extra workshop` for model operations | Base Python dependencies only |
| Database address / health | `127.0.0.1:8795` / `4295` | `127.0.0.1:8796` / `4296` |
| Database file | `data/antfly/stable/graph.aflite` | `data/antfly/latest/graph.aflite` |
| Binary selection | `ANTFLY_STABLE_BIN` or compatible installed native binary | Required `ANTFLY_LATEST_BIN`; v0.2 is rejected |
| Native inference | Not used | `http://127.0.0.1:8090` |
| Qualification | Consolidated runtime verification pending | **Unqualified, unreleased, non-distributable** |

Generated per-dataset state lives in `data/builds/<dataset>/<track>/`; source
inputs stay in `datasets/<dataset>/raw/` with one `metadata.jsonl`. Existing data,
cache and database files are preserved, never implicitly migrated or deleted.

## Exact model pins

Both tracks retain these artifacts, not mutable model aliases:

| Role | Model | Revision |
|---|---|---|
| Extraction | `fastino/gliner2.5-base-v1` | `72ac19b486cd4557424c8d61114e7530c243e9b0` |
| Embeddings | `google/embeddinggemma-2` | `914f7f89142e33e77833254d9c9b90c3cef7303b` |
| Classification | `fastino/GLiNER2.5-Decide` | `7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6` |

Latest targets source commit `c102cbeac5ae8911ffe6a13105731f381285bfc2`.
Configured pins and a version string are not runtime qualification or commit
attestation; retain the observed binary SHA and provider/model receipts.
Identical pins do not prove numerical parity across providers.

## Shared graph and Decide contract

The graph contains source, passage, entity surface-bucket, mention-occurrence and
assertion records, connected by `from_source`, `mentions`, `in_passage`,
`refers_to`, `subject`, `object` and `supported_by` structural edges. Source text
and exact offsets support every accepted semantic assertion.

Base extraction proposes ordered endpoint pairs. Decide differentiates each valid
typed pair among compatible semantic predicates plus `no_relation` and `unclear`.
Unresolved endpoints are never fabricated; rejected proposals and every decision
remain in passage analysis. Accepted assertions have status
`model_classified_not_verified`. A base proposal is not an accepted assertion,
and missing pairs cannot be recovered by Decide. Same-type normalized surfaces
are buckets, not disambiguated people/places. Neither embeddings nor metadata
supply identity merging or semantic edges.

Queries run the same analysis as passages and attach a transient graph without
index writes. Antfly handles text/vector ranking and graph traversal. The separate
optional `decide` command/viewer panel compares the query text/model graph and
passage text/model graph with provenance. Defaults are dataset-specific, not a
universal support/conflict checker. One to three independent comparisons retain
their exact inputs and label scores; there is no generated answer, aggregate truth
label, reranking or graph mutation. Full inputs must fit the 512-token/request
limits; overflow is visible, never silently truncated.

The [local viewer](../README.md#plain-querygraph-page) starts with focused
name-group relationships and one independent top text match. Selecting a
connection opens its supporting source excerpt; selecting another name recenters
the graph, with Back navigation. Diagnostics, runtime selection and the optional
Decide experiment live in one initially closed Developer section. Normal graph
inspection does not expose JSON. Use one worker and one active provider;
unavailable tracks remain visible errors, never another model implementation.

## Distribution

Read the [source origins and rights](../README.md#source-origins-and-bundling)
before sharing anything. Bundle ASoIaF with article/revision attribution,
CC BY-SA 4.0 and modification notices, and Apollo with NASA/formatter attribution.
Pizza raw text and containing graph builds remain local because permission is
unverified. Latest is non-distributable regardless of source rights. Upstream
model terms apply separately; download instructions do not grant weight-bundling
permission. A partial build is never the full-dataset workshop artifact.

## Verification status

Source collection and exact-offset preparation passed for all three datasets:
56 ASoIaF sources / 4,641 initial passages, 21 pizza sources / 268 initial
passages, and two Apollo sources / 5,302 initial passages. ASoIaF bodies match
the preserved corpus byte-for-byte. Both NASA raw and download hashes match the
approved snapshots, including transcript normalized-character bounds
`[209583,635356)`. The parser recognizes 4,335 primary headers and 111 inline
candidates, not the planning estimate of 109: punctuation-delimited `Yes.`,
`Hey,` and bracketed `[???]` tokens are preserved with unknown-role warnings.

The real stable 64-passage ASoIaF smoke passed indexed-vector round-trip:
379 mentions, 145 proposed ordered pairs, 92 actual Decide calls and 90
accepted model assertions. It is explicitly partial, not full qualification.
It was subsequently rebuilt with current code as
`12e7bcae1075020db818b63a0df04c3922191c49a592fd16a8579507718d5b20`.
Before the focused-page redesign, the actual `make web` default ASoIaF search
completed with five cited passages and an explicit partial-build warning:
42 nodes/61 edges, none omitted.
First-query time was 49.33 s: extraction 19.58 s, no relationship calls,
embedding 20.40 s and retrieval 9.20 s. This is not a latency-goal pass.
A subsequent five-request retrieval diagnostic reused the real cached extraction
and embedding, with zero model calls and no resident models. Its first hybrid
database request took 5,813 ms; the next four took 23, 7, 7 and 7 ms.
Complete retrieval, including source validation, took 5,834 ms initially and
35, 20, 20 and 19 ms afterward. This default query had no extracted entity
anchors, so these measurements do not qualify graph-traversal latency.
Actual browser repeats also showed variability: 3,083 ms, 858 ms and 21 ms
for retrieval. [INFERENCE] Index/page-cache warming or host memory pressure
may explain the slow initial/idle requests; the native internal cause is unproven.
The UI now presents rounded stage durations in a muted line at the page bottom.

The focused-page redesign was exercised against the existing ASoIaF partial and
pizza full builds, without recomputing graphs. The Tyrion question has no exact
indexed name group in that partial selection; the page shows the independent top
text match and explicit Explore suggestions instead of inventing an answer.
The real gluten-free pizza question offered ambiguous literal names; selecting
the dietary-option group retained the question and narrowed the graph to three
incoming `has_dietary_option` relationships without changing the top text match.
Selecting an edge opened its actual cited excerpt with highlighted mentions.
Recenter made zero model calls; Back restored question, selected edge and text.
Desktop and 390-pixel browser views showed one top text card, a closed Developer
section and no JSON outside it. Raw citations opened as `text/plain`.

The hidden Decide experiment succeeded after Back with edited two-label inputs;
its model label is not an accuracy claim. An oversized instruction failed visibly
at 2,057 tokens against the 512-token limit, preserving the complete 936-character
classifier input, 11,700-character instruction and prior successful outcome.
Stale navigation returned 409; nonentity focus, missing Apollo/stable and missing
ASoIaF/latest builds returned actionable 422 errors without runtime fallback.
The integrated deterministic suites passed 32 dataset tests and six Wikipedia
tests. These checks do not qualify full ASoIaF/Apollo coverage or extraction accuracy.

An actual Apollo diagnostic pair offered `suspected_cause_of`,
`determined_cause_of`, `mitigates`, `no_relation`, and `unclear`; Decide selected
`suspected_cause_of` (0.5213), retaining all base proposals and an unresolved
untyped endpoint. These are model predictions, not verified causal claims.

The full stable pizza graph passed: 21 sources, 360 indexed passages, 3,787
mentions, 536 base proposals, 527 pair decisions and 224 accepted model assertions.
All prepared source words and exact offsets were checked against indexed records;
the indexed-vector round-trip passed. It remains local/non-distributable.
An initial full pizza build exposed compiled-token overflow in Stark Street PDF
dot leaders; 46 recorded capacity refinements preserve every source word without
truncation. Full ASoIaF/Apollo model builds are **not completed**. The ASoIaF run
was stopped with its cache preserved; no full receipt is claimed. On the 8 GiB
host, concurrent model workloads caused approximately 11 GiB swap use; the pizza
rebuild took 4,595 seconds under contention. Serialize model work or use a larger
host. This is not latency or memory-admission qualification.

Observed execution prerequisites:

- `antfly --version`: `antfly 0.2.5 (zig runtime)`.
- `zig version`: `0.16.0`; `sysctl hw.memsize`: `8589934592` (8 GiB).
- `ANTFLY_LATEST_BIN` is not set; no selected pinned-main binary is available.
- A real pinned-base stable Python extraction on “Tyrion Lannister is the son of
  Tywin Lannister.” produced no entities and a `parent_of` proposal Tywin→Tyrion,
  score 0.9189. This does not prove endpoint resolution, Decide, indexed retrieval
  or a completed dataset build.

Latest remains **unqualified**. Its source requires Zig 0.17.0; the installed
0.16.0 and 8 GiB host are not qualified build/inference prerequisites. A stable
binary, fake native endpoint, configured source hash or cached stable result must
not be presented as native-main execution. Follow the latest guide on a properly
provisioned host and record actual receipts before changing that status.

Integrated regression commands passed (27 dataset boundary cases and six
Wikipedia parser/access cases), including capacity refinement, unknown speakers
and a reproduced/fixed off-site cached-URL access gap:

```sh
uv run python scripts/test_datasets.py
uv run python scripts/test_wikipedia.py
```

Passing boundary tests alone does not qualify a model runtime or full corpus.
The browser completed pizza graph search with five cited results, inspected an
entity and a directed semantic edge, and opened exact saved passage analysis and
raw source text. The projection showed 50 nodes/78 edges and explicitly reported
seven omitted nodes/12 omitted edges. A cached repeat took 839.83 ms (no model
calls); this is not evidence of fresh-query latency or the sub-500 ms goal.
Registered Apollo Markdown served as `text/plain` with `nosniff`; unknown/traversal
source IDs were rejected. Missing latest builds produced actionable HTTP 422
without fallback. Optional-panel overflow retained both independent failures,
edits, original submitted query and retrieval order. The compact serializer was
exercised directly with the real Decide model on the shortest Life of Pie result:
relevant 0.3973, not_relevant 0.3185, unclear 0.2842. Positive compact-format
API/browser qualification is **not completed**. The first uncached pizza graph
query took 137.35 seconds under contention (15.21 s extraction, 71.99 s relation
classification, 48.06 s embedding, 2.02 s retrieval); no fresh-query performance
goal is claimed.
