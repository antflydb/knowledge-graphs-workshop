# Dataset inputs and CLI

One self-contained version-2 configuration per dataset feeds `kg.dataset`:

| Slug | Configuration | Expected source inventory |
|---|---|---|
| `asoiaf` | [`asoiaf/config.yaml`](asoiaf/config.yaml) | 56 frozen Wikipedia prose files |
| `pizza-portland` | [`pizza-portland/config.yaml`](pizza-portland/config.yaml) | 21 menu/about files for 11 selected places |
| `apollo13` | [`apollo13/config.yaml`](apollo13/config.yaml) | `raw/air-to-ground.md` and `raw/review-board.md` only |

See the root [source inventory and rights](../README.md#source-origins-and-bundling)
for every pizza URL and the Wikipedia/Apollo origins and attribution.

## Commands

The only public pipeline is:

```text
python -m kg.dataset datasets/<slug>/config.yaml --track stable|latest COMMAND
```

Track defaults to `stable`. Preparation needs only base dependencies; stable
model operations need `uv run --extra workshop` (or `uv sync --extra workshop`).
Latest uses native models, never the Python model extra as a fallback.

| Command | Behavior |
|---|---|
| `collect [--refresh]` | Materialize fixed configured sources; validate existing snapshots unless refresh is explicit |
| `validate` | Check configuration, manifest, raw paths, bytes and hashes without inference |
| `prepare` | Derive exact source slices and overlapping whole-word passages without inference |
| `doctor` | Inspect selected binary/provider prerequisites; not proof of successful inference |
| `db` | Start the track's Antfly database |
| `build [--max-passages N]` | Extract, differentiate proposed relationships, embed and publish a checked build |
| `query TEXT [--mode text\|hybrid\|graph]` | Analyze a transient query graph and retrieve cited passages; default graph |
| `decide TEXT [--instruction TEXT] [--labels JSON] [--results 1\|2\|3]` | Independently compare the query and retrieved passage graphs; default one result |

Example, with `db` running in another terminal:

```sh
uv run python -m kg.dataset datasets/pizza-portland/config.yaml collect
uv run python -m kg.dataset datasets/pizza-portland/config.yaml validate
uv run python -m kg.dataset datasets/pizza-portland/config.yaml prepare
uv run --extra workshop python -m kg.dataset datasets/pizza-portland/config.yaml --track stable doctor
uv run python -m kg.dataset datasets/pizza-portland/config.yaml --track stable db
# In the other terminal:
uv run --extra workshop python -m kg.dataset datasets/pizza-portland/config.yaml --track stable build
uv run --extra workshop python -m kg.dataset datasets/pizza-portland/config.yaml --track stable query \
  'Which pizzas have a gluten-free option?'
uv run --extra workshop python -m kg.dataset datasets/pizza-portland/config.yaml --track stable decide \
  'Which pizzas have a gluten-free option?' --results 1
```

Other diagnostic questions are “Who is Tyrion Lannister's father?” and
“How did the crew adapt the carbon dioxide scrubbers?”. Diagnostics are probes,
not expected-label truth or a guarantee of successful entity/relation extraction.
There is no offline extraction substitute, hosted claim checker or dry-run build.

## Input and collection contract

All paths resolve relative to `config.yaml`. `corpus: metadata.jsonl` names a
manifest with exactly one row per UTF-8 `text/markdown` file beneath `raw/`.
Required fields are `id`, `path`, `title`, `url`, `media_type`, `sha256`,
`fetched_at`, `license` (string or null), `attribution` and `redistribution`
(`allowed` or `permission_not_verified`). Collector rows also identify
`source_kind`; available download URLs/hashes, revision/page identifiers,
modifications and section information remain provenance. Rows do not embed text.

Validation rejects duplicate IDs/paths, missing/empty files, unregistered raw
files, path escapes, invalid encodings and media/extension/hash mismatches.
Markdown is read exactly, without injecting titles or frontmatter. Prepared
passages retain absolute raw character offsets; each passage must equal
`raw[source_start:source_end]`. Extraction spans inside a passage use local Python
character offsets, not UTF-8 bytes.

`collect` reuses valid existing files and fetches missing files only from the
fixed inventory, respecting pacing, robots and access boundaries. A changed
snapshot/hash requires explicit `collect --refresh` and graph regeneration.
Collection failures must not delete valid old files, publish successful rows for
failed sources, bypass access controls or silently substitute sources.

ASoIaF freezes the existing article/revision list and prose-exclusion profile;
there is no new category discovery. Pizza HTML retains actual visible menu,
branding/location/contact/hours text, not scripts/navigation/cookie forms or
invented summaries. Only pizza source PDFs undergo text extraction; no-text
pages are recorded and an entirely no-text document fails without OCR. Apollo
uses publisher-supplied text; transcript turns preserve exact utterance-body
slices, MET and raw speaker tokens. Empty turns remain provenance, not model
calls. Unknown markers/roles remain inspectable, not repaired or named by guess.

ASoIaF and Apollo attributed raw files are bundled. Pizza raw text is local and
ignored because permission is unverified. Preserve existing `data/` and databases;
collection does not migrate or delete old corpora.

## Configuration boundaries

Version 2 uses `dataset`, `title`, `corpus`, `output`, `passages`, `extraction`,
`relation_decision`, `query_decision`, `models`, `revisions`, `dimensions`,
`retrieval`, `tracks` and `collection`. Use the three checked-in configs as the
complete examples; obsolete ownership/regex/hosted-checker schemas do not apply.

- Passage maxima: 550 UTF-8 bytes, 75 words, 10 overlap words; whole words and
  exact raw offsets are preserved. If extraction or pair classification exceeds
  compiled token capacity, build subdivides the entire rejected corpus window at
  whole words with bounded overlap, retaining all source words and recording
  `selection.capacity_splits` in the receipt. Re-extraction and classification of
  refined windows run in separate model phases. Duplicate occurrence windows are
  indexed once. A single indivisible over-capacity word fails explicitly.
  `prepare` reports initial windows; a full build can therefore contain more
  passages. No text is truncated; `--max-passages` still caps a partial build.
- `extraction.entities` lists types. `extraction.threshold` is 0.5.
  `extraction.relations` maps directed predicates to descriptions and permitted
  head/tail types. `no_relation` and `unclear` are reserved, not semantic predicates.
- `relation_decision.instruction` controls automatic pair differentiation.
  `query_decision` supplies optional experimental instruction/label descriptions
  and defaults to one result; it is not relationship construction.
- Model IDs and immutable revisions are explicit; embeddings have 768 dimensions.
  Retrieval defaults are 40 candidates, 100 graph candidates, 5 evidence passages
  and 1 reserved text passage. Require `0 <= text_reserve < evidence <= candidates`.
- Unknown keys, invalid IDs, geometry, endpoints, revisions, track selection and
  experimental label maps fail before inference/table creation. Experimental
  labels need at least two distinct nonempty names with string descriptions.

The corpus digest includes canonical manifest records and raw hashes. The build
signature includes extraction/schema, relationship instruction/context, pins,
provider/code/lock identities and dimensions. Query-only experimental instructions
or retrieval tuning do not force graph regeneration.

## Generated state and publication

Generated state lives at `data/builds/<dataset>/<track>/`. Databases are separate
by track: `data/antfly/stable/graph.aflite` and `data/antfly/latest/graph.aflite`.
Datasets share their track's server, not tables. Tables are fresh content-addressed
`kg_<dataset_with_underscores>_<track>_<hash[:16]>` names.

Build phases run base extraction, pair classification and embedding in order,
releasing resident models between phases. Caching retains call provenance and
actual relationship-call counts, including zero. Publication checks graph targets,
finite/dimension-correct vectors, embedding identity and indexed vector round-trip
before atomically replacing `build.json`. A failed build must preserve the previous
usable receipt; matching complete builds replay, conflicting tables are not dropped.
`--max-passages` always marks the build partial. Latest and builds containing
permission-unverified sources are not distributable.

Query modes select Antfly BM25, BM25/vector fusion or two-hop graph-scoped hybrid
retrieval with a text reserve. **Every mode** runs the same query analysis used
for corpus passages. Results return saved passage analysis and generic source
citations; the viewer must not re-extract retrieved text. Extraction,
classification, embedding, retrieval and total timings expose cache behavior;
a latency goal never permits skipping required work.

## Verification status

Collection and exact-offset preparation passed for all three datasets. A full
stable pizza graph passed indexed-vector and complete prepared-word coverage
checks. Full ASoIaF/Apollo builds and complete optional-panel/browser qualification
are **not completed**; the partial ASoIaF smoke is not full acceptance.
Latest remains unqualified: observed Antfly is v0.2.5, installed Zig is 0.16.0
rather than required 0.17.0, RAM is 8 GiB and `ANTFLY_LATEST_BIN` is unset.
The parent verification record belongs in the
[workshop status](../workshops/README.md#verification-status).
