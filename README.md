# Core datasets and interactive graph search

Three datasets share one model-backed generation/query pipeline and one plain
HTML graph/search page:

| Dataset | Configuration | Corpus and rights |
|---|---|---|
| ASoIaF / Game of Thrones | [`datasets/asoiaf/config.yaml`](datasets/asoiaf/config.yaml) | 56 frozen Wikipedia articles; attributed CC BY-SA 4.0 prose, mixed book/TV continuity |
| Portland pizza | [`datasets/pizza-portland/config.yaml`](datasets/pizza-portland/config.yaml) | 21 menu/about files covering 11 places; redistribution permission **not verified** |
| Apollo 13 | [`datasets/apollo13/config.yaml`](datasets/apollo13/config.yaml) | Bounded air-to-ground transcript and NASA-supplied Review Board main-report text |

Each dataset has `config.yaml`, a per-file `metadata.jsonl` and Markdown-only
`raw/`. Source metadata is provenance, not extracted facts. See the
[dataset contract](datasets/README.md) and [runtime tracks](workshops/README.md).

## Run

Run from the repository root with Python 3.12, `uv` and Antfly v0.2.5 installed:

```sh
uv sync --python 3.12 --extra workshop
uv run python -m kg.dataset datasets/asoiaf/config.yaml collect
uv run python -m kg.dataset datasets/asoiaf/config.yaml validate
uv run python -m kg.dataset datasets/asoiaf/config.yaml prepare
uv run --extra workshop python -m kg.dataset datasets/asoiaf/config.yaml --track stable doctor

# Separate terminal; leave this database running:
uv run python -m kg.dataset datasets/asoiaf/config.yaml --track stable db

uv run --extra workshop python -m kg.dataset datasets/asoiaf/config.yaml --track stable build
uv run --extra workshop python -m kg.dataset datasets/asoiaf/config.yaml --track stable query \
  "Who is Tyrion Lannister's father?"
uv run --extra workshop python -m kg.dataset datasets/asoiaf/config.yaml --track stable decide \
  "Who is Tyrion Lannister's father?" --results 1
```

Replace `asoiaf` with `pizza-portland` or `apollo13`. Pizza requires collection
on each fresh checkout because its raw text is not bundled. Stable is the default
track; graph is the default query mode. `build --max-passages 64` is a **partial**
iteration, never a full-dataset qualification. All commands and configuration
boundaries are described in [datasets/README.md](datasets/README.md).

The [stable guide](workshops/stable/README.md) uses Python inference and released
Antfly v0.2.5. The [latest guide](workshops/latest/README.md) requires an explicitly
selected pinned-main binary plus native inference; it is **unqualified,
unreleased and not distributable**. There is no Python fallback on latest.

## Plain query/graph page

With the selected track's database and a completed build available:

```sh
uv run --extra workshop uvicorn kg.web.app:app --host 127.0.0.1 --port 8000 --workers 1
```

Open `http://127.0.0.1:8000`. If the port is occupied, select a free port rather
than stopping another service. The local trusted tool has dataset/track selectors,
a search field, actual timings, query analysis, ranked citations and saved
passage analysis. Its deterministic SVG distinguishes mention links and accepted
directional predicates; click nodes/edges to inspect evidence. The projection is
bounded to 50 nodes/100 edges and reports omitted counts. Registered source
Markdown opens as plain text. It is not an authentication/multi-user product.

The optional Decide panel separately compares the last successful query with
one to three retrieved passages, serially and independently. Instructions and
JSON label descriptions are editable. Each outcome retains the exact input,
labels, scores and citation, including complete query/passage model graphs.
Graphs use compact plain text containing every entity's surface/type and every
accepted directed triple, with explicit “model predictions, not verified” labels.
It does not change retrieval order, write to either graph or generate an answer.
Model scores are not calibrated truth. The combined request must fit the existing
512-token and request-size limits; an oversized request returns visible 422,
without hidden truncation. Shorten the instruction/labels or select a shorter
query/passage. Stale builds return 409; unavailable native inference stays visible
rather than switching providers.

## What the graph means

GLiNER 2.5 base proposes entities and ordered relationship pairs. Decide classifies
each resolvable typed pair against **all type-compatible configured predicates**
plus `no_relation` and `unclear`, not just the proposal's original predicate.
No proposal means no relationship classification call; there is no all-pairs or
co-occurrence fallback. Rejected and unresolved decisions remain inspectable.
Accepted assertions are `model_classified_not_verified` predictions, not facts.

Corpus passages and queries use the same extraction/classification pipeline.
Queries form a transient graph and never add indexed records. Graph retrieval
attaches matching type+normalized-surface buckets and uses Antfly traversal and
hybrid ranking with a text reserve. Unmatched entities remain visible; no bucket
match falls back to normal hybrid retrieval. Empty evidence stays empty.

The `surface_bucket_not_disambiguated` warning matters: normalization groups
same-type/same-surface mentions; it does not resolve homonyms, aliases, surnames,
pronouns or book/TV continuity. Source titles, Wikipedia categories/links,
restaurant inventory addresses and transcript roles are not automatically
semantic edges. Apollo hypotheses must not become Review Board determinations
merely because a report quotes them. No absence-based dietary/allergen safety
claim is justified by a menu.

## Source origins and bundling

Project code and project-authored documentation use the [MIT license](LICENSE).
That grant does **not** cover third-party datasets, quotations, model weights,
dependencies or Antfly. See [NOTICE](NOTICE) for licensing scope and attribution.

Exact file hashes, article/revision URLs, fetch times, attribution, modifications
and rights are recorded in each dataset's `metadata.jsonl`. Preserve that metadata
with redistributed material. Existing `data/` artifacts are not migrated or deleted.

### Wikipedia

ASoIaF prose originates at [English Wikipedia](https://en.wikipedia.org/), using
the frozen 56-article/revision inventory recorded in
[`datasets/asoiaf/metadata.jsonl`](datasets/asoiaf/metadata.jsonl). The bundled
raw bodies preserve the original normalized article export byte-for-byte.
Collection does not discover new articles or expand categories. Categories,
headings, link targets, infoboxes and tables
are not graph facts. This is not a complete census or a single unified canon.

Bundled prose retains article/revision attribution, modification notices and
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), subject to
[Wikimedia's reuse terms](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content).
The article license is not blanket clearance for third-party media or quotations.

### Portland pizza: fixed source inventory

Only the following sources are selected. Each place has a menu and about/location
file, except Ken's single page, which serves both roles. Selected city locations
and source roles are provenance in the configuration, not injected model prose.

| Place | Menu source | About/location source |
|---|---|---|
| Apizza Scholls | https://www.apizzascholls.com/Menus/ | https://www.apizzascholls.com/ |
| Ken's Artisan Pizza | https://kensartisanpizza.com/pizza | Same page; one file |
| HOTLIPS Hawthorne | https://www.hotlipspizza.com/menu | https://www.hotlipspizza.com/hawthorne |
| Life of Pie | https://lifeofpiepizza.com/pdfs/Menu-Life_Of_Pie.pdf | https://lifeofpiepizza.com/ |
| Red Sauce Pizza | https://www.redsaucepizza.com/menu | https://www.redsaucepizza.com/ |
| Double Mountain Woodstock | https://doublemountainbrewery.com/wp-content/uploads/2026/10/WS_Menu_10_16_26.pdf | https://doublemountainbrewery.com/woodstock/ |
| The Blind Onion | https://www.blindonion.com/house.html | https://www.blindonion.com/index.html |
| Mississippi Pizza | https://mississippipizza.com/mississippi-pizza-menu/ | https://mississippipizza.com/about-mississippi-pizza/ |
| Stark Street Pizza | https://www.starkstreetpizza.com/stark-street-menu.pdf | https://www.starkstreetpizza.com/ |
| Pizza Thief NW Vaughn | https://pizzathief.com/menu/ | https://pizzathief.com/ |
| Amalfi's | https://www.amalfisrestaurant.com/pdf/dinner-menu.pdf | https://www.amalfisrestaurant.com/ |

**No selected pizza source has verified redistribution permission.** Commit only
its configuration and metadata; `datasets/pizza-portland/raw/` stays ignored and
local, as do generated builds under `data/`. Never force-add that text. An explicit
permission covers only its identified source, not every restaurant. PDF menus
are converted using available text in page order, not OCR; page-level extracted
character counts and no-text omissions are recorded. With pypdf 6.21.1, Life of
Pie's second page yields only a 38-character ordering footer, not menu coverage.
No full image-menu coverage is claimed. The dated Double Mountain URL is the selected snapshot,
not a pattern for silently substituting a newer menu. Failed/blocked sources
must remain failures, not synthetic summaries or alternate restaurants.

### Apollo 13

- Air-to-ground source: [curated NASA transcript text](https://apollojournals.org/alsj/a13/AS13_TEC.txt),
  with [transcription/formatting attribution to Heiko Kueffen](https://apollojournals.org/alsj/a13/a13trans.html)
  and [NASA NTRS record 20160014370](https://ntrs.nasa.gov/citations/20160014370)
  ([record API](https://ntrs.nasa.gov/api/citations/20160014370)).
- Review Board source: [NASA-supplied report fulltext](https://ntrs.nasa.gov/api/citations/19700076776/downloads/19700076776.txt),
  from [NTRS record 19700076776](https://ntrs.nasa.gov/citations/19700076776),
  identified as a US Government work with public use permitted.

These two attributed Markdown artifacts are bundled. The transcript retains the
continuous inclusive MET `055:50:05`–`094:33:21` interval, ending before the next
`094:33:31 LMP` marker; Windows-1252 is decoded and line endings normalized to LF.
The supplied wording, garbling and `C02` spelling are preserved. Citations use
actual MET/speaker and raw character offsets; `CC` means CapCom, not one named
person. Unknown roles are not repaired. The report includes the supplied front
matter and Chapters 1–5, not the listed appendices A–H. Printed labels remain
source text, not guessed PDF-page anchors. There is no onboard/PAO, editorial
Flight Journal, Vatican or NIST corpus, no PDF copy and no local OCR pipeline.

## Verification status

All three source inventories passed collection, hashes and exact-offset
preparation. ASoIaF reproduces the preserved 56 bodies byte-for-byte, and all 56
download hashes match intact preserved HTML. Both NASA artifacts match the
approved raw/download hashes and continuous transcript bounds.
The parser recognizes 4,335 primary headers and 111 inline candidates; unknown
punctuation-delimited/bracketed speakers remain source text, not repaired roles.

The full stable pizza build passed: 21 sources, 360 indexed passages, 3,787
mentions, 527 proposed ordered pairs, 224 accepted model assertions and 303
unresolved decisions. Indexed-vector round-trip and every prepared source word's
coverage/exact raw offsets were checked. The build is non-distributable because
pizza permission remains unverified. Compiled-token capacity refinement produced
46 recorded whole-window splits; no source words were truncated.

All 27 dataset boundary tests and six Wikipedia parser/access tests passed,
including capacity refinement, unknown speakers and off-site cache rejection.
A current-code 64-passage ASoIaF rebuild also passed indexed-vector checks.
The actual `make web` default search returned five cited passages and displayed
its partial-build warning (42 nodes/61 edges). This does not qualify the full corpus.
The exact public snapshot also passed those 33 tests, source digest/offset checks
for bundled Wikipedia/NASA text, the expected missing-local-pizza error, and a
wheel build containing LICENSE and NOTICE. Full ASoIaF/Apollo model builds and
complete optional-panel/browser qualification are **not completed**. The full
ASoIaF run was stopped before publication; its local cache is preserved. Do not
treat this repository as full-corpus or performance qualification. See the
[execution evidence](workshops/README.md#verification-status).

Latest remains **unqualified**: no explicitly selected main binary is available,
and the pinned source requires Zig 0.17.0, not the installed 0.16.0. An 8 GiB
host is not proof of compilation/inference resource admission. Doctor/build
commands on all three latest configs fail explicitly for the missing native
binary; query/decide and the browser report the missing qualified build without
Python fallback.
