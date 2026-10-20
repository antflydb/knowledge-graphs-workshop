# Dataset inputs and CLI

Each dataset is a directory: `config.yaml` (what you tune), `collection.yaml` (what
`collect` fetches), a `metadata.jsonl` manifest and Markdown under `raw/`. The local
Antfly runtime (ports, data directory) is shared in the repo's `workshop.yaml`.

| Slug | Sources |
|---|---|
| `asoiaf` | 56 frozen Wikipedia articles (bundled) |
| `pizza-portland` | 21 menu/about files for 11 places (local only; run `collect`) |
| `apollo13` | `raw/air-to-ground.md` and `raw/review-board.md` (bundled) |
| `antfly-prs` | 4 pull requests and 2 release notes, each timestamped (bundled) |

## Commands

```text
python -m kg.dataset datasets/<slug>/config.yaml COMMAND
```

| Command | Behavior |
|---|---|
| `collect [--refresh]` | Fetch missing configured sources; validate existing snapshots unless `--refresh` |
| `validate` / `prepare` | Check manifest, paths and hashes; derive exact-offset passages (no inference) |
| `doctor` | Check the Antfly binary is v0.3+ |
| `inference` / `db` | Start native inference (`:8090`) / the Antfly server (`:8796`) |
| `build [--max-passages N]` | Write passages and wait for Antfly to extract, embed and link them |
| `query TEXT [--mode text\|hybrid\|graph]` | Retrieve cited passages for a question |

The Makefile wraps these (`make build DATASET=apollo13`).

## Input contract

- Paths resolve relative to the dataset directory. `metadata.jsonl` has one row per UTF-8
  Markdown file under `raw/` with `id`, `path`, `title`, `url`, `media_type`, `sha256`,
  `fetched_at`, `license`, `attribution` and `redistribution` (`allowed` or
  `permission_not_verified`). Metadata is provenance, never graph facts.
- Validation rejects duplicate IDs/paths, unregistered or missing files, path escapes
  and hash mismatches. A changed source requires `collect --refresh` and a rebuild.
- Passages are whole-word windows (`passages` in `config.yaml`) that keep absolute raw
  offsets: every passage equals `raw[source_start:source_end]`. Transcript passages join
  consecutive radio turns that fit one window. Antfly's extractor
  windows anything longer than GLiNER's limit; nothing is truncated.
- Collection never deletes valid files, substitutes sources or bypasses access controls.
  Pizza PDFs use embedded text only (no OCR).

## Configuration

`config.yaml` is only what you'd tune:

- `extraction`: entity types, a resolver confidence floor and directed predicates.
  Antfly's extractor sees only the predicate names; descriptions and allowed head/tail
  types are used by the on-demand Decide tool, whose default instruction is
  `relation_decision`.
- `passages`: window size in bytes and words, and overlap.
- Optional: `explicit_links` (patterns for links written in the text), `transcript_epoch`
  (launch time, so transcript lines get UTC timestamps), `feed` (read one source as
  messages against a record) and `dashboard: false`.

Model pins, retrieval sizes and paths are fixed in `kg/dataset_config.py`. Unknown keys
fail before any table is created.

## Generated state

- `data/builds/<dataset>/build.json`: the receipt the web app and CLI trust. A failed
  build leaves the previous receipt in place.
- `data/antfly/`: the Antfly server's data directory, shared by all datasets. Each build
  writes one content-addressed `kg_<dataset>_<hash>` table holding passages and their
  resolved entities.
- `--max-passages` marks a build partial. Builds that include permission-unverified
  sources are marked not distributable.
