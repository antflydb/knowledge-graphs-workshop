# For coding agents

**Set up and run** (macOS arm64 or Linux):

1. Antfly v0.3.0-rc.1: `antfly --version` must report 0.3. Otherwise
   `npm install -g @antfly/cli@next`, or without npm
   `curl -fsSL https://releases.antfly.io/antfly/latest/install.sh | sh -s -- v0.3.0-rc.1`.
   If an older `antfly` stays first on `PATH`, export `ANTFLY_BIN` to the rc.1 binary.
2. `uv`: `curl -LsSf https://astral.sh/uv/install.sh | sh` if missing.
3. `make up` from the repository root. It is long-running: start it in the background and
   wait for the `Dashboard: http://127.0.0.1:8000` line (first run: about 2.4 GB of
   downloads). Ctrl-C (SIGINT) or SIGTERM stops every server it started.

Don't run `antfly inference pull` (or `make decide`) while `make up` is running: rc.1's
inference server breaks when its model files are re-verified underneath it.

**Add the user's documents as a dataset** (with `make up` running):

1. `make new DATASET=<slug>` (lowercase, hyphens).
2. Copy their files into `datasets/<slug>/raw/` as UTF-8 Markdown (`.md`; plain text
   renamed to `.md` is fine).
3. Edit `datasets/<slug>/config.yaml`: set `title`, and replace the template's entity types
   and relations with ones that fit their domain (short lowercase names; each relation's
   head/tail types must be listed entity types).
4. `make collect build DATASET=<slug>`, then reload the page and pick the dataset.

**Change what the graph extracts:** edit any `datasets/*/config.yaml`, then `make build-all`
(only changed datasets rebuild). `make test` runs the unit tests.
