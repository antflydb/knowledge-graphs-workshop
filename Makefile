DATASET ?= asoiaf
WEB_PORT ?= 8000
# A prebuilt database for every bundled dataset (`make fetch-db`), instead of `make build-all`.
DB_URL ?= https://github.com/antflydb/knowledge-graphs-workshop/releases/download/db-v0.3.0-rc.1/antfly-db.tar.gz
CONFIG = datasets/$(DATASET)/config.yaml
RUN = uv run python -m kg.dataset $(CONFIG)
# Every dataset the dashboard shows (no `dashboard: false`).
DASHBOARD = uv run python -c 'from kg.dataset_config import dashboard_datasets; print(*dashboard_datasets())'

.PHONY: help up decide new collect validate prepare doctor inference db build build-all fetch-db query web playground test
help:
	@printf '%s\n' 'Antfly v0.3.0-rc.1 (set ANTFLY_BIN if `antfly` on PATH is older)' \
		'make up                        # everything in one terminal: models, database, dashboard (Ctrl-C stops)' \
		'                               # (BUILD=1 builds the graph here instead of downloading it)' \
		'make decide                    # optional: pull GLiNER2.5-Decide (1.8 GB) for the Decide drawer (while make up is stopped)' \
		'-- or step by step --' \
		'make inference                 # native inference on :8090 (leave running)' \
		'make db                        # Antfly server on :8796 (leave running)' \
		'make collect|validate|prepare|doctor|build DATASET=asoiaf' \
		'make build-all                 # collect + rebuild every dashboard dataset (after editing configs)' \
		'make new DATASET=my-docs       # scaffold datasets/my-docs/ for your own Markdown files' \
		'make fetch-db                  # download the prebuilt database instead (before the first make db)' \
		'make query TEXT="question"' 'make web WEB_PORT=8000' 'make test' \
		'make playground                # optional: rebuild the prebuilt text-to-graph page (needs pnpm)'
up:
	@command -v uv >/dev/null || { echo "make up needs uv: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
	BUILD=$(BUILD) DECIDE=$(DECIDE) WEB_PORT=$(WEB_PORT) uv run python -m kg.up
decide:
	uv run python -c 'from kg.up import pull; pull("decide")'
collect:
	$(RUN) collect
validate:
	$(RUN) validate
prepare:
	$(RUN) prepare
doctor:
	$(RUN) doctor
inference:
	$(RUN) inference
db:
	$(RUN) db
build:
	$(RUN) build
build-all:
	@for d in $$($(DASHBOARD)); do echo "== $$d"; $(MAKE) --no-print-directory collect build DATASET=$$d || exit 1; done
fetch-db:
	@test ! -e data/antfly || { echo "data/antfly already exists; stop make db and move it aside first"; exit 1; }
	mkdir -p data && curl -fL $(DB_URL) | tar -xzf - -C data
	@echo "Database and build receipts unpacked into data/; now run make db"
new:
	@test ! -e datasets/$(DATASET) || { echo "datasets/$(DATASET) already exists"; exit 1; }
	mkdir -p datasets/$(DATASET)/raw
	sed 's/^title: My documents$$/title: $(DATASET)/' datasets/config.template.yaml > datasets/$(DATASET)/config.yaml
	@echo "Add Markdown files to datasets/$(DATASET)/raw/, then: make collect build DATASET=$(DATASET)"
query:
	$(RUN) query "$(TEXT)"
web:
	uv run uvicorn kg.web.app:app --host 127.0.0.1 --port $(WEB_PORT) --workers 1
playground:
	cd playground && pnpm install --frozen-lockfile && pnpm build
test:
	uv run python scripts/test_datasets.py
	uv run python scripts/test_wikipedia.py
