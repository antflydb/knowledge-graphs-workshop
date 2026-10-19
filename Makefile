DATASET ?= asoiaf
WEB_PORT ?= 8000
CONFIG = datasets/$(DATASET)/config.yaml
RUN = uv run python -m kg.dataset $(CONFIG)

.PHONY: help collect validate prepare doctor inference db build query web playground test
help:
	@printf '%s\n' 'Antfly v0.3.0-rc.1 (set ANTFLY_BIN if `antfly` on PATH is older)' \
		'make inference                 # native inference on :8090 (leave running)' \
		'make db                        # Antfly server on :8796 (leave running)' \
		'make collect|validate|prepare|doctor|build DATASET=asoiaf' \
		'make query TEXT="question"' 'make web WEB_PORT=8000' 'make test' \
		'make playground                # optional: rebuild the prebuilt text-to-graph page (needs pnpm)'
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
query:
	$(RUN) query "$(TEXT)"
web:
	uv run uvicorn kg.web.app:app --host 127.0.0.1 --port $(WEB_PORT) --workers 1
playground:
	cd playground && pnpm install --frozen-lockfile && pnpm build
test:
	uv run python scripts/test_datasets.py
	uv run python scripts/test_wikipedia.py
