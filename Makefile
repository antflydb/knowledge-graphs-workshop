DATASET ?= asoiaf
TRACK ?= stable
WEB_PORT ?= 8000
CONFIG = datasets/$(DATASET)/config.yaml
RUN = uv run $(if $(filter stable,$(TRACK)),--extra workshop) python -m kg.dataset $(CONFIG) --track $(TRACK)

.PHONY: help collect validate prepare doctor db build query decide web test
help:
	@printf '%s\n' 'make collect|validate|prepare|doctor|db|build DATASET=asoiaf TRACK=stable' 'make query TEXT="question"' 'make decide TEXT="question"' 'make web WEB_PORT=8000' 'make test'
collect:
	$(RUN) collect
validate:
	$(RUN) validate
prepare:
	$(RUN) prepare
doctor:
	$(RUN) doctor
db:
	$(RUN) db
build:
	$(RUN) build
query:
	$(RUN) query "$(TEXT)"
decide:
	$(RUN) decide "$(TEXT)"
web:
	uv run --extra workshop uvicorn kg.web.app:app --host 127.0.0.1 --port $(WEB_PORT) --workers 1
test:
	uv run python scripts/test_datasets.py
	uv run python scripts/test_wikipedia.py
