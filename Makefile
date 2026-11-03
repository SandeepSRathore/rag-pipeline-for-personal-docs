.PHONY: help install up down up-all ingest ingest-dry status ask serve test lint typecheck audit eval eval-llm scenario-apply scenario-revert demo-incremental clean

help:            ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:         ## Install deps (Python 3.12 venv via uv)
	uv sync

up:              ## Start Qdrant in Docker
	docker compose up -d qdrant

down:            ## Stop containers
	docker compose --profile app down

up-all:          ## Run Qdrant + API in Docker (http://localhost:8000)
	docker compose --profile app up -d --build

ingest:          ## Incrementally ingest data/corpus
	uv run rag ingest data/corpus

ingest-dry:      ## Show what would change, without changing anything
	uv run rag ingest data/corpus --dry-run

status:          ## Index, manifest, quarantined chunks, spend
	uv run rag status

ask:             ## make ask Q="How many PTO days do I get?" [ROLE=hr]
	uv run rag ask "$(Q)" --role $(or $(ROLE),employee) --debug

serve:           ## Run API + UI locally on :8000
	uv run rag serve

test:            ## Unit + integration tests (offline, free)
	uv run pytest

lint:            ## Ruff lint + format check
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

typecheck:       ## mypy
	uv run mypy src

audit:           ## Known-vulnerability scan of dependencies
	uv run pip-audit --strict --progress-spinner off

eval:            ## Retrieval + adversarial eval (free with EMBEDDINGS_PROVIDER=local)
	uv run rag eval

eval-llm:        ## Also evaluate real answers + LLM-as-judge (costs ~1-2 cents)
	uv run rag eval --llm --judge

scenario-apply:  ## Edit/delete/add corpus files to demo incremental ingestion
	uv run python scripts/scenario.py apply

scenario-revert: ## Restore the original corpus
	uv run python scripts/scenario.py revert

demo-incremental: ## Full incremental-ingestion demo
	uv run rag ingest data/corpus
	uv run python scripts/scenario.py apply
	uv run rag ingest data/corpus --dry-run
	uv run rag ingest data/corpus
	uv run python scripts/scenario.py revert
	uv run rag ingest data/corpus

clean:           ## Delete local state (manifest, cache, embedded index)
	rm -rf var/ragkit.sqlite3* var/qdrant_local var/uploads var/ingest.lock
