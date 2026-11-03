"""FastAPI app: /query, /ingest/*, /health, /ready, and the static web UI.

Handlers are plain `def` (not async): the work is blocking (DB, model inference), so
FastAPI runs them in a threadpool instead of blocking the event loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ragkit.api.auth import Principal, require_admin, require_principal
from ragkit.config import Settings, get_settings
from ragkit.embeddings.providers import build_dense, build_sparse
from ragkit.ingestion.loaders import LoaderError, validate_file_bytes
from ragkit.ingestion.manifest import Manifest
from ragkit.ingestion.pipeline import IngestBusyError, Ingestor
from ragkit.observability import cost
from ragkit.observability.logging import (
    configure_logging,
    get_logger,
    new_request_id,
    request_id_var,
)
from ragkit.service import RAGService
from ragkit.store.qdrant_store import PipelineVersionMismatchError, QdrantStore

log = get_logger(__name__)
STATIC = Path(__file__).parent / "static"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")


@dataclass
class Components:
    settings: Settings
    store: QdrantStore
    service: RAGService
    ingestor: Ingestor


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    debug: bool = False


def build_components(settings: Settings) -> Components:
    # One store + one set of embedders shared by query and ingest paths (embedded Qdrant
    # can only be opened once per process; models are expensive to load twice).
    store = QdrantStore(settings)
    dense = build_dense(settings)
    sparse = build_sparse(settings)
    return Components(
        settings,
        store,
        RAGService(settings, store=store, dense=dense, sparse=sparse),
        Ingestor(settings, store=store, dense=dense, sparse=sparse),
    )


def create_app(components: Components | None = None) -> FastAPI:
    configure_logging("INFO")
    app = FastAPI(title="Lumora Labs RAG", version="0.1.0", docs_url="/docs", redoc_url=None)
    state: dict[str, Components] = {}
    if components is not None:
        state["c"] = components

    def comp() -> Components:
        if "c" not in state:
            state["c"] = build_components(get_settings())
        return state["c"]

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Any) -> Any:
        rid = new_request_id()
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        # No inline scripts, no external origins: stops XSS and image-based exfiltration.
        if not request.url.path.startswith("/docs"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; img-src 'self'; script-src 'self'; style-src 'self'; "
                "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
            )
        response.headers["X-Request-ID"] = rid
        return response

    @app.exception_handler(Exception)
    async def unhandled(_request: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled_error", error_type=type(exc).__name__)
        # Never leak stack traces or internals to clients.
        return JSONResponse(
            {"detail": "internal error", "request_id": request_id_var.get()}, status_code=500
        )

    # ---------------------------------------------------------------- health
    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> JSONResponse:
        c = comp()
        ok = c.store.healthy() and c.store.alias_target() is not None
        return JSONResponse(
            {"ready": ok, "collection": c.store.alias_target() if ok else None},
            status_code=200 if ok else 503,
        )

    # ---------------------------------------------------------------- query
    @app.post("/query")
    def query(body: QueryIn, p: Principal = Depends(require_principal)) -> dict:
        c = comp()
        try:
            res = c.service.ask(body.question, p.role, request_id=request_id_var.get())
        except cost.BudgetExceededError as e:
            raise HTTPException(429, str(e)) from e
        out = res.to_dict(include_debug=body.debug)
        out["role"] = p.role
        return out

    # ---------------------------------------------------------------- ingest
    @app.post("/ingest/sync")
    def ingest_sync(dry_run: bool = False, _p: Principal = Depends(require_admin)) -> dict:
        """Incrementally sync the server-side corpus directory."""
        c = comp()
        return _run_ingest(c, c.settings.corpus_dir, "corpus", dry_run)

    @app.post("/ingest/upload")
    def ingest_upload(
        file: UploadFile = File(...),
        acl: Literal["public", "hr"] = Form("public"),
        _p: Principal = Depends(require_admin),
    ) -> dict:
        c = comp()
        name = Path(file.filename or "").name
        if not _SAFE_NAME.match(name) or name.endswith(".meta.yaml"):
            raise HTTPException(400, "invalid file name")
        data = file.file.read(c.settings.max_upload_bytes + 1)
        if len(data) > c.settings.max_upload_bytes:
            raise HTTPException(413, "file too large")
        try:
            validate_file_bytes(Path(name).suffix, data)
        except LoaderError as e:
            raise HTTPException(415, str(e)) from e
        updir = c.settings.data_dir / "uploads"
        updir.mkdir(parents=True, exist_ok=True)
        dest = (updir / name).resolve()
        if not dest.is_relative_to(updir.resolve()):
            raise HTTPException(400, "invalid path")
        dest.write_bytes(data)
        dest.with_name(dest.name + ".meta.yaml").write_text(f"acl: {acl}\n")
        return _run_ingest(c, updir, "uploads", False)

    @app.get("/ingest/status")
    def ingest_status(_p: Principal = Depends(require_admin)) -> dict:
        c = comp()
        return {
            "collection": c.store.alias_target(),
            "points": c.store.count(),
            "pipeline_version": c.settings.pipeline_version,
            "documents": [
                {
                    "path": e.path,
                    "status": e.status,
                    "chunks": len(e.chunk_ids),
                    "flagged": e.flagged_chunks,
                    "error": e.error,
                }
                for e in Manifest(c.settings.db_path).all_entries()
            ],
            "quarantined": c.store.flagged_chunks(),
            "spend_today_usd": round(cost.spent_today(c.settings), 6),
            "daily_budget_usd": c.settings.daily_budget_usd,
        }

    # ---------------------------------------------------------------- UI
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return app


def _run_ingest(c: Components, root: Path, source: str, dry_run: bool) -> dict:
    try:
        return c.ingestor.ingest(root, source=source, dry_run=dry_run).to_dict()
    except IngestBusyError as e:
        raise HTTPException(409, str(e)) from e
    except PipelineVersionMismatchError as e:
        raise HTTPException(409, str(e)) from e


app = create_app()
