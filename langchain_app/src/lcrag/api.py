"""FastAPI app for the LangChain edition. Same endpoints and response shape as the custom
app, so the same web UI works unchanged (served from the ragkit package).

LangChain has LangServe for exposing chains as REST, but it's in maintenance mode and has no
auth/ACL story, so a hand-written FastAPI layer is the common production choice.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import ragkit.api
from lcrag.chain import RAGChain
from lcrag.config import Settings, get_settings
from lcrag.indexing import IngestBusyError, LCIndexer
from lcrag.vectorstore import PipelineVersionMismatchError
from ragkit.api.auth import SlidingWindowLimiter
from ragkit.ingestion.loaders import LoaderError, validate_file_bytes

STATIC = Path(ragkit.api.__path__[0]) / "static"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")
limiter = SlidingWindowLimiter()


@dataclass(frozen=True)
class Principal:
    key_id: str
    role: str


@dataclass
class Components:
    settings: Settings
    rag: RAGChain
    indexer: LCIndexer


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    debug: bool = False


def build_components(settings: Settings) -> Components:
    indexer = LCIndexer(settings)
    rag = RAGChain(settings, client=indexer.client, dense=indexer.base_dense, sparse=indexer.sparse)
    return Components(settings, rag, indexer)


def create_app(components: Components | None = None) -> FastAPI:
    app = FastAPI(title="Lumora Labs RAG (LangChain edition)", version="0.1.0", redoc_url=None)
    state: dict[str, Components] = {}
    if components is not None:
        state["c"] = components

    def comp() -> Components:
        if "c" not in state:
            state["c"] = build_components(get_settings())
        return state["c"]

    def principal(x_api_key: str | None = Header(default=None)) -> Principal:
        if not x_api_key:
            raise HTTPException(401, "missing X-API-Key header")
        match = None
        for key, role in comp().settings.api_key_roles().items():
            if hmac.compare_digest(key.encode(), x_api_key.encode()):
                match = Principal(hashlib.sha256(key.encode()).hexdigest()[:8], role)
        if match is None:
            raise HTTPException(401, "invalid API key")
        limiter.check(match.key_id, comp().settings.rate_limit_per_minute)
        return match

    def admin(p: Principal = Depends(principal)) -> Principal:
        if p.role != "admin":
            raise HTTPException(403, "admin role required")
        return p

    @app.middleware("http")
    async def headers(request: Request, call_next: Any) -> Any:
        rid = uuid.uuid4().hex[:12]
        request.state.rid = rid
        response = await call_next(request)
        response.headers.update(
            {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "Referrer-Policy": "no-referrer",
                "X-Request-ID": rid,
            }
        )
        if not request.url.path.startswith("/docs"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; img-src 'self'; script-src 'self'; style-src 'self'; "
                "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
            )
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            {"detail": "internal error", "request_id": getattr(request.state, "rid", None)},
            status_code=500,
        )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "edition": "langchain"}

    @app.get("/ready")
    def ready() -> JSONResponse:
        target = comp().indexer.collections.target()
        return JSONResponse(
            {"ready": target is not None, "collection": target}, status_code=200 if target else 503
        )

    @app.post("/query")
    def query(body: QueryIn, request: Request, p: Principal = Depends(principal)) -> dict:
        res = comp().rag.ask(body.question, p.role, request_id=request.state.rid)
        if res["reason"] == "budget_exceeded":
            raise HTTPException(429, res["answer"])
        if not body.debug:
            res.pop("debug", None)
        return res

    def run_ingest(root: Path, source: str, dry_run: bool) -> dict:
        try:
            return comp().indexer.ingest(root, source=source, dry_run=dry_run).to_dict()
        except (IngestBusyError, PipelineVersionMismatchError) as e:
            raise HTTPException(409, str(e)) from e

    @app.post("/ingest/sync")
    def ingest_sync(dry_run: bool = False, _p: Principal = Depends(admin)) -> dict:
        return run_ingest(comp().settings.corpus_dir, "corpus", dry_run)

    @app.post("/ingest/upload")
    def ingest_upload(
        file: UploadFile = File(...),
        acl: Literal["public", "hr"] = Form("public"),
        _p: Principal = Depends(admin),
    ) -> dict:
        s = comp().settings
        name = Path(file.filename or "").name
        if not _SAFE_NAME.match(name) or name.endswith(".meta.yaml"):
            raise HTTPException(400, "invalid file name")
        data = file.file.read(s.max_upload_bytes + 1)
        if len(data) > s.max_upload_bytes:
            raise HTTPException(413, "file too large")
        try:
            validate_file_bytes(Path(name).suffix, data)
        except LoaderError as e:
            raise HTTPException(415, str(e)) from e
        updir = s.lc_data_dir / "uploads"
        updir.mkdir(parents=True, exist_ok=True)
        dest = (updir / name).resolve()
        if not dest.is_relative_to(updir.resolve()):
            raise HTTPException(400, "invalid path")
        dest.write_bytes(data)
        dest.with_name(dest.name + ".meta.yaml").write_text(f"acl: {acl}\n")
        return run_ingest(updir, "uploads", False)

    @app.get("/ingest/status")
    def ingest_status(_p: Principal = Depends(admin)) -> dict:
        c = comp()
        target = c.indexer.collections.target()
        points = c.indexer.client.count(target, exact=True).count if target else 0
        return {
            "collection": target,
            "points": points,
            "pipeline_version": c.settings.pipeline_version,
            "spend_today_usd": round(c.rag.ledger.spent_today(), 6),
        }

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> HTMLResponse:
        page = (STATIC / "index.html").read_text()
        return HTMLResponse(
            page.replace(
                "<span>knowledge assistant</span>",
                "<span>knowledge assistant · LangChain edition</span>",
            )
        )

    return app


app = create_app()
