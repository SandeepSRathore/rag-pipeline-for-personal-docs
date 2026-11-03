"""The RAG request pipeline, in one place:

    input guard -> budget check -> retrieve (hybrid + ACL + rerank + threshold)
      -> build spotlighted prompt -> LLM (structured) -> validate citations -> output guard

Each stage can short-circuit with a safe answer, and the cheap stages run first.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from ragkit.config import Settings
from ragkit.embeddings.base import DenseEmbedder, SparseEmbedder
from ragkit.embeddings.providers import build_dense, build_sparse
from ragkit.generation.citations import Citation, validate_citations
from ragkit.generation.llm import LLM, build_llm
from ragkit.generation.prompts import ABSTAIN_MESSAGE, build_prompt
from ragkit.guardrails.input_guard import check_input
from ragkit.guardrails.output_guard import check_output
from ragkit.observability import cost
from ragkit.observability.logging import get_logger, new_request_id, redact_query
from ragkit.retrieval.hybrid import Retriever
from ragkit.retrieval.rerank import Reranker, get_reranker
from ragkit.store.qdrant_store import QdrantStore

log = get_logger(__name__)
NO_GROUNDED_ANSWER = "I couldn't produce an answer grounded in the knowledge base."
_UNSET: Any = object()


@dataclass
class QueryResult:
    request_id: str
    answer: str
    abstained: bool = False
    blocked: bool = False
    reason: str | None = None
    citations: list[Citation] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_debug: bool = False) -> dict[str, Any]:
        d = asdict(self)
        if not include_debug:
            d.pop("debug")
        return d


class RAGService:
    def __init__(
        self,
        settings: Settings,
        store: QdrantStore | None = None,
        dense: DenseEmbedder | None = None,
        sparse: SparseEmbedder | None = None,
        reranker: Reranker | None = _UNSET,
        llm: LLM | None = None,
    ):
        self.s = settings
        self.store = store or QdrantStore(settings)
        if reranker is _UNSET:
            reranker = get_reranker(settings.rerank_model) if settings.rerank_enabled else None
        self.retriever = Retriever(
            settings,
            self.store,
            dense or build_dense(settings),
            sparse or build_sparse(settings),
            reranker,
        )
        self.llm = llm or build_llm(settings)

    def ask(
        self,
        question: str,
        role: str,
        use_moderation: bool = True,
        request_id: str | None = None,
    ) -> QueryResult:
        rid = request_id or new_request_id()
        timings: dict[str, float] = {}
        debug: dict[str, Any] = {"role": role, "timings_ms": timings}

        def lap(name: str, t0: float) -> None:
            timings[name] = round((time.perf_counter() - t0) * 1000, 1)

        # 1. input guard -------------------------------------------------------
        t = time.perf_counter()
        verdict = check_input(question, self.s, use_moderation=use_moderation)
        lap("input_guard", t)
        debug["input_guard"] = verdict.details
        if not verdict.allowed:
            log.warning("query_blocked", reason=verdict.reason, role=role, **redact_query(question))
            return QueryResult(
                rid, verdict.message or "", blocked=True, reason=verdict.reason, debug=debug
            )

        # 2. budget -------------------------------------------------------------
        try:
            cost.check_budget(self.s)
        except cost.BudgetExceededError as e:
            return QueryResult(rid, str(e), blocked=True, reason="budget_exceeded", debug=debug)

        # 3. retrieval ------------------------------------------------------------
        t = time.perf_counter()
        retrieval = self.retriever.retrieve(verdict.clean_query, role)
        lap("retrieval", t)
        debug["retrieved"] = [
            {
                "path": c.path,
                "section": c.section,
                "fused_score": round(c.fused_score, 4),
                "rerank_score": None if c.rerank_score is None else round(c.rerank_score, 3),
                "kept": c in retrieval.chunks,
                "superseded_by": c.superseded_by,
                "text": c.text[:400],
            }
            for c in (retrieval.candidates or retrieval.chunks)[:10]
        ]
        if not retrieval.chunks:
            log.info("abstain_no_context", reason=retrieval.abstain_reason, role=role)
            return QueryResult(
                rid, ABSTAIN_MESSAGE, abstained=True, reason=retrieval.abstain_reason, debug=debug
            )

        # 4. generation -----------------------------------------------------------
        bundle = build_prompt(verdict.clean_query, retrieval.chunks)
        t = time.perf_counter()
        result = self.llm.generate(bundle)
        lap("llm", t)
        debug["llm"] = {
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "usd": round(result.usd, 6),
        }
        if result.parsed is None:
            log.error("llm_no_output", error=result.error)
            return QueryResult(
                rid, NO_GROUNDED_ANSWER, abstained=True, reason="llm_error", debug=debug
            )

        parsed = result.parsed
        # 5. citations ------------------------------------------------------------
        citations, invalid = validate_citations(parsed.citations, bundle.ref_map)
        debug["invalid_citations"] = invalid
        if parsed.abstained:
            return QueryResult(
                rid, ABSTAIN_MESSAGE, abstained=True, reason="model_abstained", debug=debug
            )
        if not citations:
            return QueryResult(
                rid, NO_GROUNDED_ANSWER, abstained=True, reason="no_valid_citations", debug=debug
            )

        # 6. output guard -----------------------------------------------------------
        out = check_output(parsed.answer, bundle.canary)
        debug["output_guard"] = {"actions": out.actions, "reason": out.reason}
        if not out.allowed:
            log.error("output_blocked", reason=out.reason, role=role)
            return QueryResult(rid, out.text, blocked=True, reason=out.reason, debug=debug)

        log.info(
            "query_answered",
            role=role,
            citations=len(citations),
            timings=timings,
            **redact_query(question),
        )
        return QueryResult(rid, out.text, citations=citations, debug=debug)
