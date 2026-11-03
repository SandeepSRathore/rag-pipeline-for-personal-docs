"""The RAG pipeline as an LCEL (LangChain Expression Language) chain.

    {question, role}
      | assign(guard)                      input guard
      | branch: blocked? -> refusal
      | assign(retrieval)                  hybrid retriever + rerank + threshold + recency
      | branch: nothing relevant? -> abstain (no LLM call)
      | assign(bundle)                     spotlighted prompt variables + canary
      | assign(llm_out)                    ChatPromptTemplate | ChatOpenAI.with_structured_output
      | finalize                           citation validation + output guard

Each step is a Runnable. The state is a dict that grows as it flows: that's what
`RunnablePassthrough.assign(...)` does. Callbacks passed in `config` reach every nested
call, which is how the cost/budget handler sees the ChatOpenAI call.

The prompt text itself is imported from the custom app, so both apps send the model
exactly the same instructions. Only the orchestration differs.
"""

from __future__ import annotations

import html
import re
import secrets
import time
import uuid
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough
from qdrant_client import QdrantClient

from lcrag.callbacks import BudgetExceededError, CostCallbackHandler, Ledger
from lcrag.config import Settings
from lcrag.embeddings import build_base_dense, build_sparse
from lcrag.retrieval import apply_threshold, build_retriever, mark_superseded
from lcrag.vectorstore import make_client, vector_store
from ragkit.generation.llm import LLMAnswer
from ragkit.generation.prompts import ABSTAIN_MESSAGE, SYSTEM_TEMPLATE
from ragkit.guardrails.injection_detect import detect_injection
from ragkit.guardrails.input_guard import REFUSAL_INJECTION, REFUSAL_MODERATION
from ragkit.guardrails.output_guard import check_output
from ragkit.ingestion.normalize import normalize_text

NO_GROUNDED_ANSWER = "I couldn't produce an answer grounded in the knowledge base."
_UNSET: Any = object()

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_TEMPLATE), ("human", "{user_message}")]
)


# ------------------------------------------------------------------ generators
def build_generator(settings: Settings) -> Runnable:
    """ChatOpenAI with structured output. include_raw=True keeps the AIMessage so we can
    read token usage and see parsing errors instead of getting an exception."""
    if settings.llm_provider == "fake":
        return RunnableLambda(_fake_generate)
    from langchain_openai import ChatOpenAI

    if settings.openai_api_key is None:
        raise RuntimeError("OPENAI_API_KEY is not set (or use LLM_PROVIDER=fake)")
    kwargs: dict[str, Any] = {}
    if settings.llm_reasoning_effort and settings.llm_model.startswith(("gpt-5", "o")):
        kwargs["reasoning_effort"] = settings.llm_reasoning_effort
    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        max_completion_tokens=settings.max_output_tokens,
        use_responses_api=True,
        store=False,  # don't retain prompts (internal docs) on OpenAI's side
        max_retries=2,
        timeout=60,
        **kwargs,
    )
    return llm.with_structured_output(LLMAnswer, method="json_schema", include_raw=True)


_DOC = re.compile(r'<document ref="(S\d+)"[^>]*>\n(.*?)\n</document>', re.S)


def _fake_generate(prompt: PromptValue) -> dict[str, Any]:
    """Offline stand-in: 'answers' with the first sentence of the top document."""
    human = prompt.to_messages()[-1].content
    m = _DOC.search(str(human))
    if not m:
        parsed = LLMAnswer(answer=ABSTAIN_MESSAGE, citations=[], abstained=True)
    else:
        body = html.unescape(m.group(2)).split("\n\n", 1)[-1]
        parsed = LLMAnswer(
            answer=body.split(". ")[0][:300], citations=[m.group(1)], abstained=False
        )
    return {"raw": AIMessage(content=""), "parsed": parsed, "parsing_error": None}


# ------------------------------------------------------------------ the chain
class RAGChain:
    def __init__(
        self,
        settings: Settings,
        client: QdrantClient | None = None,
        dense: Any = None,
        sparse: Any = None,
        scorer: Any = _UNSET,
        generator: Runnable | None = None,
    ):
        self.s = settings
        self.ledger = Ledger(settings)
        self.client = client or make_client(settings)
        dense = dense or build_base_dense(settings, self.ledger)
        self.vs = vector_store(
            self.client, settings.lc_collection_alias, dense, sparse or build_sparse(settings)
        )
        if scorer is _UNSET:
            from ragkit.retrieval.rerank import get_reranker

            scorer = get_reranker(settings.rerank_model) if settings.rerank_enabled else None
        self.scorer = scorer
        self.generator = generator or build_generator(settings)
        self.chain = self._build()

    # ---------------------------------------------------------------- steps
    def _input_guard(self, state: dict) -> dict:
        t = time.perf_counter()
        norm = normalize_text(state["question"])
        q = norm.text
        out: dict[str, Any] = {"allowed": True, "clean": q}
        if not q:
            out.update(allowed=False, reason="empty", message="Please enter a question.")
        elif len(q) > self.s.max_query_chars:
            out.update(
                allowed=False,
                reason="too_long",
                message=f"Questions are limited to {self.s.max_query_chars} characters.",
            )
        else:
            inj = detect_injection(q)
            out["details"] = {"injection_score": inj.score, "injection_reasons": inj.reasons}
            if inj.is_injection():
                out.update(allowed=False, reason="prompt_injection", message=REFUSAL_INJECTION)
            elif state.get("moderate", True) and self._moderation_flagged(q):
                out.update(allowed=False, reason="moderation", message=REFUSAL_MODERATION)
        state["timings"]["input_guard"] = _ms(t)
        return out

    def _moderation_flagged(self, q: str) -> bool:
        if not self.s.moderation_enabled or self.s.openai_api_key is None:
            return False
        try:  # LangChain has no first-class moderation runnable in v1: call the SDK
            from openai import OpenAI

            client = OpenAI(api_key=self.s.openai_api_key.get_secret_value(), timeout=10)
            return bool(
                client.moderations.create(model="omni-moderation-latest", input=q)
                .results[0]
                .flagged
            )
        except Exception:
            return False  # fail open; other layers still apply

    def _retrieve(self, state: dict) -> dict:
        t = time.perf_counter()
        self.ledger.check()  # query embedding may be paid
        retriever = build_retriever(self.vs, self.s, state["role"], self.scorer)
        candidates = retriever.invoke(state["guard"]["clean"])
        kept = apply_threshold(list(candidates), self.s, reranked=self.scorer is not None)
        mark_superseded(kept)
        state["timings"]["retrieval"] = _ms(t)
        return {"docs": kept, "candidates": candidates}

    def _bundle(self, state: dict) -> dict:
        nonce = secrets.token_hex(8)
        canary = "CANARY-" + secrets.token_hex(8)
        ref_map: dict[str, Document] = {}
        parts = []
        for i, d in enumerate(state["retrieval"]["docs"], start=1):
            ref = f"S{i}"
            ref_map[ref] = d
            md = d.metadata
            status = (
                f"superseded by {md['superseded_by']}" if md.get("superseded_by") else "current"
            )

            def a(v: Any) -> str:
                return html.escape(str(v or ""), quote=True)

            parts.append(
                f'<document ref="{ref}" title="{a(md.get("title"))}" '
                f'section="{a(md.get("section"))}" '
                f'effective_date="{a(md.get("effective_date"))}" status="{a(status)}">\n'
                f"{html.escape(d.page_content, quote=False)}\n</document>"
            )
        user_message = (
            f'<context boundary="{nonce}">\n'
            + "\n".join(parts)
            + f'\n</context boundary="{nonce}">'
            f"\n\nEmployee question: {state['guard']['clean']}"
        )
        return {
            "vars": {
                "nonce": nonce,
                "abstain": ABSTAIN_MESSAGE,
                "canary": canary,
                "user_message": user_message,
            },
            "canary": canary,
            "ref_map": ref_map,
        }

    def _finalize(self, state: dict) -> dict:
        out = state["llm_out"]
        parsed: LLMAnswer | None = out.get("parsed")
        if parsed is None:
            return self._result(state, NO_GROUNDED_ANSWER, abstained=True, reason="llm_error")
        refs, invalid, seen = [], [], set()
        for raw in parsed.citations:
            ref = raw.strip().strip("[]").upper()
            if ref in seen:
                continue
            seen.add(ref)
            d = state["bundle"]["ref_map"].get(ref)
            if d is None:
                invalid.append(raw)
            else:
                refs.append(
                    {
                        "ref": ref,
                        "title": d.metadata["title"],
                        "section": d.metadata.get("section", ""),
                        "path": d.metadata["source"],
                        "chunk_id": d.id,
                    }
                )
        state["debug"]["invalid_citations"] = invalid
        if parsed.abstained:
            return self._result(state, ABSTAIN_MESSAGE, abstained=True, reason="model_abstained")
        if not refs:
            return self._result(
                state, NO_GROUNDED_ANSWER, abstained=True, reason="no_valid_citations"
            )
        guard = check_output(parsed.answer, state["bundle"]["canary"])
        state["debug"]["output_guard"] = {"actions": guard.actions, "reason": guard.reason}
        if not guard.allowed:
            return self._result(state, guard.text, blocked=True, reason=guard.reason)
        return self._result(state, guard.text, citations=refs)

    # ---------------------------------------------------------------- assembly
    def _build(self) -> Runnable:
        generate = RunnableLambda(lambda s: s["bundle"]["vars"]) | PROMPT | self.generator
        rag = RunnablePassthrough.assign(retrieval=RunnableLambda(self._retrieve)) | RunnableBranch(
            (lambda s: not s["retrieval"]["docs"], RunnableLambda(self._abstain_no_context)),
            RunnablePassthrough.assign(bundle=RunnableLambda(self._bundle))
            | RunnablePassthrough.assign(llm_out=_timed(generate, "llm"))
            | RunnableLambda(self._finalize),
        )
        return (
            RunnablePassthrough.assign(guard=RunnableLambda(self._input_guard))
            | RunnableBranch(
                (lambda s: not s["guard"]["allowed"], RunnableLambda(self._refuse)),
                rag,
            )
        ).with_config(run_name="lumora_rag")

    def ask(
        self, question: str, role: str, request_id: str | None = None, use_moderation: bool = True
    ) -> dict:
        rid = request_id or uuid.uuid4().hex[:12]
        cost_cb = CostCallbackHandler(self.ledger, self.s.llm_model)
        state = {
            "question": question,
            "role": role,
            "request_id": rid,
            "moderate": use_moderation,
            "timings": {},
            "debug": {"role": role},
        }
        try:
            result = self.chain.invoke(
                state,
                config={"callbacks": [cost_cb], "metadata": {"role": role, "request_id": rid}},
            )
        except BudgetExceededError as e:
            return self._result(state, str(e), blocked=True, reason="budget_exceeded")
        if cost_cb.last:
            result["debug"]["llm"] = cost_cb.last
        return result

    # ---------------------------------------------------------------- results
    def _result(
        self,
        state: dict,
        answer: str,
        *,
        abstained: bool = False,
        blocked: bool = False,
        reason: str | None = None,
        citations: list | None = None,
    ) -> dict:
        debug = state["debug"]
        debug["timings_ms"] = state["timings"]
        if "guard" in state:
            debug["input_guard"] = state["guard"].get("details", {})
        retrieval = state.get("retrieval")
        if retrieval:
            kept_ids = {d.id for d in retrieval["docs"]}
            debug["retrieved"] = [
                {
                    "path": d.metadata["source"],
                    "section": d.metadata.get("section", ""),
                    "fused_score": None,  # VectorStoreRetriever doesn't expose scores
                    "rerank_score": (
                        round(d.metadata["rerank_score"], 3)
                        if "rerank_score" in d.metadata
                        else None
                    ),
                    "kept": d.id in kept_ids,
                    "superseded_by": d.metadata.get("superseded_by"),
                    "text": d.page_content[:400],
                }
                for d in list(retrieval["candidates"])[:10]
            ]
        return {
            "request_id": state["request_id"],
            "answer": answer,
            "abstained": abstained,
            "blocked": blocked,
            "reason": reason,
            "citations": citations or [],
            "debug": debug,
            "role": state["role"],
        }

    def _refuse(self, state: dict) -> dict:
        g = state["guard"]
        return self._result(state, g["message"], blocked=True, reason=g["reason"])

    def _abstain_no_context(self, state: dict) -> dict:
        return self._result(
            state, ABSTAIN_MESSAGE, abstained=True, reason="below_relevance_threshold"
        )


def _ms(t: float) -> float:
    return round((time.perf_counter() - t) * 1000, 1)


def _timed(runnable: Runnable, name: str) -> Runnable:
    """Wrap a runnable and record its latency into state['timings']."""

    def run(state: dict, config: Any = None) -> Any:
        t = time.perf_counter()
        out = runnable.invoke(state, config=config)
        state["timings"][name] = _ms(t)
        return out

    return RunnableLambda(run)
