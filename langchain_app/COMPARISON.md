# Custom vs LangChain: the same RAG app built twice

Both apps use the same corpus, prompt, security rules, eval sets and web UI. Only the
orchestration differs, so you can see exactly what LangChain gives you and what it doesn't.

## Results (same eval, same models)

| | Custom (`src/ragkit`) | LangChain (`langchain_app/src/lcrag`) |
|---|---|---|
| Retrieval hit@5 / MRR / recall@5 (local embeddings) | 1.00 / 0.961 / 1.00 | 1.00 / 0.961 / 1.00 |
| Adversarial attacks blocked | 13/13 | 13/13 |
| Chunks from the corpus | 98 | 98 |
| Cost of one PTO question (gpt-5-nano) | ~$0.00009 | ~$0.00009 |
| Tests | 61 | 17 (+ shared security code is already covered by the 61) |
| Own source lines (non-blank, non-comment) | ~2,460 | ~1,330, plus ~450 reused from `ragkit` (loaders, normalizer, guardrails) |

Retrieval quality is identical because the important choices are identical: hybrid
dense+BM25 with RRF, the same cross-encoder and threshold, and the same chunk size. **The
framework doesn't make retrieval better or worse. Your choices do.**

## Stage by stage

| Stage | Custom app | LangChain app | Verdict |
|---|---|---|---|
| **Loading** | `ingestion/loaders.py`: hardened parsers | `BaseLoader` subclass wrapping the same parsers | LangChain's loaders lived in `langchain-community`, **sunset May 2026**. Its `BSHTMLLoader` keeps hidden HTML, where injections live, and `Docx2txtLoader` drops headings. The interface is useful; the implementations weren't. |
| **Chunking** | `chunker.py` (97 lines): heading split + token packing | `MarkdownHeaderTextSplitter` → `RecursiveCharacterTextSplitter.from_tiktoken_encoder` (41 lines) | **LangChain wins.** Battle-tested splitters, less code. The breadcrumb header and injection flag are still added by hand. |
| **Incremental ingestion** | `scanner.py` + `manifest.py` + `pipeline.py` + `cache.py` (451 lines) | `index()` + `SQLRecordManager`, `cleanup="scoped_full"` (186 lines) | **LangChain wins on the core**: hashing, skip unchanged, upsert, delete stale, in one call. Gaps we filled: deleted *files* (`scoped_full` can't see them), dry run, Qdrant-safe IDs (default sha1 hex isn't a valid Qdrant ID), locking, and keeping the last good version when a file fails. |
| **Embedding cache** | SQLite table keyed by (model, sha256) | `CacheBackedEmbeddings` + `LocalFileStore` | Equivalent. LangChain's is ready-made. |
| **Vector store / hybrid search** | `qdrant_store.py`: prefetch + RRF query by hand | `QdrantVectorStore(retrieval_mode=HYBRID)` | **LangChain wins**: hybrid search in one argument. Alias and blue/green management is still hand-written (`Collections`). |
| **ACL filter** | Qdrant `Filter` in both prefetches | Same `Filter` via `search_kwargs={"filter": ...}`; keys become `metadata.acl` | Same mechanism. Note LangChain nests metadata under `metadata.` in the payload. |
| **Rerank + threshold** | `Reranker` protocol, scores kept for debug | `BaseDocumentCompressor` + `ContextualCompressionRetriever` | Clean extension point. Trade-off: `VectorStoreRetriever` hides fusion scores, so the debug view loses them. |
| **Prompt** | f-string built in `prompts.py` | `ChatPromptTemplate` with the **same** template text | Equivalent. Variable values aren't re-parsed as templates, so escaped document text is safe in both. |
| **LLM + structured output** | `client.responses.parse(text_format=LLMAnswer)` | `ChatOpenAI(...).with_structured_output(LLMAnswer, include_raw=True)` | Equivalent. `include_raw=True` is important: it gives token usage and parse errors instead of exceptions. |
| **Pipeline orchestration** | `service.py`: plain function with early returns | LCEL: `RunnablePassthrough.assign` + `RunnableBranch` | Matter of taste. LCEL gives streaming, batching, async and tracing for free. The plain function is easier to debug and step through. |
| **Cost + budget** | Explicit `check_budget()` before each paid call | `BaseCallbackHandler` with `raise_error=True`; `on_chat_model_start` checks, `on_llm_end` records | Callbacks are elegant for LLM calls, but **embeddings emit no callbacks**, so we wrap them anyway. |
| **Guardrails** (injection, canary, PII, output stripping) | `guardrails/` | **Reused from `ragkit`** | LangChain has no built-in equivalent. Security code should exist once, not as two diverging copies. |
| **Moderation** | OpenAI SDK | OpenAI SDK | No first-class moderation runnable in LangChain v1. |
| **API / UI** | FastAPI | FastAPI, same UI | LangServe is in maintenance mode and has no auth/ACL story. |
| **Observability** | structlog JSON logs + request IDs | Callbacks; set `LANGSMITH_TRACING=true` + key for full traces | LangChain's tracing (LangSmith) is a real advantage if you use it. |

## Where LangChain helped

1. **`index()` + `RecordManager`.** The biggest win. Chunk-level hashing and stale cleanup is
   the hardest part to get right, and it's one call.
2. **Hybrid search in one argument** (`RetrievalMode.HYBRID`).
3. **Splitters** that handle edge cases you'd otherwise discover the hard way.
4. **Uniform interfaces** (`Embeddings`, `BaseRetriever`, `Runnable`), so swapping OpenAI for
   a local model or Qdrant for pgvector is a one-line change.
5. **Free streaming/async/batch/tracing** on any LCEL chain.

## Where it got in the way

1. **API churn.** v1 moved helpers to `langchain-classic`. `langchain-community` was sunset
   mid-2026, taking loaders and many integrations with it. Pin versions and expect migrations.
2. **Hidden defaults with security impact.** sha1 IDs, HTML loaders that keep hidden text,
   metadata nested under `metadata.`. You must read the source to know what happens.
3. **Gaps around production concerns.** Locking, dry runs, deleted-file cleanup, aliases,
   budget for embeddings, ACLs, auth: all still yours to build.
4. **Debuggability.** Stack traces go through `RunnableSequence`/`RunnableBranch`
   internals, and state lives in a growing dict instead of named variables.
5. **More dependencies.** ~15 `langchain*`/transitive packages on top of the SDKs.

## When to pick which

- **Pick LangChain** if you need many integrations quickly, want LangSmith tracing, are
  moving toward agents/LangGraph, or the team already knows it.
- **Pick plain SDKs** for a focused, long-lived service where you need full control of
  security, cost and failure behavior, and fewer moving parts.
- **Or do both, as here:** use LangChain for the commodity parts (splitters, indexing,
  vector store) and keep security, ACLs, cost control and the API in your own code.

## Things to try

1. Diff the two orchestrations side by side: `src/ragkit/service.py` vs `src/lcrag/chain.py`.
2. Turn on tracing: `export LANGSMITH_TRACING=true LANGSMITH_API_KEY=...` (free tier),
   ask a question, and look at the nested run tree.
3. Make LCEL stream: `RAGChain.chain.stream(...)`, then think about why the output guard
   makes streaming the final answer unsafe.
4. Swap `cleanup="scoped_full"` for `"full"`, break a PDF and re-ingest. Watch the last
   good version disappear (the reason we didn't use `full`).
