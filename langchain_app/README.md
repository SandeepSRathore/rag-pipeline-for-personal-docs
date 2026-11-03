# lcrag: the Lumora Labs RAG app, rebuilt with LangChain

The same app as the repo root (`src/ragkit`), reimplemented with LangChain 1.x so you can
compare the two approaches. **Start with [COMPARISON.md](COMPARISON.md).**

It shares the root's corpus (`../data/corpus`), eval sets (`../eval`), prompt text,
guardrails and web UI. It uses its own Qdrant alias (`lc_docs`) and state dir (`var/`),
so both apps can run side by side on one Qdrant.

## Where each LangChain concept lives

| File | LangChain pieces |
|---|---|
| `loaders.py` | `BaseLoader` → `Document`s |
| `splitting.py` | `MarkdownHeaderTextSplitter`, `RecursiveCharacterTextSplitter.from_tiktoken_encoder` |
| `embeddings.py` | `OpenAIEmbeddings`, `CacheBackedEmbeddings` + `LocalFileStore`, `FastEmbedSparse`, custom `Embeddings` |
| `vectorstore.py` | `QdrantVectorStore(retrieval_mode=RetrievalMode.HYBRID)` |
| `indexing.py` | `index()` + `SQLRecordManager` with `cleanup="scoped_full"`, custom `key_encoder` |
| `retrieval.py` | `as_retriever(search_kwargs=...)`, custom `BaseDocumentCompressor`, `ContextualCompressionRetriever` |
| `chain.py` | LCEL: `RunnablePassthrough.assign`, `RunnableBranch`, `RunnableLambda`, `ChatPromptTemplate`, `ChatOpenAI.with_structured_output(include_raw=True)` |
| `callbacks.py` | `BaseCallbackHandler` (budget check + cost ledger) |
| `api.py`, `cli.py` | FastAPI + Typer (not LangChain, same as the custom app) |

## Run it

```bash
cd langchain_app
uv sync
# uses the repo-root .env (and OPENAI_API_KEY from your shell); Qdrant via `make up` at the root

uv run lcrag ingest                    # incremental, via index()
uv run lcrag ingest --dry-run
uv run lcrag ask "How many PTO days do I get?" --debug
uv run lcrag ask "What are the salary bands?" --role hr
uv run lcrag status
uv run lcrag eval                      # same golden + adversarial sets as the custom app
uv run lcrag serve                     # http://localhost:8001 (custom app is :8000)
uv run pytest                          # offline, free
```

Zero-cost mode works the same as the root app:
`EMBEDDINGS_PROVIDER=local LLM_PROVIDER=fake MODERATION_ENABLED=false`.

Optional tracing: `export LANGSMITH_TRACING=true LANGSMITH_API_KEY=...` to see every
step of the chain as a run tree in LangSmith.
