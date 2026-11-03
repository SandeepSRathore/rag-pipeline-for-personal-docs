# Architecture and design decisions

Each section covers a decision, the reason for it, and the trade-off.

## 1. Ingestion is a diff, not a rebuild

**Identity.**
- `doc_id = sha256(source + ":" + relative_path)` stays stable when the content changes.
- `content_hash = sha256(file bytes + sidecar metadata bytes)` changes whenever the
  content or the metadata changes.

**Manifest (SQLite `documents` table).** Each run compares the files on disk against the
manifest and sorts every file into one bucket:

| Bucket | Condition | Action |
|---|---|---|
| new | doc_id not in manifest | process |
| changed | hash differs, or pipeline_version differs, or last run failed | re-process, upsert, delete stale chunks |
| unchanged | same hash and version | skip (0 work, $0) |
| deleted | in manifest, not on disk | delete all its points |

**Chunk identity.** `chunk_id = uuid5(doc_id, sha256(chunk_text), occurrence)`.
- An unchanged paragraph keeps its point ID.
- The **embedding cache** is keyed by `(model, sha256(text))`. Editing one paragraph of a
  document re-embeds one chunk, not the whole document. The `test_edit_reembeds_only_changed_chunks`
  test asserts exactly 1 embedding call.

**Write order.** New chunks are upserted first, then the doc's points that aren't in the
new set are deleted (`must: doc_id`, `must_not: has_id`). A document is never missing
mid-update. Deleting by filter, rather than by the manifest's ID list, also cleans up any
orphan points from crashed runs.

**Failure isolation.**
- One bad file doesn't abort the run.
- A changed document that fails to parse keeps serving its last good version.
- The failed document is retried on the next run (`status=failed`).

**Concurrency.** A file lock means only one ingest runs at a time. Two concurrent syncs
would race on the manifest.

**Trade-offs.**
- SQLite works for a single node. With multiple API replicas, move the manifest, cache and
  ledger to Postgres.
- Scanning hashes every file on each run. For millions of files, use mtime/size as a cheap
  pre-filter or event-driven ingestion (S3 notifications, webhooks).

## 2. `pipeline_version` and blue/green reindexing

`pipeline_version` is a hash of the embedding model, BM25 model, chunk size/overlap and
normalizer version. Physical collections are named `docs_<pipeline_version>_<timestamp>`,
and queries always go through the alias `docs`.

When you change something that affects stored vectors, such as a new embedding model with
different dimensions, incremental ingest refuses to mix old and new vectors
(`PipelineVersionMismatchError`). `rag reindex` then:
1. builds a fresh collection from all registered sources,
2. atomically swaps the alias (one `update_collection_aliases` call),
3. drops the old collection.

Queries never see a half-built index. To roll back, point the alias at the old collection
(skip step 3 if you want that option).

## 3. Chunking

- Every format is converted to Markdown-ish text with `#` headings: HTML `h1–h6`, DOCX
  "Heading N" styles, front matter.
- Text is split on headings first, so a chunk never straddles two topics.
- Within a section, paragraphs are packed greedily up to about 450 tokens, falling back to
  sentences and then raw tokens. Up to 60 tokens of whole trailing units are carried over
  as overlap.
- Each chunk gets a breadcrumb header (`Paid Time Off Policy (2025) > Carryover`). This
  makes short chunks retrievable, because the header supplies context the body lacks, and
  gives the LLM a sense of where the text came from.

The corpus is small, so most chunks are one section. With long documents, chunk size is
the first thing to tune with `rag eval`.

## 4. Retrieval

```
query ─┬─ dense embedding (semantic) ─┐
       └─ BM25 sparse (keywords)     ─┴─ RRF fusion (top 20, ACL-filtered) ─> cross-encoder ─> threshold ─> top 5
```

- **Hybrid search.** Dense retrieval understands "time off" ≈ "PTO" but is weak on exact
  tokens like `ERR-4291`. BM25 is the reverse. Reciprocal Rank Fusion merges the two
  ranked lists without needing comparable scores.
- **ACL as a filter.** `acl ∈ allowed_acls(role)` is applied in both prefetches.
  Restricted text never leaves the database for unauthorized roles, so no prompt can leak
  it. Never rely on the LLM to "not mention" restricted content.
- **Quarantine filter.** Chunks flagged at ingestion as likely injections are excluded
  (`EXCLUDE_FLAGGED_CHUNKS`).
- **Cross-encoder rerank.** `ms-marco-MiniLM-L-6-v2` runs locally. It reads the query and
  the chunk together, which is far more accurate than vector similarity. Its scores are
  absolute, so a threshold works: relevant chunks scored above -3 and off-topic ones below
  -7 on this corpus, so the default cutoff is -5. Below the threshold the system abstains
  **without calling the LLM** (cheaper and safer).
- **Recency.** Documents can share a `doc_family`. Older versions in the results are
  marked `status="superseded by …"` in the prompt rather than dropped, so a user can still
  ask about the old policy on purpose.

## 5. Generation

- The rules live in `instructions` (system/developer level). The question and documents
  go in the user turn.
- Documents are HTML-escaped and wrapped in `<document ref="S1" …>` inside a
  `<context boundary="<random nonce>">` block (spotlighting).
- Output is structured (`{answer, citations[], abstained}`) and enforced by a JSON schema
  through the Responses API `parse`.
- Citations are validated against the refs actually provided. An answer with no valid
  citation is replaced with "I couldn't produce an answer grounded in the knowledge base."
- `store=False`, so prompts containing internal documents aren't retained by the API
  provider for later retrieval.
- **No token streaming.** Output guards (canary, link stripping, PII redaction) need the
  complete text before anything is shown. Streaming first and checking later would show
  the user exactly what the guard was meant to stop. For a nano model the latency cost is
  small.

## 6. Observability

- JSON logs carry a `request_id`, which is also returned as the `X-Request-ID` header and
  in the response body.
- Raw questions are **not** logged, only a hash and the length.
- Per-stage latency is reported in `debug.timings_ms`.
- Every paid call is written to `cost_ledger` (tokens and USD), which drives the daily
  budget cap and `rag status`.

## 7. What you'd add for real production

| Area | Upgrade |
|---|---|
| Auth | OIDC/JWT from your IdP; groups → ACL tags; per-document ACL lists |
| State | Postgres for manifest/cache/ledger; Redis for rate limiting |
| Ingestion | Queue-based workers (one doc per job), source connectors (Drive, Confluence), webhooks |
| Retrieval | Query rewriting / multi-query, parent-document retrieval, metadata filters from the UI |
| Safety | A trained prompt-injection classifier, LLM-based output grounding check |
| Ops | OpenTelemetry tracing (e.g. Langfuse/Phoenix), dashboards, alerts on spend and abstain rate |
| Eval | Larger golden set from real user questions; track metrics per release |
