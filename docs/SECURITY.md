# Security model

Threat model: employees ask questions through an authenticated API, and admins add documents.
- **Attacker 1:** a user trying to make the bot misbehave or read data they shouldn't.
- **Attacker 2:** someone who gets malicious text into a document (a vendor PDF, a wiki
  page, a pasted email). This is called *indirect* prompt injection, and in RAG it is the
  more dangerous of the two.

No single control is enough. Each layer below assumes the others can fail.

## Mapping to OWASP Top 10 for LLM Applications

| Risk | Controls | Where | Test |
|---|---|---|---|
| **LLM01 Prompt injection (direct)** | Unicode normalization; heuristic detector incl. obfuscation (spaced letters, zero-width splits); query length cap; refusal without calling the LLM | `guardrails/input_guard.py`, `injection_detect.py` | `test_injection_attacks_are_blocked`, `eval/adversarial.jsonl` |
| **LLM01 Prompt injection (indirect)** | HTML loader drops scripts, comments and hidden elements; invisible/tag characters stripped; chunks scanned at ingestion and **quarantined**; spotlighting with HTML-escaped docs inside nonce-delimited tags; explicit "documents are data" rule | `ingestion/loaders.py`, `normalize.py`, `pipeline.py`, `generation/prompts.py` | `test_html_drops_scripts…`, `test_injected_chunk_is_quarantined`, `test_prompt_escapes_document_markup` |
| **LLM02 Sensitive info disclosure** | ACL filter in the vector query (role → allowed tags); secret and PII redaction on output (keys, cards with Luhn, SSN, non-company emails) | `retrieval/hybrid.py`, `guardrails/pii.py` | `test_acl_hides_restricted_docs_from_employees`, `test_pii_redaction…` |
| **LLM04 Data poisoning** | Admin-only ingestion; extension + magic-byte allowlist; size, PDF page and DOCX zip-bomb limits; `yaml.safe_load`; filename sanitization and path-traversal checks; symlinks escaping the root ignored | `api/main.py`, `ingestion/loaders.py`, `scanner.py` | `test_upload_validation`, `test_docx_zip_bomb_is_rejected`, `test_yaml_is_loaded_safely` |
| **LLM05 Improper output handling** | Markdown images, links, HTML and raw URLs stripped; UI uses `textContent` only; strict CSP (`default-src 'self'`), so even injected HTML couldn't load external images | `guardrails/output_guard.py`, `api/static/app.js`, CSP middleware | `test_output_guard_strips_exfiltration_vectors`, `test_security_headers` |
| **LLM06 Excessive agency** | The model has **no tools**; it can only return JSON text. The answer is validated before use | design | n/a |
| **LLM07 System prompt leakage** | Per-request canary token; output containing it or prompt fingerprints is blocked; rule forbidding disclosure. The prompt holds no secrets anyway | `generation/prompts.py`, `output_guard.py` | `test_canary_leak_is_blocked` |
| **LLM09 Misinformation** | Relevance threshold → abstain; structured output; citations validated against provided refs; superseded versions labeled | `retrieval/hybrid.py`, `generation/citations.py` | `test_answer_without_valid_citations_is_rejected` |
| **LLM10 Unbounded consumption** | API keys; per-key rate limit; hard daily USD budget checked before every paid call; `max_output_tokens`; query length cap | `api/auth.py`, `observability/cost.py` | `test_rate_limit`, `test_cost_recording_and_budget_cap` |

## Other practices

- **Secrets.**
  - Secrets come from `.env`, which is git-ignored, and are held as pydantic `SecretStr`,
    so they never appear in reprs or logs.
  - Use `.env.example` as the template.
  - API keys are compared with `hmac.compare_digest`.
- **Logging.**
  - Raw questions are not logged (hash + length only).
  - Errors return a generic message with a request ID, never a stack trace.
- **Network.**
  - Docker ports are bound to `127.0.0.1`.
  - The container runs as a non-root user.
  - Qdrant has no auth in local dev. Set `QDRANT_API_KEY` (and TLS) anywhere else.
- **Supply chain.**
  - `uv.lock` pins every dependency.
  - `pip-audit` runs in CI.
  - CI runs with no API keys.
- **Data retention.** OpenAI calls use `store=False`.

## Known limitations (be honest about these)

- **Heuristics miss things.** The injection detector can be bypassed by creative phrasing,
  other languages or paraphrase. It's the first layer, not the only one. Upgrade path: a
  trained classifier.
- **Spotlighting only lowers the odds.** It reduces, but does not eliminate, the chance a
  model follows injected instructions. That's why the model has no tools and the output is
  checked.
- **Quarantine is coarse.** It removes the whole flagged chunk, legitimate text included.
  Review quarantined chunks with `rag status` and fix the source document.
- **The rate limiter is per-process,** so it resets on restart. Use Redis or the gateway
  in production.
- **Dev API keys are static.** The keys in `.env.example` are for local use only.
