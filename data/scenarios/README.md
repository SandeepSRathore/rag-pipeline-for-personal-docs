# Incremental ingestion scenario

`uv run python scripts/scenario.py apply` changes `data/corpus` like a real week of edits:

| Change | File |
|---|---|
| modified | `hr/pto-policy-2025.md` (carryover 10 → 12 days, new "Volunteer days" section) |
| deleted  | `hr/travel-policy.md` |
| added    | `hr/learning-budget.md` |

Then `rag ingest --dry-run` should report exactly 1 new, 1 changed, 1 deleted, and
`rag ingest` should embed only the chunks that actually changed.

`uv run python scripts/scenario.py revert` restores the original corpus.
