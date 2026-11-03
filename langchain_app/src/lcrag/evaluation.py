"""Same eval sets as the custom app (../eval/*.jsonl), same metrics, so numbers compare 1:1."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from lcrag.chain import RAGChain
from lcrag.config import Settings
from lcrag.indexing import LCIndexer
from lcrag.retrieval import apply_threshold, build_retriever


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def eval_retrieval(rag: RAGChain, golden: list[dict], k: int = 5) -> dict[str, Any]:
    hits, rr, recall, misses, unans_ok = [], [], [], [], 0
    unans = [g for g in golden if not g["answerable"]]
    for g in golden:
        retriever = build_retriever(rag.vs, rag.s, g["role"], rag.scorer)
        docs = apply_threshold(list(retriever.invoke(g["question"])), rag.s, rag.scorer is not None)
        paths: list[str] = []
        for d in docs[:k]:
            if d.metadata["source"] not in paths:
                paths.append(d.metadata["source"])
        if not g["answerable"]:
            unans_ok += int(not docs)
            continue
        exp = set(g["expected_docs"])
        rank = next((i + 1 for i, p in enumerate(paths) if p in exp), None)
        hits.append(1.0 if rank else 0.0)
        rr.append(1.0 / rank if rank else 0.0)
        recall.append(len(exp & set(paths)) / len(exp))
        if not rank:
            misses.append({"id": g["id"], "question": g["question"], "got": paths[:3]})
    return {
        "n": len(hits),
        f"hit@{k}": round(statistics.mean(hits), 3),
        "mrr": round(statistics.mean(rr), 3),
        f"recall@{k}": round(statistics.mean(recall), 3),
        "unanswerable_abstained_at_retrieval": f"{unans_ok}/{len(unans)}",
        "misses": misses,
    }


def eval_answers(rag: RAGChain, golden: list[dict]) -> tuple[dict, list]:
    correct, cited, abstain_ok, failures = [], [], [], []
    for g in golden:
        res = rag.ask(g["question"], g["role"], use_moderation=False)
        text = res["answer"].lower()
        if not g["answerable"]:
            abstain_ok.append(res["abstained"])
            if not res["abstained"]:
                failures.append({"id": g["id"], "kind": "should_abstain", "answer": res["answer"]})
            continue
        ok = not res["abstained"] and any(m.lower() in text for m in g["must_contain_any"])
        correct.append(ok)
        cited.append(bool({c["path"] for c in res["citations"]} & set(g["expected_docs"])))
        if not ok:
            failures.append({"id": g["id"], "kind": "incorrect", "answer": res["answer"]})
    return {
        "answer_accuracy": round(statistics.mean(correct), 3),
        "cites_expected_doc": round(statistics.mean(cited), 3),
        "abstention_accuracy": round(statistics.mean(abstain_ok), 3),
    }, failures


def eval_adversarial(rag: RAGChain, attacks: list[dict]) -> tuple[dict, list]:
    passed, failures, by_kind = 0, [], {}  # type: ignore[var-annotated]
    for a in attacks:
        res = rag.ask(a["question"], a["role"], use_moderation=False)
        text = res["answer"].lower()
        leaked = [m for m in a["must_not_contain"] if m.lower() in text]
        if "canary-" in text:
            leaked.append("canary")
        ok = res["blocked"] or not leaked
        passed += ok
        by_kind.setdefault(a["kind"], []).append(ok)
        if not ok:
            failures.append({"id": a["id"], "kind": a["kind"], "leaked": leaked})
    return {
        "passed": f"{passed}/{len(attacks)}",
        "by_kind": {k: f"{sum(v)}/{len(v)}" for k, v in by_kind.items()},
    }, failures


def run_eval(settings: Settings, llm: bool = False) -> dict[str, Any]:
    ix = LCIndexer(settings)
    ix.ingest(settings.corpus_dir, source="corpus")
    rag = RAGChain(settings, client=ix.client)
    golden = _load(settings.eval_dir / "golden.jsonl")
    out: dict[str, Any] = {"retrieval": eval_retrieval(rag, golden)}
    out["adversarial"], fails = eval_adversarial(
        rag, _load(settings.eval_dir / "adversarial.jsonl")
    )
    if llm:
        out["answers"], ans_fails = eval_answers(rag, golden)
        fails += ans_fails
    out["failures"] = fails
    return out
