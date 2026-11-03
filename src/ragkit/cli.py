"""Command line interface: `rag --help`."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ragkit.config import get_settings
from ragkit.observability.logging import configure_logging

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Lumora Labs RAG toolkit")
console = Console()


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Show JSON logs")) -> None:
    configure_logging("INFO" if verbose else "WARNING", json=True)


@app.command()
def ingest(
    path: Path = typer.Argument(Path("data/corpus"), help="Directory to ingest"),
    source: str | None = typer.Option(None, help="Source name (default: directory name)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only show what would change"),
    force: bool = typer.Option(False, help="Re-process every file even if unchanged"),
) -> None:
    """Incrementally ingest a directory: only new/changed/deleted files are processed."""
    from ragkit.ingestion.pipeline import Ingestor

    rep = Ingestor(get_settings()).ingest(path, source=source, dry_run=dry_run, force=force)
    t = Table(title=f"Ingest plan for source '{rep.source}'" + (" (dry run)" if dry_run else ""))
    for col in ("new", "changed", "unchanged", "deleted"):
        t.add_column(col)
    t.add_row(*(str(rep.plan[c]) for c in ("new", "changed", "unchanged", "deleted")))
    console.print(t)
    for kind, paths in rep.planned.items():
        for p in paths:
            console.print(f"  [bold]{kind:>8}[/bold]  {p}")
    if not dry_run:
        console.print(
            f"chunks written: {rep.chunks_written} | embedded via API/model: "
            f"{rep.embedded_via_api} | embedding cache hits: {rep.embedding_cache_hits} | "
            f"quarantined chunks: {rep.flagged_chunks} | {rep.duration_s}s"
        )
    for w in rep.warnings:
        console.print(f"[yellow]warning:[/yellow] {w}")
    for f in rep.failed:
        console.print(f"[red]failed:[/red] {f['path']}: {f['error']}")
    if rep.failed:
        raise typer.Exit(1)


@app.command()
def reindex() -> None:
    """Rebuild everything into a new collection, then atomically swap the alias."""
    from ragkit.ingestion.pipeline import Ingestor

    res = Ingestor(get_settings()).reindex()
    console.print(f"new collection: {res['new_collection']}  (dropped: {res['dropped']})")


@app.command()
def ask(
    question: str,
    role: str = typer.Option("employee", help="employee | hr | admin"),
    debug: bool = typer.Option(False, help="Show retrieved chunks, scores, guardrails, cost"),
) -> None:
    """Ask a question through the full guarded pipeline."""
    from ragkit.service import RAGService

    res = RAGService(get_settings()).ask(question, role)
    style = "red" if res.blocked else ("yellow" if res.abstained else "green")
    console.print(f"[{style}]{res.answer}[/{style}]")
    for c in res.citations:
        console.print(f"  [{c.ref}] {c.title} — {c.section or '(intro)'}  ({c.path})")
    if res.reason:
        console.print(f"[dim]reason: {res.reason}[/dim]")
    if debug:
        console.print_json(json.dumps(res.debug, default=str))


@app.command()
def status() -> None:
    """Show index, manifest, quarantined chunks and spend."""
    from ragkit.ingestion.manifest import Manifest
    from ragkit.observability import cost
    from ragkit.store.qdrant_store import QdrantStore

    s = get_settings()
    store = QdrantStore(s)
    console.print(f"pipeline_version: {s.pipeline_version}  embeddings: {s.embedding_model_name}")
    console.print(
        f"alias '{s.collection_alias}' -> {store.alias_target()}  points: {store.count()}"
    )
    t = Table(title="Manifest")
    for col in ("path", "status", "chunks", "flagged", "version"):
        t.add_column(col)
    for e in Manifest(s.db_path).all_entries():
        t.add_row(
            e.path,
            e.status if not e.error else f"{e.status}: {e.error[:40]}",
            str(len(e.chunk_ids)),
            str(e.flagged_chunks),
            e.pipeline_version,
        )
    console.print(t)
    for fc in store.flagged_chunks():
        console.print(
            f"[red]quarantined:[/red] {fc.get('path')} › {fc.get('section')} "
            f"{fc.get('flag_reasons')}"
        )
    console.print(
        f"spend today: ${cost.spent_today(s):.5f}  total: ${cost.spent_total(s):.5f}  "
        f"(daily cap ${s.daily_budget_usd:.2f})"
    )


@app.command("eval")
def evaluate(
    llm: bool = typer.Option(False, help="Also run full answers through the LLM (costs ~$0.01)"),
    judge: bool = typer.Option(False, help="LLM-as-judge faithfulness (needs --llm)"),
    min_hit_rate: float = typer.Option(0.85, help="Fail if retrieval hit@5 is below this"),
) -> None:
    """Run retrieval, answer and adversarial evals. Exits non-zero on regression."""
    from ragkit.evaluation import run_eval

    s = get_settings()
    rep = run_eval(s, llm=llm, judge=judge)
    console.rule("Retrieval")
    console.print_json(json.dumps({k: v for k, v in rep.retrieval.items() if k != "misses"}))
    for m in rep.retrieval["misses"]:
        console.print(f"  [yellow]miss[/yellow] {m['id']}: {m['question']} -> {m['got']}")
    if rep.answers:
        console.rule("Answers")
        console.print_json(json.dumps(rep.answers))
    console.rule("Adversarial")
    console.print_json(json.dumps(rep.adversarial))
    for f in rep.failures:
        console.print(f"[red]FAIL[/red] {json.dumps(f)[:300]}")
    adv_ok = rep.adversarial["passed"].split("/")
    if rep.retrieval["hit@5"] < min_hit_rate or adv_ok[0] != adv_ok[1]:
        console.print("[red]eval gate failed[/red]")
        raise typer.Exit(1)
    console.print("[green]eval gate passed[/green]")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the API + web UI."""
    import uvicorn

    uvicorn.run("ragkit.api.main:app", host=host, port=port, proxy_headers=False)


if __name__ == "__main__":
    app()
