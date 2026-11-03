"""CLI for the LangChain edition: `lcrag --help`. Mirrors `rag` from the custom app."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from lcrag.config import get_settings

app = typer.Typer(
    no_args_is_help=True, add_completion=False, help="Lumora Labs RAG — LangChain edition"
)
console = Console()


@app.command()
def ingest(
    path: Path | None = typer.Argument(None, help="Directory (default: ../data/corpus)"),
    source: str | None = typer.Option(None, help="Source name (default: directory name)"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, help="Re-write every chunk"),
) -> None:
    """Incremental ingest via LangChain's index() + SQLRecordManager."""
    from lcrag.indexing import LCIndexer

    s = get_settings()
    rep = LCIndexer(s).ingest(path or s.corpus_dir, source=source, dry_run=dry_run, force=force)
    t = Table(title=f"Plan for '{rep.source}'" + (" (dry run)" if dry_run else ""))
    for col in ("new", "changed", "unchanged", "deleted"):
        t.add_column(col)
    t.add_row(*(str(rep.plan[c]) for c in ("new", "changed", "unchanged", "deleted")))
    console.print(t)
    for kind, paths in rep.planned.items():
        for p in paths:
            console.print(f"  [bold]{kind:>8}[/bold]  {p}")
    if not dry_run:
        console.print(f"LangChain index(): {rep.indexing_result}")
        console.print(
            f"embedded via model/API: {rep.embedded_via_api} | quarantined chunks: "
            f"{rep.flagged_chunks} | {rep.duration_s}s"
        )
    for w in rep.warnings:
        console.print(f"[yellow]warning:[/yellow] {w}")
    for f in rep.failed:
        console.print(f"[red]failed:[/red] {f['path']}: {f['error']}")
    if rep.failed:
        raise typer.Exit(1)


@app.command()
def reindex() -> None:
    """Blue/green rebuild into a new collection, then swap the alias."""
    from lcrag.indexing import LCIndexer

    res = LCIndexer(get_settings()).reindex()
    console.print(f"new collection: {res['new_collection']}  (dropped: {res['dropped']})")


@app.command()
def ask(question: str, role: str = typer.Option("employee"), debug: bool = False) -> None:
    """Ask through the LCEL chain."""
    from lcrag.chain import RAGChain

    res = RAGChain(get_settings()).ask(question, role)
    style = "red" if res["blocked"] else ("yellow" if res["abstained"] else "green")
    console.print(f"[{style}]{res['answer']}[/{style}]")
    for c in res["citations"]:
        console.print(f"  [{c['ref']}] {c['title']} — {c['section'] or '(intro)'}  ({c['path']})")
    if res["reason"]:
        console.print(f"[dim]reason: {res['reason']}[/dim]")
    if debug:
        console.print_json(json.dumps(res["debug"], default=str))


@app.command()
def status() -> None:
    """Alias target, point count, spend."""
    from lcrag.callbacks import Ledger
    from lcrag.vectorstore import Collections, make_client

    s = get_settings()
    client = make_client(s)
    target = Collections(client, s).target()
    points = client.count(target, exact=True).count if target else 0
    console.print(f"pipeline_version: {s.pipeline_version}  embeddings: {s.embedding_model_name}")
    console.print(f"alias '{s.lc_collection_alias}' -> {target}  points: {points}")
    console.print(f"spend today: ${Ledger(s).spent_today():.5f}  (cap ${s.daily_budget_usd:.2f})")


@app.command("eval")
def evaluate(
    llm: bool = typer.Option(False, help="Also score real answers (costs ~1 cent)"),
    min_hit_rate: float = typer.Option(0.85),
) -> None:
    """Same golden + adversarial sets as the custom app."""
    from lcrag.evaluation import run_eval

    rep = run_eval(get_settings(), llm=llm)
    console.rule("Retrieval")
    console.print_json(json.dumps({k: v for k, v in rep["retrieval"].items() if k != "misses"}))
    for m in rep["retrieval"]["misses"]:
        console.print(f"  [yellow]miss[/yellow] {m['id']}: {m['question']} -> {m['got']}")
    if "answers" in rep:
        console.rule("Answers")
        console.print_json(json.dumps(rep["answers"]))
    console.rule("Adversarial")
    console.print_json(json.dumps(rep["adversarial"]))
    for f in rep["failures"]:
        console.print(f"[red]FAIL[/red] {json.dumps(f)[:300]}")
    passed, total = rep["adversarial"]["passed"].split("/")
    if rep["retrieval"]["hit@5"] < min_hit_rate or passed != total:
        console.print("[red]eval gate failed[/red]")
        raise typer.Exit(1)
    console.print("[green]eval gate passed[/green]")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8001) -> None:
    """API + web UI on :8001 (the custom app uses :8000)."""
    import uvicorn

    uvicorn.run("lcrag.api:app", host=host, port=port)


if __name__ == "__main__":
    app()
