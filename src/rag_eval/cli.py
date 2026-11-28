import hashlib
import os
import subprocess
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from rag_eval import __version__
from rag_eval.adapters.chatbot_rag import AdapterError, ChatbotRagClient
from rag_eval.adapters.jsonl import DataFileError, read_dataset, read_run
from rag_eval.adapters.sources import SourceError, load_sources
from rag_eval.judge.base import JudgeError
from rag_eval.judge.cache import CachedJudge
from rag_eval.judge.groq import DEFAULT_MODEL, GroqJudge
from rag_eval.models import EvalConfig, Report, RunItem
from rag_eval.quotes import check_quotes
from rag_eval.report import ReportError, build_report, compare_reports, render_comparison

app = typer.Typer(no_args_is_help=True, help="Evaluate RAG systems against a reference dataset.")

ExistingFile = Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)]


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"rag-eval {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_print_version, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    pass


@app.command()
def evaluate(
    dataset: Annotated[Path, typer.Option(exists=True, dir_okay=False, help="Dataset JSONL.")],
    run: Annotated[Path, typer.Option(exists=True, dir_okay=False, help="Run JSONL.")],
    out: Annotated[Path, typer.Option(help="Where to write the JSON report.")],
    k: Annotated[int, typer.Option(min=1, help="Contexts considered per question.")] = 5,
    threshold: Annotated[
        float, typer.Option(min=0.01, max=1.0, help="Minimum quote coverage.")
    ] = 0.6,
    judge: Annotated[
        bool, typer.Option("--judge", help="Also score answers with an LLM judge (Groq).")
    ] = False,
    judge_model: Annotated[str, typer.Option(help="Groq model used as judge.")] = DEFAULT_MODEL,
    cache_dir: Annotated[
        Path, typer.Option(help="Judgement cache; delete it to re-ask the judge.")
    ] = Path(".rag_eval_cache"),
) -> None:
    """Score a run against the dataset and write a JSON report."""
    cached: CachedJudge | None = None
    if judge:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            _fail("--judge needs the GROQ_API_KEY environment variable")
        try:
            cached = CachedJudge(GroqJudge(api_key, model=judge_model), cache_dir)
        except JudgeError as e:
            _fail(str(e))

    try:
        report = build_report(
            read_dataset(dataset),
            read_run(run),
            EvalConfig(k=k, threshold=threshold),
            dataset_sha256=_sha256(dataset),
            run_sha256=_sha256(run),
            git_commit=_git_commit(),
            judge=cached,
        )
    except (DataFileError, ReportError, JudgeError) as e:
        _fail(str(e))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    for name, est in report.aggregates.items():
        typer.echo(f"{name:<14} {est.mean:.3f}  [{est.ci_low:.3f}, {est.ci_high:.3f}]  n={est.n}")
    if cached is not None:
        typer.echo(f"Judge {cached.model}: {cached.misses} calls, {cached.hits} from cache")
    typer.echo(f"Report written to {out}")


@app.command()
def compare(
    baseline: ExistingFile,
    candidate: ExistingFile,
    out: Annotated[Path | None, typer.Option(help="Write the Markdown table here.")] = None,
) -> None:
    """Compare two reports built from the same dataset and settings."""
    try:
        base = Report.model_validate_json(baseline.read_text(encoding="utf-8"))
        cand = Report.model_validate_json(candidate.read_text(encoding="utf-8"))
        rows = compare_reports(base, cand)
    except (ValueError, ReportError) as e:
        _fail(str(e))

    markdown = render_comparison(rows, baseline.stem, candidate.stem, base)
    if out is None:
        typer.echo(markdown)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")
        typer.echo(f"Comparison written to {out}")


@app.command()
def collect(
    dataset: Annotated[Path, typer.Option(exists=True, dir_okay=False, help="Dataset JSONL.")],
    url: Annotated[str, typer.Option(help="chatbot-rag API base, e.g. https://host/api")],
    out: Annotated[Path, typer.Option(help="Where to write the run JSONL.")],
) -> None:
    """Ask chatbot-rag every dataset question and save its answers and contexts."""
    try:
        items = read_dataset(dataset)
    except DataFileError as e:
        _fail(str(e))

    client = ChatbotRagClient(url)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = len(items)

    def progress(run_item: RunItem) -> None:
        latency = f"{run_item.latency_ms:.0f} ms" if run_item.latency_ms is not None else "-"
        typer.echo(f"{run_item.id}: {len(run_item.contexts)} contexts, {latency}")

    try:
        runs = client.collect(items, on_progress=progress)
    except AdapterError as e:
        _fail(str(e))

    out.write_text(
        "".join(r.model_dump_json() + "\n" for r in runs),
        encoding="utf-8",
    )
    typer.echo(f"{total} answers written to {out}")


@app.command("check-quotes")
def check_quotes_command(
    dataset: Annotated[Path, typer.Option(exists=True, dir_okay=False, help="Dataset JSONL.")],
    sources_dir: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            help="Folder with the source documents. PDFs need a NAME.pdf.txt beside them "
            "(pdftotext NAME.pdf NAME.pdf.txt); DOCX, TXT and MD are read directly.",
        ),
    ],
) -> None:
    """Check that every evidence quote is verbatim in its source and long enough."""
    try:
        items = read_dataset(dataset)
        names = {ev.source for item in items for ev in item.evidence}
        problems = check_quotes(items, load_sources(sources_dir, names))
    except (DataFileError, SourceError) as e:
        _fail(str(e))

    for p in problems:
        typer.echo(f"{p.item_id}: {p.issue} ({p.detail}) [{p.source}] {p.quote!r}")
    quotes = sum(len(item.evidence) for item in items)
    if problems:
        _fail(f"{len(problems)} problems in {quotes} quotes")
    typer.echo(f"All {quotes} quotes are verbatim and have at least 8 words")


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=1)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    # Records which version of rag-eval produced the report when running from a
    # checkout; installed copies have no repository and report None.
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None
