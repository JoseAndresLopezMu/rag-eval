import hashlib
import subprocess
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from rag_eval import __version__
from rag_eval.adapters.jsonl import DataFileError, read_dataset, read_run
from rag_eval.models import EvalConfig, Report
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
) -> None:
    """Score a run against the dataset and write a JSON report."""
    try:
        report = build_report(
            read_dataset(dataset),
            read_run(run),
            EvalConfig(k=k, threshold=threshold),
            dataset_sha256=_sha256(dataset),
            run_sha256=_sha256(run),
            git_commit=_git_commit(),
        )
    except (DataFileError, ReportError) as e:
        _fail(str(e))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    for name, est in report.aggregates.items():
        typer.echo(f"{name:<14} {est.mean:.3f}  [{est.ci_low:.3f}, {est.ci_high:.3f}]  n={est.n}")
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
