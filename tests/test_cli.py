import json
from pathlib import Path

from typer.testing import CliRunner

from rag_eval import __version__
from rag_eval.cli import app
from rag_eval.models import Report

runner = CliRunner()

DATASET: list[dict[str, object]] = [
    {
        "id": "q1",
        "question": "¿Qué servicios incluye la nube privada?",
        "reference_answer": "Recuperación de datos.",
        "evidence": [{"source": "pliego.pdf", "quote": "servicio de recuperacion de datos"}],
    },
    {
        "id": "q2",
        "question": "¿Cuál es el presupuesto de marketing?",
        "reference_answer": "No consta.",
        "answerable": False,
    },
]


def run_with(text_q1: str) -> list[dict[str, object]]:
    return [
        {
            "id": "q1",
            "answer": "a",
            "contexts": [{"rank": 1, "source": "pliego.pdf", "text": text_q1}],
        },
        {"id": "q2", "answer": "No tengo esa información."},
    ]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return path


def evaluate(
    directory: Path, name: str, text_q1: str, dataset: list[dict[str, object]] = DATASET
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    dataset_path = write_jsonl(directory / "dataset.jsonl", dataset)
    run = write_jsonl(directory / f"{name}.jsonl", run_with(text_q1))
    out = directory / "reports" / f"{name}.json"
    result = runner.invoke(
        app, ["evaluate", "--dataset", str(dataset_path), "--run", str(run), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    return out


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"rag-eval {__version__}"


def test_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Evaluate RAG systems" in result.output


def test_evaluate_writes_report_and_prints_summary(tmp_path: Path) -> None:
    out = evaluate(tmp_path, "good", "... servicio de recuperacion de datos ...")
    report = Report.model_validate_json(out.read_text(encoding="utf-8"))
    assert report.aggregates["hit@5"].mean == 1.0
    assert report.config.k == 5
    assert len(report.dataset_sha256) == 64


def test_evaluate_reports_data_errors_without_traceback(tmp_path: Path) -> None:
    dataset = write_jsonl(tmp_path / "dataset.jsonl", DATASET)
    run = write_jsonl(tmp_path / "run.jsonl", run_with("x")[:1])
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--dataset",
            str(dataset),
            "--run",
            str(run),
            "--out",
            str(tmp_path / "r.json"),
        ],
    )
    assert result.exit_code == 1
    assert "error: run does not match dataset; missing from run: q2" in result.output
    assert "Traceback" not in result.output


def test_compare_prints_markdown(tmp_path: Path) -> None:
    bad = evaluate(tmp_path, "semantic", "texto que no contiene la cita")
    good = evaluate(tmp_path, "hybrid", "servicio de recuperacion de datos")
    result = runner.invoke(app, ["compare", str(bad), str(good)])
    assert result.exit_code == 0, result.output
    assert "# hybrid vs semantic" in result.output
    assert "| hit@5 |" in result.output


def test_compare_writes_file(tmp_path: Path) -> None:
    a = evaluate(tmp_path, "a", "x")
    b = evaluate(tmp_path, "b", "x")
    out = tmp_path / "cmp.md"
    result = runner.invoke(app, ["compare", str(a), str(b), "--out", str(out)])
    assert result.exit_code == 0
    assert "inconclusive" in out.read_text(encoding="utf-8")


def test_compare_rejects_reports_from_different_datasets(tmp_path: Path) -> None:
    a = evaluate(tmp_path / "one", "a", "x")
    edited = [{**DATASET[0], "question": "otra pregunta"}, DATASET[1]]
    b = evaluate(tmp_path / "two", "b", "x", dataset=edited)
    result = runner.invoke(app, ["compare", str(a), str(b)])
    assert result.exit_code == 1
    assert "different datasets" in result.output
