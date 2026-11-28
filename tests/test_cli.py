import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from rag_eval import __version__
from rag_eval.cli import app
from rag_eval.judge.fake import FakeJudge
from rag_eval.models import Report, RunItem

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


def judge_args(tmp_path: Path) -> list[str]:
    dataset = write_jsonl(tmp_path / "dataset.jsonl", DATASET)
    run = write_jsonl(tmp_path / "run.jsonl", run_with("servicio de recuperacion de datos"))
    return [
        "evaluate",
        "--dataset",
        str(dataset),
        "--run",
        str(run),
        "--out",
        str(tmp_path / "r.json"),
        "--judge",
        "--cache-dir",
        str(tmp_path / "cache"),
    ]


def test_judge_requires_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    result = runner.invoke(app, judge_args(tmp_path))
    assert result.exit_code == 1
    assert "GROQ_API_KEY" in result.output


def test_malformed_api_key_gives_a_clean_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_aquí_tu_clave_completa")
    result = runner.invoke(app, judge_args(tmp_path))
    assert result.exit_code == 1
    assert "error: GROQ_API_KEY contains spaces or non-ASCII characters" in result.output
    assert "Traceback" not in result.output


def test_judge_scores_are_added_and_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeJudge(
        {
            "faithfulness": lambda p: {"claims": [{"claim": "c", "supported": True}]},
            "correctness": lambda p: {"correct": True, "rationale": "r"},
            "abstention": lambda p: {"abstained": True, "rationale": "r"},
        },
        model="fake-model",
    )
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr("rag_eval.cli.GroqJudge", lambda key, model: fake)

    first = runner.invoke(app, judge_args(tmp_path))
    assert first.exit_code == 0, first.output
    assert "Judge fake-model: 4 calls, 0 from cache" in first.output

    second = runner.invoke(app, judge_args(tmp_path))
    assert "Judge fake-model: 0 calls, 4 from cache" in second.output

    report = Report.model_validate_json((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert report.aggregates["correctness"].mean == 1.0
    assert report.judge is not None
    assert report.judge.model == "fake-model"


def test_check_quotes_passes_and_fails(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "pliego.pdf.txt").write_text(
        "• Servicio de recuperación de datos.\n• Soporte in situ a usuarios.", encoding="utf-8"
    )
    good: list[dict[str, object]] = [
        {
            "id": "q1",
            "question": "?",
            "reference_answer": "r",
            "evidence": [
                {
                    "source": "pliego.pdf",
                    "quote": "Servicio de recuperación de datos. Soporte in situ a usuarios",
                }
            ],
        }
    ]
    dataset = write_jsonl(tmp_path / "good.jsonl", good)
    args = ["check-quotes", "--dataset", str(dataset), "--sources-dir", str(tmp_path / "docs")]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "All 1 quotes are verbatim" in result.output

    write_jsonl(tmp_path / "good.jsonl", DATASET)
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "q1: too short (5 words, minimum 8)" in result.output


def test_collect_writes_run_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class StubClient:
        def __init__(self, url: str) -> None:
            assert url == "https://bot.example/api"

        def collect(self, items: object, on_progress: Any) -> list[RunItem]:
            runs = [RunItem(id="q1", answer="a"), RunItem(id="q2", answer="b", latency_ms=5)]
            for r in runs:
                on_progress(r)
            return runs

    monkeypatch.setattr("rag_eval.cli.ChatbotRagClient", StubClient)
    dataset = write_jsonl(tmp_path / "dataset.jsonl", DATASET)
    out = tmp_path / "runs" / "hybrid.jsonl"
    result = runner.invoke(
        app,
        [
            "collect",
            "--dataset",
            str(dataset),
            "--url",
            "https://bot.example/api",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "q2: 0 contexts, 5 ms" in result.output
    assert [json.loads(line)["id"] for line in out.read_text().splitlines()] == ["q1", "q2"]
