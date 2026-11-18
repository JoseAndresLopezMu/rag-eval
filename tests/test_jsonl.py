import json
from pathlib import Path

import pytest

from rag_eval.adapters.jsonl import DataFileError, read_dataset, read_run

ITEM = {
    "id": "q001",
    "question": "¿Qué certificaciones exige el pliego?",
    "reference_answer": "ISO 27001.",
    "evidence": [{"source": "pliego.pdf", "page": 7, "quote": "ISO/IEC 27001"}],
}
UNANSWERABLE = {
    "id": "q002",
    "question": "¿Cuál es el presupuesto de marketing?",
    "reference_answer": "No aparece en los documentos.",
    "answerable": False,
}
RUN = {
    "id": "q001",
    "answer": "ISO 27001",
    "contexts": [{"rank": 1, "source": "pliego.pdf", "page": 7, "text": "... ISO/IEC 27001 ..."}],
    "latency_ms": 1200,
}


def write_lines(path: Path, *lines: object) -> Path:
    path.write_text(
        "\n".join(line if isinstance(line, str) else json.dumps(line) for line in lines),
        encoding="utf-8",
    )
    return path


def test_reads_dataset_in_file_order(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "d.jsonl", ITEM, UNANSWERABLE)
    items = read_dataset(path)
    assert [i.id for i in items] == ["q001", "q002"]
    assert items[1].answerable is False


def test_reads_run(tmp_path: Path) -> None:
    run = read_run(write_lines(tmp_path / "r.jsonl", RUN))
    assert run[0].contexts[0].source == "pliego.pdf"


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "d.jsonl", ITEM, "", "   ", UNANSWERABLE, "")
    assert len(read_dataset(path)) == 2


def test_error_reports_line_id_and_field(tmp_path: Path) -> None:
    broken = {**UNANSWERABLE, "answerable": "yes please"}
    path = write_lines(tmp_path / "d.jsonl", ITEM, broken)
    with pytest.raises(DataFileError) as exc:
        read_dataset(path)
    message = str(exc.value)
    assert "d.jsonl:2" in message
    assert "id=q002" in message
    assert "answerable" in message


def test_invalid_json_reports_line(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "d.jsonl", ITEM, '{"id": "q002",')
    with pytest.raises(DataFileError, match=r"d\.jsonl:2: invalid JSON"):
        read_dataset(path)


def test_non_object_line_is_rejected(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "d.jsonl", [1, 2])
    with pytest.raises(DataFileError, match="expected a JSON object"):
        read_dataset(path)


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "d.jsonl", ITEM, UNANSWERABLE, ITEM)
    with pytest.raises(DataFileError, match=r"d\.jsonl:3.*duplicate id, first seen on line 1"):
        read_dataset(path)


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "d.jsonl", "")
    with pytest.raises(DataFileError, match="no records"):
        read_dataset(path)
