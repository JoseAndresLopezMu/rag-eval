import zipfile
from pathlib import Path

from rag_eval.adapters.sources import load_sources
from rag_eval.models import DatasetItem, Evidence
from rag_eval.quotes import Issue, check_quotes

# Verbatim excerpt from pliego_tecnico_madrimasd_S_2026_003.pdf, page 7.
PLIEGO = (
    "seguir accesible el sistema de bloqueos que ofrece CIFS.\n"
    "• Servicio de recuperación de datos.\n"
    "• Soporte in situ a usuarios."
)


def item(*quotes: str, source: str = "pliego.pdf") -> DatasetItem:
    return DatasetItem(
        id="q1",
        question="?",
        reference_answer="r",
        evidence=tuple(Evidence(source=source, quote=q) for q in quotes),
    )


def test_verbatim_quote_across_line_breaks_and_bullets_passes() -> None:
    quote = "Servicio de recuperación de datos. Soporte in situ a usuarios"
    assert check_quotes([item(quote)], {"pliego.pdf": PLIEGO}) == []


def test_paraphrased_quote_is_reported_with_best_coverage() -> None:
    quote = "Servicio de recuperación de la información y soporte in situ a usuarios"
    (problem,) = check_quotes([item(quote)], {"pliego.pdf": PLIEGO})
    assert problem.issue == Issue.NOT_VERBATIM
    # 12 quote words; the longest run in the source is "soporte in situ a usuarios" (5).
    assert problem.detail == f"longest matching run covers {5 / 12:.0%}"


def test_short_quote_is_reported() -> None:
    (problem,) = check_quotes([item("Servicio de recuperación de datos")], {"pliego.pdf": PLIEGO})
    assert problem.issue == Issue.TOO_SHORT
    assert problem.detail == "5 words, minimum 8"


def test_missing_source_is_reported() -> None:
    (problem,) = check_quotes([item("x " * 8, source="otra.pdf")], {"pliego.pdf": PLIEGO})
    assert problem.issue == Issue.MISSING_SOURCE


def test_quote_must_match_whole_words() -> None:
    # "datos soporte" exists, but not as the start of "atos soporte".
    quote = "atos. Soporte in situ a usuarios y algo más"
    (problem,) = check_quotes([item(quote)], {"pliego.pdf": PLIEGO})
    assert problem.issue == Issue.NOT_VERBATIM


def test_unanswerable_questions_have_nothing_to_check() -> None:
    unanswerable = DatasetItem(id="q2", question="?", reference_answer="No.", answerable=False)
    assert check_quotes([unanswerable], {}) == []


def write_docx(path: Path, *paragraphs: str) -> None:
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    xml = f'<w:document xmlns:w="{ns}"><w:body>{body}</w:body></w:document>'
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", xml)


def test_load_sources_reads_txt_next_to_pdf_and_docx(tmp_path: Path) -> None:
    (tmp_path / "pliego.pdf").write_bytes(b"%PDF-1.7 binary")
    (tmp_path / "pliego.pdf.txt").write_text(PLIEGO, encoding="utf-8")
    write_docx(tmp_path / "oferta.docx", "SLA de 4 horas", "para incidencias críticas")

    texts = load_sources(tmp_path, {"pliego.pdf", "oferta.docx", "falta.pdf"})

    assert texts["pliego.pdf"] == PLIEGO
    assert texts["oferta.docx"] == "SLA de 4 horas\npara incidencias críticas"
    assert "falta.pdf" not in texts


def test_pdf_without_extracted_text_is_not_loaded(tmp_path: Path) -> None:
    (tmp_path / "pliego.pdf").write_bytes(b"%PDF-1.7 binary")
    assert load_sources(tmp_path, {"pliego.pdf"}) == {}
