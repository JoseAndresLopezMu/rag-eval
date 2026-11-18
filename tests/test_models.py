import pytest
from pydantic import ValidationError

from rag_eval.models import Context, DatasetItem, Evidence, RunItem

EVIDENCE = {"source": "pliego.pdf", "page": 7, "quote": "certificación ISO/IEC 27001"}


def make_item(**overrides: object) -> DatasetItem:
    data: dict[str, object] = {
        "id": "q001",
        "question": "¿Qué certificaciones exige el pliego?",
        "reference_answer": "ISO 27001.",
        "evidence": [EVIDENCE],
    }
    data.update(overrides)
    return DatasetItem.model_validate(data)


class TestDatasetItem:
    def test_valid_answerable_item(self) -> None:
        item = make_item()
        assert item.answerable is True
        assert item.evidence == (Evidence(source="pliego.pdf", page=7, quote=EVIDENCE["quote"]),)

    def test_valid_unanswerable_item(self) -> None:
        item = make_item(answerable=False, evidence=[])
        assert item.evidence == ()

    def test_answerable_without_evidence_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at least one evidence"):
            make_item(evidence=[])

    def test_unanswerable_with_evidence_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not have evidence"):
            make_item(answerable=False)

    def test_unknown_field_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="refrence_answer"):
            make_item(refrence_answer="typo")

    @pytest.mark.parametrize("field", ["id", "question", "reference_answer"])
    def test_blank_strings_are_rejected(self, field: str) -> None:
        with pytest.raises(ValidationError):
            make_item(**{field: "   "})


class TestEvidence:
    def test_page_is_optional(self) -> None:
        # DOCX files have no pages; matching relies on source and quote only.
        evidence = Evidence(source="oferta.docx", quote="SLA de 4 horas")
        assert evidence.page is None

    def test_page_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            Evidence(source="pliego.pdf", page=0, quote="texto")


class TestRunItem:
    def test_contexts_with_consecutive_ranks(self) -> None:
        run = RunItem(
            id="q001",
            answer="ISO 27001",
            contexts=(
                Context(rank=1, source="a.pdf", text="uno"),
                Context(rank=2, source="b.pdf", text="dos"),
            ),
        )
        assert [c.rank for c in run.contexts] == [1, 2]

    def test_empty_contexts_are_allowed(self) -> None:
        assert RunItem(id="q001", answer="No tengo esa información.").contexts == ()

    @pytest.mark.parametrize("ranks", [[2, 1], [1, 3], [1, 1], [0]])
    def test_bad_ranks_are_rejected(self, ranks: list[int]) -> None:
        contexts = [{"rank": r, "source": "a.pdf", "text": "x"} for r in ranks]
        with pytest.raises(ValidationError):
            RunItem.model_validate({"id": "q001", "answer": "x", "contexts": contexts})

    def test_negative_latency_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RunItem(id="q001", answer="x", latency_ms=-1)
