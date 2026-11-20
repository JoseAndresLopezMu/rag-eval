import pytest

from rag_eval.metrics.retrieval import RetrievalScores, retrieval_scores
from rag_eval.models import Context, Evidence

# The relevance rule is tested on its own; here a context "covers" an evidence
# item when its text contains the evidence quote verbatim.
E1 = Evidence(source="pliego.pdf", quote="E1")
E2 = Evidence(source="pliego.pdf", quote="E2")


def covers(context: Context, evidence: Evidence) -> bool:
    return evidence.quote in context.text


def ctx(*texts: str) -> list[Context]:
    return [Context(rank=i, source="pliego.pdf", text=t) for i, t in enumerate(texts, start=1)]


def test_first_context_relevant() -> None:
    scores = retrieval_scores([E1], ctx("E1", "x", "y"), k=3, covers=covers)
    assert scores == RetrievalScores(hit=1.0, recall=1.0, precision=1 / 3, mrr=1.0)


def test_relevant_context_at_rank_three() -> None:
    scores = retrieval_scores([E1], ctx("x", "y", "E1"), k=5, covers=covers)
    assert scores.mrr == pytest.approx(1 / 3)
    assert scores.hit == 1.0


def test_relevant_context_beyond_k_does_not_count() -> None:
    scores = retrieval_scores([E1], ctx("x", "y", "E1"), k=2, covers=covers)
    assert scores == RetrievalScores(hit=0.0, recall=0.0, precision=0.0, mrr=0.0)


def test_partial_recall_with_two_evidence_items() -> None:
    scores = retrieval_scores([E1, E2], ctx("x", "E2"), k=5, covers=covers)
    assert scores.recall == 0.5
    assert scores.mrr == 0.5


def test_one_context_covering_both_evidence_items() -> None:
    scores = retrieval_scores([E1, E2], ctx("E1 E2", "x"), k=2, covers=covers)
    assert scores.recall == 1.0
    assert scores.precision == 0.5


def test_two_contexts_covering_same_evidence_count_once_for_recall() -> None:
    scores = retrieval_scores([E1, E2], ctx("E1", "E1 again"), k=2, covers=covers)
    assert scores.recall == 0.5
    assert scores.precision == 1.0


def test_no_contexts() -> None:
    scores = retrieval_scores([E1], [], k=5, covers=covers)
    assert scores == RetrievalScores(hit=0.0, recall=0.0, precision=0.0, mrr=0.0)


def test_precision_uses_returned_contexts_when_fewer_than_k() -> None:
    scores = retrieval_scores([E1], ctx("E1", "x"), k=5, covers=covers)
    assert scores.precision == 0.5


def test_contexts_are_ordered_by_rank_not_by_position() -> None:
    contexts = [
        Context(rank=2, source="pliego.pdf", text="E1"),
        Context(rank=1, source="pliego.pdf", text="x"),
    ]
    assert retrieval_scores([E1], contexts, k=1, covers=covers).hit == 0.0


def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="k must be"):
        retrieval_scores([E1], ctx("E1"), k=0, covers=covers)


def test_evidence_is_required() -> None:
    with pytest.raises(ValueError, match="at least one evidence"):
        retrieval_scores([], ctx("E1"), k=1, covers=covers)
