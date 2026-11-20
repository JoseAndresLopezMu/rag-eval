from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rag_eval.models import Context, Evidence

Covers = Callable[[Context, Evidence], bool]


@dataclass(frozen=True)
class RetrievalScores:
    hit: float
    recall: float
    precision: float
    mrr: float


def retrieval_scores(
    evidence: Sequence[Evidence],
    contexts: Sequence[Context],
    k: int,
    covers: Covers,
) -> RetrievalScores:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not evidence:
        raise ValueError("retrieval metrics need at least one evidence item")

    top = sorted(contexts, key=lambda c: c.rank)[:k]
    matches = [[covers(c, e) for e in evidence] for c in top]

    relevant_ranks = [i for i, row in enumerate(matches, start=1) if any(row)]
    covered = sum(1 for j in range(len(evidence)) if any(row[j] for row in matches))

    return RetrievalScores(
        hit=1.0 if relevant_ranks else 0.0,
        recall=covered / len(evidence),
        # Divided by the contexts actually returned, not by k: a system that
        # returns 3 good contexts when asked for 5 is not penalised for brevity.
        precision=len(relevant_ranks) / len(top) if top else 0.0,
        mrr=1.0 / relevant_ranks[0] if relevant_ranks else 0.0,
    )
