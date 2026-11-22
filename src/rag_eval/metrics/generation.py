from collections.abc import Sequence
from dataclasses import dataclass

from rag_eval.judge import prompts
from rag_eval.judge.base import (
    AbstentionJudgement,
    CorrectnessJudgement,
    FaithfulnessJudgement,
    Judge,
    Judgement,
)


@dataclass(frozen=True)
class Scored[J: Judgement]:
    # None when the metric does not apply, e.g. faithfulness of an answer
    # with no factual claims. The judgement is kept for auditing either way.
    score: float | None
    judgement: J


def faithfulness(
    judge: Judge, answer: str, contexts: Sequence[str]
) -> Scored[FaithfulnessJudgement]:
    j = judge.ask(prompts.faithfulness(answer, contexts), FaithfulnessJudgement)
    if not j.claims:
        return Scored(None, j)
    supported = sum(1 for c in j.claims if c.supported)
    return Scored(supported / len(j.claims), j)


def correctness(
    judge: Judge, question: str, answer: str, reference: str
) -> Scored[CorrectnessJudgement]:
    j = judge.ask(prompts.correctness(question, answer, reference), CorrectnessJudgement)
    return Scored(1.0 if j.correct else 0.0, j)


def abstention(judge: Judge, question: str, answer: str) -> Scored[AbstentionJudgement]:
    j = judge.ask(prompts.abstention(question, answer), AbstentionJudgement)
    return Scored(1.0 if j.abstained else 0.0, j)
