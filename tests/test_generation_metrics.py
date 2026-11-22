from typing import Any

import pytest

from rag_eval.judge.base import Prompt
from rag_eval.judge.fake import FakeJudge
from rag_eval.metrics.generation import abstention, correctness, faithfulness


def claims(*supported: bool) -> dict[str, Any]:
    return {"claims": [{"claim": f"c{i}", "supported": s} for i, s in enumerate(supported)]}


def judge_returning(name: str, payload: dict[str, Any]) -> FakeJudge:
    return FakeJudge({name: lambda _: payload})


class TestFaithfulness:
    def test_share_of_supported_claims(self) -> None:
        judge = judge_returning("faithfulness", claims(True, True, False, True))
        assert faithfulness(judge, "a", ["ctx"]).score == 0.75

    def test_fully_supported(self) -> None:
        judge = judge_returning("faithfulness", claims(True))
        assert faithfulness(judge, "a", ["ctx"]).score == 1.0

    def test_answer_without_claims_is_not_scored(self) -> None:
        judge = judge_returning("faithfulness", claims())
        result = faithfulness(judge, "No tengo esa información.", ["ctx"])
        assert result.score is None
        assert result.judgement.claims == []

    def test_sends_answer_and_contexts_to_the_judge(self) -> None:
        seen: list[Prompt] = []

        def handler(p: Prompt) -> dict[str, Any]:
            seen.append(p)
            return claims(True)

        faithfulness(FakeJudge({"faithfulness": handler}), "la respuesta", ["uno", "dos"])
        assert "[2] dos" in seen[0].user
        assert "la respuesta" in seen[0].user


@pytest.mark.parametrize(("correct", "expected"), [(True, 1.0), (False, 0.0)])
def test_correctness(correct: bool, expected: float) -> None:
    judge = judge_returning("correctness", {"correct": correct, "rationale": "r"})
    result = correctness(judge, "q", "a", "ref")
    assert result.score == expected
    assert result.judgement.rationale == "r"


@pytest.mark.parametrize(("abstained", "expected"), [(True, 1.0), (False, 0.0)])
def test_abstention(abstained: bool, expected: float) -> None:
    judge = judge_returning("abstention", {"abstained": abstained, "rationale": "r"})
    assert abstention(judge, "q", "a").score == expected


def test_malformed_judge_output_fails_validation() -> None:
    judge = judge_returning("correctness", {"verdict": "yes"})
    with pytest.raises(ValueError, match="correct"):
        correctness(judge, "q", "a", "ref")
