import pytest

from rag_eval.judge import prompts
from rag_eval.judge.base import CorrectnessJudgement, JudgeError, Prompt
from rag_eval.judge.fake import FakeJudge


def test_faithfulness_prompt_numbers_contexts() -> None:
    p = prompts.faithfulness("respuesta", ["primero", "segundo"])
    assert p.name == "faithfulness"
    assert p.version == prompts.PROMPT_VERSION
    assert "[1] primero\n\n[2] segundo" in p.user
    assert p.user.endswith("ANSWER:\nrespuesta")


def test_faithfulness_prompt_without_contexts() -> None:
    assert "(no context)" in prompts.faithfulness("x", []).user


def test_correctness_prompt_includes_reference() -> None:
    p = prompts.correctness("¿Pregunta?", "respuesta", "referencia")
    assert "REFERENCE ANSWER:\nreferencia" in p.user


@pytest.mark.parametrize(
    "prompt",
    [
        prompts.faithfulness("a", ["c"]),
        prompts.correctness("q", "a", "r"),
        prompts.abstention("q", "a"),
    ],
)
def test_prompts_ask_for_json(prompt: Prompt) -> None:
    assert "single JSON object" in prompt.system


def test_fake_judge_validates_and_records_calls() -> None:
    judge = FakeJudge({"correctness": lambda p: {"correct": True, "rationale": "ok"}})
    prompt = prompts.correctness("q", "a", "r")
    assert judge.ask(prompt, CorrectnessJudgement) == CorrectnessJudgement(
        correct=True, rationale="ok"
    )
    assert judge.calls == [prompt]


def test_fake_judge_without_handler_fails() -> None:
    with pytest.raises(JudgeError, match="no fake handler"):
        FakeJudge({}).ask(prompts.abstention("q", "a"), CorrectnessJudgement)
