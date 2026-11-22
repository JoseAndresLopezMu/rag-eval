from pathlib import Path

from rag_eval.judge import prompts
from rag_eval.judge.base import AbstentionJudgement, CorrectnessJudgement, Prompt
from rag_eval.judge.cache import CachedJudge
from rag_eval.judge.fake import FakeJudge


def counting_judge(model: str = "m1") -> FakeJudge:
    return FakeJudge({"correctness": lambda p: {"correct": True, "rationale": p.user[-1]}}, model)


def test_second_call_is_served_from_disk(tmp_path: Path) -> None:
    inner = counting_judge()
    judge = CachedJudge(inner, tmp_path)
    prompt = prompts.correctness("q", "a", "r")

    first = judge.ask(prompt, CorrectnessJudgement)
    second = judge.ask(prompt, CorrectnessJudgement)

    assert first == second
    assert len(inner.calls) == 1
    assert (judge.hits, judge.misses) == (1, 1)


def test_cache_survives_a_new_instance(tmp_path: Path) -> None:
    prompt = prompts.correctness("q", "a", "r")
    CachedJudge(counting_judge(), tmp_path).ask(prompt, CorrectnessJudgement)

    inner = counting_judge()
    CachedJudge(inner, tmp_path).ask(prompt, CorrectnessJudgement)
    assert inner.calls == []


def test_different_prompt_model_or_version_misses(tmp_path: Path) -> None:
    base = prompts.correctness("q", "a", "r")
    CachedJudge(counting_judge("m1"), tmp_path).ask(base, CorrectnessJudgement)

    other_model = counting_judge("m2")
    CachedJudge(other_model, tmp_path).ask(base, CorrectnessJudgement)
    assert len(other_model.calls) == 1

    inner = counting_judge("m1")
    judge = CachedJudge(inner, tmp_path)
    judge.ask(prompts.correctness("q", "other answer", "r"), CorrectnessJudgement)
    judge.ask(
        Prompt(base.name, "2", base.system, base.user),
        CorrectnessJudgement,
    )
    assert len(inner.calls) == 2


def test_corrupt_entry_is_replaced(tmp_path: Path) -> None:
    prompt = prompts.correctness("q", "a", "r")
    CachedJudge(counting_judge(), tmp_path).ask(prompt, CorrectnessJudgement)
    (entry,) = tmp_path.rglob("*.json")
    entry.write_text("{not json", encoding="utf-8")

    inner = counting_judge()
    result = CachedJudge(inner, tmp_path).ask(prompt, CorrectnessJudgement)
    assert result.correct is True
    assert len(inner.calls) == 1
    assert CorrectnessJudgement.model_validate_json(entry.read_text(encoding="utf-8"))


def test_schema_is_part_of_the_key(tmp_path: Path) -> None:
    inner = FakeJudge(
        {"correctness": lambda _: {"correct": True, "abstained": False, "rationale": "r"}}
    )
    judge = CachedJudge(inner, tmp_path)
    prompt = prompts.correctness("q", "a", "r")
    judge.ask(prompt, CorrectnessJudgement)
    judge.ask(prompt, AbstentionJudgement)
    assert len(inner.calls) == 2
