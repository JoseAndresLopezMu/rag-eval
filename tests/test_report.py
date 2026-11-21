import pytest

from rag_eval.models import Context, DatasetItem, EvalConfig, Evidence, Report, RunItem
from rag_eval.report import (
    ReportError,
    Verdict,
    build_report,
    compare_reports,
    render_comparison,
)

SOURCE = "pliego.pdf"
CONFIG = EvalConfig(k=2, resamples=200)


def question(qid: str, quote: str | None) -> DatasetItem:
    if quote is None:
        return DatasetItem(id=qid, question="?", reference_answer="No consta.", answerable=False)
    return DatasetItem(
        id=qid,
        question="?",
        reference_answer="r",
        evidence=(Evidence(source=SOURCE, quote=quote),),
    )


def answer(qid: str, *texts: str) -> RunItem:
    contexts = tuple(Context(rank=i, source=SOURCE, text=t) for i, t in enumerate(texts, start=1))
    return RunItem(id=qid, answer="a", contexts=contexts)


DATASET = (
    question("q1", "servicio de recuperacion de datos"),
    question("q2", "soporte in situ a usuarios"),
    question("q3", None),
)


def report(run: tuple[RunItem, ...], config: EvalConfig = CONFIG, sha: str = "d" * 64) -> Report:
    return build_report(DATASET, run, config, dataset_sha256=sha, run_sha256="r" * 64)


GOOD_RUN = (
    answer("q1", "servicio de recuperacion de datos", "otro"),
    answer("q2", "nada", "soporte in situ a usuarios"),
    answer("q3"),
)
BAD_RUN = (
    answer("q1", "otro", "nada"),
    answer("q2", "nada", "otro"),
    answer("q3"),
)


class TestBuildReport:
    def test_per_question_scores(self) -> None:
        r = report(GOOD_RUN)
        q1, q2, q3 = r.questions
        assert q1.scores == {"hit@2": 1.0, "recall@2": 1.0, "precision@2": 0.5, "mrr@2": 1.0}
        assert q2.scores["mrr@2"] == 0.5
        assert q3.scores == {}
        assert q3.answerable is False

    def test_aggregates_skip_unanswerable_questions(self) -> None:
        agg = report(GOOD_RUN).aggregates
        assert agg["mrr@2"].mean == 0.75
        assert agg["mrr@2"].n == 2
        assert list(agg) == ["hit@2", "recall@2", "precision@2", "mrr@2"]

    def test_records_config_and_provenance(self) -> None:
        r = report(GOOD_RUN)
        assert r.config == CONFIG
        assert r.dataset_sha256 == "d" * 64
        assert r.run_sha256 == "r" * 64

    def test_is_reproducible(self) -> None:
        assert report(GOOD_RUN) == report(GOOD_RUN)

    def test_run_order_does_not_matter(self) -> None:
        assert report(GOOD_RUN).questions == report(GOOD_RUN[::-1]).questions

    def test_missing_answers_are_an_error(self) -> None:
        with pytest.raises(ReportError, match="missing from run: q3"):
            report(GOOD_RUN[:2])

    def test_unknown_answers_are_an_error(self) -> None:
        with pytest.raises(ReportError, match="not in dataset: q9"):
            report((*GOOD_RUN, answer("q9")))

    def test_threshold_is_applied(self) -> None:
        half = (answer("q1", "servicio de recuperacion"), answer("q2"), answer("q3"))
        lenient = report(half, EvalConfig(k=2, threshold=0.4, resamples=200))
        # 3 of 5 quote tokens = 0.6: passes at 0.4, fails at 0.7.
        strict = report(half, EvalConfig(k=2, threshold=0.7, resamples=200))
        assert lenient.questions[0].scores["hit@2"] == 1.0
        assert strict.questions[0].scores["hit@2"] == 0.0

    def test_survives_json_round_trip(self) -> None:
        r = report(GOOD_RUN)
        assert Report.model_validate_json(r.model_dump_json()) == r


class TestCompareReports:
    def test_identical_runs_are_inconclusive_with_zero_difference(self) -> None:
        rows = compare_reports(report(GOOD_RUN), report(GOOD_RUN))
        assert {r.verdict for r in rows} == {Verdict.INCONCLUSIVE}
        assert all(r.diff.mean == 0 for r in rows)

    def test_consistently_better_candidate(self) -> None:
        rows = {r.metric: r for r in compare_reports(report(BAD_RUN), report(GOOD_RUN))}
        assert rows["hit@2"].verdict == Verdict.BETTER
        assert rows["hit@2"].diff.mean == 1.0

    def test_consistently_worse_candidate(self) -> None:
        rows = {r.metric: r for r in compare_reports(report(GOOD_RUN), report(BAD_RUN))}
        assert rows["recall@2"].verdict == Verdict.WORSE

    def test_mixed_results_are_inconclusive(self) -> None:
        mixed = (answer("q1", "servicio de recuperacion de datos"), answer("q2"), answer("q3"))
        flipped = (answer("q1"), answer("q2", "soporte in situ a usuarios"), answer("q3"))
        rows = {r.metric: r for r in compare_reports(report(mixed), report(flipped))}
        assert rows["hit@2"].verdict == Verdict.INCONCLUSIVE

    def test_different_datasets_are_not_comparable(self) -> None:
        with pytest.raises(ReportError, match="different datasets"):
            compare_reports(report(GOOD_RUN), report(GOOD_RUN, sha="x" * 64))

    def test_different_k_is_not_comparable(self) -> None:
        other = report(GOOD_RUN, EvalConfig(k=3, resamples=200))
        with pytest.raises(ReportError, match="different k: 2 vs 3"):
            compare_reports(report(GOOD_RUN), other)


def test_render_comparison_markdown() -> None:
    base, cand = report(BAD_RUN), report(GOOD_RUN)
    md = render_comparison(compare_reports(base, cand), "semantic", "hybrid", base)
    assert md.startswith("# hybrid vs semantic\n")
    assert "| Metric | semantic | hybrid | Difference | Verdict |" in md
    assert "| hit@2 | 0.00 [0.00, 0.00] (n=2) | 1.00 [1.00, 1.00] (n=2) " in md
    assert "+1.00 [+1.00, +1.00] (n=2) | better |" in md
