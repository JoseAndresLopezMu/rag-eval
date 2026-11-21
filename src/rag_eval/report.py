from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import partial

from rag_eval import __version__
from rag_eval.metrics.retrieval import retrieval_scores
from rag_eval.models import (
    DatasetItem,
    EvalConfig,
    MetricEstimate,
    QuestionResult,
    Report,
    RunItem,
)
from rag_eval.relevance import covers
from rag_eval.stats import bootstrap_ci, paired_bootstrap_diff


class ReportError(ValueError):
    pass


def build_report(
    dataset: Sequence[DatasetItem],
    run: Sequence[RunItem],
    config: EvalConfig,
    dataset_sha256: str,
    run_sha256: str,
    git_commit: str | None = None,
) -> Report:
    runs_by_id = {r.id: r for r in run}
    _check_same_ids([d.id for d in dataset], list(runs_by_id))

    covers_at = partial(covers, threshold=config.threshold)
    questions = []
    for item in dataset:
        scores: dict[str, float] = {}
        if item.answerable:
            r = retrieval_scores(item.evidence, runs_by_id[item.id].contexts, config.k, covers_at)
            k = config.k
            scores = {
                f"hit@{k}": r.hit,
                f"recall@{k}": r.recall,
                f"precision@{k}": r.precision,
                f"mrr@{k}": r.mrr,
            }
        questions.append(QuestionResult(id=item.id, answerable=item.answerable, scores=scores))

    return Report(
        rag_eval_version=__version__,
        git_commit=git_commit,
        dataset_sha256=dataset_sha256,
        run_sha256=run_sha256,
        config=config,
        aggregates=_aggregate(questions, config),
        questions=tuple(questions),
    )


def _check_same_ids(dataset_ids: list[str], run_ids: list[str]) -> None:
    missing = [i for i in dataset_ids if i not in set(run_ids)]
    extra = [i for i in run_ids if i not in set(dataset_ids)]
    problems = []
    if missing:
        problems.append(f"missing from run: {_preview(missing)}")
    if extra:
        problems.append(f"not in dataset: {_preview(extra)}")
    if problems:
        raise ReportError("run does not match dataset; " + "; ".join(problems))


def _preview(ids: list[str], limit: int = 5) -> str:
    shown = ", ".join(ids[:limit])
    return f"{shown} (+{len(ids) - limit} more)" if len(ids) > limit else shown


def _metric_names(questions: Sequence[QuestionResult]) -> list[str]:
    names: dict[str, None] = {}
    for q in questions:
        names.update(dict.fromkeys(q.scores))
    return list(names)


def _aggregate(
    questions: Sequence[QuestionResult], config: EvalConfig
) -> dict[str, MetricEstimate]:
    result = {}
    for name in _metric_names(questions):
        values = [q.scores[name] for q in questions if name in q.scores]
        est = bootstrap_ci(values, config.confidence, config.resamples, config.seed)
        result[name] = MetricEstimate(
            mean=est.mean, ci_low=est.ci_low, ci_high=est.ci_high, n=est.n
        )
    return result


class Verdict(StrEnum):
    BETTER = "better"
    WORSE = "worse"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class MetricComparison:
    metric: str
    baseline: MetricEstimate
    candidate: MetricEstimate
    diff: MetricEstimate
    verdict: Verdict


def compare_reports(baseline: Report, candidate: Report) -> list[MetricComparison]:
    _check_comparable(baseline, candidate)
    config = baseline.config
    base_q = {q.id: q for q in baseline.questions}
    cand_q = {q.id: q for q in candidate.questions}

    rows = []
    for name in baseline.aggregates:
        if name not in candidate.aggregates:
            continue
        ids = [i for i, q in base_q.items() if name in q.scores and name in cand_q[i].scores]
        est = paired_bootstrap_diff(
            [base_q[i].scores[name] for i in ids],
            [cand_q[i].scores[name] for i in ids],
            config.confidence,
            config.resamples,
            config.seed,
        )
        diff = MetricEstimate(mean=est.mean, ci_low=est.ci_low, ci_high=est.ci_high, n=est.n)
        rows.append(
            MetricComparison(
                metric=name,
                baseline=baseline.aggregates[name],
                candidate=candidate.aggregates[name],
                diff=diff,
                verdict=_verdict(diff),
            )
        )
    return rows


def _check_comparable(a: Report, b: Report) -> None:
    if a.dataset_sha256 != b.dataset_sha256:
        raise ReportError("reports were built from different datasets")
    for field in ("k", "threshold", "confidence", "resamples", "seed"):
        va, vb = getattr(a.config, field), getattr(b.config, field)
        if va != vb:
            raise ReportError(f"reports use different {field}: {va} vs {vb}")
    if {q.id for q in a.questions} != {q.id for q in b.questions}:
        raise ReportError("reports cover different questions")


def _verdict(diff: MetricEstimate) -> Verdict:
    if diff.ci_low > 0:
        return Verdict.BETTER
    if diff.ci_high < 0:
        return Verdict.WORSE
    return Verdict.INCONCLUSIVE


def render_comparison(
    rows: Sequence[MetricComparison],
    baseline_name: str,
    candidate_name: str,
    report: Report,
) -> str:
    c = report.config
    lines = [
        f"# {candidate_name} vs {baseline_name}",
        "",
        f"Dataset `{report.dataset_sha256[:12]}`, k={c.k}, threshold={c.threshold}, "
        f"{c.confidence:.0%} bootstrap intervals ({c.resamples} resamples, seed {c.seed}).",
        "",
        f"| Metric | {baseline_name} | {candidate_name} | Difference | Verdict |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.metric} | {_fmt(r.baseline)} | {_fmt(r.candidate)} "
            f"| {_fmt(r.diff, signed=True)} | {r.verdict} |"
        )
    lines += [
        "",
        "A difference is `better` or `worse` only when its whole interval is above or "
        "below zero; otherwise it may be noise and is marked `inconclusive`.",
        "",
    ]
    return "\n".join(lines)


def _fmt(e: MetricEstimate, signed: bool = False) -> str:
    f = "{:+.2f}" if signed else "{:.2f}"
    return f"{f.format(e.mean)} [{f.format(e.ci_low)}, {f.format(e.ci_high)}] (n={e.n})"
