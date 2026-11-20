import random
from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_RESAMPLES = 2000
DEFAULT_SEED = 0


@dataclass(frozen=True)
class Estimate:
    mean: float
    ci_low: float
    ci_high: float
    n: int


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean of an empty sequence")
    return sum(values) / len(values)


def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Estimate:
    """Mean with a percentile bootstrap confidence interval.

    The seed is fixed so the same input always produces the same report.
    """
    if not values:
        raise ValueError("bootstrap of an empty sequence")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    rng = random.Random(seed)
    n = len(values)
    means = sorted(mean(rng.choices(values, k=n)) for _ in range(resamples))
    tail = (1 - confidence) / 2
    return Estimate(
        mean=mean(values),
        ci_low=_percentile(means, tail),
        ci_high=_percentile(means, 1 - tail),
        n=n,
    )


def paired_bootstrap_diff(
    baseline: Sequence[float],
    candidate: Sequence[float],
    confidence: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Estimate:
    """Interval for mean(candidate - baseline), resampling questions, not scores.

    Pairing keeps each question's two scores together, so the interval reflects
    how consistently the candidate beats the baseline on the same questions.
    """
    if len(baseline) != len(candidate):
        raise ValueError(f"paired samples differ in length: {len(baseline)} vs {len(candidate)}")
    diffs = [c - b for b, c in zip(baseline, candidate, strict=True)]
    return bootstrap_ci(diffs, confidence=confidence, resamples=resamples, seed=seed)


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    # Linear interpolation between closest ranks, same as numpy's default.
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)
