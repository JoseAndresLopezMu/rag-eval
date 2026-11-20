from dataclasses import astuple

import pytest

from rag_eval.stats import Estimate, bootstrap_ci, mean, paired_bootstrap_diff


def test_mean() -> None:
    assert mean([1.0, 0.0, 0.5]) == 0.5


def test_mean_of_empty_sequence_fails() -> None:
    with pytest.raises(ValueError, match="empty"):
        mean([])


class TestBootstrapCI:
    def test_constant_values_give_a_zero_width_interval(self) -> None:
        assert astuple(bootstrap_ci([0.7] * 10)) == pytest.approx((0.7, 0.7, 0.7, 10))

    def test_single_value(self) -> None:
        assert bootstrap_ci([1.0]) == Estimate(1.0, 1.0, 1.0, 1)

    def test_interval_contains_the_mean(self) -> None:
        values = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
        est = bootstrap_ci(values)
        assert est.ci_low < est.mean < est.ci_high
        assert est.mean == 0.7

    def test_is_deterministic_for_a_given_seed(self) -> None:
        values = [0.1, 0.9, 0.4, 0.6, 0.3]
        assert bootstrap_ci(values) == bootstrap_ci(values)
        # With few values two seeds can coincide; across several they should not all.
        assert len({bootstrap_ci(values, seed=s) for s in range(5)}) > 1

    def test_more_data_gives_a_narrower_interval(self) -> None:
        small = bootstrap_ci([0.0, 1.0] * 5)
        large = bootstrap_ci([0.0, 1.0] * 50)
        assert large.ci_high - large.ci_low < small.ci_high - small.ci_low

    def test_binary_interval_matches_textbook_width(self) -> None:
        # p = 0.5, n = 100: normal approximation gives 0.5 +- 1.96 * 0.05.
        est = bootstrap_ci([0.0, 1.0] * 50)
        assert est.ci_low == pytest.approx(0.402, abs=0.02)
        assert est.ci_high == pytest.approx(0.598, abs=0.02)

    def test_rejects_empty_input_and_bad_confidence(self) -> None:
        with pytest.raises(ValueError):
            bootstrap_ci([])
        with pytest.raises(ValueError, match="confidence"):
            bootstrap_ci([1.0], confidence=1.0)


class TestPairedBootstrapDiff:
    def test_consistent_improvement(self) -> None:
        baseline = [0.0, 0.5, 0.2, 0.4]
        candidate = [b + 0.5 for b in baseline]
        assert astuple(paired_bootstrap_diff(baseline, candidate)) == pytest.approx(
            (0.5, 0.5, 0.5, 4)
        )

    def test_mixed_results_are_inconclusive(self) -> None:
        baseline = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        candidate = [0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
        est = paired_bootstrap_diff(baseline, candidate)
        assert est.mean == 0.0
        assert est.ci_low < 0 < est.ci_high

    def test_pairing_detects_small_consistent_gains(self) -> None:
        # Scores vary a lot between questions, but the candidate is always 0.1
        # better on the same question. Unpaired intervals would overlap.
        baseline = [0.0, 0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6]
        candidate = [b + 0.1 for b in baseline]
        b_ci, c_ci = bootstrap_ci(baseline), bootstrap_ci(candidate)
        assert c_ci.ci_low < b_ci.ci_high
        assert paired_bootstrap_diff(baseline, candidate).ci_low > 0

    def test_lengths_must_match(self) -> None:
        with pytest.raises(ValueError, match="differ in length"):
            paired_bootstrap_diff([1.0], [1.0, 0.0])
