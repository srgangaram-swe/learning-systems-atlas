"""Exact tie handling, invariants, and oracle checks for binary ROC metrics."""

import numpy as np
import pytest
from sklearn import metrics as sklearn_metrics

from learning_atlas.supervised.metrics import binary_roc_curve, roc_auc_score

pytestmark = pytest.mark.unit


def test_binary_roc_curve_matches_hand_computed_perfect_ranking() -> None:
    observed = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])

    false_positive_rates, true_positive_rates, thresholds = binary_roc_curve(observed, scores)

    np.testing.assert_array_equal(false_positive_rates, [0.0, 0.0, 0.0, 0.5, 1.0])
    np.testing.assert_array_equal(true_positive_rates, [0.0, 0.5, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(thresholds, [np.inf, 0.9, 0.8, 0.2, 0.1])
    assert roc_auc_score(observed, scores) == 1.0
    assert roc_auc_score(observed, -scores) == 0.0


def test_binary_roc_curve_groups_ties_without_row_order_bias() -> None:
    scores = np.full(6, 0.5)
    first_observed = np.asarray([0, 0, 0, 1, 1, 1])
    second_observed = first_observed[::-1].copy()

    first = binary_roc_curve(first_observed, scores)
    second = binary_roc_curve(second_observed, scores)

    np.testing.assert_array_equal(first[0], [0.0, 1.0])
    np.testing.assert_array_equal(first[1], [0.0, 1.0])
    np.testing.assert_array_equal(first[2], [np.inf, 0.5])
    for first_array, second_array in zip(first, second, strict=True):
        np.testing.assert_array_equal(second_array, first_array)
    assert roc_auc_score(first_observed, scores) == pytest.approx(0.5)
    assert roc_auc_score(second_observed, scores) == pytest.approx(0.5)


@pytest.mark.parametrize("seed", range(8))
def test_binary_roc_and_auc_match_sklearn_with_quantized_ties(seed: int) -> None:
    generator = np.random.default_rng(seed)
    observed = np.repeat([0, 1], 30)
    generator.shuffle(observed)
    scores = generator.integers(-3, 4, size=len(observed)).astype(np.float64)

    false_positive_rates, true_positive_rates, thresholds = binary_roc_curve(observed, scores)
    expected_fpr, expected_tpr, expected_thresholds = sklearn_metrics.roc_curve(
        observed, scores, drop_intermediate=False
    )

    np.testing.assert_allclose(false_positive_rates, expected_fpr, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(true_positive_rates, expected_tpr, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(thresholds[1:], expected_thresholds[1:], rtol=0.0, atol=0.0)
    assert roc_auc_score(observed, scores) == pytest.approx(
        sklearn_metrics.roc_auc_score(observed, scores), abs=1e-15
    )


@pytest.mark.parametrize("seed", range(6))
def test_auc_equals_pairwise_ranking_probability_with_half_credit_for_ties(seed: int) -> None:
    generator = np.random.default_rng(seed)
    negative_scores = generator.integers(-2, 3, size=12).astype(np.float64)
    positive_scores = generator.integers(-2, 3, size=9).astype(np.float64)
    observed = np.concatenate([np.zeros(len(negative_scores)), np.ones(len(positive_scores))])
    scores = np.concatenate([negative_scores, positive_scores])
    pairwise_difference = positive_scores[:, None] - negative_scores[None, :]
    expected = float(np.mean(pairwise_difference > 0.0) + 0.5 * np.mean(pairwise_difference == 0.0))

    assert roc_auc_score(observed, scores) == pytest.approx(expected, abs=1e-15)


def test_roc_invariants_hold_under_permutation_and_rank_preserving_transform() -> None:
    observed = np.asarray([0, 1, 0, 1, 1, 0, 0, 1])
    scores = np.asarray([-2.0, 0.5, -1.0, 2.0, 0.5, 0.0, -2.0, 3.0])
    permutation = np.asarray([7, 2, 5, 1, 4, 0, 6, 3])

    baseline_auc = roc_auc_score(observed, scores)
    permuted_auc = roc_auc_score(observed[permutation], scores[permutation])
    transformed_auc = roc_auc_score(observed, 7.0 + 3.0 * scores)
    inverted_auc = roc_auc_score(1 - observed, -scores)

    assert 0.0 <= baseline_auc <= 1.0
    assert permuted_auc == pytest.approx(baseline_auc)
    assert transformed_auc == pytest.approx(baseline_auc)
    assert inverted_auc == pytest.approx(baseline_auc)


def test_roc_outputs_have_stable_dtype_endpoints_and_monotonicity() -> None:
    observed = np.asarray([0, 1, 0, 1, 0, 1])
    scores = np.asarray([0.1, 0.8, 0.4, 0.4, -0.2, 1.0], dtype=np.float32)

    false_positive_rates, true_positive_rates, thresholds = binary_roc_curve(observed, scores)

    assert false_positive_rates.dtype == np.float64
    assert true_positive_rates.dtype == np.float64
    assert thresholds.dtype == np.float64
    np.testing.assert_array_equal([false_positive_rates[0], true_positive_rates[0]], [0.0, 0.0])
    np.testing.assert_array_equal([false_positive_rates[-1], true_positive_rates[-1]], [1.0, 1.0])
    assert np.all(np.diff(false_positive_rates) >= 0.0)
    assert np.all(np.diff(true_positive_rates) >= 0.0)
    assert np.all(np.diff(thresholds) < 0.0)
    assert thresholds[0] > float(np.max(scores))


def test_roc_thresholds_remain_strictly_descending_at_float64_extremes() -> None:
    observed = np.asarray([0, 1, 0, 1])
    limit = np.finfo(np.float64).max
    scores = np.asarray([-limit, 1e20, -1e20, limit])

    _, _, thresholds = binary_roc_curve(observed, scores)

    assert np.isposinf(thresholds[0])
    assert np.all(np.diff(thresholds) < 0.0)
    assert roc_auc_score(observed, scores) == 1.0


def test_roc_metrics_do_not_mutate_caller_arrays() -> None:
    observed = np.asarray([0, 1, 0, 1])
    scores = np.asarray([0.4, 0.7, 0.1, 0.8])
    observed_before = observed.copy()
    scores_before = scores.copy()

    binary_roc_curve(observed, scores)
    roc_auc_score(observed, scores)

    np.testing.assert_array_equal(observed, observed_before)
    np.testing.assert_array_equal(scores, scores_before)


@pytest.mark.parametrize(
    ("observed", "scores", "exception", "message"),
    [
        ([], [], ValueError, "at least one"),
        ([[0, 1]], [0.1, 0.9], ValueError, "observed targets must be 1D"),
        ([0, 1], [[0.1, 0.9]], ValueError, "targets must be 1D"),
        ([0, 1], [0.1], ValueError, "inconsistent"),
        ([0, 1], [0.1, np.nan], ValueError, "finite"),
        ([0, 1], [0.1, np.inf], ValueError, "finite"),
        ([0, 1], ["low", "high"], TypeError, "real numeric"),
        ([0, 1], [0.1 + 0.2j, 0.8], TypeError, "real numeric"),
        ([0, 0], [0.1, 0.2], ValueError, r"\{0, 1\}"),
        ([1, 1], [0.1, 0.2], ValueError, r"\{0, 1\}"),
        ([-1, 1], [0.1, 0.2], ValueError, r"\{0, 1\}"),
        ([1, 2], [0.1, 0.2], ValueError, r"\{0, 1\}"),
    ],
)
def test_roc_metrics_reject_invalid_labels_and_scores(
    observed: object,
    scores: object,
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        binary_roc_curve(observed, scores)  # type: ignore[arg-type]
    with pytest.raises(exception, match=message):
        roc_auc_score(observed, scores)  # type: ignore[arg-type]
