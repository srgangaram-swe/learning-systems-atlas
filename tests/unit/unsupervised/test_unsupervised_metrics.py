"""Mathematical, metamorphic, and failure-contract tests for clustering metrics."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.metrics import (
    adjusted_rand_score as sklearn_adjusted_rand_score,
)
from sklearn.metrics import normalized_mutual_info_score as sklearn_nmi_score
from sklearn.metrics import silhouette_score as sklearn_silhouette_score

from learning_atlas.unsupervised.metrics import (
    adjusted_rand_score,
    contingency_matrix,
    mean_partition_stability,
    normalized_mutual_info_score,
    pairwise_euclidean_distances,
    pairwise_partition_stability,
    silhouette_analysis,
    silhouette_score,
)

pytestmark = pytest.mark.unit


def test_hand_contingency_and_perfect_partition_scores() -> None:
    truth = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
    renamed = np.array([9, 9, 9, -4, -4, 7], dtype=np.int64)

    np.testing.assert_array_equal(
        contingency_matrix(truth, renamed),
        np.array([[0, 0, 3], [2, 0, 0], [0, 1, 0]], dtype=np.int64),
    )
    assert adjusted_rand_score(truth, renamed) == 1.0
    assert normalized_mutual_info_score(truth, renamed) == 1.0
    assert adjusted_rand_score([0], [99]) == 1.0
    assert normalized_mutual_info_score([0, 0], [8, 8]) == 1.0


def test_hand_silhouette_matches_definition_and_independent_reference() -> None:
    features = np.array([[0.0], [1.0], [9.0], [10.0]], dtype=np.float64)
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    expected = (1.0 - 1.0 / 9.5 + 1.0 - 1.0 / 8.5) / 2.0

    result = silhouette_analysis(features, labels)

    assert result.defined
    assert result.coverage == 1.0
    assert result.n_clusters == 2
    assert result.n_samples == 4
    assert result.score == pytest.approx(expected)
    assert result.score == pytest.approx(sklearn_silhouette_score(features, labels))
    assert silhouette_score(features, labels) == result.score


def test_silhouette_noise_coverage_singletons_and_degenerate_contract() -> None:
    features = np.array([[0.0], [0.1], [10.0], [10.1], [100.0]], dtype=np.float64)
    labels = np.array([0, 0, 1, 1, -1], dtype=np.int64)

    excluded = silhouette_analysis(features, labels)
    included = silhouette_analysis(features, labels, include_noise=True)

    assert excluded.defined
    assert excluded.coverage == 0.8
    assert excluded.sample_scores[-1] == 0.0
    assert included.defined
    assert included.coverage == 1.0
    assert included.sample_scores[-1] == 0.0
    with pytest.raises(ValueError):
        included.sample_scores[0] = 99.0

    for degenerate in (
        np.full(5, -1, dtype=np.int64),
        np.zeros(5, dtype=np.int64),
        np.arange(5, dtype=np.int64),
    ):
        result = silhouette_analysis(features, degenerate)
        assert result.score == -1.0
        assert not result.defined
        assert np.all(np.isfinite(result.sample_scores))


def test_duplicate_points_have_finite_zero_silhouettes() -> None:
    features = np.zeros((4, 2), dtype=np.float64)
    labels = np.array([0, 0, 1, 1], dtype=np.int64)

    result = silhouette_analysis(features, labels)

    assert result.defined
    assert result.score == 0.0
    np.testing.assert_array_equal(result.sample_scores, np.zeros(4))


@pytest.mark.parametrize("seed", range(8))
def test_metrics_match_sklearn_on_random_partitions(seed: int) -> None:
    rng = np.random.default_rng(seed)
    truth = rng.integers(-2, 4, size=120, dtype=np.int64)
    predicted = rng.integers(10, 17, size=120, dtype=np.int64)

    assert adjusted_rand_score(truth, predicted) == pytest.approx(
        sklearn_adjusted_rand_score(truth, predicted), abs=1e-14
    )
    assert normalized_mutual_info_score(truth, predicted) == pytest.approx(
        sklearn_nmi_score(truth, predicted), abs=1e-14
    )


def test_random_label_adjusted_rand_is_centered_near_zero() -> None:
    rng = np.random.default_rng(91)
    truth = np.repeat(np.arange(5, dtype=np.int64), 40)
    scores = [adjusted_rand_score(truth, rng.permutation(truth)) for _ in range(160)]

    assert abs(float(np.mean(scores))) < 0.01


@settings(max_examples=40, derandomize=True, database=None, deadline=None)
@given(
    st.lists(
        st.tuples(st.integers(-2, 4), st.integers(7, 13)),
        min_size=2,
        max_size=80,
    )
)
def test_external_metrics_are_invariant_to_rows_and_label_names(
    pairs: list[tuple[int, int]],
) -> None:
    truth = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
    predicted = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    order = np.arange(len(truth) - 1, -1, -1)

    expected_ari = adjusted_rand_score(truth, predicted)
    expected_nmi = normalized_mutual_info_score(truth, predicted)
    renamed_truth = truth * 31 + 101
    renamed_predicted = predicted * -17 - 3

    assert adjusted_rand_score(renamed_truth[order], renamed_predicted[order]) == pytest.approx(
        expected_ari, abs=1e-15
    )
    assert normalized_mutual_info_score(
        renamed_truth[order], renamed_predicted[order]
    ) == pytest.approx(expected_nmi, abs=1e-15)


def test_noise_can_be_included_or_jointly_excluded() -> None:
    truth = np.array([0, 0, 1, 1, -1], dtype=np.int64)
    predicted = np.array([4, 4, 9, -1, -1], dtype=np.int64)

    included = adjusted_rand_score(truth, predicted)
    excluded = adjusted_rand_score(truth, predicted, ignore_noise=True)

    assert included != excluded
    assert excluded == 1.0
    assert normalized_mutual_info_score(truth, predicted, ignore_noise=True) == 1.0
    np.testing.assert_array_equal(
        contingency_matrix(truth, predicted, ignore_noise=True),
        np.eye(2, dtype=np.int64) * np.array([[2], [1]], dtype=np.int64),
    )
    assert adjusted_rand_score(truth, predicted) == adjusted_rand_score(predicted, truth)
    assert normalized_mutual_info_score(truth, predicted) == normalized_mutual_info_score(
        predicted, truth
    )


def test_partition_stability_is_symmetric_and_label_invariant() -> None:
    first = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    equivalent = np.array([8, 8, -4, -4, 12, 12], dtype=np.int64)
    different = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)

    matrix = pairwise_partition_stability((first, equivalent, different))

    np.testing.assert_allclose(matrix, matrix.T)
    np.testing.assert_array_equal(np.diag(matrix), np.ones(3))
    assert matrix[0, 1] == 1.0
    assert mean_partition_stability((first, equivalent)) == 1.0


def test_large_counts_do_not_overflow_adjusted_rand_arithmetic() -> None:
    labels = np.repeat(np.arange(10, dtype=np.int64), 10_000)
    renamed = labels * 1_000_003 + 97

    assert adjusted_rand_score(labels, renamed) == 1.0
    assert normalized_mutual_info_score(labels, renamed) == 1.0


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: adjusted_rand_score([], []), "empty"),
        (lambda: adjusted_rand_score([0, 1], [0]), "inconsistent"),
        (lambda: adjusted_rand_score([0.5, 1.0], [0, 1]), "integer-valued"),
        (lambda: adjusted_rand_score([True, False], [0, 1]), "booleans"),
        (lambda: normalized_mutual_info_score([0, np.nan], [0, 1]), "finite"),
        (
            lambda: adjusted_rand_score([-1, -1], [-1, -1], ignore_noise=True),
            "no samples remain",
        ),
        (lambda: mean_partition_stability(([0, 1],)), "at least two"),
        (lambda: pairwise_partition_stability(()), "at least one"),
        (
            lambda: pairwise_partition_stability(([0, 1], [0, 1, 2])),
            "same number",
        ),
    ],
)
def test_metric_label_failures_are_actionable(call: object, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        call()  # type: ignore[operator]


def test_pairwise_distance_validation_budget_and_overflow() -> None:
    distances = pairwise_euclidean_distances([[0.0, 0.0], [3.0, 4.0]])
    np.testing.assert_allclose(distances, [[0.0, 5.0], [5.0, 0.0]])

    with pytest.raises(MemoryError, match="requires 9 elements"):
        pairwise_euclidean_distances(np.zeros((3, 2)), max_pairwise_elements=8)
    with pytest.raises(ValueError, match="positive integer"):
        pairwise_euclidean_distances(np.zeros((2, 1)), max_pairwise_elements=0)
    with pytest.raises(ValueError, match="overflowed"):
        pairwise_euclidean_distances([[-1.0e308], [1.0e308]])
    with pytest.raises(ValueError, match="inconsistent samples"):
        silhouette_analysis([[0.0], [1.0]], [0])
    with pytest.raises(ValueError, match="finite"):
        silhouette_analysis([[0.0], [np.inf]], [0, 1])


def test_contingency_allocation_guard_is_explicit() -> None:
    with pytest.raises(MemoryError, match="requires 9 cells"):
        contingency_matrix([0, 1, 2], [3, 4, 5], max_cells=8)
