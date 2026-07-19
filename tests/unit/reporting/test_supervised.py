"""Numerical and labeling contracts for supervised evidence plots."""

from pathlib import Path

import numpy as np
import pytest

from learning_atlas.core.artifacts import ArtifactStore
from learning_atlas.reporting.supervised import _binary_roc, decision_boundary_plot

pytestmark = pytest.mark.unit


def test_binary_roc_groups_tied_scores_without_row_order_bias() -> None:
    probability = np.full(4, 0.5, dtype=np.float64)
    first_observed = np.asarray([0.0, 0.0, 1.0, 1.0])
    second_observed = first_observed[::-1].copy()

    first_fpr, first_tpr = _binary_roc(first_observed, probability)
    second_fpr, second_tpr = _binary_roc(second_observed, probability)

    np.testing.assert_array_equal(first_fpr, [0.0, 1.0])
    np.testing.assert_array_equal(first_tpr, [0.0, 1.0])
    np.testing.assert_array_equal(second_fpr, first_fpr)
    np.testing.assert_array_equal(second_tpr, first_tpr)
    assert np.trapezoid(first_tpr, first_fpr) == pytest.approx(0.5)


def test_decision_boundary_rejects_reference_width_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="same columns"):
        decision_boundary_plot(
            {},
            np.zeros((2, 2)),
            np.zeros(2),
            np.zeros((2, 3)),
            context_label="seed=42 | train n=2 | holdout n=2 | CV=2 folds",
            artifacts=ArtifactStore(tmp_path),
            relative_path="unused.png",
        )
