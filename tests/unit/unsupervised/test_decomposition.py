"""Mathematical, metamorphic, and failure-contract tests for PCA."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from learning_atlas.unsupervised.base import NotFittedError
from learning_atlas.unsupervised.decomposition import PCA, DecompositionConvergenceError

pytestmark = pytest.mark.unit


def _anisotropic_data(seed: int = 17) -> np.ndarray:
    generator = np.random.default_rng(seed)
    latent = generator.normal(size=(80, 4))
    mixing = np.array(
        [
            [4.0, 1.0, 0.2, 0.0, 0.3],
            [0.2, 2.5, 0.1, 0.4, 0.0],
            [0.0, 0.3, 1.2, 0.1, 0.2],
            [0.1, 0.0, 0.2, 0.5, 0.1],
        ],
        dtype=np.float64,
    )
    return np.asarray(latent @ mixing, dtype=np.float64)


def test_pca_matches_thin_svd_and_owns_standard_diagnostics() -> None:
    features = _anisotropic_data()
    centered = features - np.mean(features, axis=0)
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)

    model = PCA(3).fit(features)
    projection = model.transform(features)

    np.testing.assert_allclose(model.components_ @ model.components_.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.mean(projection, axis=0), 0.0, atol=1e-13)
    np.testing.assert_allclose(model.singular_values_, singular_values[:3], rtol=1e-12)
    np.testing.assert_allclose(
        model.explained_variance_,
        singular_values[:3] ** 2 / (len(features) - 1),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        model.components_.T @ model.components_,
        right_vectors[:3].T @ right_vectors[:3],
        atol=1e-12,
    )
    assert np.all(np.isfinite(model.explained_variance_ratio_))
    assert np.all(np.diff(model.explained_variance_) <= 0.0)
    assert np.all(model.explained_variance_ratio_ >= 0.0)
    assert np.sum(model.explained_variance_ratio_) <= 1.0 + 1e-12
    assert model.rank_ == 4
    assert model.n_components_ == 3


def test_sign_orientation_is_deterministic_and_dominant_loading_is_nonnegative() -> None:
    features = _anisotropic_data()
    first = PCA(4).fit(features)
    second = PCA(4).fit(features.copy())

    np.testing.assert_array_equal(first.components_, second.components_)
    dominant = np.argmax(np.abs(first.components_), axis=1)
    assert np.all(first.components_[np.arange(4), dominant] >= 0.0)


def test_reconstruction_error_is_non_increasing_with_component_count() -> None:
    features = _anisotropic_data()
    errors = []
    for count in range(1, 6):
        model = PCA(count).fit(features)
        reconstructed = model.inverse_transform(model.transform(features))
        errors.append(float(np.linalg.norm(features - reconstructed)))

    assert np.all(np.diff(errors) <= 1e-11)
    assert errors[-1] < 1e-11


def test_whitening_has_identity_sample_covariance_and_round_trips() -> None:
    features = _anisotropic_data()
    model = PCA(4, whiten=True)
    whitened = model.fit_transform(features)

    np.testing.assert_allclose(np.cov(whitened, rowvar=False), np.eye(4), atol=1e-12)
    reconstructed = model.inverse_transform(whitened)
    unwhitened = PCA(4).fit(features)
    expected = unwhitened.inverse_transform(unwhitened.transform(features))
    np.testing.assert_allclose(reconstructed, expected, atol=1e-12)


def test_constant_data_is_explicit_and_whitening_rejects_zero_variance() -> None:
    features = np.full((12, 3), 7.5, dtype=np.float64)
    model = PCA(3).fit(features)

    assert model.rank_ == 0
    np.testing.assert_array_equal(model.explained_variance_, np.zeros(3))
    np.testing.assert_array_equal(model.explained_variance_ratio_, np.zeros(3))
    np.testing.assert_array_equal(model.transform(features), np.zeros((12, 3)))
    np.testing.assert_array_equal(
        model.inverse_transform(model.transform(features)),
        features,
    )
    with pytest.raises(ValueError, match=r"whitening.*rank 0"):
        PCA(1, whiten=True).fit(features)


def test_rank_deficient_data_allows_projection_but_guards_whitening() -> None:
    coordinate = np.linspace(-3.0, 3.0, 20)
    features = np.column_stack((coordinate, 2.0 * coordinate, -coordinate))

    model = PCA(3).fit(features)
    assert model.rank_ == 1
    assert np.all(np.isfinite(model.transform(features)))
    assert np.linalg.norm(features - model.inverse_transform(model.transform(features))) < 1e-12
    assert np.all(np.isfinite(PCA(1, whiten=True).fit_transform(features)))
    with pytest.raises(ValueError, match=r"requested 2 component.*rank 1"):
        PCA(2, whiten=True).fit(features)


def test_translation_changes_only_the_mean_and_reconstruction_origin() -> None:
    features = _anisotropic_data()
    translation = np.array([11.0, -7.0, 0.5, 3.0, 9.0])
    original = PCA(3).fit(features)
    shifted = PCA(3).fit(features + translation)

    np.testing.assert_allclose(original.components_, shifted.components_, atol=1e-12)
    np.testing.assert_allclose(original.explained_variance_, shifted.explained_variance_)
    np.testing.assert_allclose(
        original.transform(features),
        shifted.transform(features + translation),
        atol=1e-12,
    )
    np.testing.assert_allclose(shifted.mean_, original.mean_ + translation)


def test_orthogonal_feature_rotation_preserves_variance_and_reconstruction_error() -> None:
    features = _anisotropic_data()
    q, _ = np.linalg.qr(np.random.default_rng(9).normal(size=(5, 5)))
    rotated = features @ q
    original = PCA(3).fit(features)
    transformed = PCA(3).fit(rotated)

    np.testing.assert_allclose(original.explained_variance_, transformed.explained_variance_)
    original_error = np.linalg.norm(
        features - original.inverse_transform(original.transform(features))
    )
    rotated_error = np.linalg.norm(
        rotated - transformed.inverse_transform(transformed.transform(rotated))
    )
    assert rotated_error == pytest.approx(original_error, rel=1e-12, abs=1e-12)


@settings(max_examples=12, deadline=None)
@given(
    rows=st.integers(min_value=3, max_value=12),
    columns=st.integers(min_value=1, max_value=6),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_full_thin_svd_reconstruction_is_bounded_by_roundoff(
    rows: int,
    columns: int,
    seed: int,
) -> None:
    features = np.random.default_rng(seed).normal(size=(rows, columns))
    model = PCA(min(rows, columns)).fit(features)
    reconstructed = model.inverse_transform(model.transform(features))
    tolerance = 1e-11 * max(1.0, float(np.linalg.norm(features)))
    assert np.linalg.norm(features - reconstructed) <= tolerance


def test_serialization_preserves_fitted_state_and_projection() -> None:
    features = _anisotropic_data()
    model = PCA(3, whiten=True).fit(features)
    restored = pickle.loads(pickle.dumps(model))

    np.testing.assert_array_equal(restored.components_, model.components_)
    np.testing.assert_array_equal(restored.transform(features), model.transform(features))
    assert restored.n_features_in_ == model.n_features_in_


def test_fitted_arrays_are_defensive_copies() -> None:
    model = PCA(2).fit(_anisotropic_data())
    components = model.components_
    components[:] = 0.0
    mean = model.mean_
    mean[:] = 0.0

    assert np.any(model.components_ != 0.0)
    assert np.any(model.mean_ != 0.0)


def test_state_and_shape_contracts_are_actionable() -> None:
    model = PCA(2)
    with pytest.raises(NotFittedError, match="PCA is not fitted"):
        _ = model.components_
    with pytest.raises(NotFittedError, match="PCA is not fitted"):
        model.transform(np.ones((2, 3)))
    with pytest.raises(NotFittedError, match="PCA is not fitted"):
        model.inverse_transform(np.ones((2, 2)))

    model.fit(np.ones((3, 3)))
    with pytest.raises(ValueError, match="fitted estimator expects 3"):
        model.transform(np.ones((3, 2)))
    with pytest.raises(ValueError, match="fitted estimator expects 2"):
        model.inverse_transform(np.ones((3, 3)))


@pytest.mark.parametrize("value", [True, 1.5, "2", 0, -1])
def test_invalid_component_counts_fail_early(value: object) -> None:
    expected = TypeError if value in {True, 1.5, "2"} else ValueError
    with pytest.raises(expected, match="n_components"):
        PCA(value)  # type: ignore[arg-type]


def test_fit_rejects_excess_components_invalid_data_and_invalid_whiten() -> None:
    with pytest.raises(ValueError, match="thin-SVD limit"):
        PCA(4).fit(np.ones((3, 5)))
    with pytest.raises(ValueError, match="at least 2"):
        PCA().fit(np.ones((1, 2)))
    with pytest.raises(ValueError, match="finite"):
        PCA().fit([[1.0, np.inf], [2.0, 3.0]])
    with pytest.raises(ValueError, match="2D"):
        PCA().fit([1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="whiten"):
        PCA(whiten=1)  # type: ignore[arg-type]


def test_svd_failure_has_domain_specific_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> object:
        raise np.linalg.LinAlgError("synthetic failure")

    monkeypatch.setattr(np.linalg, "svd", fail)
    with pytest.raises(DecompositionConvergenceError, match="did not converge"):
        PCA().fit(np.eye(3))


def test_centering_and_variance_overflow_have_actionable_errors() -> None:
    with pytest.raises(DecompositionConvergenceError, match="centering overflowed"):
        PCA().fit([[1e308, 0.0], [1e308, 1.0], [-1e308, 2.0]])
    with pytest.raises(DecompositionConvergenceError, match="variance overflowed"):
        PCA().fit([[1e200, 0.0], [-1e200, 1.0], [0.0, -1.0]])


def test_extreme_finite_scale_uses_stable_variances_and_ratios() -> None:
    scale = 1.225e154
    features = scale * np.vstack((np.eye(3), -np.eye(3)))

    model = PCA(3).fit(features)

    assert np.all(np.isfinite(model.explained_variance_))
    np.testing.assert_allclose(model.explained_variance_ratio_, np.full(3, 1.0 / 3.0))


def test_whitening_rejects_variance_that_underflows_float64() -> None:
    coordinate = np.linspace(-1e-200, 1e-200, 10)

    with pytest.raises(ValueError, match="representable positive variance"):
        PCA(1, whiten=True).fit(coordinate[:, np.newaxis])


def test_non_finite_svd_state_has_domain_specific_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        np.linalg,
        "svd",
        lambda *_args, **_kwargs: (
            np.eye(2),
            np.array([np.nan, 0.0]),
            np.eye(2),
        ),
    )
    with pytest.raises(DecompositionConvergenceError, match="non-finite state"):
        PCA().fit(np.eye(2))


def test_failed_refit_does_not_replace_previous_fitted_state() -> None:
    features = _anisotropic_data()
    model = PCA(2).fit(features)
    previous = model.components_

    with pytest.raises(ValueError, match="finite"):
        model.fit([[1.0, np.nan], [2.0, 3.0]])

    np.testing.assert_array_equal(model.components_, previous)
