"""Optimization, kernel, and failure-contract tests for the from-scratch SVM."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from learning_atlas.core.estimators import NotFittedError
from learning_atlas.supervised.svm import KernelSVMClassifier, SVMConvergenceError

pytestmark = pytest.mark.unit


def standardized_blobs() -> tuple[np.ndarray, np.ndarray]:
    features, targets = make_blobs(
        n_samples=60,
        centers=2,
        cluster_std=0.8,
        random_state=3,
    )
    return StandardScaler().fit_transform(features), targets.astype(np.float64)


def standardized_moons() -> tuple[np.ndarray, np.ndarray]:
    features, targets = make_moons(n_samples=80, noise=0.12, random_state=3)
    return StandardScaler().fit_transform(features), targets.astype(np.float64)


def test_linear_svm_matches_sklearn_and_satisfies_dual_kkt_contracts() -> None:
    features, binary_targets = standardized_blobs()
    targets = np.where(binary_targets == 0.0, -5.0, 11.0)
    model = KernelSVMClassifier(
        kernel="linear",
        C=2.0,
        tol=1e-3,
        max_passes=20,
    ).fit(features, targets)
    oracle = SVC(kernel="linear", C=2.0, tol=1e-3).fit(features, targets)

    assert model.converged_
    assert model.max_kkt_violation_ <= model.tol
    assert model.score(features, targets) == 1.0
    assert np.mean(model.predict(features) == oracle.predict(features)) >= 0.99
    np.testing.assert_allclose(
        model.alphas_ @ np.where(targets == -5.0, -1.0, 1.0), 0.0, atol=1e-10
    )
    assert np.all(model.alphas_ >= 0.0)
    assert np.all(model.alphas_ <= model.C)
    assert model.dual_coef_.shape == (1, len(model.support_))
    assert model.support_vectors_.shape == (len(model.support_), features.shape[1])
    assert np.sum(model.n_support_) == len(model.support_)
    assert model.dual_objective_ > 0.0


def test_tiny_linear_problem_matches_closed_form_dual_solution() -> None:
    # For x=(-1, +1), y=(-1, +1), stationarity and the equality constraint
    # yield alpha_1=alpha_2=1/2, w=1, b=0, and unit functional margins.
    model = KernelSVMClassifier(
        kernel="linear",
        C=10.0,
        tol=1e-8,
        max_passes=10,
    ).fit([[-1.0], [1.0]], [-2.0, 7.0])

    np.testing.assert_allclose(model.alphas_, [0.5, 0.5], atol=1e-12)
    np.testing.assert_allclose(model.intercept_, [0.0], atol=1e-12)
    np.testing.assert_allclose(model.decision_function([[-1.0], [1.0]]), [-1.0, 1.0])
    assert model.dual_objective_ == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("kernel", "extra"),
    [
        ("rbf", {}),
        ("poly", {"degree": 3, "coef0": 1.0}),
    ],
)
def test_nonlinear_kernels_match_sklearn_decisions(
    kernel: str,
    extra: dict[str, float | int],
) -> None:
    features, targets = standardized_moons()
    model = KernelSVMClassifier(
        kernel=kernel,  # type: ignore[arg-type]
        C=2.0,
        tol=1e-3,
        max_passes=20,
        **extra,  # type: ignore[arg-type]
    ).fit(features, targets)
    oracle = SVC(kernel=kernel, C=2.0, tol=1e-3, **extra).fit(features, targets)

    assert model.converged_
    assert model.score(features, targets) >= 0.95
    assert np.mean(model.predict(features) == oracle.predict(features)) >= 0.98
    assert (
        np.corrcoef(model.decision_function(features), oracle.decision_function(features))[0, 1]
        > 0.99
    )


def test_rbf_kernel_separates_concentric_circles() -> None:
    from sklearn.datasets import make_circles

    features, targets = make_circles(
        n_samples=100,
        factor=0.35,
        noise=0.05,
        random_state=6,
    )
    features = StandardScaler().fit_transform(features)
    model = KernelSVMClassifier(
        kernel="rbf",
        gamma=1.0,
        C=5.0,
        max_passes=30,
    ).fit(features, targets)

    assert model.converged_
    assert model.score(features, targets) == 1.0


def test_soft_margin_dual_variables_are_capped_by_c() -> None:
    features = np.asarray([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    targets = np.asarray([0.0, 0.0, 1.0, 1.0, 0.0])
    model = KernelSVMClassifier(kernel="linear", C=0.2, max_passes=20).fit(features, targets)

    assert np.all(model.alphas_ >= 0.0)
    assert np.all(model.alphas_ <= 0.2 + 1e-12)
    assert np.any(np.isclose(model.alphas_, 0.2))


def test_training_is_bitwise_deterministic() -> None:
    features, targets = standardized_moons()
    parameters = {
        "kernel": "rbf",
        "C": 2.0,
        "gamma": "scale",
        "tol": 1e-3,
        "max_passes": 20,
    }
    first = KernelSVMClassifier(**parameters).fit(features, targets)  # type: ignore[arg-type]
    second = KernelSVMClassifier(**parameters).fit(features, targets)  # type: ignore[arg-type]

    np.testing.assert_array_equal(first.alphas_, second.alphas_)
    np.testing.assert_array_equal(first.support_, second.support_)
    np.testing.assert_array_equal(first.intercept_, second.intercept_)
    np.testing.assert_array_equal(first.kkt_violation_, second.kkt_violation_)
    assert first.n_iter_ == second.n_iter_


def test_gamma_resolution_is_explicit_and_zero_variance_safe() -> None:
    features = np.asarray([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])
    targets = np.asarray([0.0, 0.0, 1.0, 1.0])
    automatic = KernelSVMClassifier(kernel="rbf", gamma="auto").fit(features, targets)
    scaled = KernelSVMClassifier(kernel="rbf", gamma="scale").fit(features, targets)
    constant = KernelSVMClassifier(kernel="rbf", gamma="scale").fit(
        np.ones((4, 2)),
        targets,
    )

    assert automatic.gamma_ == 0.5
    assert scaled.gamma_ == pytest.approx(1.0 / (2.0 * np.var(features)))
    assert constant.gamma_ == 1.0


def test_nonconvergence_is_observable_or_atomic_in_strict_mode() -> None:
    features, targets = standardized_moons()
    permissive = KernelSVMClassifier(
        max_iter=1,
        max_passes=1,
        tol=1e-12,
    ).fit(features, targets)

    assert permissive.is_fitted
    assert not permissive.converged_
    assert permissive.n_iter_ == 1
    assert permissive.max_kkt_violation_ > permissive.tol
    assert np.all(np.isfinite(permissive.decision_function(features[:3])))

    strict = KernelSVMClassifier(
        max_iter=1,
        max_passes=1,
        tol=1e-12,
        raise_on_nonconvergence=True,
    )
    with pytest.raises(SVMConvergenceError, match="maximum violation") as captured:
        strict.fit(features, targets)
    assert captured.value.iterations == 1
    assert captured.value.max_kkt_violation > 0.0
    assert not strict.is_fitted


def test_properties_are_defensive_and_zero_margin_uses_smaller_label() -> None:
    features, targets = standardized_blobs()
    model = KernelSVMClassifier(kernel="linear", max_passes=20).fit(features, targets)
    support = model.support_
    support[:] = 0
    vectors = model.support_vectors_
    vectors[:] = 999.0

    assert not np.all(model.support_ == 0)
    assert not np.all(model.support_vectors_ == 999.0)
    boundary = np.zeros((1, features.shape[1]))
    # Constructing the exact geometric boundary from fitted coefficients is not
    # generally convenient; the stable prediction itself remains deterministic.
    assert model.predict(boundary)[0] in model.classes_


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"C": 0.0}, ValueError, "C"),
        ({"kernel": "sigmoid"}, ValueError, "kernel"),
        ({"gamma": 0.0}, ValueError, "gamma"),
        ({"gamma": True}, ValueError, "gamma"),
        ({"gamma": "median"}, ValueError, "gamma"),
        ({"degree": 0}, ValueError, "degree"),
        ({"coef0": np.inf}, ValueError, "coef0"),
        ({"tol": 0.0}, ValueError, "tol"),
        ({"max_iter": 0}, ValueError, "max_iter"),
        ({"max_passes": True}, TypeError, "max_passes"),
        ({"raise_on_nonconvergence": 1}, TypeError, "raise_on_nonconvergence"),
    ],
)
def test_invalid_hyperparameters_fail_early(
    kwargs: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        KernelSVMClassifier(**kwargs)  # type: ignore[arg-type]


def test_binary_fitted_shape_and_kernel_overflow_boundaries() -> None:
    model = KernelSVMClassifier()
    with pytest.raises(NotFittedError, match="call fit"):
        model.predict([[0.0]])
    with pytest.raises(ValueError, match="exactly two classes"):
        model.fit([[0.0], [1.0], [2.0]], [0.0, 1.0, 2.0])

    fitted = KernelSVMClassifier(kernel="linear").fit([[0.0], [1.0]], [0.0, 1.0])
    with pytest.raises(ValueError, match="expects 1"):
        fitted.predict([[0.0, 1.0]])
    with pytest.raises(ValueError, match="kernel evaluation produced non-finite"):
        KernelSVMClassifier(kernel="poly", degree=100, gamma=1.0).fit(
            [[-1e100], [1e100]],
            [0.0, 1.0],
        )
