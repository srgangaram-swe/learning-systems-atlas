"""Probability, optimization, replay, and failure tests for exact t-SNE."""

from __future__ import annotations

import inspect
import pickle
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score

from learning_atlas.unsupervised.base import NotFittedError
from learning_atlas.unsupervised.manifold import (
    TSNE,
    PerplexitySearchError,
    TSNEConvergenceError,
    TSNENumericalError,
    compute_affinities,
    kl_divergence,
    pairwise_squared_distances,
    student_t_probabilities,
    tsne_gradient,
)

pytestmark = pytest.mark.unit


def _small_features() -> np.ndarray:
    return np.array(
        [
            [-2.0, 0.0, 0.5],
            [-1.5, 0.2, 0.4],
            [-0.8, -0.1, 0.6],
            [0.7, 0.1, -0.5],
            [1.4, -0.2, -0.4],
            [2.1, 0.0, -0.6],
        ],
        dtype=np.float64,
    )


def test_pairwise_squared_distances_match_hand_fixture() -> None:
    features = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 0.0]])
    expected = np.array([[0.0, 25.0, 9.0], [25.0, 0.0, 16.0], [9.0, 16.0, 0.0]])

    distances = pairwise_squared_distances(features)

    np.testing.assert_array_equal(distances, expected)
    np.testing.assert_array_equal(distances, distances.T)
    np.testing.assert_array_equal(np.diag(distances), np.zeros(3))


def test_pairwise_distances_are_translation_stable_and_guard_overflow() -> None:
    features = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 0.0]])
    translated = features + np.array([1e10, -1e10])

    np.testing.assert_allclose(
        pairwise_squared_distances(translated),
        pairwise_squared_distances(features),
        rtol=0.0,
        atol=1e-5,
    )
    with pytest.raises(ValueError, match="distances overflowed"):
        pairwise_squared_distances([[1e308, 0.0], [-1e308, 0.0]])


def test_affinities_satisfy_row_entropy_and_joint_probability_invariants() -> None:
    perplexity = 3.0
    tolerance = 1e-6
    affinities = compute_affinities(
        _small_features(),
        perplexity=perplexity,
        tolerance=tolerance,
    )

    conditional = affinities.conditional_probabilities
    joint = affinities.joint_probabilities
    np.testing.assert_allclose(np.sum(conditional, axis=1), 1.0, atol=1e-14)
    np.testing.assert_array_equal(np.diag(conditional), np.zeros(len(conditional)))
    assert np.all(np.isfinite(conditional))
    assert np.all(conditional >= 0.0)
    assert np.all(np.isfinite(affinities.precisions))
    assert np.all(affinities.precisions >= 0.0)
    assert np.max(np.abs(np.log(affinities.achieved_perplexities) - np.log(perplexity))) <= (
        tolerance + 1e-12
    )
    np.testing.assert_allclose(joint, joint.T, atol=0.0)
    np.testing.assert_array_equal(np.diag(joint), np.zeros(len(joint)))
    assert np.sum(joint) == pytest.approx(1.0, abs=1e-15)
    assert np.all(joint >= 0.0)
    for diagnostic in (
        affinities.conditional_probabilities,
        affinities.joint_probabilities,
        affinities.precisions,
        affinities.achieved_perplexities,
    ):
        assert diagnostic.flags.writeable is False
        with pytest.raises(ValueError, match="read-only"):
            diagnostic.flat[0] = 0.0


def test_uniform_distance_row_supports_only_maximum_perplexity() -> None:
    identical = np.ones((4, 2), dtype=np.float64)
    affinities = compute_affinities(identical, perplexity=3.0)

    np.testing.assert_array_equal(affinities.precisions, np.zeros(4))
    np.testing.assert_array_equal(affinities.achieved_perplexities, np.full(4, 3.0))
    with pytest.raises(PerplexitySearchError, match="row 0") as captured:
        compute_affinities(identical, perplexity=2.0)
    assert captured.value.row == 0
    assert captured.value.target_perplexity == 2.0
    assert captured.value.achieved_perplexity == 3.0


def test_row_search_exhaustion_reports_the_failing_row() -> None:
    with pytest.raises(PerplexitySearchError) as captured:
        compute_affinities(_small_features(), perplexity=2.5, tolerance=1e-15, max_iter=1)

    assert captured.value.row == 0
    assert captured.value.iterations == 1
    assert "closest=" in str(captured.value)


def test_student_probabilities_and_kl_are_finite_and_normalized() -> None:
    embedding = np.array([[-1.0, 0.0], [0.0, 0.2], [1.0, 0.0], [0.0, 1.0]])
    joint = compute_affinities(_small_features()[:4], perplexity=2.0).joint_probabilities
    low = student_t_probabilities(embedding)

    np.testing.assert_allclose(low, low.T)
    np.testing.assert_array_equal(np.diag(low), np.zeros(4))
    assert np.sum(low) == pytest.approx(1.0)
    assert np.all(low >= 0.0)
    assert np.isfinite(kl_divergence(joint, embedding))


def test_exact_gradient_matches_central_finite_differences() -> None:
    features = _small_features()[:5]
    joint = compute_affinities(features, perplexity=2.0).joint_probabilities
    embedding = np.random.default_rng(5).normal(scale=0.2, size=(5, 2))
    analytical = tsne_gradient(embedding, joint)
    numerical = np.zeros_like(embedding)
    step = 1e-6

    for row in range(len(embedding)):
        for column in range(embedding.shape[1]):
            above = embedding.copy()
            below = embedding.copy()
            above[row, column] += step
            below[row, column] -= step
            numerical[row, column] = (kl_divergence(joint, above) - kl_divergence(joint, below)) / (
                2.0 * step
            )

    np.testing.assert_allclose(analytical, numerical, rtol=2e-6, atol=2e-8)
    np.testing.assert_allclose(np.sum(analytical, axis=0), 0.0, atol=1e-14)


def test_reference_embedding_improves_post_exaggeration_kl_and_neighborhoods() -> None:
    features, retrospective_labels = make_blobs(
        n_samples=75,
        centers=3,
        n_features=5,
        cluster_std=0.65,
        random_state=8,
    )
    model = TSNE(
        perplexity=15.0,
        learning_rate=30.0,
        max_iter=260,
        early_exaggeration_iter=70,
        init="pca",
        seed=12,
    )

    embedding = model.fit_transform(features)

    assert embedding.shape == (75, 2)
    assert np.all(np.isfinite(embedding))
    np.testing.assert_allclose(np.mean(embedding, axis=0), 0.0, atol=1e-13)
    assert model.converged_ is True
    assert model.kl_divergence_ < model.post_exaggeration_start_kl_
    assert model.kl_divergence_ == model.kl_history_[-1]
    assert model.n_iter_ == len(model.kl_history_) == len(model.gradient_norm_history_)
    assert silhouette_score(embedding, retrospective_labels) > 0.85
    assert trustworthiness(features, embedding, n_neighbors=7) > 0.95


def test_same_seed_and_configuration_replay_exactly() -> None:
    features = _small_features()
    options = dict(
        perplexity=2.0,
        learning_rate=15.0,
        max_iter=90,
        early_exaggeration_iter=25,
        init="random",
        seed=91,
    )
    first = TSNE(**options).fit(features)
    second = TSNE(**options).fit(features.copy())

    np.testing.assert_array_equal(first.embedding_, second.embedding_)
    np.testing.assert_array_equal(first.joint_probabilities_, second.joint_probabilities_)
    assert first.kl_history_ == second.kl_history_
    assert first.gradient_norm_history_ == second.gradient_norm_history_
    assert first.stop_reason_ == second.stop_reason_


def test_random_initialization_changes_with_seed_but_pca_does_not() -> None:
    features = _small_features()
    common = dict(
        perplexity=2.0,
        learning_rate=10.0,
        max_iter=50,
        early_exaggeration_iter=15,
    )
    random_one = TSNE(**common, init="random", seed=1).fit_transform(features)
    random_two = TSNE(**common, init="random", seed=2).fit_transform(features)
    pca_one = TSNE(**common, init="pca", seed=1).fit_transform(features)
    pca_two = TSNE(**common, init="pca", seed=2).fit_transform(features)

    assert not np.array_equal(random_one, random_two)
    np.testing.assert_array_equal(pca_one, pca_two)


def test_transductive_contract_exposes_no_out_of_sample_transform() -> None:
    model = TSNE(
        perplexity=2.0,
        learning_rate=10.0,
        max_iter=40,
        early_exaggeration_iter=10,
    )
    returned = model.fit(_small_features())

    assert returned is model
    assert not hasattr(model, "transform")
    assert tuple(inspect.signature(model.fit).parameters) == ("features",)


def test_fitted_probability_diagnostics_are_defensive_copies() -> None:
    model = TSNE(
        perplexity=2.0,
        learning_rate=10.0,
        max_iter=40,
        early_exaggeration_iter=10,
    ).fit(_small_features())
    joint = model.joint_probabilities_
    joint[:] = 0.0
    embedding = model.embedding_
    embedding[:] = 0.0

    assert np.sum(model.joint_probabilities_) == pytest.approx(1.0)
    assert np.any(model.embedding_ != 0.0)
    np.testing.assert_allclose(model.perplexities_, 2.0, rtol=1e-5)


def test_serialization_preserves_embedding_and_diagnostics() -> None:
    model = TSNE(
        perplexity=2.0,
        learning_rate=10.0,
        max_iter=50,
        early_exaggeration_iter=15,
        init="random",
        seed=7,
    ).fit(_small_features())
    restored = pickle.loads(pickle.dumps(model))

    np.testing.assert_array_equal(restored.embedding_, model.embedding_)
    np.testing.assert_array_equal(restored.precisions_, model.precisions_)
    assert restored.kl_history_ == model.loss_history_
    assert restored.kl_divergence_ == model.kl_divergence_


def test_strict_nonconvergence_raises_with_history() -> None:
    model = TSNE(
        perplexity=2.0,
        learning_rate=np.finfo(np.float64).tiny,
        max_iter=3,
        early_exaggeration=1.0,
        early_exaggeration_iter=0,
        init="random",
        seed=1,
        raise_on_nonconvergence=True,
    )

    with pytest.raises(TSNEConvergenceError, match="did not improve") as captured:
        model.fit(_small_features())
    assert captured.value.iterations == 3
    assert captured.value.final_kl == captured.value.post_exaggeration_start
    assert len(captured.value.history) == 3


def test_non_strict_nonconvergence_is_visible_in_fitted_state() -> None:
    model = TSNE(
        perplexity=2.0,
        learning_rate=np.finfo(np.float64).tiny,
        max_iter=3,
        early_exaggeration=1.0,
        early_exaggeration_iter=0,
        init="random",
        seed=1,
    ).fit(_small_features())

    assert model.converged_ is False
    assert model.stop_reason_ == "post_exaggeration_objective_did_not_improve"


def test_non_finite_gradient_raises_iteration_attributed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from learning_atlas.unsupervised import manifold as module

    monkeypatch.setattr(
        module,
        "_gradient_from_probabilities",
        lambda *_args, **_kwargs: np.full((6, 2), np.inf),
    )
    model = TSNE(
        perplexity=2.0,
        max_iter=20,
        early_exaggeration_iter=5,
    )
    with pytest.raises(TSNENumericalError, match="iteration 0") as captured:
        model.fit(_small_features())
    assert captured.value.stage == "gradient"


def test_low_dimensional_overflow_is_wrapped_as_optimizer_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from learning_atlas.unsupervised import manifold as module

    monkeypatch.setattr(
        module,
        "_gradient_from_probabilities",
        lambda *_args, **_kwargs: np.arange(12, dtype=np.float64).reshape(6, 2),
    )
    model = TSNE(
        perplexity=2.0,
        learning_rate=1e200,
        max_iter=2,
        early_exaggeration_iter=0,
        init="random",
    )
    with pytest.raises(TSNENumericalError) as captured:
        model.fit(_small_features())
    assert captured.value.stage == "low-dimensional affinities"


def test_probability_helpers_reject_malformed_state() -> None:
    embedding = np.ones((3, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="shape"):
        kl_divergence(np.ones((2, 2)), embedding)
    with pytest.raises(ValueError, match="symmetric"):
        kl_divergence(
            np.array([[0.0, 0.4, 0.1], [0.2, 0.0, 0.1], [0.1, 0.1, 0.0]]),
            embedding,
        )
    with pytest.raises(ValueError, match="zero diagonal"):
        kl_divergence(np.eye(3) / 3.0, embedding)
    with pytest.raises(ValueError, match="sum to one"):
        kl_divergence(np.zeros((3, 3)), embedding)
    with pytest.raises(ValueError, match="non-negative"):
        tsne_gradient(embedding, -np.ones((3, 3)))
    with pytest.raises(TypeError, match="real numeric"):
        kl_divergence(np.full((3, 3), "bad"), embedding)
    with pytest.raises(TypeError, match="real numeric"):
        kl_divergence(np.eye(3, dtype=np.complex128), embedding)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"n_components": 0}, ValueError, "n_components"),
        ({"n_components": True}, TypeError, "n_components"),
        ({"perplexity": 0.0}, ValueError, "perplexity"),
        ({"learning_rate": 0.0}, ValueError, "learning_rate"),
        ({"max_iter": 0}, ValueError, "max_iter"),
        ({"early_exaggeration": 0.9}, ValueError, "early_exaggeration"),
        ({"initial_momentum": 1.0}, ValueError, "initial_momentum"),
        ({"final_momentum": -0.1}, ValueError, "final_momentum"),
        ({"min_gain": 0.0}, ValueError, "min_gain"),
        ({"min_grad_norm": -1.0}, ValueError, "min_grad_norm"),
        ({"n_iter_without_progress": 0}, ValueError, "n_iter_without_progress"),
        ({"kl_tolerance": -1.0}, ValueError, "kl_tolerance"),
        ({"perplexity_tolerance": 0.0}, ValueError, "perplexity_tolerance"),
        ({"perplexity_search_max_iter": 0}, ValueError, "perplexity_search_max_iter"),
        ({"init": "spectral"}, ValueError, "init"),
        ({"seed": -1}, ValueError, "seed"),
        ({"seed": "seed"}, TypeError, "seed"),
        ({"raise_on_nonconvergence": 1}, TypeError, "raise_on_nonconvergence"),
    ],
)
def test_invalid_configuration_fails_at_construction(
    kwargs: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        TSNE(**kwargs)  # type: ignore[arg-type]


def test_invalid_sample_and_perplexity_regimes_fail_before_optimization() -> None:
    with pytest.raises(ValueError, match="smaller than max_iter"):
        TSNE(max_iter=10, early_exaggeration_iter=10)
    with pytest.raises(ValueError, match="at least 3"):
        TSNE(perplexity=1.0, max_iter=2, early_exaggeration_iter=0).fit(np.eye(2))
    with pytest.raises(ValueError, match="available neighbors"):
        TSNE(perplexity=6.0, max_iter=2, early_exaggeration_iter=0).fit(_small_features())
    with pytest.raises(ValueError, match="finite"):
        TSNE(perplexity=2.0, max_iter=2, early_exaggeration_iter=0).fit(
            [[0.0, 1.0], [1.0, np.nan], [2.0, 3.0]]
        )


def test_pca_initialization_rejects_wider_embedding_but_random_supports_it() -> None:
    features = np.arange(12, dtype=np.float64).reshape(6, 2)
    pca_model = TSNE(
        n_components=3,
        perplexity=2.0,
        max_iter=2,
        early_exaggeration_iter=0,
        init="pca",
    )
    with pytest.raises(ValueError, match="use init='random'"):
        pca_model.fit(features)

    random_embedding = TSNE(
        n_components=3,
        perplexity=2.0,
        learning_rate=1.0,
        max_iter=2,
        early_exaggeration_iter=0,
        init="random",
    ).fit_transform(features)
    assert random_embedding.shape == (6, 3)


def test_fitted_state_is_guarded_before_fit() -> None:
    model = TSNE(max_iter=2, early_exaggeration_iter=0)
    properties = (
        "embedding_",
        "joint_probabilities_",
        "conditional_probabilities_",
        "precisions_",
        "perplexities_",
        "kl_history_",
        "gradient_norm_history_",
        "initial_kl_",
        "post_exaggeration_start_kl_",
        "kl_divergence_",
        "n_iter_",
        "converged_",
        "stop_reason_",
    )
    for name in properties:
        with pytest.raises(NotFittedError, match="TSNE is not fitted"):
            getattr(model, name)


def test_source_module_has_no_sklearn_or_scipy_dependency() -> None:
    source = Path(inspect.getfile(TSNE)).read_text(encoding="utf-8")
    assert "sklearn" not in source
    assert "scipy" not in source
