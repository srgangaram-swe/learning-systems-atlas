"""Leakage-safe, from-scratch supervised model comparison experiments."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, cast, runtime_checkable

import joblib
import numpy as np
from numpy.typing import ArrayLike
from pydantic import JsonValue

from learning_atlas.core.config import (
    ScratchClassificationBenchmarkConfig,
    ScratchRegressionBenchmarkConfig,
)
from learning_atlas.core.contracts import (
    CandidateResult,
    LearningParadigm,
    RunContext,
    RunResult,
    SourceKind,
    SourceMetadata,
)
from learning_atlas.core.data import array_fingerprint
from learning_atlas.core.estimators import Estimator
from learning_atlas.core.validation import FloatArray, validate_features
from learning_atlas.reporting.supervised import (
    classification_diagnostics_plot,
    cv_distribution_plot,
    decision_boundary_plot,
    ensemble_evidence_plot,
    heldout_comparison_plot,
    optimization_plot,
    regression_diagnostics_plot,
    regularization_path_plot,
)
from learning_atlas.supervised.baselines import MeanRegressor, PriorClassifier
from learning_atlas.supervised.datasets import make_classification, make_regression
from learning_atlas.supervised.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from learning_atlas.supervised.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from learning_atlas.supervised.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from learning_atlas.supervised.model_selection import kfold_indices, stratified_kfold_indices
from learning_atlas.supervised.naive_bayes import GaussianNB, MultinomialNB
from learning_atlas.supervised.neighbors import KNeighborsClassifier, KNeighborsRegressor
from learning_atlas.supervised.preprocessing import SplitData, StandardScaler, train_test_split
from learning_atlas.supervised.svm import KernelSVMClassifier
from learning_atlas.supervised.tree import DecisionTreeClassifier, DecisionTreeRegressor

PreprocessingKind = Literal["standard", "nonnegative"]


class CandidateExecutionError(RuntimeError):
    """Add model and partition context to an estimator failure."""

    def __init__(self, candidate: str, partition: str) -> None:
        self.candidate = candidate
        self.partition = partition
        super().__init__(f"candidate {candidate!r} failed while fitting {partition}")


@runtime_checkable
class ProbabilisticClassifier(Protocol):
    """Classifier that owns a genuine normalized probability model."""

    def predict_proba(self, features: ArrayLike) -> FloatArray:
        """Return one probability column per sorted fitted class."""


class FeatureTransformer(Protocol):
    """Fitted feature transformation used by a serialized comparison model."""

    def transform(self, features: ArrayLike) -> FloatArray:
        """Transform inference rows with training-only state."""


class _NonnegativeShift:
    """Training-fitted shift for MultinomialNB with clipped unseen minima."""

    def __init__(self) -> None:
        self._minimum: FloatArray | None = None

    def fit(self, features: ArrayLike) -> _NonnegativeShift:
        validated = validate_features(features)
        self._minimum = np.asarray(np.min(validated, axis=0), dtype=np.float64)
        return self

    def fit_transform(self, features: ArrayLike) -> FloatArray:
        return self.fit(features).transform(features)

    def transform(self, features: ArrayLike) -> FloatArray:
        if self._minimum is None:
            msg = "nonnegative shift is not fitted"
            raise RuntimeError(msg)
        validated = validate_features(features, expected_features=len(self._minimum))
        return np.asarray(np.maximum(validated - self._minimum, 0.0), dtype=np.float64)


@dataclass(slots=True)
class FittedPipeline:
    """Serializable training-fitted transform and from-scratch estimator pair."""

    transformer: FeatureTransformer
    estimator: Estimator

    def predict(self, features: ArrayLike) -> FloatArray:
        return self.estimator.predict(self.transformer.transform(features))

    def predict_proba(self, features: ArrayLike) -> FloatArray:
        if not isinstance(self.estimator, ProbabilisticClassifier):
            msg = f"{type(self.estimator).__name__} does not expose modeled probabilities"
            raise TypeError(msg)
        return self.estimator.predict_proba(self.transformer.transform(features))


@dataclass(frozen=True, slots=True)
class _CandidateSpec:
    name: str
    factory: Callable[[int], Estimator]
    parameters: dict[str, JsonValue]
    preprocessing: PreprocessingKind = "standard"


@dataclass(slots=True)
class _Evaluation:
    result: CandidateResult
    fold_scores: tuple[float, ...]
    pipeline: FittedPipeline
    train_predictions: FloatArray
    test_predictions: FloatArray
    test_probabilities: FloatArray | None = None


@dataclass(frozen=True, slots=True)
class _CandidateSeeds:
    folds: tuple[int, ...]
    final: int


def _named_seed(seed: int, *namespace: str) -> int:
    """Derive a stable stream whose value is independent of candidate ordering."""

    encoded = "\x1f".join(namespace).encode()
    digest = hashlib.sha256(encoded).digest()
    words = [int.from_bytes(digest[offset : offset + 4], "big") for offset in range(0, 16, 4)]
    sequence = np.random.SeedSequence([seed, *words])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _candidate_seed_plan(
    seed: int,
    experiment: str,
    specs: Sequence[_CandidateSpec],
    fold_count: int,
) -> dict[str, _CandidateSeeds]:
    """Materialize every model seed so an artifact can reproduce each fit."""

    return {
        spec.name: _CandidateSeeds(
            folds=tuple(
                _named_seed(seed, experiment, "candidate", spec.name, f"fold-{fold_index}")
                for fold_index in range(fold_count)
            ),
            final=_named_seed(seed, experiment, "candidate", spec.name, "final"),
        )
        for spec in specs
    }


def _serialized_seed_plan(plan: dict[str, _CandidateSeeds]) -> dict[str, JsonValue]:
    return {
        name: {"folds": list(seeds.folds), "final": seeds.final} for name, seeds in plan.items()
    }


def _candidate_parameters(specs: Sequence[_CandidateSpec]) -> dict[str, JsonValue]:
    return {
        spec.name: {
            "preprocessing": spec.preprocessing,
            "estimator": spec.parameters,
        }
        for spec in specs
    }


def _fit_pipeline(
    spec: _CandidateSpec,
    features: FloatArray,
    targets: FloatArray,
    *,
    seed: int,
) -> FittedPipeline:
    if spec.preprocessing == "nonnegative":
        transformer: StandardScaler | _NonnegativeShift = _NonnegativeShift()
    else:
        transformer = StandardScaler()
    transformed = transformer.fit_transform(features)
    estimator = spec.factory(seed)
    estimator.fit(transformed, targets)
    return FittedPipeline(transformer=transformer, estimator=estimator)


def _write_candidate_tables(
    evaluations: Sequence[_Evaluation],
    specs: Sequence[_CandidateSpec],
    context: RunContext,
    *,
    prefix: str,
) -> tuple[str, str]:
    json_path = f"metrics/{prefix}_candidates.json"
    specifications = {spec.name: spec for spec in specs}
    records = [
        {
            "name": evaluation.result.name,
            "preprocessing": specifications[evaluation.result.name].preprocessing,
            "parameters": specifications[evaluation.result.name].parameters,
            "metrics": evaluation.result.metrics,
        }
        for evaluation in evaluations
    ]
    context.artifacts.write_json(json_path, records)
    metric_names = sorted(
        {name for evaluation in evaluations for name in evaluation.result.metrics}
    )
    csv_path = f"metrics/{prefix}_candidates.csv"
    with context.artifacts.atomic_target(csv_path) as temporary:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["candidate", "preprocessing", "parameters_json", *metric_names],
            )
            writer.writeheader()
            for evaluation in evaluations:
                spec = specifications[evaluation.result.name]
                writer.writerow(
                    {
                        "candidate": evaluation.result.name,
                        "preprocessing": spec.preprocessing,
                        "parameters_json": json.dumps(spec.parameters, sort_keys=True),
                        **evaluation.result.metrics,
                    }
                )
    return json_path, csv_path


def _loss_history(estimator: Estimator) -> tuple[float, ...]:
    raw = getattr(estimator, "loss_history_", ())
    return tuple(float(value) for value in cast(Sequence[float], raw))


def _validation_loss(estimator: Estimator) -> tuple[float, ...]:
    raw = getattr(estimator, "validation_loss_", ())
    return tuple(float(value) for value in cast(Sequence[float], raw))


def _regression_specs(config: ScratchRegressionBenchmarkConfig) -> tuple[_CandidateSpec, ...]:
    return (
        _CandidateSpec(
            name="mean_baseline",
            factory=lambda _seed: MeanRegressor(),
            parameters={},
        ),
        _CandidateSpec(
            name="linear_lstsq",
            factory=lambda _seed: LinearRegression(solver="lstsq"),
            parameters={
                "solver": "lstsq",
                "fit_intercept": True,
                "learning_rate": None,
                "max_iter": 10_000,
                "tol": 1e-8,
            },
        ),
        _CandidateSpec(
            name="linear_gradient_descent",
            factory=lambda seed: LinearRegression(
                solver="gradient_descent", max_iter=8_000, tol=1e-8, seed=seed
            ),
            parameters={
                "solver": "gradient_descent",
                "fit_intercept": True,
                "learning_rate": None,
                "max_iter": 8_000,
                "tol": 1e-8,
            },
        ),
        _CandidateSpec(
            name="ridge",
            factory=lambda _seed: Ridge(alpha=1.0),
            parameters={"alpha": 1.0, "fit_intercept": True},
        ),
        _CandidateSpec(
            name="lasso",
            factory=lambda _seed: Lasso(alpha=0.15, max_iter=15_000, tol=1e-7),
            parameters={
                "alpha": 0.15,
                "fit_intercept": True,
                "max_iter": 15_000,
                "tol": 1e-7,
            },
        ),
        _CandidateSpec(
            name="cart",
            factory=lambda seed: DecisionTreeRegressor(
                max_depth=config.max_depth,
                min_samples_leaf=4,
                random_state=seed,
            ),
            parameters={
                "criterion": "squared_error",
                "max_depth": config.max_depth,
                "min_samples_split": 2,
                "min_samples_leaf": 4,
                "max_features": None,
                "min_impurity_decrease": 0.0,
            },
        ),
        _CandidateSpec(
            name="random_forest",
            factory=lambda seed: RandomForestRegressor(
                n_estimators=config.forest_estimators,
                max_depth=config.max_depth,
                min_samples_leaf=3,
                max_features=0.75,
                oob_score=True,
                random_state=seed,
            ),
            parameters={
                "n_estimators": config.forest_estimators,
                "criterion": "squared_error",
                "max_depth": config.max_depth,
                "min_samples_split": 2,
                "min_samples_leaf": 3,
                "max_features": 0.75,
                "bootstrap": True,
                "max_samples": None,
                "oob_score": True,
            },
        ),
        _CandidateSpec(
            name="gradient_boosting",
            factory=lambda seed: GradientBoostingRegressor(
                n_estimators=config.boosting_estimators,
                learning_rate=0.08,
                max_depth=2,
                min_samples_leaf=3,
                early_stopping_rounds=8,
                validation_fraction=0.15,
                random_state=seed,
            ),
            parameters={
                "n_estimators": config.boosting_estimators,
                "learning_rate": 0.08,
                "max_depth": 2,
                "min_samples_split": 2,
                "min_samples_leaf": 3,
                "early_stopping_rounds": 8,
                "validation_fraction": 0.15,
                "tol": 1e-4,
            },
        ),
        _CandidateSpec(
            name="knn_distance",
            factory=lambda _seed: KNeighborsRegressor(
                n_neighbors=7, weights="distance", metric="euclidean"
            ),
            parameters={
                "n_neighbors": 7,
                "weights": "distance",
                "metric": "euclidean",
                "p": 2.0,
            },
        ),
    )


def _classification_specs(
    config: ScratchClassificationBenchmarkConfig,
) -> tuple[_CandidateSpec, ...]:
    return (
        _CandidateSpec(
            name="prior_baseline",
            factory=lambda _seed: PriorClassifier(),
            parameters={},
        ),
        _CandidateSpec(
            name="logistic_regression",
            factory=lambda seed: LogisticRegression(
                alpha=0.03, max_iter=8_000, tol=1e-7, seed=seed
            ),
            parameters={
                "alpha": 0.03,
                "fit_intercept": True,
                "learning_rate": None,
                "max_iter": 8_000,
                "tol": 1e-7,
            },
        ),
        _CandidateSpec(
            name="cart",
            factory=lambda seed: DecisionTreeClassifier(
                criterion="gini",
                max_depth=config.max_depth,
                min_samples_leaf=4,
                random_state=seed,
            ),
            parameters={
                "criterion": "gini",
                "max_depth": config.max_depth,
                "min_samples_split": 2,
                "min_samples_leaf": 4,
                "max_features": None,
                "min_impurity_decrease": 0.0,
            },
        ),
        _CandidateSpec(
            name="random_forest",
            factory=lambda seed: RandomForestClassifier(
                n_estimators=config.forest_estimators,
                max_depth=config.max_depth,
                min_samples_leaf=3,
                max_features="sqrt",
                oob_score=True,
                random_state=seed,
            ),
            parameters={
                "n_estimators": config.forest_estimators,
                "criterion": "gini",
                "max_depth": config.max_depth,
                "min_samples_split": 2,
                "min_samples_leaf": 3,
                "max_features": "sqrt",
                "bootstrap": True,
                "max_samples": None,
                "oob_score": True,
            },
        ),
        _CandidateSpec(
            name="gradient_boosting",
            factory=lambda seed: GradientBoostingClassifier(
                n_estimators=config.boosting_estimators,
                learning_rate=0.08,
                max_depth=2,
                min_samples_leaf=3,
                early_stopping_rounds=8,
                validation_fraction=0.15,
                random_state=seed,
            ),
            parameters={
                "n_estimators": config.boosting_estimators,
                "learning_rate": 0.08,
                "max_depth": 2,
                "min_samples_split": 2,
                "min_samples_leaf": 3,
                "early_stopping_rounds": 8,
                "validation_fraction": 0.15,
                "tol": 1e-4,
            },
        ),
        _CandidateSpec(
            name="knn_distance",
            factory=lambda _seed: KNeighborsClassifier(
                n_neighbors=9, weights="distance", metric="euclidean"
            ),
            parameters={
                "n_neighbors": 9,
                "weights": "distance",
                "metric": "euclidean",
                "p": 2.0,
            },
        ),
        _CandidateSpec(
            name="linear_svm",
            factory=lambda _seed: KernelSVMClassifier(
                C=1.0,
                kernel="linear",
                max_iter=5_000,
                max_passes=30,
                raise_on_nonconvergence=True,
            ),
            parameters={
                "C": 1.0,
                "kernel": "linear",
                "gamma": "scale",
                "degree": 3,
                "coef0": 0.0,
                "tol": 1e-3,
                "max_iter": 5_000,
                "max_passes": 30,
                "raise_on_nonconvergence": True,
            },
        ),
        _CandidateSpec(
            name="rbf_svm",
            factory=lambda _seed: KernelSVMClassifier(
                C=2.0,
                kernel="rbf",
                gamma="scale",
                max_iter=5_000,
                max_passes=30,
                raise_on_nonconvergence=True,
            ),
            parameters={
                "C": 2.0,
                "kernel": "rbf",
                "gamma": "scale",
                "degree": 3,
                "coef0": 0.0,
                "tol": 1e-3,
                "max_iter": 5_000,
                "max_passes": 30,
                "raise_on_nonconvergence": True,
            },
        ),
        _CandidateSpec(
            name="gaussian_nb",
            factory=lambda _seed: GaussianNB(var_smoothing=1e-9),
            parameters={"var_smoothing": 1e-9, "class_prior": None},
        ),
        _CandidateSpec(
            name="multinomial_nb",
            factory=lambda _seed: MultinomialNB(alpha=1.0),
            parameters={"alpha": 1.0, "fit_prior": True, "class_prior": None},
            preprocessing="nonnegative",
        ),
    )


class ScratchRegressionBenchmark:
    """Compare every applicable Sprint 2 regressor behind one held-out boundary."""

    def __init__(self, config: ScratchRegressionBenchmarkConfig) -> None:
        self._config = config

    def run(self, context: RunContext) -> RunResult:
        experiment = self._config.experiment
        data_seed = _named_seed(context.seed, experiment, "data")
        split_seed = _named_seed(context.seed, experiment, "split")
        fold_seed = _named_seed(context.seed, experiment, "folds")
        dataset = make_regression(
            n_samples=self._config.n_samples,
            n_features=self._config.n_features,
            n_informative=self._config.n_informative,
            noise=self._config.noise,
            coefficient_scale=18.0,
            seed=data_seed,
        )
        split = train_test_split(
            dataset.features,
            dataset.targets,
            test_size=self._config.test_size,
            seed=split_seed,
        )
        folds = kfold_indices(len(split.x_train), self._config.cv_folds, seed=fold_seed)
        specs = _regression_specs(self._config)
        seed_plan = _candidate_seed_plan(context.seed, experiment, specs, len(folds))
        evaluations: list[_Evaluation] = []
        for spec in specs:
            spec_seeds = seed_plan[spec.name]
            fold_scores: list[float] = []
            for fold_index, fold in enumerate(folds):
                try:
                    pipeline = _fit_pipeline(
                        spec,
                        split.x_train[fold.train],
                        split.y_train[fold.train],
                        seed=spec_seeds.folds[fold_index],
                    )
                    predicted = pipeline.predict(split.x_train[fold.validation])
                except Exception as error:
                    raise CandidateExecutionError(spec.name, f"CV fold {fold_index}") from error
                fold_scores.append(
                    root_mean_squared_error(split.y_train[fold.validation], predicted)
                )
            try:
                final = _fit_pipeline(
                    spec,
                    split.x_train,
                    split.y_train,
                    seed=spec_seeds.final,
                )
                train_prediction = final.predict(split.x_train)
                test_prediction = final.predict(split.x_test)
            except Exception as error:
                raise CandidateExecutionError(spec.name, "full training partition") from error
            metrics = {
                "cv_rmse_mean": float(np.mean(fold_scores)),
                "cv_rmse_std": float(np.std(fold_scores, ddof=1)),
                "train_rmse": root_mean_squared_error(split.y_train, train_prediction),
                "train_r2": r2_score(split.y_train, train_prediction),
                "test_mae": mean_absolute_error(split.y_test, test_prediction),
                "test_rmse": root_mean_squared_error(split.y_test, test_prediction),
                "test_r2": r2_score(split.y_test, test_prediction),
            }
            if isinstance(final.estimator, RandomForestRegressor):
                metrics["oob_r2"] = float(final.estimator.oob_score_)
                metrics["oob_coverage"] = float(final.estimator.oob_coverage_)
            evaluations.append(
                _Evaluation(
                    result=CandidateResult(name=spec.name, metrics=metrics),
                    fold_scores=tuple(fold_scores),
                    pipeline=final,
                    train_predictions=train_prediction,
                    test_predictions=test_prediction,
                )
            )

        winner = min(
            evaluations,
            key=lambda evaluation: (
                evaluation.result.metrics["cv_rmse_mean"],
                evaluation.result.name,
            ),
        )
        baseline = next(
            evaluation for evaluation in evaluations if evaluation.result.name == "mean_baseline"
        )
        artifacts = self._publish_evidence(
            evaluations,
            specs,
            winner,
            split,
            context,
            cv_folds=self._config.cv_folds,
        )
        return RunResult(
            experiment=self._config.experiment,
            paradigm=LearningParadigm.SUPERVISED,
            seed=context.seed,
            source=SourceMetadata(
                kind=SourceKind.GENERATOR,
                name="learning_atlas.make_regression",
                version="0.2.0",
                fingerprint_sha256=array_fingerprint(dataset.features, dataset.targets),
                target_used_for_fit=True,
                details={
                    "sample_count": self._config.n_samples,
                    "feature_count": self._config.n_features,
                    "informative_features": self._config.n_informative,
                    "train_samples": len(split.x_train),
                    "test_samples": len(split.x_test),
                    "cv_folds": self._config.cv_folds,
                    "coefficient_l2_norm": float(np.linalg.norm(dataset.coefficients)),
                    "candidate_parameters": _candidate_parameters(specs),
                    "seed_streams": {
                        "root": context.seed,
                        "derivation": "SHA-256 namespace plus NumPy SeedSequence to uint32",
                        "data": data_seed,
                        "split": split_seed,
                        "folds": fold_seed,
                        "candidates": _serialized_seed_plan(seed_plan),
                    },
                },
            ),
            selected_model=winner.result.name,
            metrics={
                **winner.result.metrics,
                "rmse_reduction_vs_mean": 1.0
                - winner.result.metrics["test_rmse"] / baseline.result.metrics["test_rmse"],
            },
            candidates=tuple(evaluation.result for evaluation in evaluations),
            artifacts=artifacts,
            notes=(
                "The winner is selected by training-fold RMSE; holdout results are descriptive.",
                "Every candidate receives the same folds and fold-fitted preprocessing.",
                "The synthetic generator exposes the true sparse coefficient vector.",
            ),
        )

    @staticmethod
    def _publish_evidence(
        evaluations: Sequence[_Evaluation],
        specs: Sequence[_CandidateSpec],
        winner: _Evaluation,
        split: SplitData,
        context: RunContext,
        *,
        cv_folds: int,
    ) -> dict[str, str]:
        x_train = split.x_train
        y_train = split.y_train
        y_test = split.y_test
        names = [evaluation.result.name for evaluation in evaluations]
        context_label = (
            f"root seed={context.seed} | train n={len(split.x_train)} | "
            f"holdout n={len(split.x_test)} | CV={cv_folds} folds"
        )
        model_path = "models/scratch_regression.joblib"
        with context.artifacts.atomic_target(model_path) as temporary:
            joblib.dump(winner.pipeline, temporary)
        json_path, csv_path = _write_candidate_tables(
            evaluations,
            specs,
            context,
            prefix="regression",
        )
        cv_path = cv_distribution_plot(
            names,
            [evaluation.fold_scores for evaluation in evaluations],
            metric_label="Validation RMSE (lower is better)",
            title="Regression cross-validation distributions — training partition only",
            context_label=context_label,
            artifacts=context.artifacts,
            relative_path="plots/scratch_regression_cv.png",
        )
        comparison_path = heldout_comparison_plot(
            names,
            [evaluation.result.metrics["cv_rmse_mean"] for evaluation in evaluations],
            [evaluation.result.metrics["cv_rmse_std"] for evaluation in evaluations],
            [evaluation.result.metrics["test_rmse"] for evaluation in evaluations],
            metric_label="RMSE (lower is better)",
            title="Regression selection evidence vs. untouched test",
            context_label=context_label,
            artifacts=context.artifacts,
            relative_path="plots/scratch_regression_comparison.png",
        )
        diagnostics_path = regression_diagnostics_plot(
            y_test,
            winner.test_predictions,
            model_name=winner.result.name,
            context_label=context_label,
            artifacts=context.artifacts,
            relative_path="plots/scratch_regression_diagnostics.png",
        )
        curves = {
            evaluation.result.name: _loss_history(evaluation.pipeline.estimator)
            for evaluation in evaluations
            if len(_loss_history(evaluation.pipeline.estimator)) > 1
        }
        optimization_path = optimization_plot(
            curves,
            title="Regression optimization and stagewise training loss",
            ylabel="Training objective (log scale)",
            context_label=context_label,
            artifacts=context.artifacts,
            relative_path="plots/scratch_regression_optimization.png",
        )

        scaled_train = StandardScaler().fit_transform(x_train)
        alphas = np.logspace(-3, 1, 9)
        ridge_norms: list[float] = []
        lasso_nonzero: list[int] = []
        for alpha in alphas:
            ridge = Ridge(alpha=float(alpha)).fit(scaled_train, y_train)
            lasso = Lasso(alpha=float(alpha), max_iter=20_000, tol=1e-7).fit(scaled_train, y_train)
            ridge_norms.append(float(np.linalg.norm(ridge.coef_)))
            lasso_nonzero.append(int(np.count_nonzero(lasso.coef_)))
        regularization_path = regularization_path_plot(
            alphas.tolist(),
            ridge_norms,
            lasso_nonzero,
            context_label=context_label,
            artifacts=context.artifacts,
            relative_path="plots/scratch_regression_regularization.png",
        )
        forest = next(
            evaluation for evaluation in evaluations if evaluation.result.name == "random_forest"
        )
        boosting = next(
            evaluation
            for evaluation in evaluations
            if evaluation.result.name == "gradient_boosting"
        )
        ensemble_path = ensemble_evidence_plot(
            {
                "random_forest": (
                    forest.result.metrics["oob_r2"],
                    forest.result.metrics["test_r2"],
                )
            },
            _loss_history(boosting.pipeline.estimator),
            _validation_loss(boosting.pipeline.estimator),
            score_label="R²",
            context_label=context_label,
            artifacts=context.artifacts,
            relative_path="plots/scratch_regression_ensembles.png",
        )
        return {
            "model": model_path,
            "candidate_metrics_json": json_path,
            "candidate_metrics_csv": csv_path,
            "cv_distribution_plot": cv_path,
            "comparison_plot": comparison_path,
            "diagnostics_plot": diagnostics_path,
            "optimization_plot": optimization_path,
            "regularization_plot": regularization_path,
            "ensemble_plot": ensemble_path,
        }


class ScratchClassificationBenchmark:
    """Compare every Sprint 2 classifier without manufacturing SVM probabilities."""

    def __init__(self, config: ScratchClassificationBenchmarkConfig) -> None:
        self._config = config

    def run(self, context: RunContext) -> RunResult:
        experiment = self._config.experiment
        data_seed = _named_seed(context.seed, experiment, "data")
        split_seed = _named_seed(context.seed, experiment, "split")
        fold_seed = _named_seed(context.seed, experiment, "folds")
        dataset = make_classification(
            n_samples=self._config.n_samples,
            n_features=self._config.n_features,
            n_informative=self._config.n_informative,
            class_sep=self._config.class_sep,
            flip_y=self._config.label_noise,
            seed=data_seed,
        )
        split = train_test_split(
            dataset.features,
            dataset.targets,
            test_size=self._config.test_size,
            seed=split_seed,
            stratify=dataset.targets,
        )
        folds = stratified_kfold_indices(
            split.y_train,
            self._config.cv_folds,
            seed=fold_seed,
        )
        specs = _classification_specs(self._config)
        seed_plan = _candidate_seed_plan(context.seed, experiment, specs, len(folds))
        evaluations: list[_Evaluation] = []
        for spec in specs:
            spec_seeds = seed_plan[spec.name]
            fold_scores: list[float] = []
            for fold_index, fold in enumerate(folds):
                try:
                    pipeline = _fit_pipeline(
                        spec,
                        split.x_train[fold.train],
                        split.y_train[fold.train],
                        seed=spec_seeds.folds[fold_index],
                    )
                    predicted = pipeline.predict(split.x_train[fold.validation])
                except Exception as error:
                    raise CandidateExecutionError(spec.name, f"CV fold {fold_index}") from error
                fold_scores.append(f1_score(split.y_train[fold.validation], predicted))
            try:
                final = _fit_pipeline(
                    spec,
                    split.x_train,
                    split.y_train,
                    seed=spec_seeds.final,
                )
                train_prediction = final.predict(split.x_train)
                test_prediction = final.predict(split.x_test)
            except Exception as error:
                raise CandidateExecutionError(spec.name, "full training partition") from error

            probability: FloatArray | None = None
            metrics = {
                "cv_macro_f1_mean": float(np.mean(fold_scores)),
                "cv_macro_f1_std": float(np.std(fold_scores, ddof=1)),
                "train_accuracy": accuracy_score(split.y_train, train_prediction),
                "train_macro_f1": f1_score(split.y_train, train_prediction),
                "test_accuracy": accuracy_score(split.y_test, test_prediction),
                "test_macro_f1": f1_score(split.y_test, test_prediction),
            }
            if isinstance(final.estimator, ProbabilisticClassifier):
                probability = final.predict_proba(split.x_test)
                metrics["test_log_loss"] = log_loss(split.y_test, probability, labels=[0.0, 1.0])
                metrics["test_brier"] = float(np.mean((split.y_test - probability[:, 1]) ** 2))
            if isinstance(final.estimator, RandomForestClassifier):
                metrics["oob_accuracy"] = float(final.estimator.oob_score_)
                metrics["oob_coverage"] = float(final.estimator.oob_coverage_)
            if isinstance(final.estimator, KernelSVMClassifier):
                metrics["solver_converged"] = float(final.estimator.converged_)
                metrics["max_kkt_violation"] = final.estimator.max_kkt_violation_
                metrics["dual_objective"] = final.estimator.dual_objective_
                metrics["support_vector_count"] = float(len(final.estimator.support_))
                metrics["support_vector_fraction"] = float(
                    len(final.estimator.support_) / len(split.x_train)
                )
            evaluations.append(
                _Evaluation(
                    result=CandidateResult(name=spec.name, metrics=metrics),
                    fold_scores=tuple(fold_scores),
                    pipeline=final,
                    train_predictions=train_prediction,
                    test_predictions=test_prediction,
                    test_probabilities=probability,
                )
            )

        winner = min(
            evaluations,
            key=lambda evaluation: (
                -evaluation.result.metrics["cv_macro_f1_mean"],
                evaluation.result.name,
            ),
        )
        baseline = next(
            evaluation for evaluation in evaluations if evaluation.result.name == "prior_baseline"
        )
        artifacts = self._publish_evidence(
            evaluations,
            specs,
            winner,
            split,
            context,
            cv_folds=self._config.cv_folds,
        )
        return RunResult(
            experiment=self._config.experiment,
            paradigm=LearningParadigm.SUPERVISED,
            seed=context.seed,
            source=SourceMetadata(
                kind=SourceKind.GENERATOR,
                name="learning_atlas.make_classification",
                version="0.2.0",
                fingerprint_sha256=array_fingerprint(dataset.features, dataset.targets),
                target_used_for_fit=True,
                details={
                    "sample_count": self._config.n_samples,
                    "feature_count": self._config.n_features,
                    "informative_features": self._config.n_informative,
                    "train_samples": len(split.x_train),
                    "test_samples": len(split.x_test),
                    "cv_folds": self._config.cv_folds,
                    "test_positive_prevalence": float(np.mean(split.y_test)),
                    "candidate_parameters": _candidate_parameters(specs),
                    "seed_streams": {
                        "root": context.seed,
                        "derivation": "SHA-256 namespace plus NumPy SeedSequence to uint32",
                        "data": data_seed,
                        "split": split_seed,
                        "folds": fold_seed,
                        "candidates": _serialized_seed_plan(seed_plan),
                    },
                },
            ),
            selected_model=winner.result.name,
            metrics={
                **winner.result.metrics,
                "macro_f1_gain_vs_prior": winner.result.metrics["test_macro_f1"]
                - baseline.result.metrics["test_macro_f1"],
            },
            candidates=tuple(evaluation.result for evaluation in evaluations),
            artifacts=artifacts,
            notes=(
                "The winner is selected by training-fold macro-F1; holdout results are descriptive.",
                "SVM candidates report decision metrics only; no uncalibrated sigmoid is presented.",
                "MultinomialNB receives a fold-fitted nonnegative shift and clipped unseen minima.",
                "Reliability curves are descriptive and do not establish formal calibration.",
            ),
        )

    @staticmethod
    def _publish_evidence(
        evaluations: Sequence[_Evaluation],
        specs: Sequence[_CandidateSpec],
        winner: _Evaluation,
        split: SplitData,
        context: RunContext,
        *,
        cv_folds: int,
    ) -> dict[str, str]:
        x_train = split.x_train
        x_test = split.x_test
        y_test = split.y_test
        names = [evaluation.result.name for evaluation in evaluations]
        context_label = (
            f"root seed={context.seed} | train n={len(split.x_train)} | "
            f"holdout n={len(split.x_test)} | stratified CV={cv_folds} folds"
        )
        model_path = "models/scratch_classification.joblib"
        with context.artifacts.atomic_target(model_path) as temporary:
            joblib.dump(winner.pipeline, temporary)
        json_path, csv_path = _write_candidate_tables(
            evaluations,
            specs,
            context,
            prefix="classification",
        )
        cv_path = cv_distribution_plot(
            names,
            [evaluation.fold_scores for evaluation in evaluations],
            metric_label="Validation macro-F1 (higher is better)",
            title="Classification cross-validation distributions — training partition only",
            context_label=context_label,
            artifacts=context.artifacts,
            relative_path="plots/scratch_classification_cv.png",
        )
        comparison_path = heldout_comparison_plot(
            names,
            [evaluation.result.metrics["cv_macro_f1_mean"] for evaluation in evaluations],
            [evaluation.result.metrics["cv_macro_f1_std"] for evaluation in evaluations],
            [evaluation.result.metrics["test_macro_f1"] for evaluation in evaluations],
            metric_label="Macro-F1 (higher is better)",
            title="Classification selection evidence vs. untouched test",
            context_label=context_label,
            artifacts=context.artifacts,
            relative_path="plots/scratch_classification_comparison.png",
        )
        probability_by_model = {
            evaluation.result.name: evaluation.test_probabilities[:, 1]
            for evaluation in evaluations
            if evaluation.test_probabilities is not None
        }
        diagnostics_path = classification_diagnostics_plot(
            y_test,
            winner.test_predictions,
            probability_by_model,
            selected_model=winner.result.name,
            reliability_models=("logistic_regression", winner.result.name),
            context_label=context_label,
            artifacts=context.artifacts,
            relative_path="plots/scratch_classification_diagnostics.png",
        )
        curves = {
            evaluation.result.name: _loss_history(evaluation.pipeline.estimator)
            for evaluation in evaluations
            if len(_loss_history(evaluation.pipeline.estimator)) > 1
        }
        optimization_path = optimization_plot(
            curves,
            title="Classification optimization and stagewise log loss",
            ylabel="Training objective (log scale)",
            context_label=context_label,
            artifacts=context.artifacts,
            relative_path="plots/scratch_classification_optimization.png",
        )
        evaluation_by_name = {
            evaluation.result.name: evaluation.pipeline for evaluation in evaluations
        }
        representative_names = (
            "logistic_regression",
            "cart",
            "random_forest",
            "knn_distance",
            "linear_svm",
            "rbf_svm",
        )
        boundaries_path = decision_boundary_plot(
            {name: evaluation_by_name[name] for name in representative_names},
            x_test,
            y_test,
            x_train,
            context_label=context_label,
            artifacts=context.artifacts,
            relative_path="plots/scratch_classification_boundaries.png",
        )
        forest = next(
            evaluation for evaluation in evaluations if evaluation.result.name == "random_forest"
        )
        boosting = next(
            evaluation
            for evaluation in evaluations
            if evaluation.result.name == "gradient_boosting"
        )
        ensemble_path = ensemble_evidence_plot(
            {
                "random_forest": (
                    forest.result.metrics["oob_accuracy"],
                    forest.result.metrics["test_accuracy"],
                )
            },
            _loss_history(boosting.pipeline.estimator),
            _validation_loss(boosting.pipeline.estimator),
            score_label="Accuracy",
            context_label=context_label,
            artifacts=context.artifacts,
            relative_path="plots/scratch_classification_ensembles.png",
        )
        return {
            "model": model_path,
            "candidate_metrics_json": json_path,
            "candidate_metrics_csv": csv_path,
            "cv_distribution_plot": cv_path,
            "comparison_plot": comparison_path,
            "diagnostics_plot": diagnostics_path,
            "optimization_plot": optimization_path,
            "decision_boundaries_plot": boundaries_path,
            "ensemble_plot": ensemble_path,
        }
