"""Leakage-safe supervised regression model comparison."""

from dataclasses import dataclass

import joblib
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from learning_atlas.core.config import RegressionBenchmarkConfig
from learning_atlas.core.contracts import (
    CandidateResult,
    LearningParadigm,
    RunContext,
    RunResult,
    SourceKind,
    SourceMetadata,
)
from learning_atlas.core.data import array_fingerprint
from learning_atlas.reporting.plots import publish_figure

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RegressionData:
    """Explicit train/test boundary used to prevent preprocessing leakage."""

    x_train: FloatArray
    x_test: FloatArray
    y_train: FloatArray
    y_test: FloatArray
    feature_names: tuple[str, ...]
    fingerprint: str


def load_regression_data(seed: int, test_size: float) -> RegressionData:
    """Load the bundled diabetes dataset and split before any preprocessing."""

    dataset = load_diabetes()
    features = np.asarray(dataset.data, dtype=np.float64)
    target = np.asarray(dataset.target, dtype=np.float64)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=seed,
    )
    return RegressionData(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=tuple(str(name) for name in dataset.feature_names),
        fingerprint=array_fingerprint(features, target),
    )


def build_candidates(config: RegressionBenchmarkConfig) -> dict[str, Pipeline]:
    """Build unfitted candidates with preprocessing contained inside each pipeline."""

    return {
        "dummy_mean": Pipeline([("model", DummyRegressor(strategy="mean"))]),
        "ridge": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=config.ridge_alpha)),
            ]
        ),
        "random_forest": Pipeline(
            [
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=config.forest_estimators,
                        max_depth=config.forest_max_depth,
                        random_state=config.seed,
                        n_jobs=1,
                    ),
                )
            ]
        ),
    }


class RegressionBenchmark:
    """Compare baseline, regularized linear, and ensemble regressors."""

    def __init__(self, config: RegressionBenchmarkConfig) -> None:
        self._config = config

    def run(self, context: RunContext) -> RunResult:
        data = load_regression_data(context.seed, self._config.test_size)
        candidates = build_candidates(self._config)
        cross_validation = KFold(
            n_splits=self._config.cv_folds,
            shuffle=True,
            random_state=context.seed,
        )

        candidate_results: list[CandidateResult] = []
        predictions: dict[str, FloatArray] = {}
        for name, pipeline in candidates.items():
            cv_scores = -cross_val_score(
                pipeline,
                data.x_train,
                data.y_train,
                scoring="neg_root_mean_squared_error",
                cv=cross_validation,
                n_jobs=1,
            )
            pipeline.fit(data.x_train, data.y_train)
            prediction = np.asarray(pipeline.predict(data.x_test), dtype=np.float64)
            predictions[name] = prediction
            candidate_results.append(
                CandidateResult(
                    name=name,
                    metrics={
                        "cv_rmse_mean": float(np.mean(cv_scores)),
                        "cv_rmse_std": float(np.std(cv_scores, ddof=1)),
                        "test_mae": float(mean_absolute_error(data.y_test, prediction)),
                        "test_rmse": float(root_mean_squared_error(data.y_test, prediction)),
                        "test_r2": float(r2_score(data.y_test, prediction)),
                    },
                )
            )

        winner = min(candidate_results, key=lambda candidate: candidate.metrics["cv_rmse_mean"])
        dummy = next(candidate for candidate in candidate_results if candidate.name == "dummy_mean")
        headline_metrics = {
            **winner.metrics,
            "rmse_reduction_vs_dummy": 1.0
            - winner.metrics["test_rmse"] / dummy.metrics["test_rmse"],
        }

        model_path = "models/regression.joblib"
        with context.artifacts.atomic_target(model_path) as temporary:
            joblib.dump(candidates[winner.name], temporary)

        diagnostics_path = self._diagnostics_plot(
            data.y_test,
            predictions[winner.name],
            winner.name,
            context,
        )
        comparison_path = self._comparison_plot(candidate_results, context)
        return RunResult(
            experiment=self._config.experiment,
            paradigm=LearningParadigm.SUPERVISED,
            seed=context.seed,
            source=SourceMetadata(
                kind=SourceKind.DATASET,
                name="scikit-learn diabetes",
                version="scikit-learn-1.9",
                fingerprint_sha256=data.fingerprint,
                target_used_for_fit=True,
                details={
                    "sample_count": len(data.x_train) + len(data.x_test),
                    "feature_count": data.x_train.shape[1],
                    "split_seed": context.seed,
                    "train_samples": len(data.x_train),
                    "test_samples": len(data.x_test),
                    "feature_names": list(data.feature_names),
                },
            ),
            selected_model=winner.name,
            metrics=headline_metrics,
            candidates=tuple(candidate_results),
            artifacts={
                "model": model_path,
                "diagnostics_plot": diagnostics_path,
                "model_comparison_plot": comparison_path,
            },
            notes=(
                "The winner is selected by training-fold cross-validation, not test performance.",
                "The bundled dataset keeps the reference run network-independent.",
            ),
        )

    @staticmethod
    def _diagnostics_plot(
        observed: FloatArray,
        predicted: FloatArray,
        model_name: str,
        context: RunContext,
    ) -> str:
        residuals = observed - predicted
        figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))

        lower = float(min(np.min(observed), np.min(predicted)))
        upper = float(max(np.max(observed), np.max(predicted)))
        axes[0].scatter(observed, predicted, alpha=0.72, edgecolor="none")
        axes[0].plot([lower, upper], [lower, upper], color="black", linestyle="--")
        axes[0].set(title="Observed vs. predicted", xlabel="Observed", ylabel="Predicted")

        axes[1].scatter(predicted, residuals, alpha=0.72, edgecolor="none")
        axes[1].axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        axes[1].set(title="Residual structure", xlabel="Prediction", ylabel="Residual")

        axes[2].hist(residuals, bins=18, alpha=0.82, edgecolor="white")
        axes[2].axvline(0.0, color="black", linewidth=1.0, linestyle="--")
        axes[2].set(title="Residual distribution", xlabel="Residual", ylabel="Count")

        for axis in axes:
            axis.grid(alpha=0.2)
        figure.suptitle(f"Held-out regression diagnostics — {model_name}")
        return publish_figure(figure, context.artifacts, "plots/regression_diagnostics.png")

    @staticmethod
    def _comparison_plot(
        candidates: list[CandidateResult],
        context: RunContext,
    ) -> str:
        names = [candidate.name.replace("_", " ") for candidate in candidates]
        cross_validated = [candidate.metrics["cv_rmse_mean"] for candidate in candidates]
        cross_validated_std = [candidate.metrics["cv_rmse_std"] for candidate in candidates]
        held_out = [candidate.metrics["test_rmse"] for candidate in candidates]
        positions = np.arange(len(candidates))
        width = 0.36

        figure, axis = plt.subplots(figsize=(8.0, 4.8))
        axis.bar(
            positions - width / 2,
            cross_validated,
            width,
            yerr=cross_validated_std,
            capsize=4,
            label="cross-validation RMSE",
        )
        axis.bar(positions + width / 2, held_out, width, label="held-out RMSE")
        axis.set(
            title="Regression candidate comparison",
            ylabel="RMSE (lower is better)",
            xticks=positions,
            xticklabels=names,
        )
        axis.legend()
        axis.grid(axis="y", alpha=0.2)
        return publish_figure(figure, context.artifacts, "plots/regression_model_comparison.png")
