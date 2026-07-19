"""Leakage-safe supervised classification model comparison."""

from dataclasses import dataclass

import joblib
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from sklearn.calibration import calibration_curve
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from learning_atlas.core.config import ClassificationBenchmarkConfig
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
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class ClassificationData:
    """Stratified train/test boundary established before preprocessing."""

    x_train: FloatArray
    x_test: FloatArray
    y_train: IntArray
    y_test: IntArray
    feature_names: tuple[str, ...]
    fingerprint: str


def load_classification_data(seed: int, test_size: float) -> ClassificationData:
    """Load the bundled breast-cancer dataset and create a stratified holdout."""

    dataset = load_breast_cancer()
    features = np.asarray(dataset.data, dtype=np.float64)
    target = np.asarray(dataset.target, dtype=np.int64)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=seed,
        stratify=target,
    )
    return ClassificationData(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=tuple(str(name) for name in dataset.feature_names),
        fingerprint=array_fingerprint(features, target),
    )


def build_candidates(config: ClassificationBenchmarkConfig) -> dict[str, Pipeline]:
    """Build baseline, linear, and nonlinear candidates with owned preprocessing."""

    return {
        "dummy_prior": Pipeline([("model", DummyClassifier(strategy="prior"))]),
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=config.logistic_c,
                        max_iter=2_000,
                        random_state=config.seed,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=config.forest_estimators,
                        max_depth=config.forest_max_depth,
                        random_state=config.seed,
                        n_jobs=1,
                    ),
                )
            ]
        ),
    }


class ClassificationBenchmark:
    """Compare naive, regularized linear, and ensemble classifiers."""

    def __init__(self, config: ClassificationBenchmarkConfig) -> None:
        self._config = config

    def run(self, context: RunContext) -> RunResult:
        data = load_classification_data(context.seed, self._config.test_size)
        candidates = build_candidates(self._config)
        cross_validation = StratifiedKFold(
            n_splits=self._config.cv_folds,
            shuffle=True,
            random_state=context.seed,
        )

        candidate_results: list[CandidateResult] = []
        probabilities: dict[str, FloatArray] = {}
        predictions: dict[str, IntArray] = {}
        for name, pipeline in candidates.items():
            cv_scores = cross_val_score(
                pipeline,
                data.x_train,
                data.y_train,
                scoring="roc_auc",
                cv=cross_validation,
                n_jobs=1,
            )
            pipeline.fit(data.x_train, data.y_train)
            probability = np.asarray(pipeline.predict_proba(data.x_test)[:, 1], dtype=np.float64)
            prediction = np.asarray(pipeline.predict(data.x_test), dtype=np.int64)
            probabilities[name] = probability
            predictions[name] = prediction
            candidate_results.append(
                CandidateResult(
                    name=name,
                    metrics={
                        "cv_roc_auc_mean": float(np.mean(cv_scores)),
                        "cv_roc_auc_std": float(np.std(cv_scores, ddof=1)),
                        "test_accuracy": float(accuracy_score(data.y_test, prediction)),
                        "test_balanced_accuracy": float(
                            balanced_accuracy_score(data.y_test, prediction)
                        ),
                        "test_f1": float(f1_score(data.y_test, prediction)),
                        "test_roc_auc": float(roc_auc_score(data.y_test, probability)),
                        "test_log_loss": float(log_loss(data.y_test, probability)),
                        "test_brier": float(brier_score_loss(data.y_test, probability)),
                    },
                )
            )

        winner = max(candidate_results, key=lambda candidate: candidate.metrics["cv_roc_auc_mean"])
        dummy = next(
            candidate for candidate in candidate_results if candidate.name == "dummy_prior"
        )
        headline_metrics = {
            **winner.metrics,
            "roc_auc_gain_vs_dummy": winner.metrics["test_roc_auc"] - dummy.metrics["test_roc_auc"],
        }

        model_path = "models/classification.joblib"
        with context.artifacts.atomic_target(model_path) as temporary:
            joblib.dump(candidates[winner.name], temporary)
        diagnostics_path = self._diagnostics_plot(
            data.y_test,
            predictions[winner.name],
            probabilities,
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
                name="scikit-learn breast cancer Wisconsin diagnostic",
                version="scikit-learn-1.9",
                fingerprint_sha256=data.fingerprint,
                target_used_for_fit=True,
                details={
                    "sample_count": len(data.x_train) + len(data.x_test),
                    "feature_count": data.x_train.shape[1],
                    "split_seed": context.seed,
                    "train_samples": len(data.x_train),
                    "test_samples": len(data.x_test),
                    "positive_test_prevalence": float(np.mean(data.y_test)),
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
                "The winner is selected by training-fold ROC AUC, never held-out performance.",
                "Threshold-dependent and probabilistic metrics are both reported.",
                "The bundled dataset keeps the reference run network-independent.",
            ),
        )

    @staticmethod
    def _diagnostics_plot(
        observed: IntArray,
        predicted: IntArray,
        probabilities: dict[str, FloatArray],
        model_name: str,
        context: RunContext,
    ) -> str:
        figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.3))

        for name, probability in probabilities.items():
            false_positive_rate, true_positive_rate, _ = roc_curve(observed, probability)
            axes[0].plot(false_positive_rate, true_positive_rate, label=name.replace("_", " "))
        axes[0].plot([0.0, 1.0], [0.0, 1.0], color="black", linestyle="--")
        axes[0].set(
            title="Held-out ROC curves", xlabel="False-positive rate", ylabel="True-positive rate"
        )
        axes[0].legend(fontsize=8)

        matrix = confusion_matrix(observed, predicted)
        axes[1].imshow(matrix, cmap="Blues")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axes[1].text(column, row, str(matrix[row, column]), ha="center", va="center")
        axes[1].set(
            title=f"Confusion matrix — {model_name}",
            xlabel="Predicted class",
            ylabel="Observed class",
            xticks=(0, 1),
            yticks=(0, 1),
        )

        selected_probability = probabilities[model_name]
        observed_rate, predicted_rate = calibration_curve(
            observed,
            selected_probability,
            n_bins=8,
            strategy="quantile",
        )
        axes[2].plot(predicted_rate, observed_rate, marker="o")
        axes[2].plot([0.0, 1.0], [0.0, 1.0], color="black", linestyle="--")
        axes[2].set(
            title="Reliability curve",
            xlabel="Mean predicted probability",
            ylabel="Observed positive rate",
        )

        for axis in axes:
            axis.grid(alpha=0.2)
        figure.suptitle("Held-out classification diagnostics")
        return publish_figure(figure, context.artifacts, "plots/classification_diagnostics.png")

    @staticmethod
    def _comparison_plot(
        candidates: list[CandidateResult],
        context: RunContext,
    ) -> str:
        names = [candidate.name.replace("_", " ") for candidate in candidates]
        positions = np.arange(len(candidates))
        width = 0.27
        figure, axis = plt.subplots(figsize=(8.0, 4.8))
        for index, (metric, label) in enumerate(
            (
                ("cv_roc_auc_mean", "cross-validation ROC AUC"),
                ("test_roc_auc", "held-out ROC AUC"),
                ("test_accuracy", "held-out accuracy"),
            )
        ):
            axis.bar(
                positions + (index - 1) * width,
                [candidate.metrics[metric] for candidate in candidates],
                width,
                label=label,
            )
        axis.set(
            title="Classification candidate comparison",
            ylabel="Score (higher is better)",
            ylim=(0.0, 1.05),
            xticks=positions,
            xticklabels=names,
        )
        axis.legend(fontsize=8)
        axis.grid(axis="y", alpha=0.2)
        return publish_figure(
            figure, context.artifacts, "plots/classification_model_comparison.png"
        )
