"""Publish complete fixed-seed parity and timing evidence into a new directory."""

import argparse
import hashlib
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from learning_atlas.core.artifacts import ArtifactStore
from learning_atlas.core.reproducibility import environment_metadata
from learning_atlas.reporting.parity import measure_reference_parity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        parser.error("output exists; use a new directory and review all evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".parity-", dir=output.parent))
    try:
        observations = measure_reference_parity()
        rows = [asdict(row) for row in observations]
        store = ArtifactStore(staging)
        store.write_json(
            "observations.json",
            {
                "schema_version": 1,
                "seeds": [17, 29, 43],
                "train_rows": 120,
                "test_rows": 40,
                "features": 5,
                "rows": rows,
                "environment": environment_metadata(),
                "selection": "None; every fixed pair and seed is reported",
                "limitations": [
                    "Synthetic numerical qualification, not domain generalization.",
                    "No stochastic ensemble, clustering, neural or RL prediction-parity claim.",
                    "Timing samples include cold calls, with no warmup; they are not CI performance gates.",
                ],
            },
        )
        sns.set_theme(style="whitegrid", palette="colorblind")
        figure, axes = plt.subplots(1, 2, figsize=(15, 6), layout="constrained")
        try:
            sns.stripplot(
                data={
                    "family": [row.family for row in observations],
                    "error / tolerance": [
                        row.max_prediction_error / row.tolerance for row in observations
                    ],
                },
                x="error / tolerance",
                y="family",
                jitter=False,
                ax=axes[0],
            )
            axes[0].axvline(1, color="black", linestyle="--", label="Agreement limit")
            axes[0].set(
                xscale="symlog",
                xlim=(
                    -0.3,
                    max(row.max_prediction_error / row.tolerance for row in observations) * 3,
                ),
                title="Every agreement result; symlog scale, ≤1 passes",
            )
            axes[0].legend()
            times = [
                {"family": row.family, "implementation": name, "fit milliseconds": seconds * 1000}
                for row in observations
                for name, seconds in (
                    ("Atlas", row.scratch_fit_seconds),
                    ("scikit-learn", row.oracle_fit_seconds),
                )
            ]
            sns.stripplot(
                data={key: [row[key] for row in times] for key in times[0]},
                x="fit milliseconds",
                y="family",
                hue="implementation",
                dodge=True,
                jitter=False,
                ax=axes[1],
            )
            axes[1].set(xscale="log", title="Observed fit time, log scale; no speedup claim")
            figure.suptitle(
                "Atlas Sprint 6: fixed synthetic oracle qualification\nSeeds 17, 29, 43; 120 train / 40 untouched test rows; 5 features; all 27 observations",
                fontsize=12,
            )
            figure.savefig(store.resolve("parity.png"), dpi=150)
        finally:
            plt.close(figure)
        store.write_json(
            "manifest.json",
            {
                "schema_version": 1,
                "files": {
                    str(path): hashlib.sha256((staging / path).read_bytes()).hexdigest()
                    for path in store.files()
                },
            },
        )
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    if not all(row.passed for row in observations):
        parser.exit(1, "Numerical agreement failed; complete report retained for inspection.\n")
    print(f"27/27 fixed-seed comparisons passed; evidence: {output}")


if __name__ == "__main__":
    main()
