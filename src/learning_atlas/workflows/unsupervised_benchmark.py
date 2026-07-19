"""Atomic publication for the Sprint 3 unsupervised evidence suite."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from learning_atlas.core.contracts import RunResult
from learning_atlas.core.reproducibility import environment_metadata
from learning_atlas.workflows.suite import run_suite

_EXPECTED_EXPERIMENTS = {
    "scratch_clustering_benchmark",
    "scratch_representation_benchmark",
}


def _validate_results(results: tuple[RunResult, ...]) -> None:
    observed = {result.experiment for result in results}
    if observed != _EXPECTED_EXPERIMENTS:
        expected = ", ".join(sorted(_EXPECTED_EXPERIMENTS))
        actual = ", ".join(sorted(observed)) or "none"
        msg = f"unsupervised benchmark requires exactly {expected}; received {actual}"
        raise ValueError(msg)


def _write_reports(root: Path, results: tuple[RunResult, ...]) -> None:
    payload = {
        result.experiment: {
            "selected_model": result.selected_model,
            "metrics": result.metrics,
            "candidates": [candidate.model_dump(mode="json") for candidate in result.candidates],
            "source": result.source.model_dump(mode="json"),
            "notes": list(result.notes),
        }
        for result in results
    }
    (root / "comparison.json").write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows: list[dict[str, str | float]] = []
    for result in results:
        for candidate in result.candidates:
            row: dict[str, str | float] = {
                "experiment": result.experiment,
                "candidate": candidate.name,
                "selected": str(candidate.name == result.selected_model).lower(),
            }
            row.update(candidate.metrics)
            rows.append(row)
    metric_columns = sorted(
        {key for row in rows for key in row if key not in {"experiment", "candidate", "selected"}}
    )
    with (root / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["experiment", "candidate", "selected", *metric_columns],
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Sprint 3 unsupervised benchmark",
        "",
        "Selection uses features, internal geometry, and named perturbation stability only.",
        "Ground-truth labels are reserved for retrospective evaluation and plot coloring.",
        "",
        "| Study | Selected method | Label-free selection evidence | Retrospective evidence |",
        "|---|---|---:|---:|",
    ]
    for result in sorted(results, key=lambda item: item.experiment):
        if result.experiment == "scratch_clustering_benchmark":
            study = "Heterogeneous clustering"
            internal = f"score {result.metrics['selection_score']:.3f}"
            external = f"mean ARI {result.metrics['retrospective_ari_mean']:.3f}"
        else:
            study = "Representation learning"
            internal = f"neighbor overlap {result.metrics['neighborhood_preservation']:.3f}"
            external = f"ARI {result.metrics['retrospective_ari']:.3f}"
        lines.append(f"| {study} | {result.selected_model} | {internal} | {external} |")
    (root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(root: Path, results: tuple[RunResult, ...]) -> None:
    manifest_path = root / "benchmark_manifest.json"
    checksums = {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    payload = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "workflow": "sprint-03-unsupervised-benchmark",
        "experiments": [
            {"name": result.experiment, "seed": result.seed}
            for result in sorted(results, key=lambda item: item.experiment)
        ],
        "environment": environment_metadata(),
        "root_reports": ["comparison.csv", "comparison.json", "comparison.md"],
        "artifacts_sha256": checksums,
    }
    manifest_path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_unsupervised_benchmark(config_dir: Path, output_dir: Path) -> tuple[RunResult, ...]:
    """Run both Sprint 3 studies and atomically publish aggregate evidence."""

    target = output_dir.resolve()
    if target.exists() and any(target.iterdir()):
        msg = f"refusing to overwrite non-empty benchmark directory: {target}"
        raise FileExistsError(msg)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        results = run_suite(config_dir, staging / "experiments")
        _validate_results(results)
        _write_reports(staging, results)
        _write_manifest(staging, results)
        if target.exists():
            target.rmdir()
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return results
