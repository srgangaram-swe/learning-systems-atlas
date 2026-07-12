"""Transactional experiment execution and manifest generation."""

import hashlib
import json
import os
import shutil
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from learning_atlas.core.artifacts import ArtifactStore
from learning_atlas.core.config import ExperimentConfig
from learning_atlas.core.contracts import RunContext, RunResult
from learning_atlas.core.registry import REGISTRY, build_experiment
from learning_atlas.core.reproducibility import environment_metadata


def _canonical_config(config: ExperimentConfig) -> tuple[dict[str, object], str]:
    payload = config.model_dump(mode="json")
    canonical = json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
    fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
    return payload, fingerprint


def _execute(config: ExperimentConfig, store: ArtifactStore) -> RunResult:
    config_payload, config_fingerprint = _canonical_config(config)
    store.write_json("resolved_config.json", config_payload)

    started = time.perf_counter()
    result = build_experiment(config).run(RunContext(seed=config.seed, artifacts=store))
    elapsed = time.perf_counter() - started
    _validate_result(config, result, store)

    common_artifacts = {
        **result.artifacts,
        "configuration": "resolved_config.json",
        "manifest": "manifest.json",
        "result": "result.json",
    }
    result = result.model_copy(update={"artifacts": common_artifacts})
    store.write_json("result.json", result.model_dump(mode="json"))

    checksums = {
        str(relative): store.sha256(str(relative))
        for relative in store.files()
        if str(relative) != "manifest.json"
    }
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "experiment": result.experiment,
        "paradigm": result.paradigm.value,
        "seed": result.seed,
        "config_sha256": config_fingerprint,
        "duration_seconds": elapsed,
        "environment": environment_metadata(),
        "artifacts_sha256": checksums,
    }
    store.write_json("manifest.json", manifest)
    return result


def _validate_result(
    config: ExperimentConfig,
    result: RunResult,
    store: ArtifactStore,
) -> None:
    """Verify experiment identity and every declared artifact before publication."""

    expected_paradigm = REGISTRY[config.experiment].paradigm
    if result.experiment != config.experiment:
        msg = f"result experiment does not match config: {result.experiment} != {config.experiment}"
        raise ValueError(msg)
    if result.seed != config.seed:
        msg = f"result seed does not match config: {result.seed} != {config.seed}"
        raise ValueError(msg)
    if result.paradigm is not expected_paradigm:
        msg = f"result paradigm does not match registry: {result.paradigm} != {expected_paradigm}"
        raise ValueError(msg)

    candidate_names = [candidate.name for candidate in result.candidates]
    if len(candidate_names) != len(set(candidate_names)):
        msg = "candidate names must be unique"
        raise ValueError(msg)
    if candidate_names.count(result.selected_model) != 1:
        msg = "selected_model must name exactly one candidate"
        raise ValueError(msg)
    selected = next(
        candidate for candidate in result.candidates if candidate.name == result.selected_model
    )
    for metric_name, metric_value in selected.metrics.items():
        if result.metrics.get(metric_name) != metric_value:
            msg = f"headline metric diverges from selected candidate: {metric_name}"
            raise ValueError(msg)

    reserved_keys = {"configuration", "manifest", "result"}
    if reserved_keys.intersection(result.artifacts):
        msg = "experiment cannot claim runner-reserved artifact keys"
        raise ValueError(msg)
    reserved = {"manifest.json", "resolved_config.json", "result.json"}
    normalized_paths: list[str] = []
    for relative_path in result.artifacts.values():
        artifact = store.resolve(relative_path)
        normalized = str(artifact.relative_to(store.root))
        if normalized in reserved:
            msg = f"experiment cannot claim runner-reserved artifact: {normalized}"
            raise ValueError(msg)
        normalized_paths.append(normalized)
        if not artifact.is_file():
            msg = f"declared artifact does not exist: {relative_path}"
            raise ValueError(msg)
    if len(normalized_paths) != len(set(normalized_paths)):
        msg = "declared artifact paths must be unique"
        raise ValueError(msg)


def run_experiment(config: ExperimentConfig, output_dir: Path) -> RunResult:
    """Run into a staging directory and atomically publish a complete artifact set."""

    target = output_dir.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and any(target.iterdir()):
        msg = f"refusing to overwrite non-empty output directory: {target}"
        raise FileExistsError(msg)

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        result = _execute(config, ArtifactStore(staging))
        if target.exists():
            target.rmdir()
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return result
