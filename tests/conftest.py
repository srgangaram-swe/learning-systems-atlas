"""Shared test fixtures for versioned experiment contracts."""

from pathlib import Path

import pytest

from learning_atlas.core.artifacts import ArtifactStore
from learning_atlas.core.contracts import (
    CandidateResult,
    LearningParadigm,
    RunContext,
    RunResult,
    SourceKind,
    SourceMetadata,
)


@pytest.fixture
def artifact_store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts")


@pytest.fixture
def run_context(artifact_store: ArtifactStore) -> RunContext:
    return RunContext(seed=42, artifacts=artifact_store)


@pytest.fixture
def sample_result() -> RunResult:
    return RunResult(
        experiment="regression_benchmark",
        paradigm=LearningParadigm.SUPERVISED,
        seed=42,
        source=SourceMetadata(
            kind=SourceKind.DATASET,
            name="fixture",
            version="1",
            fingerprint_sha256="a" * 64,
            target_used_for_fit=True,
            details={"sample_count": 10, "feature_count": 2},
        ),
        selected_model="fixture_model",
        metrics={"score": 0.5},
        candidates=(CandidateResult(name="fixture_model", metrics={"score": 0.5}),),
    )
