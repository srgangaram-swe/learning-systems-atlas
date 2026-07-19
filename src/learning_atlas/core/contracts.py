"""Paradigm-neutral experiment and artifact contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Literal, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

if TYPE_CHECKING:
    from learning_atlas.core.artifacts import ArtifactStore


LiteralSchemaVersion = Literal["1.0.0"]


class LearningParadigm(StrEnum):
    """Top-level learning paradigms represented in the atlas."""

    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT = "reinforcement"


class SourceKind(StrEnum):
    """Nature of the observations or interaction source used by an experiment."""

    DATASET = "dataset"
    GENERATOR = "generator"
    ENVIRONMENT = "environment"


class ContractModel(BaseModel):
    """Field-frozen base class for serialized run contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceMetadata(ContractModel):
    """Identity and leakage-relevant facts about a dataset, generator, or environment."""

    kind: SourceKind
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_used_for_fit: bool | None
    details: dict[str, JsonValue] = Field(default_factory=dict)


class CandidateResult(ContractModel):
    """Deterministic metrics for one model or policy candidate."""

    name: str = Field(min_length=1)
    metrics: dict[str, float] = Field(min_length=1)

    @field_validator("metrics")
    @classmethod
    def metrics_must_be_finite(cls, metrics: dict[str, float]) -> dict[str, float]:
        """Reject NaN and infinity before they enter an artifact."""

        non_finite = [name for name, value in metrics.items() if not math.isfinite(value)]
        if non_finite:
            msg = f"metrics must be finite: {', '.join(sorted(non_finite))}"
            raise ValueError(msg)
        return metrics


class RunResult(ContractModel):
    """Versioned, JSON-safe output shared by all experiment paradigms."""

    schema_version: LiteralSchemaVersion = "1.0.0"
    experiment: str = Field(min_length=1)
    paradigm: LearningParadigm
    seed: int = Field(ge=0)
    source: SourceMetadata
    selected_model: str = Field(min_length=1)
    metrics: dict[str, float] = Field(min_length=1)
    candidates: tuple[CandidateResult, ...] = Field(min_length=1)
    artifacts: dict[str, str] = Field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @field_validator("metrics")
    @classmethod
    def result_metrics_must_be_finite(cls, metrics: dict[str, float]) -> dict[str, float]:
        """Apply the same finite-value invariant to headline metrics."""

        return CandidateResult.metrics_must_be_finite(metrics)

    @model_validator(mode="after")
    def selection_must_reference_one_unique_candidate(self) -> Self:
        """Keep the headline model selection internally consistent."""

        candidate_names = [candidate.name for candidate in self.candidates]
        if len(candidate_names) != len(set(candidate_names)):
            msg = "candidate names must be unique"
            raise ValueError(msg)
        if candidate_names.count(self.selected_model) != 1:
            msg = "selected_model must name exactly one candidate"
            raise ValueError(msg)
        return self


@dataclass(frozen=True, slots=True)
class RunContext:
    """Explicit dependencies passed to an experiment at execution time."""

    seed: int
    artifacts: ArtifactStore


@runtime_checkable
class Experiment(Protocol):
    """The only shared behavioral boundary imposed across learning paradigms."""

    def run(self, context: RunContext) -> RunResult:
        """Execute the experiment and return a serializable result."""
