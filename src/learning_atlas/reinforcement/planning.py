"""Discounted Bellman planning with a posteriori residual certificates.

For gamma < 1 the Bellman operator is a sup-norm contraction. Consequently a
returned residual epsilon certifies ||V-V*||_infinity <= epsilon/(1-gamma).
Iteration limits are resource guards, never substitutes for convergence.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from learning_atlas.reinforcement.gridworld import FiniteMDP
from learning_atlas.reinforcement.validation import (
    FloatArray,
    ReinforcementError,
    integer,
    scalar,
    vector,
)


@dataclass(frozen=True, slots=True)
class PlanningResult:
    """Values, deterministic greedy actions, and residuals after each update."""

    values: FloatArray
    policy: NDArray[np.int64]
    residuals: FloatArray
    error_bound: float


def action_values(mdp: FiniteMDP, values: FloatArray, gamma: float) -> FloatArray:
    """Apply one O(S²A) Bellman expectation with no continuation from terminals."""

    scalar(gamma, "gamma", 0.0, 0.9999)
    v = vector(values, "values", size=mdp.transitions.shape[0])
    q = np.sum(mdp.transitions * (mdp.rewards + gamma * v[None, None, :] * ~mdp.terminal), axis=2)
    q[mdp.terminal] = 0.0
    if not np.isfinite(q).all():
        raise ReinforcementError("Bellman update produced non-finite values")
    return np.asarray(q, dtype=np.float64)


def evaluate_policy(mdp: FiniteMDP, policy: NDArray[np.int64], gamma: float) -> FloatArray:
    """Solve (I-gamma P_pi)V=r_pi; O(S³) time and O(S²) working memory."""

    scalar(gamma, "gamma", 0.0, 0.9999)
    states, actions, _ = mdp.transitions.shape
    choices = np.asarray(policy)
    if (
        choices.shape != (states,)
        or choices.dtype.kind not in "iu"
        or np.any(choices < 0)
        or np.any(choices >= actions)
    ):
        raise ValueError("policy must contain one valid integer action per state")
    p = mdp.transitions[np.arange(states), choices].copy()
    r = np.sum(p * mdp.rewards[np.arange(states), choices], axis=1)
    p[:, mdp.terminal] = 0.0
    p[mdp.terminal] = 0.0
    r[mdp.terminal] = 0.0
    try:
        value = np.linalg.solve(np.eye(states) - gamma * p, r)
    except np.linalg.LinAlgError as error:
        raise ReinforcementError("discounted policy system could not be solved") from error
    if not np.isfinite(value).all():
        raise ReinforcementError("policy evaluation produced non-finite values")
    return value


def plan(
    mdp: FiniteMDP,
    *,
    method: Literal["value_iteration", "policy_iteration"] = "value_iteration",
    gamma: float = 0.95,
    tolerance: float = 1e-9,
    max_iterations: int = 10_000,
) -> PlanningResult:
    """Return only when the recomputed optimal Bellman residual meets tolerance."""

    scalar(gamma, "gamma", 0.0, 0.9999)
    scalar(tolerance, "tolerance", 1e-12, 1.0)
    integer(max_iterations, "max_iterations", 1, 100_000)
    if method not in {"value_iteration", "policy_iteration"}:
        raise ValueError("unsupported planning method")
    values = np.zeros(mdp.transitions.shape[0])
    policy = np.zeros(len(values), dtype=np.int64)
    residuals: list[float] = []
    for _ in range(max_iterations):
        values = (
            action_values(mdp, values, gamma).max(axis=1)
            if method == "value_iteration"
            else evaluate_policy(mdp, policy, gamma)
        )
        q = action_values(mdp, values, gamma)
        residual = float(np.max(np.abs(q.max(axis=1) - values)))
        residuals.append(residual)
        policy = np.argmax(q, axis=1).astype(np.int64)
        if residual <= tolerance:
            return PlanningResult(values, policy, np.asarray(residuals), residual / (1 - gamma))
    raise ReinforcementError(
        f"{method} exhausted {max_iterations} iterations; residual={residuals[-1]:.6g}"
    )
