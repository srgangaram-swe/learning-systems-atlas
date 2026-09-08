"""Finite discounted MDP and bounded Gymnasium grid with exact transition model."""

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from learning_atlas.reinforcement.validation import FloatArray, integer, scalar


@dataclass(frozen=True, slots=True)
class FiniteMDP:
    """Dense P[s,a,s'] and conditional E[r|s,a,s']; terminal continuation is zero.

    Copies are immutable. Dense storage is deliberately bounded to 256 states and
    16 actions; sparse or continuous environments require a different contract.
    """

    transitions: FloatArray
    rewards: FloatArray
    terminal: NDArray[np.bool_]

    def __post_init__(self) -> None:
        p = np.asarray(self.transitions)
        r = np.asarray(self.rewards)
        terminal = np.asarray(self.terminal)
        if (
            p.ndim != 3
            or not 1 <= p.shape[0] <= 256
            or not 1 <= p.shape[1] <= 16
            or p.shape[2] != p.shape[0]
            or r.shape != p.shape
            or terminal.shape != (p.shape[0],)
            or terminal.dtype != np.bool_
            or p.dtype.kind not in "fiu"
            or r.dtype.kind not in "fiu"
        ):
            raise ValueError(
                "MDP requires bounded numeric [states,actions,states] arrays and bool terminals"
            )
        if (
            not np.isfinite(p).all()
            or not np.isfinite(r).all()
            or np.any(p < 0)
            or not np.allclose(p.sum(axis=2), 1.0, rtol=0, atol=1e-12)
            or np.max(np.abs(r)) > 1e6
        ):
            raise ValueError(
                "MDP probabilities must normalize and rewards must be finite and bounded"
            )
        for name, array, dtype in (
            ("transitions", p, np.float64),
            ("rewards", r, np.float64),
            ("terminal", terminal, np.bool_),
        ):
            copied = np.array(array, dtype=dtype, copy=True)
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)


class Gridworld(gym.Env[int, int]):
    """Four actions (up,right,down,left), optional slip and cliff reset.

    Exactly one S and G; # are walls, C are cliffs, . are traversable. Cliff
    incurs its reward and resets to S without terminating. A time limit truncates
    without converting the physical next state to a terminal state.
    """

    _moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

    def __init__(
        self,
        layout: tuple[str, ...] = ("S...", ".##.", "...G"),
        *,
        slip: float = 0.0,
        step_reward: float = -1.0,
        goal_reward: float = 0.0,
        cliff_reward: float = -100.0,
        max_steps: int = 500,
        render_mode: str | None = "ansi",
    ) -> None:
        self.metadata = {"render_modes": ["ansi"]}
        if (
            not isinstance(layout, tuple)
            or not layout
            or not all(isinstance(row, str) and row for row in layout)
            or len(layout) * len(layout[0]) > 256
            or any(len(row) != len(layout[0]) for row in layout)
            or any(char not in "SG.#C" for row in layout for char in row)
            or "".join(layout).count("S") != 1
            or "".join(layout).count("G") != 1
        ):
            raise ValueError("layout must be rectangular <=256 cells with exactly one S and G")
        if render_mode not in {None, "ansi"}:
            raise ValueError("Gridworld supports only ansi rendering")
        self.render_mode = render_mode
        self.layout = layout
        self.slip = scalar(slip, "slip", 0.0, 1.0)
        self.step_reward = scalar(step_reward, "step_reward", -1e6, 1e6)
        self.goal_reward = scalar(goal_reward, "goal_reward", -1e6, 1e6)
        self.cliff_reward = scalar(cliff_reward, "cliff_reward", -1e6, 1e6)
        self.max_steps = integer(max_steps, "max_steps", 1, 100_000)
        self.cells = tuple(
            (row, col)
            for row, line in enumerate(layout)
            for col, char in enumerate(line)
            if char not in "#C"
        )
        self._indices = {cell: index for index, cell in enumerate(self.cells)}
        self.start = next(i for i, (r, c) in enumerate(self.cells) if layout[r][c] == "S")
        self.goal = next(i for i, (r, c) in enumerate(self.cells) if layout[r][c] == "G")
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(len(self.cells))
        self._state = self.start
        self._steps = 0
        self._active = False

    def outcome(self, state: int, action: int) -> tuple[int, float, bool]:
        """One deterministic physical action, before mixing slip probabilities."""

        integer(state, "state", 0, len(self.cells) - 1)
        integer(action, "action", 0, 3)
        if state == self.goal:
            return state, 0.0, True
        row, col = self.cells[state]
        dr, dc = self._moves[action]
        cell = (row + dr, col + dc)
        if 0 <= cell[0] < len(self.layout) and 0 <= cell[1] < len(self.layout[0]):
            if self.layout[cell[0]][cell[1]] == "C":
                return self.start, self.cliff_reward, False
        next_state = self._indices.get(cell, state)
        terminal = next_state == self.goal
        return next_state, self.goal_reward if terminal else self.step_reward, terminal

    def model(self) -> FiniteMDP:
        """Exactly marginalize slip; aggregate conditional rewards for collided outcomes."""

        count = len(self.cells)
        p = np.zeros((count, 4, count))
        weighted_rewards = np.zeros_like(p)
        for state in range(count):
            for action in range(4):
                for actual in range(4):
                    probability = self.slip / 4 + (1 - self.slip) * (actual == action)
                    target, reward, _ = self.outcome(state, actual)
                    p[state, action, target] += probability
                    weighted_rewards[state, action, target] += probability * reward
        rewards = np.divide(weighted_rewards, p, out=np.zeros_like(p), where=p > 0)
        return FiniteMDP(p, rewards, np.arange(count) == self.goal)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[int, dict[str, Any]]:
        """Reset and optionally seed the environment-owned generator."""

        super().reset(seed=seed)
        if options:
            raise ValueError("Gridworld does not support reset options")
        self._state, self._steps, self._active = self.start, 0, True
        return self._state, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        """Advance once; reject steps before reset or after an episode boundary."""

        if not self._active:
            raise RuntimeError("reset is required before stepping Gridworld")
        if isinstance(action, bool) or not self.action_space.contains(action):
            raise ValueError("action must be in [0,3]")
        actual = int(action)
        if self.np_random.random() < self.slip:
            actual = int(self.np_random.integers(4))
        self._state, reward, terminated = self.outcome(self._state, actual)
        self._steps += 1
        truncated = self._steps >= self.max_steps and not terminated
        self._active = not (terminated or truncated)
        return self._state, reward, terminated, truncated, {}

    def render(self) -> str:
        """Return ASCII only; no display server or GUI dependency."""

        rows = [list(row) for row in self.layout]
        row, col = self.cells[self._state]
        rows[row][col] = "@"
        return "\n".join("".join(row) for row in rows)
