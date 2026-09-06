"""Independent identities, properties, API behavior, and numerical failure cases."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from hypothesis import given
from hypothesis import strategies as st

from learning_atlas.reinforcement.bandits import BanditAgent, simulate_bandit
from learning_atlas.reinforcement.gridworld import FiniteMDP, Gridworld
from learning_atlas.reinforcement.planning import action_values, evaluate_policy, plan
from learning_atlas.reinforcement.tabular import epsilon_greedy, td_target, train_tabular
from learning_atlas.reinforcement.validation import ReinforcementError, integer, scalar, vector

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("bad", [True, 1.2, -1, 11, "1"])
def test_integer_rejects_ambiguous_values(bad):
    with pytest.raises(ValueError, match="integer"):
        integer(bad, "count", 0, 10)


@pytest.mark.parametrize("bad", [True, "1", np.nan, np.inf, -1, 11])
def test_scalar_rejects_unsafe_values(bad):
    with pytest.raises(ValueError, match="finite"):
        scalar(bad, "rate", 0, 10)


@pytest.mark.parametrize("bad", [[], [[1]], [np.nan], ["1"], [1j]])
def test_vector_contract(bad):
    with pytest.raises(ValueError, match="vector"):
        vector(np.asarray(bad), "x")
    with pytest.raises(ValueError):
        vector(np.ones(3), "x", size=2)


def test_beta_and_gaussian_posteriors_match_conjugate_fixtures():
    beta = BanditAgent(2, "bernoulli")
    for reward in (1.0, 0.0, 1.0):
        beta.update(0, reward)
    np.testing.assert_array_equal(beta.posterior(), [[3, 1], [2, 1]])
    normal = BanditAgent(2, "gaussian", sigma=2.0)
    normal.update(0, 2.0)
    normal.update(0, 4.0)
    mean, variance = normal.posterior()
    np.testing.assert_allclose(mean, [1.0, 0.0])
    np.testing.assert_allclose(variance, [2 / 3, 1.0])
    before = beta.counts.copy()
    with pytest.raises(ValueError, match="zero or one"):
        beta.update(0, 0.5)
    np.testing.assert_array_equal(before, beta.counts)
    with pytest.raises(ValueError):
        BanditAgent(2, "invalid")
    with pytest.raises(ValueError):
        beta.choose("invalid", np.random.default_rng(1))


@pytest.mark.parametrize("distribution", ["bernoulli", "gaussian"])
@pytest.mark.parametrize("method", ["epsilon_greedy", "ucb", "thompson"])
def test_bandit_repeatability_regret_and_no_truth_access(distribution, method):
    means = np.array([0.1, 0.4, 0.9])
    first = simulate_bandit(means, distribution=distribution, method=method, steps=500, seed=7)
    other = simulate_bandit(means, distribution=distribution, method=method, steps=500, seed=7)
    np.testing.assert_array_equal(first.actions, other.actions)
    np.testing.assert_array_equal(first.rewards, other.rewards)
    np.testing.assert_allclose(first.cumulative_regret, np.cumsum(0.9 - means[first.actions]))
    assert np.all(np.diff(first.cumulative_regret) >= 0)
    assert np.mean(first.actions[-100:] == 2) > 0.6
    equal = simulate_bandit(
        np.ones(2) * 0.5, distribution=distribution, method=method, steps=10, seed=0
    )
    np.testing.assert_array_equal(equal.cumulative_regret, np.zeros(10))


@pytest.mark.parametrize("means,method", [([0, 2], "ucb"), ([0, 1e5], "ucb"), ([0, 1], "invalid")])
def test_bandit_rejects_bad_simulation(means, method):
    with pytest.raises(ValueError):
        simulate_bandit(np.array(means), distribution="bernoulli", method=method, steps=2, seed=1)


def test_gaussian_reward_budget_and_random_exploration():
    with pytest.raises(ValueError, match="budget"):
        simulate_bandit(np.array([0, 1e5]), distribution="gaussian", method="ucb", steps=2, seed=1)
    agent = BanditAgent(2, "bernoulli")
    assert agent.choose("epsilon_greedy", np.random.default_rng(1), epsilon=1.0) in (0, 1)


@pytest.mark.parametrize(
    "layout", [(), ("S", "GG"), ("SSG",), ("SxG",), ("SG" * 200,), ("",), ["SG"]]
)
def test_grid_layout_rejection(layout):
    with pytest.raises(ValueError, match="layout"):
        Gridworld(layout)


def test_grid_checker_boundaries_and_ascii():
    env = Gridworld(("S.G",), max_steps=2)
    check_env(env, skip_render_check=True)
    with pytest.raises(ValueError, match="render"):
        Gridworld(render_mode="rgb_array")
    with pytest.raises(ValueError, match="options"):
        env.reset(options={"unknown": 1})
    env.reset(seed=2)
    assert env.render() == "@.G"
    assert env.step(1)[1:4] == (-1.0, False, False)
    assert env.step(1)[1:4] == (0.0, True, False)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(1)
    env.reset(seed=2)
    with pytest.raises(ValueError, match="action"):
        env.step(4)
    env.step(0)
    assert env.step(0)[2:4] == (False, True)
    assert env.outcome(env.goal, 0) == (env.goal, 0.0, True)


def test_grid_slip_is_reproducible_and_exact_model_matches_sampling():
    env = Gridworld(("SCG", "..."), slip=0.4)
    first = []
    second = []
    for trace in (first, second):
        env.reset(seed=5)
        for _ in range(100):
            trace.append(env.step(1))
            if trace[-1][2] or trace[-1][3]:
                env.reset()
    assert first == second
    model = env.model()
    assert not model.transitions.flags.writeable
    outcomes = [env.outcome(env.start, action) for action in range(4)]
    expected_reward = sum(
        (0.1 + 0.6 * (action == 1)) * item[1] for action, item in enumerate(outcomes)
    )
    assert np.sum(model.transitions[env.start, 1] * model.rewards[env.start, 1]) == pytest.approx(
        expected_reward
    )
    env.reset(seed=123)
    samples = []
    for _ in range(5000):
        samples.append(env.step(1)[1])
        env.reset()
    assert np.mean(samples) == pytest.approx(expected_reward, abs=2.0)


@pytest.mark.parametrize(
    "mutation", ["shape", "terminal", "negative", "normalization", "nan", "reward"]
)
def test_mdp_rejects_invalid_arrays(mutation):
    p = np.ones((2, 1, 2)) / 2
    r = np.zeros_like(p)
    terminal = np.array([False, True])
    if mutation == "shape":
        p = p[:, :, :1]
    elif mutation == "terminal":
        terminal = terminal.astype(int)
    elif mutation == "negative":
        p[0, 0] = [-0.1, 1.1]
    elif mutation == "normalization":
        p[0, 0] = [0, 0]
    elif mutation == "nan":
        p[0, 0, 0] = np.nan
    else:
        r[0, 0, 0] = 1e7
    with pytest.raises(ValueError, match="MDP"):
        FiniteMDP(p, r, terminal)


def test_hand_derived_corridor_and_planner_agreement():
    mdp = Gridworld(("S..G",)).model()
    vi = plan(mdp, gamma=0.9)
    pi = plan(mdp, gamma=0.9, method="policy_iteration")
    np.testing.assert_allclose(vi.values, [-1.9, -1, 0, 0])
    np.testing.assert_allclose(pi.values, vi.values)
    assert vi.policy[0] == 1
    assert vi.residuals[-1] <= 1e-9
    assert vi.error_bound <= 1e-8
    with pytest.raises(ReinforcementError, match="exhausted"):
        plan(mdp, max_iterations=1)
    with pytest.raises(ValueError):
        plan(mdp, method="invalid")
    with pytest.raises(ValueError):
        evaluate_policy(mdp, np.array([1, 1]), 0.9)


@given(st.floats(min_value=0.0, max_value=0.8, allow_nan=False))
def test_stochastic_planning_contraction_and_differential_agreement(slip):
    mdp = Gridworld(slip=slip).model()
    vi, pi = plan(mdp), plan(mdp, method="policy_iteration")
    np.testing.assert_allclose(vi.values, pi.values, atol=2e-8)
    v = np.arange(len(vi.values), dtype=float)
    w = v[::-1].copy()
    error = np.max(np.abs(action_values(mdp, v, 0.95).max(1) - action_values(mdp, w, 0.95).max(1)))
    assert error <= 0.95 * np.max(np.abs(v - w)) + 1e-12


def test_policy_solve_failure_is_structured(monkeypatch):
    def fail(*args):
        raise np.linalg.LinAlgError("injected")

    monkeypatch.setattr(np.linalg, "solve", fail)
    with pytest.raises(ReinforcementError, match="could not be solved") as error:
        plan(Gridworld().model(), method="policy_iteration")
    assert isinstance(error.value.__cause__, np.linalg.LinAlgError)


@pytest.mark.parametrize("method", ["q_learning", "sarsa"])
def test_tabular_reproducibility_and_near_optimal_policy(method):
    kwargs = dict(method=method, episodes=300, seed=12)
    first = train_tabular(Gridworld(), **kwargs)
    second = train_tabular(Gridworld(), **kwargs)
    np.testing.assert_array_equal(first.returns, second.returns)
    np.testing.assert_array_equal(first.q_values, second.q_values)
    mdp = Gridworld().model()
    values = evaluate_policy(mdp, first.q_values.argmax(1), 0.95)
    assert values[0] == pytest.approx(plan(mdp).values[0], abs=0.1)
    assert first.lengths.max() <= 500
    assert first.cliff_falls.sum() == 0


def test_td_terminal_mask_and_invalid_inputs():
    assert td_target(1.0, 10.0, gamma=0.9, terminated=False) == 10.0
    assert td_target(1.0, 10.0, gamma=0.9, terminated=True) == 1.0
    with pytest.raises(ValueError):
        td_target(1.0, 1.0, gamma=0.9, terminated=1)
    with pytest.raises(ValueError):
        train_tabular(Gridworld(), method="invalid", episodes=2, seed=1)
    with pytest.raises(ValueError, match="budget"):
        train_tabular(Gridworld(max_steps=100_000), method="sarsa", episodes=1000, seed=1)
    rng = np.random.default_rng(2)
    choices = [epsilon_greedy(np.zeros(4), 0.0, rng) for _ in range(100)]
    assert set(choices) == {0, 1, 2, 3}
    truncated = train_tabular(Gridworld(max_steps=1), method="sarsa", episodes=2, seed=1)
    np.testing.assert_array_equal(truncated.lengths, [1, 1])
