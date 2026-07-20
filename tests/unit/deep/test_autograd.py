"""Mathematical, graph-lifecycle, and failure-contract tests for NumPy autograd."""

from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from learning_atlas.deep.autograd import AutogradError, Tensor, finite_difference_gradient

pytestmark = pytest.mark.unit


def _analytic_gradient(objective: Tensor, parameter: Tensor) -> np.ndarray:
    parameter.zero_grad()
    objective.backward()
    assert parameter.grad is not None
    return parameter.grad.copy()


def test_scalar_composite_matches_closed_form_derivative() -> None:
    value = 1.25
    tensor = Tensor(value, requires_grad=True)
    objective = ((tensor**3) + 2.0 * tensor - 4.0).tanh()

    objective.backward()

    polynomial = value**3 + 2.0 * value - 4.0
    expected = (1.0 - np.tanh(polynomial) ** 2) * (3.0 * value**2 + 2.0)
    assert tensor.grad is not None
    assert tensor.grad.item() == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_fan_out_reuse_and_repeated_backward_accumulate_one_fresh_vjp_per_call() -> None:
    tensor = Tensor(3.0, requires_grad=True)
    reused = tensor * tensor
    objective = reused + tensor

    objective.backward()
    assert tensor.grad is not None
    assert tensor.grad.item() == pytest.approx(7.0)

    objective.backward()
    assert tensor.grad is not None
    assert tensor.grad.item() == pytest.approx(14.0)

    tensor.zero_grad()
    objective.backward()
    assert tensor.grad is not None
    assert tensor.grad.item() == pytest.approx(7.0)


def test_broadcasting_reduces_gradients_to_each_operand_shape() -> None:
    features = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
    bias = Tensor([[0.5, -1.0, 2.0]], requires_grad=True)

    ((features + bias) * 2.0).sum().backward()

    assert features.grad is not None
    assert bias.grad is not None
    np.testing.assert_array_equal(features.grad, np.full((2, 3), 2.0))
    np.testing.assert_array_equal(bias.grad, np.full((1, 3), 4.0))


def test_explicit_vector_seed_computes_vector_jacobian_product() -> None:
    tensor = Tensor([-2.0, 0.5, 3.0], requires_grad=True)
    output = tensor * tensor
    upstream = np.asarray([0.5, -1.0, 2.0])

    output.backward(upstream)

    assert tensor.grad is not None
    np.testing.assert_allclose(tensor.grad, 2.0 * tensor.data * upstream)


def test_matrix_cross_entropy_gradients_match_central_differences() -> None:
    features = Tensor(
        [[0.2, -0.4, 1.0], [1.2, 0.3, -0.7], [-0.1, 0.9, 0.5]],
        requires_grad=False,
    )
    weights = Tensor(
        [[0.1, -0.2], [0.3, 0.4], [-0.5, 0.2]],
        requires_grad=True,
    )
    bias = Tensor([[0.05, -0.05]], requires_grad=True)
    one_hot = Tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

    def objective() -> Tensor:
        logits = features.matmul(weights) + bias
        return -(logits.log_softmax(axis=1) * one_hot).sum() / 3.0

    loss = objective()
    weights.zero_grad()
    bias.zero_grad()
    loss.backward()
    assert weights.grad is not None
    assert bias.grad is not None

    numeric_weights = finite_difference_gradient(objective, weights)
    numeric_bias = finite_difference_gradient(objective, bias)
    np.testing.assert_allclose(weights.grad, numeric_weights, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(bias.grad, numeric_bias, rtol=0.0, atol=1e-5)


@settings(max_examples=16, deadline=None)
@given(
    st.lists(
        st.one_of(
            st.floats(
                min_value=-3.0,
                max_value=-0.2,
                allow_nan=False,
                allow_infinity=False,
            ),
            st.floats(
                min_value=0.2,
                max_value=3.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        ),
        min_size=1,
        max_size=12,
    )
)
def test_relu_gradient_matches_central_differences_away_from_the_kink(
    values: list[float],
) -> None:
    parameter = Tensor(np.asarray(values, dtype=np.float64), requires_grad=True)
    coefficients = Tensor(np.linspace(-1.5, 1.5, len(values), dtype=np.float64))

    def objective() -> Tensor:
        return (parameter.relu() * coefficients).sum()

    objective().backward()
    assert parameter.grad is not None
    analytic = parameter.grad.copy()
    numeric = finite_difference_gradient(objective, parameter)

    np.testing.assert_allclose(analytic, numeric, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(
        analytic,
        coefficients.data * (parameter.data > 0.0),
        rtol=0.0,
        atol=0.0,
    )


@settings(max_examples=12, deadline=None)
@given(
    rows=st.integers(min_value=2, max_value=4),
    columns=st.integers(min_value=1, max_value=4),
    data=st.data(),
)
def test_positive_log_with_broadcast_bias_matches_central_differences(
    rows: int,
    columns: int,
    data: st.DataObject,
) -> None:
    feature_values = np.asarray(
        data.draw(
            st.lists(
                st.floats(
                    min_value=0.5,
                    max_value=3.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=rows * columns,
                max_size=rows * columns,
            ),
            label="features",
        ),
        dtype=np.float64,
    ).reshape(rows, columns)
    bias_values = np.asarray(
        data.draw(
            st.lists(
                st.floats(
                    min_value=0.5,
                    max_value=2.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=columns,
                max_size=columns,
            ),
            label="bias",
        ),
        dtype=np.float64,
    ).reshape(1, columns)
    features = Tensor(feature_values, requires_grad=True)
    bias = Tensor(bias_values, requires_grad=True)

    def objective() -> Tensor:
        return (features + bias).log().mean()

    objective().backward()
    assert features.grad is not None
    assert bias.grad is not None
    analytic_features = features.grad.copy()
    analytic_bias = bias.grad.copy()
    numeric_features = finite_difference_gradient(objective, features)
    numeric_bias = finite_difference_gradient(objective, bias)

    np.testing.assert_allclose(analytic_features, numeric_features, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(analytic_bias, numeric_bias, rtol=0.0, atol=1e-7)


@settings(max_examples=16, deadline=None)
@given(
    st.lists(
        st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=8,
    )
)
def test_elementwise_composite_gradient_property(values: list[float]) -> None:
    array = np.asarray(values, dtype=np.float64)
    tensor = Tensor(array, requires_grad=True)
    objective = (tensor.tanh() * tensor + 0.3 * tensor**2).mean()

    objective.backward()

    expected = (np.tanh(array) + array * (1.0 - np.tanh(array) ** 2) + 0.6 * array) / len(array)
    assert tensor.grad is not None
    np.testing.assert_allclose(tensor.grad, expected, rtol=1e-12, atol=1e-12)


def test_log_softmax_is_normalized_and_has_zero_gradient_for_probability_sum() -> None:
    logits = Tensor([[1_000.0, 999.0, -1_000.0], [-500.0, 0.0, 500.0]], requires_grad=True)
    log_probabilities = logits.log_softmax(axis=1)
    probabilities = log_probabilities.exp()

    np.testing.assert_allclose(np.sum(probabilities.data, axis=1), 1.0, atol=1e-15)
    probabilities.sum().backward()
    assert logits.grad is not None
    np.testing.assert_allclose(logits.grad, 0.0, atol=1e-15)


def test_zero_power_has_exactly_zero_gradient_at_zero() -> None:
    tensor = Tensor([0.0, -2.0, 3.0], requires_grad=True)

    (tensor**0).sum().backward()

    assert tensor.grad is not None
    np.testing.assert_array_equal(tensor.grad, np.zeros(3))
    assert np.all(np.isfinite(tensor.grad))


def test_reshape_sum_mean_and_detach_preserve_expected_contracts() -> None:
    tensor = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
    objective = tensor.reshape((3, 2)).mean(axis=0).sum()

    objective.backward()

    assert objective.shape == ()
    assert tensor.grad is not None
    np.testing.assert_allclose(tensor.grad, np.full((2, 3), 1.0 / 3.0))
    detached = tensor.detach()
    assert detached.requires_grad is False
    assert detached.grad is None
    np.testing.assert_array_equal(detached.data, tensor.data)


@pytest.mark.parametrize(
    "values",
    [
        [np.nan],
        [np.inf],
        np.asarray([1.0 + 2.0j]),
        ["1.5"],
        [[1.0], [2.0, 3.0]],
    ],
)
def test_tensor_construction_rejects_nonfinite_or_non_real_numeric_inputs(values: object) -> None:
    with pytest.raises(AutogradError):
        Tensor(values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (lambda: Tensor([1.0, 2.0], requires_grad=True).backward(), "non-scalar"),
        (lambda: Tensor(1.0).backward(), "gradient tracking"),
        (
            lambda: Tensor([1.0, 2.0], requires_grad=True).backward([1.0]),
            "shape",
        ),
        (lambda: Tensor(1.0, requires_grad=True).backward(np.nan), "non-finite"),
        (lambda: Tensor(1.0) / 0.0, "non-zero"),
        (lambda: Tensor(1.0) / np.inf, "finite"),
        (lambda: Tensor(-1.0) ** 0.5, "non-negative"),
        (lambda: Tensor(0.0) ** -1, "unbounded"),
        (lambda: Tensor([1.0]).matmul(Tensor([1.0])), "2-D"),
        (lambda: Tensor([[1.0, 2.0]]).matmul(Tensor([[1.0, 2.0]])), "dimensions"),
        (lambda: Tensor([0.0]).log(), "positive"),
        (lambda: Tensor([[1.0]]).log_softmax(axis=2), "out of range"),
        (lambda: Tensor([[1.0]]).sum(axis=2), "out of range"),
        (lambda: Tensor([1.0, 2.0]).reshape((3,)), "cannot reshape"),
        (lambda: Tensor([1.0, 2.0]).item(), "single-element"),
    ],
)
def test_invalid_operations_raise_actionable_domain_errors(
    operation: Callable[[], object], message: str
) -> None:
    with pytest.raises(AutogradError, match=message):
        operation()


def test_exponential_overflow_is_rejected_at_the_producing_operation() -> None:
    with pytest.raises(AutogradError, match=r"exp.*non-finite"):
        Tensor([1_000.0], requires_grad=True).exp()


@pytest.mark.parametrize("epsilon", [0.0, -1e-6, np.nan, np.inf])
def test_finite_difference_rejects_invalid_steps(epsilon: float) -> None:
    parameter = Tensor(1.0, requires_grad=True)
    with pytest.raises(ValueError, match="positive finite"):
        finite_difference_gradient(lambda: parameter**2, parameter, epsilon=epsilon)


def test_finite_difference_restores_parameter_when_objective_raises() -> None:
    parameter = Tensor(0.0, requires_grad=True)
    baseline = parameter.data.copy()

    with pytest.raises(AutogradError, match="positive"):
        finite_difference_gradient(parameter.log, parameter)

    np.testing.assert_array_equal(parameter.data, baseline)


def test_independent_graphs_intentionally_accumulate_into_shared_leaf() -> None:
    parameter = Tensor(2.0, requires_grad=True)
    (parameter**2).backward()
    (3.0 * parameter).backward()

    assert parameter.grad is not None
    assert parameter.grad.item() == pytest.approx(7.0)
