"""Minimal reverse-mode automatic differentiation over NumPy arrays.

The engine records a dynamic computation graph of :class:`Tensor` nodes and
differentiates a scalar objective with one reverse topological sweep. It
implements exactly the primitives a multilayer perceptron needs — broadcasting
arithmetic, matrix products, pointwise nonlinearities, reductions, and a
numerically stable log-softmax — with no dependency beyond NumPy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
TensorOperand = Union["Tensor", float, int]

_GradientRule = Callable[[FloatArray], None]


class AutogradError(RuntimeError):
    """Raised when a computation graph is used outside its differentiability contract."""


def _finite_float_array(values: ArrayLike, *, context: str) -> FloatArray:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as error:
        msg = f"{context} requires a rectangular real numeric array"
        raise AutogradError(msg) from error
    if raw.dtype.kind not in "biuf":
        msg = f"{context} requires real numeric values; received dtype {raw.dtype}"
        raise AutogradError(msg)
    array = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        msg = f"{context} produced non-finite values; refusing to continue silently"
        raise AutogradError(msg)
    return array


def _unbroadcast(gradient: FloatArray, shape: tuple[int, ...]) -> FloatArray:
    """Reduce an upstream gradient back to the shape of a broadcast operand."""

    excess = gradient.ndim - len(shape)
    if excess > 0:
        gradient = gradient.sum(axis=tuple(range(excess)))
    collapsed = tuple(
        axis for axis, extent in enumerate(shape) if extent == 1 and gradient.shape[axis] != 1
    )
    if collapsed:
        gradient = gradient.sum(axis=collapsed, keepdims=True)
    return np.asarray(gradient.reshape(shape), dtype=np.float64)


class Tensor:
    """A NumPy-backed autodiff node with reverse-mode gradient accumulation."""

    __slots__ = ("_gradient_rule", "_parents", "data", "grad", "requires_grad")

    def __init__(self, data: ArrayLike, *, requires_grad: bool = False) -> None:
        self.data = _finite_float_array(data, context="tensor construction")
        self.requires_grad = bool(requires_grad)
        self.grad: FloatArray | None = None
        self._parents: tuple[Tensor, ...] = ()
        self._gradient_rule: _GradientRule | None = None

    @classmethod
    def _from_operation(
        cls,
        data: ArrayLike,
        *,
        parents: tuple[Tensor, ...],
        gradient_rule: _GradientRule,
        context: str,
    ) -> Tensor:
        result = cls.__new__(cls)
        result.data = _finite_float_array(data, context=context)
        result.requires_grad = any(parent.requires_grad for parent in parents)
        result.grad = None
        if result.requires_grad:
            result._parents = parents
            result._gradient_rule = gradient_rule
        else:
            result._parents = ()
            result._gradient_rule = None
        return result

    # ------------------------------------------------------------------ shape

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the wrapped array."""

        return self.data.shape

    @property
    def ndim(self) -> int:
        """Number of array dimensions."""

        return self.data.ndim

    @property
    def size(self) -> int:
        """Total number of elements."""

        return int(self.data.size)

    def item(self) -> float:
        """Return the value of a single-element tensor as a Python float."""

        if self.data.size != 1:
            msg = f"item() requires a single-element tensor; shape is {self.shape}"
            raise AutogradError(msg)
        return float(self.data.reshape(()))

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"

    # ------------------------------------------------------------- graph state

    def detach(self) -> Tensor:
        """Return a graph-free tensor sharing this tensor's values."""

        return Tensor(self.data, requires_grad=False)

    def zero_grad(self) -> None:
        """Clear any accumulated gradient."""

        self.grad = None

    def _accumulate(self, contribution: FloatArray) -> None:
        if contribution.shape != self.data.shape:
            msg = (
                f"gradient contribution shape {contribution.shape} does not match "
                f"tensor shape {self.data.shape}"
            )
            raise AutogradError(msg)
        if not np.all(np.isfinite(contribution)):
            msg = "gradient contribution became non-finite"
            raise AutogradError(msg)
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        with np.errstate(over="ignore", invalid="ignore"):
            accumulated = self.grad + contribution
        if not np.all(np.isfinite(accumulated)):
            msg = "accumulated gradient became non-finite"
            raise AutogradError(msg)
        self.grad = accumulated

    def _reachable_reverse_order(self) -> list[Tensor]:
        """Return graph nodes with parents before children via iterative postorder."""

        order: list[Tensor] = []
        visited: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for parent in node._parents:
                if parent.requires_grad and id(parent) not in visited:
                    stack.append((parent, False))
        return order

    def backward(self, gradient: ArrayLike | None = None) -> None:
        """Accumulate gradients of this output into every reachable parameter."""

        if not self.requires_grad:
            msg = "backward requires a tensor that participates in gradient tracking"
            raise AutogradError(msg)
        if gradient is None:
            if self.data.size != 1:
                msg = f"backward on a non-scalar output of shape {self.shape} needs a gradient"
                raise AutogradError(msg)
            seed = np.ones_like(self.data)
        else:
            seed = _finite_float_array(gradient, context="backward seed gradient")
            if seed.shape != self.data.shape:
                msg = f"gradient shape {seed.shape} must match output shape {self.data.shape}"
                raise AutogradError(msg)
        reverse_order = self._reachable_reverse_order()
        # Graph-internal adjoints belong to one reverse sweep. Leaf adjoints
        # intentionally accumulate across calls, matching established autodiff
        # semantics while preventing stale intermediates from being propagated.
        for node in reverse_order:
            if node._parents:
                node.grad = None
        self._accumulate(seed)
        for node in reversed(reverse_order):
            if node._gradient_rule is None:
                continue
            assert node.grad is not None  # seeded above or accumulated by a child
            node._gradient_rule(node.grad)

    # -------------------------------------------------------------- arithmetic

    @staticmethod
    def _coerce(operand: TensorOperand) -> Tensor:
        if isinstance(operand, Tensor):
            return operand
        return Tensor(operand)

    def __add__(self, other: TensorOperand) -> Tensor:
        left, right = self, self._coerce(other)

        def rule(upstream: FloatArray) -> None:
            if left.requires_grad:
                left._accumulate(_unbroadcast(upstream, left.data.shape))
            if right.requires_grad:
                right._accumulate(_unbroadcast(upstream, right.data.shape))

        return Tensor._from_operation(
            left.data + right.data,
            parents=(left, right),
            gradient_rule=rule,
            context="add",
        )

    def __radd__(self, other: TensorOperand) -> Tensor:
        return self.__add__(other)

    def __mul__(self, other: TensorOperand) -> Tensor:
        left, right = self, self._coerce(other)

        def rule(upstream: FloatArray) -> None:
            if left.requires_grad:
                left._accumulate(_unbroadcast(upstream * right.data, left.data.shape))
            if right.requires_grad:
                right._accumulate(_unbroadcast(upstream * left.data, right.data.shape))

        return Tensor._from_operation(
            left.data * right.data,
            parents=(left, right),
            gradient_rule=rule,
            context="multiply",
        )

    def __rmul__(self, other: TensorOperand) -> Tensor:
        return self.__mul__(other)

    def __neg__(self) -> Tensor:
        return self.__mul__(-1.0)

    def __sub__(self, other: TensorOperand) -> Tensor:
        return self.__add__(-self._coerce(other))

    def __rsub__(self, other: TensorOperand) -> Tensor:
        return self._coerce(other).__add__(-self)

    def __truediv__(self, other: float | int) -> Tensor:
        divisor = float(other)
        if not np.isfinite(divisor) or divisor == 0.0:
            msg = "tensor division supports only finite non-zero scalar divisors"
            raise AutogradError(msg)
        return self.__mul__(1.0 / divisor)

    def __pow__(self, exponent: float | int) -> Tensor:
        power = float(exponent)
        if not np.isfinite(power):
            msg = "power requires a finite scalar exponent"
            raise AutogradError(msg)
        is_integer_power = float(power).is_integer()
        if not is_integer_power and np.any(self.data < 0.0):
            msg = "fractional powers require non-negative inputs"
            raise AutogradError(msg)
        if power < 1.0 and power != 0.0 and np.any(self.data == 0.0):
            msg = f"power {power} has an unbounded derivative at zero inputs"
            raise AutogradError(msg)
        base = self

        def rule(upstream: FloatArray) -> None:
            if power == 0.0:
                base._accumulate(np.zeros_like(base.data))
                return
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                contribution = upstream * power * np.power(base.data, power - 1.0)
            base._accumulate(np.asarray(contribution, dtype=np.float64))

        return Tensor._from_operation(
            np.power(base.data, power),
            parents=(base,),
            gradient_rule=rule,
            context="power",
        )

    def matmul(self, other: Tensor) -> Tensor:
        """Differentiable 2-D matrix product."""

        left, right = self, other
        if left.data.ndim != 2 or right.data.ndim != 2:
            msg = (
                "matmul requires two 2-D tensors; "
                f"received {left.data.ndim}-D and {right.data.ndim}-D"
            )
            raise AutogradError(msg)
        if left.data.shape[1] != right.data.shape[0]:
            msg = f"matmul inner dimensions differ: {left.data.shape} @ {right.data.shape}"
            raise AutogradError(msg)

        def rule(upstream: FloatArray) -> None:
            if left.requires_grad:
                left._accumulate(upstream @ right.data.T)
            if right.requires_grad:
                right._accumulate(left.data.T @ upstream)

        return Tensor._from_operation(
            left.data @ right.data,
            parents=(left, right),
            gradient_rule=rule,
            context="matmul",
        )

    def __matmul__(self, other: Tensor) -> Tensor:
        return self.matmul(other)

    # ------------------------------------------------------------ nonlinearity

    def relu(self) -> Tensor:
        """Rectified linear unit with the conventional zero subgradient at zero."""

        source = self

        def rule(upstream: FloatArray) -> None:
            source._accumulate(upstream * (source.data > 0.0))

        return Tensor._from_operation(
            np.maximum(source.data, 0.0),
            parents=(source,),
            gradient_rule=rule,
            context="relu",
        )

    def tanh(self) -> Tensor:
        """Hyperbolic tangent."""

        source = self
        activated = np.tanh(source.data)

        def rule(upstream: FloatArray) -> None:
            source._accumulate(upstream * (1.0 - activated**2))

        return Tensor._from_operation(
            activated,
            parents=(source,),
            gradient_rule=rule,
            context="tanh",
        )

    def exp(self) -> Tensor:
        """Elementwise exponential; overflow surfaces as an explicit error."""

        source = self
        with np.errstate(over="ignore"):
            exponent = np.exp(source.data)

        def rule(upstream: FloatArray) -> None:
            source._accumulate(upstream * exponent)

        return Tensor._from_operation(
            exponent,
            parents=(source,),
            gradient_rule=rule,
            context="exp",
        )

    def log(self) -> Tensor:
        """Natural logarithm restricted to strictly positive inputs."""

        source = self
        if np.any(source.data <= 0.0):
            msg = "log requires strictly positive inputs"
            raise AutogradError(msg)

        def rule(upstream: FloatArray) -> None:
            source._accumulate(upstream / source.data)

        return Tensor._from_operation(
            np.log(source.data),
            parents=(source,),
            gradient_rule=rule,
            context="log",
        )

    def log_softmax(self, *, axis: int = -1) -> Tensor:
        """Numerically stable log-softmax along one axis."""

        source = self
        if not -source.data.ndim <= axis < source.data.ndim:
            msg = f"log_softmax axis {axis} is out of range for shape {source.data.shape}"
            raise AutogradError(msg)
        shifted = source.data - np.max(source.data, axis=axis, keepdims=True)
        log_normalizer = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
        log_probabilities = shifted - log_normalizer

        def rule(upstream: FloatArray) -> None:
            softmax = np.exp(log_probabilities)
            source._accumulate(upstream - softmax * np.sum(upstream, axis=axis, keepdims=True))

        return Tensor._from_operation(
            log_probabilities,
            parents=(source,),
            gradient_rule=rule,
            context="log_softmax",
        )

    # --------------------------------------------------------------- reduction

    def _validated_axis(self, axis: int | None) -> int | None:
        if axis is None:
            return None
        if not -self.data.ndim <= axis < self.data.ndim:
            msg = f"reduction axis {axis} is out of range for shape {self.data.shape}"
            raise AutogradError(msg)
        return axis % self.data.ndim

    def sum(self, *, axis: int | None = None, keepdims: bool = False) -> Tensor:
        """Differentiable summation over all elements or one axis."""

        source = self
        resolved_axis = self._validated_axis(axis)

        def rule(upstream: FloatArray) -> None:
            expanded = upstream
            if resolved_axis is not None and not keepdims:
                expanded = np.expand_dims(expanded, axis=resolved_axis)
            source._accumulate(np.broadcast_to(expanded, source.data.shape).astype(np.float64))

        return Tensor._from_operation(
            np.sum(source.data, axis=resolved_axis, keepdims=keepdims),
            parents=(source,),
            gradient_rule=rule,
            context="sum",
        )

    def mean(self, *, axis: int | None = None, keepdims: bool = False) -> Tensor:
        """Differentiable arithmetic mean over all elements or one axis."""

        resolved_axis = self._validated_axis(axis)
        count = self.data.size if resolved_axis is None else self.data.shape[resolved_axis]
        return self.sum(axis=axis, keepdims=keepdims) / float(count)

    def reshape(self, shape: tuple[int, ...]) -> Tensor:
        """Differentiable reshape onto a compatible shape."""

        source = self
        try:
            reshaped = source.data.reshape(shape)
        except ValueError as error:
            msg = f"cannot reshape {source.data.shape} into {shape}"
            raise AutogradError(msg) from error

        def rule(upstream: FloatArray) -> None:
            source._accumulate(upstream.reshape(source.data.shape))

        return Tensor._from_operation(
            reshaped,
            parents=(source,),
            gradient_rule=rule,
            context="reshape",
        )


def finite_difference_gradient(
    objective: Callable[[], Tensor],
    parameter: Tensor,
    *,
    epsilon: float = 1e-6,
) -> FloatArray:
    """Estimate d(objective)/d(parameter) with central differences.

    The objective is re-evaluated with the parameter perturbed one element at a
    time, so it must be a deterministic function of the current parameter values.
    """

    if not np.isfinite(epsilon) or epsilon <= 0.0:
        msg = "epsilon must be a positive finite step"
        raise ValueError(msg)
    baseline = parameter.data.copy()
    estimate = np.zeros_like(baseline)
    iterator = np.nditer(baseline, flags=["multi_index"])
    try:
        while not iterator.finished:
            index = iterator.multi_index
            parameter.data[index] = baseline[index] + epsilon
            upper = objective().item()
            parameter.data[index] = baseline[index] - epsilon
            lower = objective().item()
            estimate[index] = (upper - lower) / (2.0 * epsilon)
            iterator.iternext()
    finally:
        parameter.data[...] = baseline
    return estimate
