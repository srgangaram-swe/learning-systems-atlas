"""Contracts and padding invariants for the deep sequence reference models."""

from collections.abc import Callable

import pytest
import torch

from learning_atlas.deep.sequence import PaddedSequenceMLP, SequenceLSTM

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sequence_lstm_returns_finite_logits_and_gradients(dtype: torch.dtype) -> None:
    torch.manual_seed(17)
    model = SequenceLSTM(input_size=2, hidden_size=5, num_layers=2, n_classes=3).to(dtype=dtype)
    padded = torch.randn(4, 6, 2, dtype=dtype, requires_grad=True)
    lengths = torch.tensor([6, 2, 5, 3], dtype=torch.int64)

    logits = model(padded, lengths)
    logits.square().mean().backward()

    assert logits.shape == (4, 3)
    assert logits.dtype == dtype
    assert bool(torch.all(torch.isfinite(logits)))
    assert padded.grad is not None
    assert bool(torch.all(torch.isfinite(padded.grad)))
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert bool(torch.all(torch.isfinite(parameter.grad)))


def test_sequence_lstm_is_invariant_to_values_after_each_true_length() -> None:
    torch.manual_seed(23)
    model = SequenceLSTM(input_size=1, hidden_size=4, n_classes=2).eval()
    original = torch.randn(3, 6, 1)
    lengths = torch.tensor([6, 2, 4], dtype=torch.int64)
    perturbed = original.clone()
    perturbed[1, 2:] = 10_000.0
    perturbed[2, 4:] = -10_000.0

    with torch.no_grad():
        original_logits = model(original, lengths)
        perturbed_logits = model(perturbed, lengths)

    torch.testing.assert_close(perturbed_logits, original_logits, rtol=0.0, atol=0.0)


def test_sequence_lstm_padding_positions_have_zero_input_gradient() -> None:
    torch.manual_seed(29)
    model = SequenceLSTM(input_size=1, hidden_size=4, n_classes=2)
    padded = torch.randn(2, 5, 1, requires_grad=True)
    lengths = torch.tensor([5, 2], dtype=torch.int64)

    model(padded, lengths).sum().backward()

    assert padded.grad is not None
    torch.testing.assert_close(padded.grad[1, 2:], torch.zeros_like(padded.grad[1, 2:]))
    assert bool(torch.any(padded.grad[1, :2] != 0.0))


def test_padded_sequence_mlp_has_explicit_position_bound_contract() -> None:
    torch.manual_seed(31)
    model = PaddedSequenceMLP(max_length=5, input_size=2, hidden_units=7, n_classes=3)
    padded = torch.randn(4, 5, 2)

    first = model(padded, torch.tensor([5, 4, 3, 2]))
    second = model(padded, torch.tensor([1, 1, 1, 1]))

    assert first.shape == (4, 3)
    assert first.dtype == padded.dtype
    assert bool(torch.all(torch.isfinite(first)))
    torch.testing.assert_close(second, first, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: SequenceLSTM(input_size=0), "input_size"),
        (lambda: SequenceLSTM(hidden_size=True), "hidden_size"),
        (lambda: SequenceLSTM(num_layers=-1), "num_layers"),
        (lambda: SequenceLSTM(n_classes=0), "n_classes"),
        (lambda: PaddedSequenceMLP(max_length=False), "max_length"),
        (lambda: PaddedSequenceMLP(max_length=3, input_size=0), "input_size"),
        (lambda: PaddedSequenceMLP(max_length=3, hidden_units=0), "hidden_units"),
        (lambda: PaddedSequenceMLP(max_length=3, n_classes=-2), "n_classes"),
    ],
)
def test_sequence_constructor_rejects_nonpositive_or_boolean_widths(
    factory: Callable[[], object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.parametrize(
    ("padded", "lengths", "message"),
    [
        (torch.zeros(2, 3), torch.tensor([3, 3]), "shaped"),
        (torch.zeros(2, 3, 1), torch.tensor([[3], [3]]), "one length"),
        (torch.zeros(2, 3, 1), torch.tensor([3]), "one length"),
        (torch.zeros(2, 3, 1), torch.tensor([0, 3]), "at least 1"),
        (torch.zeros(2, 3, 1), torch.tensor([4, 3]), "exceed"),
        (torch.zeros(0, 3, 1), torch.zeros(0, dtype=torch.int64), "sample|non-empty"),
    ],
)
def test_sequence_models_reject_invalid_batch_shapes_and_lengths(
    padded: torch.Tensor, lengths: torch.Tensor, message: str
) -> None:
    model = SequenceLSTM(input_size=1)
    with pytest.raises(ValueError, match=message):
        model(padded, lengths)


@pytest.mark.parametrize(
    ("lengths", "message"),
    [
        (torch.tensor([3.0, 2.0]), "integer"),
        (torch.tensor([True, True]), "integer|boolean"),
    ],
)
def test_sequence_models_reject_noninteger_length_dtypes(
    lengths: torch.Tensor, message: str
) -> None:
    model = SequenceLSTM(input_size=1)
    with pytest.raises((TypeError, ValueError), match=message):
        model(torch.zeros(2, 3, 1), lengths)


@pytest.mark.parametrize(
    ("padded", "message"),
    [
        (torch.full((2, 3, 1), torch.nan), "finite"),
        (torch.ones(2, 3, 1, dtype=torch.int64), "floating"),
    ],
)
def test_sequence_models_reject_nonfinite_or_nonfloating_batches(
    padded: torch.Tensor, message: str
) -> None:
    model = SequenceLSTM(input_size=1)
    with pytest.raises((TypeError, ValueError), match=message):
        model(padded, torch.tensor([3, 2]))


def test_sequence_lstm_rejects_configured_feature_width_mismatch() -> None:
    model = SequenceLSTM(input_size=2)
    with pytest.raises(ValueError, match=r"feature|input_size"):
        model(torch.zeros(2, 4, 1), torch.tensor([4, 3]))


@pytest.mark.parametrize("shape", [(2, 4, 2), (2, 5, 1), (2, 5, 3)])
def test_padded_sequence_mlp_rejects_layout_mismatch(shape: tuple[int, ...]) -> None:
    model = PaddedSequenceMLP(max_length=5, input_size=2)
    with pytest.raises(ValueError, match="sized for"):
        model(torch.zeros(shape), torch.tensor([1, 1]))
