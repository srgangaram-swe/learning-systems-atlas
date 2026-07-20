"""Unit contracts for compact image-classification reference models."""

from __future__ import annotations

import io
import math
from collections.abc import Callable

import pytest
import torch
from torch import nn

from learning_atlas.deep.vision import CompactCNN, VisionMLP

pytestmark = pytest.mark.unit

ModelFactory = Callable[[], nn.Module]


def _factories() -> tuple[ModelFactory, ...]:
    return (
        lambda: VisionMLP(image_size=8, hidden_units=16, n_classes=10),
        lambda: CompactCNN(image_size=8, channels=4, hidden_units=16, n_classes=10),
    )


def _seeded(factory: ModelFactory, *, seed: int) -> nn.Module:
    previous_state = torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        return factory()
    finally:
        torch.set_rng_state(previous_state)


@pytest.mark.parametrize("factory", _factories())
def test_forward_returns_finite_class_logits_on_cpu(factory: ModelFactory) -> None:
    model = _seeded(factory, seed=17)
    images = torch.linspace(0.0, 1.0, steps=3 * 8 * 8).reshape(3, 1, 8, 8)

    logits = model(images)

    assert logits.shape == (3, 10)
    assert logits.dtype == torch.float32
    assert logits.device.type == "cpu"
    assert torch.isfinite(logits).all()
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


@pytest.mark.parametrize("factory", _factories())
@pytest.mark.parametrize(
    "images",
    [
        torch.ones(1, 8, 8),
        torch.ones(2, 2, 8, 8),
        torch.ones(2, 1, 7, 8),
        torch.ones(2, 1, 8, 7),
    ],
)
def test_forward_rejects_wrong_image_shapes(factory: ModelFactory, images: torch.Tensor) -> None:
    with pytest.raises(ValueError, match=r"shaped|shape"):
        factory()(images)


@pytest.mark.parametrize("factory", _factories())
def test_forward_rejects_empty_nonfloating_and_nonfinite_batches(
    factory: ModelFactory,
) -> None:
    model = factory()

    with pytest.raises(ValueError, match=r"empty|non-empty"):
        model(torch.empty((0, 1, 8, 8), dtype=torch.float32))
    with pytest.raises((TypeError, ValueError), match=r"floating|dtype"):
        model(torch.ones((2, 1, 8, 8), dtype=torch.long))
    for value in (math.nan, math.inf, -math.inf):
        images = torch.zeros((2, 1, 8, 8), dtype=torch.float32)
        images[0, 0, 0, 0] = value
        with pytest.raises(ValueError, match="finite"):
            model(images)


@pytest.mark.parametrize("factory", _factories())
def test_forward_rejects_non_tensor_input(factory: ModelFactory) -> None:
    with pytest.raises(TypeError, match="tensor"):
        factory()([[[[0.0]]]])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("factory", "field"),
    [
        (lambda: VisionMLP(image_size=0), "image_size"),
        (lambda: VisionMLP(image_size=True), "image_size"),
        (lambda: VisionMLP(hidden_units=0), "hidden_units"),
        (lambda: VisionMLP(hidden_units=1.5), "hidden_units"),
        (lambda: VisionMLP(n_classes=0), "n_classes"),
        (lambda: CompactCNN(image_size=0), "image_size"),
        (lambda: CompactCNN(channels=0), "channels"),
        (lambda: CompactCNN(channels=True), "channels"),
        (lambda: CompactCNN(hidden_units=0), "hidden_units"),
        (lambda: CompactCNN(n_classes=0), "n_classes"),
    ],
)
def test_constructors_reject_invalid_dimensions(factory: ModelFactory, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        factory()


def test_cnn_rejects_image_size_incompatible_with_pooling_stages() -> None:
    with pytest.raises(ValueError, match=r"image_size|pool"):
        CompactCNN(image_size=6)


@pytest.mark.parametrize("factory", _factories())
def test_seeded_initialization_and_logits_are_exactly_repeatable(
    factory: ModelFactory,
) -> None:
    first = _seeded(factory, seed=99)
    second = _seeded(factory, seed=99)
    images = torch.arange(2 * 8 * 8, dtype=torch.float32).reshape(2, 1, 8, 8) / 128.0

    assert first.state_dict().keys() == second.state_dict().keys()
    for name in first.state_dict():
        torch.testing.assert_close(
            first.state_dict()[name], second.state_dict()[name], rtol=0.0, atol=0.0
        )
    torch.testing.assert_close(first(images), second(images), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("factory", _factories())
def test_state_dict_reload_restores_identical_logits(factory: ModelFactory) -> None:
    original = _seeded(factory, seed=123)
    original.eval()
    images = torch.linspace(-1.0, 1.0, steps=4 * 8 * 8).reshape(4, 1, 8, 8)
    expected = original(images).detach().clone()
    buffer = io.BytesIO()
    torch.save(original.state_dict(), buffer)
    buffer.seek(0)

    restored = _seeded(factory, seed=456)
    state = torch.load(buffer, map_location="cpu", weights_only=True)
    restored.load_state_dict(state)
    restored.eval()

    torch.testing.assert_close(restored(images), expected, rtol=0.0, atol=0.0)
    for name in original.state_dict():
        torch.testing.assert_close(
            original.state_dict()[name], restored.state_dict()[name], rtol=0.0, atol=0.0
        )


def test_first_layer_filters_are_a_detached_defensive_copy() -> None:
    model = _seeded(lambda: CompactCNN(channels=5, hidden_units=12, n_classes=10), seed=77)
    assert isinstance(model, CompactCNN)
    original_weight = model.features[0].weight.detach().cpu().clone()  # type: ignore[union-attr]

    filters = model.first_layer_filters()

    assert filters.shape == (5, 1, 3, 3)
    assert filters.device.type == "cpu"
    assert not filters.requires_grad
    filters.zero_()
    torch.testing.assert_close(
        model.features[0].weight.detach().cpu(),  # type: ignore[union-attr]
        original_weight,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("factory", _factories())
def test_backward_produces_finite_parameter_gradients(factory: ModelFactory) -> None:
    model = _seeded(factory, seed=88)
    images = torch.rand((4, 1, 8, 8), generator=torch.Generator().manual_seed(5))
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    loss = nn.functional.cross_entropy(model(images), labels)
    loss.backward()

    gradients = [parameter.grad for parameter in model.parameters()]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
