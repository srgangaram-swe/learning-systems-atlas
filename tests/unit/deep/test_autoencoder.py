"""Mathematical, state, and anomaly-score contracts for the autoencoder."""

from collections.abc import Callable

import pytest
import torch

from learning_atlas.deep.autoencoder import BottleneckAutoencoder, reconstruction_errors

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_autoencoder_shapes_dtype_finiteness_and_gradient_flow(dtype: torch.dtype) -> None:
    torch.manual_seed(41)
    model = BottleneckAutoencoder(n_features=8, hidden_units=5, latent_dim=2).to(dtype=dtype)
    inputs = torch.randn(6, 8, dtype=dtype, requires_grad=True)

    latent = model.encode(inputs)
    reconstructed = model(inputs)
    torch.mean((reconstructed - inputs) ** 2).backward()

    assert model.n_features == 8
    assert latent.shape == (6, 2)
    assert reconstructed.shape == inputs.shape
    assert latent.dtype == dtype
    assert reconstructed.dtype == dtype
    assert bool(torch.all(torch.isfinite(latent)))
    assert bool(torch.all(torch.isfinite(reconstructed)))
    assert inputs.grad is not None
    assert bool(torch.all(torch.isfinite(inputs.grad)))
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert bool(torch.all(torch.isfinite(parameter.grad)))


def test_reconstruction_errors_equal_hand_computed_per_sample_mse() -> None:
    model = BottleneckAutoencoder(n_features=4, hidden_units=2, latent_dim=1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, -2.0, 0.0, 2.0]])

    errors = reconstruction_errors(model, inputs)

    torch.testing.assert_close(errors, torch.tensor([7.5, 2.0]))
    assert errors.shape == (2,)
    assert errors.dtype == inputs.dtype
    assert errors.requires_grad is False
    assert all(parameter.grad is None for parameter in model.parameters())


def test_reconstruction_errors_rank_more_corrupted_inputs_higher() -> None:
    model = BottleneckAutoencoder(n_features=4, hidden_units=2, latent_dim=1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    inliers = torch.full((5, 4), 0.1)
    anomalies = torch.full((5, 4), 4.0)

    inlier_errors = reconstruction_errors(model, inliers)
    anomaly_errors = reconstruction_errors(model, anomalies)

    assert float(torch.min(anomaly_errors)) > float(torch.max(inlier_errors))


@pytest.mark.parametrize("training", [False, True])
def test_reconstruction_errors_restores_the_callers_model_mode(training: bool) -> None:
    model = BottleneckAutoencoder(n_features=4, hidden_units=2, latent_dim=1)
    model.train(training)

    reconstruction_errors(model, torch.zeros(2, 4))

    assert model.training is training


def test_reconstruction_errors_restores_training_mode_after_nonfinite_output() -> None:
    model = BottleneckAutoencoder(n_features=4, hidden_units=2, latent_dim=1).train()
    final_layer = model.decoder[-1]
    assert isinstance(final_layer, torch.nn.Linear)
    with torch.no_grad():
        final_layer.bias.fill_(torch.nan)

    with pytest.raises(ValueError, match="non-finite"):
        reconstruction_errors(model, torch.zeros(2, 4))

    assert model.training is True


def test_reconstruction_errors_restores_mode_when_forward_validation_fails() -> None:
    model = BottleneckAutoencoder(n_features=4, hidden_units=2, latent_dim=1).train()

    with pytest.raises(ValueError, match="shaped"):
        reconstruction_errors(model, torch.zeros(2, 3))

    assert model.training is True


def test_reconstruction_errors_rejects_non_tensor_inputs_actionably() -> None:
    model = BottleneckAutoencoder(n_features=4, hidden_units=3, latent_dim=2)

    with pytest.raises(TypeError, match="torch tensor"):
        reconstruction_errors(model, [[0.0, 0.0, 0.0, 0.0]])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: BottleneckAutoencoder(n_features=False), "n_features"),
        (lambda: BottleneckAutoencoder(n_features=4, hidden_units=0), "hidden_units"),
        (lambda: BottleneckAutoencoder(n_features=4, hidden_units=2, latent_dim=-1), "latent_dim"),
        (
            lambda: BottleneckAutoencoder(n_features=4, hidden_units=3, latent_dim=3),
            "latent_dim < hidden_units < n_features",
        ),
        (
            lambda: BottleneckAutoencoder(n_features=4, hidden_units=4, latent_dim=1),
            "latent_dim < hidden_units < n_features",
        ),
    ],
)
def test_autoencoder_constructor_enforces_true_bottleneck(
    factory: Callable[[], BottleneckAutoencoder], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        (torch.zeros(4), "shaped"),
        (torch.zeros(2, 5), "shaped"),
        (torch.zeros(0, 4), "sample|non-empty"),
        (torch.full((2, 4), torch.nan), "finite"),
        (torch.ones(2, 4, dtype=torch.int64), "floating"),
    ],
)
def test_autoencoder_rejects_invalid_batches(inputs: torch.Tensor, message: str) -> None:
    model = BottleneckAutoencoder(n_features=4, hidden_units=2, latent_dim=1)
    with pytest.raises((TypeError, ValueError), match=message):
        model(inputs)
