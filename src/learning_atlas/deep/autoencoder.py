"""A bottleneck autoencoder with reconstruction-error anomaly scoring."""

from __future__ import annotations

import torch
from torch import nn


def _positive(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        msg = f"{name} must be a positive integer"
        raise ValueError(msg)
    return value


def _validated_flat_batch(inputs: torch.Tensor, *, n_features: int, context: str) -> torch.Tensor:
    if not isinstance(inputs, torch.Tensor):
        msg = f"{context} requires a torch tensor input batch"
        raise TypeError(msg)
    if inputs.ndim != 2 or inputs.shape[1] != n_features:
        msg = f"{context} expects batches shaped (n, {n_features}); received {tuple(inputs.shape)}"
        raise ValueError(msg)
    if inputs.shape[0] == 0:
        msg = f"{context} requires a non-empty input batch"
        raise ValueError(msg)
    if not inputs.is_floating_point():
        msg = f"{context} requires floating-point inputs"
        raise TypeError(msg)
    if not bool(torch.all(torch.isfinite(inputs))):
        msg = f"{context} requires finite input values"
        raise ValueError(msg)
    return inputs


class BottleneckAutoencoder(nn.Module):
    """A symmetric dense autoencoder whose latent width is the model's capacity dial."""

    def __init__(
        self, *, n_features: int = 64, hidden_units: int = 32, latent_dim: int = 8
    ) -> None:
        super().__init__()
        self._n_features = _positive(n_features, name="n_features")
        hidden = _positive(hidden_units, name="hidden_units")
        latent = _positive(latent_dim, name="latent_dim")
        if not latent < hidden < self._n_features:
            msg = "expected latent_dim < hidden_units < n_features for a true bottleneck"
            raise ValueError(msg)
        self.encoder = nn.Sequential(
            nn.Linear(self._n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self._n_features),
        )

    @property
    def n_features(self) -> int:
        """Width of the input and reconstruction space."""

        return self._n_features

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Map inputs into the latent bottleneck."""

        validated = _validated_flat_batch(inputs, n_features=self._n_features, context="encode")
        latent: torch.Tensor = self.encoder(validated)
        return latent

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Reconstruct inputs through the bottleneck."""

        return self.decoder(self.encode(inputs))  # type: ignore[no-any-return]


def reconstruction_errors(model: BottleneckAutoencoder, inputs: torch.Tensor) -> torch.Tensor:
    """Return per-sample mean-squared reconstruction error without gradients."""

    validated = _validated_flat_batch(
        inputs,
        n_features=model.n_features,
        context="reconstruction_errors",
    )
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            device = next(model.parameters()).device
            moved = validated.to(device)
            reconstructed = model(moved)
            errors = torch.mean((reconstructed - moved) ** 2, dim=1).detach().cpu()
    finally:
        model.train(was_training)
    if not bool(torch.all(torch.isfinite(errors))):
        msg = "reconstruction errors became non-finite"
        raise ValueError(msg)
    return errors
