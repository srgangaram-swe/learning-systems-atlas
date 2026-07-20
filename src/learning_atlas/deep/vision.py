"""Compact convolutional and dense reference models for 8x8 digit images."""

from __future__ import annotations

import torch
from torch import nn


def _positive(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        msg = f"{name} must be a positive integer"
        raise ValueError(msg)
    return value


def _validated_images(images: torch.Tensor, *, image_size: int, context: str) -> torch.Tensor:
    if not isinstance(images, torch.Tensor):
        msg = f"{context} requires a torch tensor image batch"
        raise TypeError(msg)
    expected = (1, image_size, image_size)
    if images.ndim != 4 or tuple(images.shape[1:]) != expected:
        msg = (
            f"{context} expects batches shaped (n, 1, {image_size}, {image_size}); "
            f"received {tuple(images.shape)}"
        )
        raise ValueError(msg)
    if images.shape[0] == 0:
        msg = f"{context} requires a non-empty image batch"
        raise ValueError(msg)
    if not images.is_floating_point():
        msg = f"{context} requires floating-point images"
        raise TypeError(msg)
    if not bool(torch.all(torch.isfinite(images))):
        msg = f"{context} requires finite image values"
        raise ValueError(msg)
    return images


class VisionMLP(nn.Module):
    """A dense baseline that consumes the same image tensors as the CNN."""

    def __init__(self, *, image_size: int = 8, hidden_units: int = 32, n_classes: int = 10) -> None:
        super().__init__()
        self._image_size = _positive(image_size, name="image_size")
        hidden = _positive(hidden_units, name="hidden_units")
        classes = _positive(n_classes, name="n_classes")
        n_inputs = self._image_size * self._image_size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_inputs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of single-channel images."""

        validated = _validated_images(images, image_size=self._image_size, context="VisionMLP")
        logits: torch.Tensor = self.classifier(validated)
        return logits


class CompactCNN(nn.Module):
    """A compact convolutional stack and small head sized for 8x8 grayscale digits."""

    def __init__(
        self,
        *,
        image_size: int = 8,
        channels: int = 8,
        hidden_units: int = 32,
        n_classes: int = 10,
    ) -> None:
        super().__init__()
        self._image_size = _positive(image_size, name="image_size")
        base_channels = _positive(channels, name="channels")
        hidden = _positive(hidden_units, name="hidden_units")
        classes = _positive(n_classes, name="n_classes")
        if self._image_size % 4 != 0:
            msg = "image_size must be divisible by 4 for the two pooling stages"
            raise ValueError(msg)
        pooled = self._image_size // 4
        self.features = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.Conv2d(base_channels, 2 * base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * base_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(2 * base_channels, 4 * base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((pooled, pooled)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * base_channels * pooled * pooled, hidden),
            nn.ReLU(),
            nn.Linear(hidden, classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of single-channel images."""

        validated = _validated_images(images, image_size=self._image_size, context="CompactCNN")
        logits: torch.Tensor = self.classifier(self.features(validated))
        return logits

    def first_layer_filters(self) -> torch.Tensor:
        """Return learned first-convolution kernels for evidence plots."""

        first = self.features[0]
        assert isinstance(first, nn.Conv2d)
        return first.weight.detach().cpu().clone()
