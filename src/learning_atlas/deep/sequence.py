"""Sequence classifiers that make padding and true lengths explicit."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


def _positive(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        msg = f"{name} must be a positive integer"
        raise ValueError(msg)
    return value


def _validated_sequence_batch(
    padded: torch.Tensor, lengths: torch.Tensor, *, context: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(padded, torch.Tensor) or not isinstance(lengths, torch.Tensor):
        msg = f"{context} requires tensor sequences and lengths"
        raise TypeError(msg)
    if padded.ndim != 3:
        msg = f"{context} expects padded batches shaped (n, time, features)"
        raise ValueError(msg)
    if lengths.ndim != 1 or lengths.shape[0] != padded.shape[0]:
        msg = f"{context} expects one length per sequence"
        raise ValueError(msg)
    if padded.shape[0] == 0:
        msg = f"{context} requires a non-empty sequence batch"
        raise ValueError(msg)
    if not padded.is_floating_point():
        msg = f"{context} requires floating-point sequence values"
        raise TypeError(msg)
    if not bool(torch.all(torch.isfinite(padded))):
        msg = f"{context} requires finite sequence values"
        raise ValueError(msg)
    if lengths.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        msg = f"{context} requires integer sequence lengths"
        raise TypeError(msg)
    if not bool(torch.all(lengths >= 1)):
        msg = f"{context} requires every sequence length to be at least 1"
        raise ValueError(msg)
    if not bool(torch.all(lengths <= padded.shape[1])):
        msg = f"{context} received lengths exceeding the padded time dimension"
        raise ValueError(msg)
    return padded, lengths


class SequenceLSTM(nn.Module):
    """An LSTM classifier that packs padded batches so padding never enters the cell."""

    def __init__(
        self,
        *,
        input_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 1,
        n_classes: int = 2,
    ) -> None:
        super().__init__()
        _positive(input_size, name="input_size")
        _positive(hidden_size, name="hidden_size")
        _positive(num_layers, name="num_layers")
        _positive(n_classes, name="n_classes")
        self._input_size = input_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, n_classes)

    def forward(self, padded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Classify from the final hidden state at each sequence's true end."""

        validated, valid_lengths = _validated_sequence_batch(
            padded, lengths, context="SequenceLSTM"
        )
        if validated.shape[2] != self._input_size:
            msg = (
                f"SequenceLSTM was sized for {self._input_size} features; "
                f"received {validated.shape[2]}"
            )
            raise ValueError(msg)
        packed = pack_padded_sequence(
            validated,
            valid_lengths.detach().cpu().to(torch.int64),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (final_hidden, _) = self.lstm(packed)
        logits: torch.Tensor = self.head(final_hidden[-1])
        return logits


class PaddedSequenceMLP(nn.Module):
    """A deliberately position-bound baseline over the zero-padded layout.

    The model flattens the padded tensor, so its weights attach to absolute time
    positions and it receives no length information. On variable-length tasks
    whose informative token moves with the sequence end, this baseline is
    structurally unable to track the signal — that failure is the evidence the
    LSTM comparison needs.
    """

    def __init__(
        self,
        *,
        max_length: int,
        input_size: int = 1,
        hidden_units: int = 64,
        n_classes: int = 2,
    ) -> None:
        super().__init__()
        self._max_length = _positive(max_length, name="max_length")
        self._input_size = _positive(input_size, name="input_size")
        hidden = _positive(hidden_units, name="hidden_units")
        classes = _positive(n_classes, name="n_classes")
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._max_length * self._input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, classes),
        )

    def forward(self, padded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Classify from the padded layout alone; lengths are intentionally unused."""

        validated, _ = _validated_sequence_batch(padded, lengths, context="PaddedSequenceMLP")
        if validated.shape[1] != self._max_length or validated.shape[2] != self._input_size:
            msg = (
                "PaddedSequenceMLP was sized for "
                f"(n, {self._max_length}, {self._input_size}); "
                f"received {tuple(validated.shape)}"
            )
            raise ValueError(msg)
        logits: torch.Tensor = self.classifier(validated)
        return logits
