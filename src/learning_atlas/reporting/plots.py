"""Atomic Matplotlib artifact publication."""

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from learning_atlas.core.artifacts import ArtifactStore


def publish_figure(
    figure: Figure,
    artifacts: ArtifactStore,
    relative_path: str,
    *,
    dpi: int = 160,
) -> str:
    """Save and close a figure, publishing only a complete PNG."""

    try:
        with artifacts.atomic_target(relative_path) as temporary:
            figure.savefig(temporary, format="png", dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(figure)
    return relative_path
