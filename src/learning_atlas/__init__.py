"""Learning Systems Atlas public package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("learning-systems-atlas")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
