"""Safe, atomic storage for small local experiment artifacts."""

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class ArtifactStore:
    """Restrict writes to one run directory and publish files atomically."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        """Return the absolute run root for library integrations that need a path."""

        return self._root

    def resolve(self, relative_path: str) -> Path:
        """Resolve a relative artifact path while preventing directory traversal."""

        candidate = Path(relative_path)
        if not relative_path.strip() or candidate == Path("."):
            msg = "artifact path must name a file within the run directory"
            raise ValueError(msg)
        if candidate.is_absolute() or ".." in candidate.parts:
            msg = f"artifact path must stay within the run directory: {relative_path}"
            raise ValueError(msg)
        resolved = (self._root / candidate).resolve()
        if not resolved.is_relative_to(self._root):
            msg = f"artifact path escapes the run directory: {relative_path}"
            raise ValueError(msg)
        return resolved

    @contextmanager
    def atomic_target(self, relative_path: str) -> Iterator[Path]:
        """Yield a temporary path and atomically promote it after a successful write."""

        destination = self.resolve(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            yield temporary
            os.replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def write_json(self, relative_path: str, payload: object) -> Path:
        """Serialize strict JSON with stable key ordering."""

        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        return self.write_text(relative_path, f"{encoded}\n")

    def write_text(self, relative_path: str, content: str) -> Path:
        """Write UTF-8 text atomically."""

        with self.atomic_target(relative_path) as temporary:
            temporary.write_text(content, encoding="utf-8")
        return self.resolve(relative_path)

    def files(self) -> tuple[Path, ...]:
        """List published files relative to the store root in stable order."""

        return tuple(
            sorted(path.relative_to(self._root) for path in self._root.rglob("*") if path.is_file())
        )

    def sha256(self, relative_path: str) -> str:
        """Calculate a published artifact's SHA-256 digest."""

        digest = hashlib.sha256()
        with self.resolve(relative_path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
