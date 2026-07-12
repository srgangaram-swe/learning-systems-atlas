"""Artifact isolation and atomic-publication tests."""

import hashlib
from pathlib import Path

import pytest

from learning_atlas.core.artifacts import ArtifactStore

pytestmark = pytest.mark.unit


def test_json_is_stable_and_checksummed(artifact_store: ArtifactStore) -> None:
    published = artifact_store.write_json("nested/result.json", {"z": 1, "a": 2})

    expected = '{\n  "a": 2,\n  "z": 1\n}\n'
    assert published.read_text(encoding="utf-8") == expected
    assert (
        artifact_store.sha256("nested/result.json") == hashlib.sha256(expected.encode()).hexdigest()
    )
    assert artifact_store.files() == (Path("nested/result.json"),)


@pytest.mark.parametrize(
    "path",
    ["", ".", "../escape.json", "/tmp/escape.json", "nested/../../escape"],
)
def test_path_traversal_is_rejected(artifact_store: ArtifactStore, path: str) -> None:
    with pytest.raises(ValueError, match=r"run directory|name a file"):
        artifact_store.write_text(path, "unsafe")


def test_failed_atomic_write_leaves_no_partial_artifact(artifact_store: ArtifactStore) -> None:
    with pytest.raises(RuntimeError, match="simulated"):
        with artifact_store.atomic_target("models/model.bin") as temporary:
            temporary.write_bytes(b"partial")
            raise RuntimeError("simulated failure")

    assert not artifact_store.resolve("models/model.bin").exists()
    assert artifact_store.files() == ()


def test_failed_replacement_preserves_existing_artifact(artifact_store: ArtifactStore) -> None:
    artifact_store.write_text("model.txt", "known-good")
    with pytest.raises(RuntimeError):
        with artifact_store.atomic_target("model.txt") as temporary:
            temporary.write_text("corrupt")
            raise RuntimeError("stop")
    assert artifact_store.resolve("model.txt").read_text() == "known-good"
    assert not tuple(artifact_store.root.glob("*.tmp"))


def test_json_rejects_nan_without_publishing(artifact_store: ArtifactStore) -> None:
    with pytest.raises(ValueError, match="JSON compliant"):
        artifact_store.write_json("bad.json", {"metric": float("nan")})
    assert artifact_store.files() == ()


def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "run")
    outside = tmp_path / "outside"
    outside.mkdir()
    (store.root / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        store.write_text("link/escaped.txt", "unsafe")
    assert not (outside / "escaped.txt").exists()


def test_root_is_resolved_and_text_is_published(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "parent" / ".." / "run")
    assert store.root == (tmp_path / "run").resolve()
    assert store.write_text("note.txt", "complete").read_text() == "complete"
