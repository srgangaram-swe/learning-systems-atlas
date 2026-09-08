"""Execute the three trusted, committed tours without retaining output artifacts."""

import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("supervised", "unsupervised", "reinforcement"):
        notebook = nbformat.read(root / "notebooks" / f"{name}.ipynb", as_version=4)
        with tempfile.TemporaryDirectory(prefix=f"atlas-{name}-") as directory:
            NotebookClient(
                notebook,
                timeout=180,
                kernel_name="python3",
                resources={"metadata": {"path": directory}},
                allow_errors=False,
            ).execute()
        print(f"Executed {name}: {len(notebook.cells)} cells")


if __name__ == "__main__":
    main()
