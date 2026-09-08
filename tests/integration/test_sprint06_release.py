"""Release evidence remains complete, transactional and traceable."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


def test_parity_report_preserves_failures_and_refuses_overwrite(tmp_path):
    output = tmp_path / "report"
    command = [sys.executable, "scripts/reference_parity.py", str(output)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90)
    report = json.loads((output / "observations.json").read_text())
    assert len(report["rows"]) == 27
    assert result.returncode == int(any(not row["passed"] for row in report["rows"]))
    manifest = json.loads((output / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    previous = (output / "manifest.json").read_bytes()
    repeat = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert repeat.returncode == 2
    assert (output / "manifest.json").read_bytes() == previous
    assert not list(tmp_path.glob(".parity-*"))


def test_every_public_class_has_current_generated_api_reference():
    result = subprocess.run(
        [sys.executable, "scripts/build_api_reference.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    inventory = json.loads((ROOT / "docs/api/inventory.json").read_text())
    assert len(inventory) > 100
    for item in inventory:
        page = (ROOT / "docs/api" / item["page"]).read_text()
        assert f"::: {item['module']}.{item['class']}" in page
        assert "../model-cards.md" in page


def test_notebooks_are_output_free_thin_clients():
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert len(notebooks) == 3
    for path in notebooks:
        notebook = json.loads(path.read_text())
        cells = notebook["cells"]
        assert any("Model-selection discussion" in "".join(cell["source"]) for cell in cells)
        code = "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")
        assert "run_experiment(config, output)" in code
        for cell in cells:
            if cell["cell_type"] == "code":
                assert cell["outputs"] == [] and cell["execution_count"] is None
