"""Tests for the explicit raw branch-coverage release gate."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _report(tmp_path: Path, branch_rate: str) -> Path:
    path = tmp_path / "coverage.xml"
    path.write_text(f'<coverage branch-rate="{branch_rate}" />', encoding="utf-8")
    return path


def _run(report: Path, *, minimum: str = "90") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/check_branch_coverage.py",
            str(report),
            "--minimum",
            minimum,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_gate_accepts_coverage_at_the_floor(tmp_path: Path) -> None:
    completed = _run(_report(tmp_path, "0.9"))

    assert completed.returncode == 0
    assert "Raw branch coverage: 90.00%" in completed.stdout
    assert completed.stderr == ""


@pytest.mark.parametrize("branch_rate", ["nan", "-0.1", "1.1", "not-a-number"])
def test_gate_rejects_invalid_branch_rates(tmp_path: Path, branch_rate: str) -> None:
    completed = _run(_report(tmp_path, branch_rate))

    assert completed.returncode == 2
    assert "branch-rate" in completed.stderr


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("<coverage />", "does not declare"),
        ("<coverage", "cannot read"),
    ],
)
def test_gate_rejects_unusable_reports(tmp_path: Path, contents: str, message: str) -> None:
    path = tmp_path / "coverage.xml"
    path.write_text(contents, encoding="utf-8")
    completed = _run(path)

    assert completed.returncode == 2
    assert message in completed.stderr


def test_gate_reports_a_floor_failure(tmp_path: Path) -> None:
    completed = _run(_report(tmp_path, "0.9"), minimum="90.01")

    assert completed.returncode == 1
    assert "required: 90.01%" in completed.stdout
    assert "floor was not met" in completed.stderr
