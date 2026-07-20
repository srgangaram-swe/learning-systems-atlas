"""Enforce a raw branch-edge coverage floor from coverage.py XML output."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from xml.etree import ElementTree


class CoverageGateError(ValueError):
    """Raised when a coverage report cannot support a trustworthy decision."""


def read_branch_rate(path: Path) -> float:
    """Return the validated raw branch rate encoded by coverage.py."""

    try:
        root = ElementTree.parse(path).getroot()
    except (OSError, ElementTree.ParseError) as error:
        msg = f"cannot read coverage report {path}: {error}"
        raise CoverageGateError(msg) from error
    raw_rate = root.get("branch-rate")
    if raw_rate is None:
        msg = f"coverage report {path} does not declare branch-rate"
        raise CoverageGateError(msg)
    try:
        rate = float(raw_rate)
    except ValueError as error:
        msg = f"coverage report {path} has invalid branch-rate {raw_rate!r}"
        raise CoverageGateError(msg) from error
    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        msg = f"coverage report {path} has out-of-range branch-rate {raw_rate!r}"
        raise CoverageGateError(msg)
    return rate


def main(argv: list[str] | None = None) -> int:
    """Check one XML report and return a process-compatible status code."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="coverage.py XML report")
    parser.add_argument(
        "--minimum",
        type=float,
        default=90.0,
        help="minimum raw branch coverage percentage (default: 90.0)",
    )
    arguments = parser.parse_args(argv)
    if not math.isfinite(arguments.minimum) or not 0.0 <= arguments.minimum <= 100.0:
        parser.error("--minimum must be finite and lie in [0, 100]")
    try:
        observed = 100.0 * read_branch_rate(arguments.report)
    except CoverageGateError as error:
        parser.error(str(error))
    print(f"Raw branch coverage: {observed:.2f}% (required: {arguments.minimum:.2f}%)")
    if observed + 1e-12 < arguments.minimum:
        print("Raw branch coverage floor was not met.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
