"""Verify a local reference manifest and publish only safe, explicitly allowed evidence.

Run after inspecting the generated plots. No model, checkpoint, host log, cache,
or arbitrary run file can enter the public reference through this allowlist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


def publish(source: Path, destination: Path) -> None:
    """Copy safe aggregates/plots only after hash and clean-source validation."""

    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((source / "result.json").read_text(encoding="utf-8"))
    if (
        manifest["experiment"] != "reinforcement_benchmark"
        or result["experiment"] != manifest["experiment"]
    ):
        raise ValueError("source is not a reinforcement benchmark")
    if manifest["environment"]["git"]["dirty"] is not False:
        raise ValueError("reference requires a recorded clean source commit")
    plot_names = (
        "bandit_posteriors",
        "bandit_rewards",
        "bernoulli_regret",
        "cliff_learning",
        "cliff_risk",
        "dqn_ablation",
        "dqn_td_loss",
        "gaussian_regret",
        "gridworld_policy",
        "heldout_return_distribution",
        "heldout_seed_means",
        "interaction_cost",
        "navigation_learning",
        "neural_learning",
        "neural_validation",
        "planning_residuals",
        "ppo_optimization",
        "reinforce_gradient_variance",
        "sample_efficiency",
        "tabular_policy_gap",
    )
    allowed = {
        "evidence.json",
        "comparison.csv",
        "comparison.md",
        "resolved_config.json",
        "result.json",
    }
    allowed.update(f"plots/{name}.png" for name in plot_names)
    if set(manifest["artifacts_sha256"]) != allowed:
        raise ValueError("reference manifest contains missing or unexpected artifacts")
    for name in sorted(allowed):
        path = source / name
        if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()):
            raise ValueError("reference files cannot escape the source directory")
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("reference artifact exceeds the public evidence size budget")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest["artifacts_sha256"][name]:
            raise ValueError(f"reference checksum mismatch: {name}")
    target = destination.resolve()
    if target.exists():
        raise FileExistsError("reference destination must not exist")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        for name in sorted(allowed):
            output = staging / name
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, output)
        # The runner manifest contains versions/platform/commit, not credentials
        # or user paths. Keep timing provenance separate from deterministic data.
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    try:
        publish(arguments.source, arguments.destination)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"reference publication failed: {error}\n")


if __name__ == "__main__":
    main()
