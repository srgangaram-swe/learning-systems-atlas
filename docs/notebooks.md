# Guided notebooks

Each notebook is a thin client of the typed experiment runner. It loads a committed
configuration, publishes into a temporary run directory, displays the recorded
metrics and Seaborn-generated plots, and ends with a model-selection discussion.
No estimator mathematics is hidden inside notebook cells.

- [Supervised tour](https://github.com/srgangaram-swe/learning-systems-atlas/blob/main/notebooks/supervised.ipynb): training-only cross-validation, untouched-test reporting and naive baselines.
- [Unsupervised tour](https://github.com/srgangaram-swe/learning-systems-atlas/blob/main/notebooks/unsupervised.ipynb): label-free selection and retrospective label diagnostics.
- [Reinforcement tour](https://github.com/srgangaram-swe/learning-systems-atlas/blob/main/notebooks/reinforcement.ipynb): exploration versus evaluation, finite interaction budgets and a random policy baseline.

```bash
uv sync --locked --all-groups
uv run python scripts/execute_notebooks.py
```

The nbclient runner gives each notebook an isolated working directory and each
cell a hard timeout. Execution failures fail the documentation job. Executed
outputs remain temporary and are never committed. To explore interactively,
open a notebook using the environment's Python kernel; the repository root is
found from the installed package. Reference data is bundled or generated offline.
