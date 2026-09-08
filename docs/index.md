# Learning Systems Atlas

**Signal. Structure. Strategy.**

A mathematical research-engineering library with from-scratch supervised and
unsupervised models, reverse-mode autodiff and neural training, and an offline
reinforcement-learning control laboratory. The shared experiment boundary is
`Experiment.run(RunContext) -> RunResult`; artifacts include configuration,
environment and content hashes. These are learning systems and research evidence,
not clinical, financial, robotic or other production deployment qualifications.

Use Python 3.11–3.13 and the pinned uv version in `pyproject.toml`:

```bash
git clone https://github.com/srgangaram-swe/learning-systems-atlas.git
cd learning-systems-atlas
uv sync --locked --all-groups
uv run learning-atlas run-all --config-dir configs --output-dir runs/quickstart
```

The reference profiles need no network data or accelerator. Output directories
must be empty; the runner publishes complete evidence atomically. A fresh run
records timing and provenance, so those fields need not be byte-identical across
machines. See [reproducibility](reproducibility.md) for the precise contract.

Start with the [guided notebooks](notebooks.md), examine the
[family cards](model-cards.md), then follow the source-derived
[API reference](api/index.md). The three main navigation panels contain supervised
(including deep learning), unsupervised and reinforcement systems.
The [release parity study](sprint-06.md) records both numerical agreement and
timing differences against independent implementations.

Distributed and generative-AI milestones remain planned. Their existence on the
roadmap is not evidence that a model, cluster speedup or deployment exists.
