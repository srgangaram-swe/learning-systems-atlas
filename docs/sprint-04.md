# Sprint 4 — Deep-learning systems

Sprint 4 connects mathematical first principles to a reusable training system.
It implements reverse-mode automatic differentiation and a multilayer
perceptron in NumPy, then uses a shared PyTorch trainer for image, sequence, and
representation-learning studies. The four reference experiments run through
the same `Experiment.run(RunContext) -> RunResult` boundary and publish one
transactional, hash-manifested evidence suite.

The architectural separation between educational differentiation, operational
training state, and evaluation state is recorded in
[ADR 0006](adr/0006-deep-learning-training-and-evaluation-boundaries.md).
All figures below are deterministic point estimates from the committed CPU
reference profile with root seed 42. They are evidence for the stated synthetic
and bundled-data tasks, not estimates of frontier-scale performance.

## Work-item traceability

| Work item | Delivered surface |
|---|---|
| [#24 — minimal reverse-mode autograd engine](https://github.com/srgangaram-swe/learning-systems-atlas/issues/24) | Dynamic tensor graph, reverse topological differentiation, broadcasting-aware vector-Jacobian products, stable log-softmax, finite-difference checks, graph-reuse semantics, and explicit numerical failures |
| [#25 — MLP classifier on the autograd engine](https://github.com/srgangaram-swe/learning-systems-atlas/issues/25) | Multilayer ReLU/tanh network, softmax cross-entropy, named initialization and shuffle streams, minibatch SGD with momentum, transactional refits, XOR/two-moons comparison, and optimization plots |
| [#26 — PyTorch training-loop infrastructure](https://github.com/srgangaram-swe/learning-systems-atlas/issues/26) | Typed trainer configuration, task objectives, device resolution, deterministic loading, sample-weighted metrics, numerical guards, early stopping, atomic checkpoints, exact resume state, and best-weight restoration |
| [#27 — CNN image classifier](https://github.com/srgangaram-swe/learning-systems-atlas/issues/27) | Compact convolutional network and dense baseline on bundled 8×8 digits, validation-only selection, checkpoint reload audit, confusion matrix, error gallery, and learned-filter evidence |
| [#28 — LSTM sequence model](https://github.com/srgangaram-swe/learning-systems-atlas/issues/28) | Packed variable-length LSTM and position-bound padded MLP on temporal XOR, true-length validation, padding-invariance audit, and accuracy-by-length evidence |
| [#29 — autoencoder for representation and anomaly detection](https://github.com/srgangaram-swe/learning-systems-atlas/issues/29) | Dense bottleneck autoencoder, clean-only reconstruction training, independently corrupted validation/test anomalies, PCA reconstruction baseline, AUROC, latent structure, and reconstruction diagnostics |

## Reference profile and observed evidence

The configuration directory `configs/deep/sprint-04` contains four bounded,
network-independent profiles:

| Study | Data and split | Model budget | Optimization |
|---|---|---|---|
| Scratch MLP | 320 XOR and 320 two-moons samples; 20% validation and 25% test per task | one 16-unit tanh hidden layer | 300 epochs, batch 32, SGD 0.30, momentum 0.90 |
| Vision | 1,200 stratified bundled digits; 16% validation and 20% test | 12 base CNN channels, 48-unit head; matched 48-unit dense baseline | at most 30 epochs, batch 64, Adam 0.003, patience 8 |
| Sequence | 1,200 temporal-XOR sequences of length 8–32; 16% validation and 20% test | 48-state LSTM; 64-unit padded MLP baseline | at most 120 epochs, batch 64, Adam 0.005, patience 40 |
| Autoencoder | 1,200 bundled digits; 16% validation and 20% clean test inliers; 50% as many independently corrupted anomalies | 64→32→8→32→64 bottleneck; rank-8 PCA baseline | at most 60 epochs, batch 64, Adam 0.003, patience 8 |

Observed reference results are:

| Study | Validation selection evidence | Untouched-test evidence | Diagnostic |
|---|---:|---:|---:|
| Autograd MLP | mean accuracy **1.0000** | mean accuracy **1.0000** | maximum finite-difference gradient error **1.5634×10⁻⁹** |
| Logistic baseline | mean accuracy **0.5938** | mean accuracy **0.6000** | linear reference on the same planar partitions |
| Compact CNN | accuracy **0.9793** | accuracy **0.9707** | selected over the dense image MLP |
| Dense image MLP | accuracy **0.9637** | accuracy **0.9414** | test gap to CNN **0.0293** |
| Packed LSTM | accuracy **1.0000** | accuracy **1.0000** | appended-padding maximum logit delta **0.0** |
| Padded sequence MLP | accuracy **0.5052** | accuracy **0.5625** | test gap to LSTM **0.4375** |
| Bottleneck autoencoder | anomaly AUROC **1.0000** | anomaly AUROC **1.0000** | mean-error ratio **9.4343**; latent silhouette **0.1947** |
| Rank-8 PCA reconstruction | anomaly AUROC **1.0000** | anomaly AUROC **1.0000** | error ratio **7.4342**; autoencoder wins the declared validation tie-break |

These numbers require restrained interpretation. The nonlinear planar tasks are
small and synthetic. The CNN improvement is 2.93 percentage points on one
fixed split, not a confidence interval. Temporal XOR is deliberately designed
to reward memory of the first token and correct handling of a moving final
token; the result demonstrates the intended recurrent inductive bias, not
universal LSTM superiority. Pixel permutation is an easy, synthetic anomaly
mechanism. Both the autoencoder and PCA obtain perfect test AUROC, so this run
does **not** establish an AUROC advantage for nonlinear reconstruction. The
latent silhouette of 0.1947 indicates some retrospective class structure, but
not clean class separation.

## System boundaries

The sprint intentionally does not force all deep-learning code into one
abstraction:

```text
resolved YAML + root seed
          |
          v
registry -> paradigm-native Experiment -> train/validation/test evidence
                         |                         |
             +-----------+-----------+             v
             |                       |       immutable RunResult
      NumPy autograd MLP       PyTorch Trainer            |
      owns differentiation     owns runtime state          v
             |                       |             staged artifacts
             +-----------+-----------+             + SHA-256 manifest
```

- `deep.autograd` and `deep.mlp` are auditable from-scratch modules. They do not
  import PyTorch, scikit-learn, plotting, configuration, CLI, or artifact code.
- `deep.training` is explicitly PyTorch-native. It centralizes the operational
  concerns that should not be copied into every model.
- `deep.vision`, `deep.sequence`, and `deep.autoencoder` define architecture and
  input contracts. They do not own experiment selection or artifact paths.
- `deep.benchmarks` owns dataset partitioning, candidate construction,
  validation-only selection, test evaluation, and scientific diagnostics.
- `workflows.deep_benchmark` requires exactly four experiments, stages the
  complete suite, writes aggregate reports and hashes, and atomically publishes
  only after every experiment succeeds.

## Reverse-mode automatic differentiation

### Values, adjoints, and vector-Jacobian products

For a directed acyclic computation graph with scalar loss $L$ and intermediate
tensor $v$, reverse mode stores the adjoint

$$
\bar v = \frac{\partial L}{\partial v}.
$$

An operation $z=f(x_1,\ldots,x_k)$ does not materialize its full Jacobian.
Instead, its local rule receives $\bar z$ and contributes the
vector-Jacobian product

$$
\bar x_j \mathrel{+}=
\bar z\,\frac{\partial z}{\partial x_j}.
$$

The implementation supports the exact primitives needed by the scratch MLP.
`unbroadcast` means summing axes introduced or expanded by NumPy broadcasting
until the contribution again has the parent's shape.

| Forward operation | Reverse contribution |
|---|---|
| $z=x+y$ | $\bar x\mathrel{+}=\operatorname{unbroadcast}(\bar z)$ and likewise for $y$ |
| $z=x\odot y$ | $\bar x\mathrel{+}=\operatorname{unbroadcast}(\bar z\odot y)$; $\bar y\mathrel{+}=\operatorname{unbroadcast}(\bar z\odot x)$ |
| $z=x/c$ for finite nonzero scalar $c$ | $\bar x\mathrel{+}=\bar z/c$ |
| $z=x^p$ | $\bar x\mathrel{+}=\bar z\odot p x^{p-1}$; $p=0$ has the exact zero derivative |
| $Z=XY$ | $\bar X\mathrel{+}=\bar ZY^\top$; $\bar Y\mathrel{+}=X^\top\bar Z$ |
| $z=\operatorname{ReLU}(x)$ | $\bar x\mathrel{+}=\bar z\odot\mathbf{1}[x>0]$; the chosen subgradient at zero is zero |
| $z=\tanh x$ | $\bar x\mathrel{+}=\bar z\odot(1-z^2)$ |
| $z=\exp x$ | $\bar x\mathrel{+}=\bar z\odot z$ |
| $z=\log x$ | $\bar x\mathrel{+}=\bar z/x$, restricted to $x>0$ |
| $z=\log\operatorname{softmax}(x)$ | $\bar x\mathrel{+}=\bar z-\operatorname{softmax}(x)\sum_j\bar z_j$ along the selected axis |
| $z=\sum x$ | broadcast $\bar z$ back to the input shape |
| $z=\operatorname{mean}(x)$ | the sum rule divided by the reduced element count |
| $z=\operatorname{reshape}(x)$ | reshape $\bar z$ back to the original shape |

Log-softmax subtracts the per-axis maximum before exponentiation. This leaves
the mathematical probability unchanged while preventing overflow for large
finite logits. Construction and every operation reject non-real, ragged, or
non-finite values. Invalid domains, axes, matrix shapes, backward seeds, and
gradient shapes raise `AutogradError` rather than leaking a low-context NumPy
exception. Gradient contributions and their accumulated sums must remain
finite.

### Why fan-out is correct

Each result records references to the already-existing parent tensors that
created it. A postorder depth-first traversal visits each reachable node once.
Reversing that order places every node before each of its parents, so all paths
from the loss into a shared node have contributed to that node's adjoint before
its local rule executes.

For example, if $u=x^2$ and $L=u+x$, the direct branch contributes $1$ to
$\bar x$, while the branch through $u$ contributes $2x$. Accumulation therefore
produces $\bar x=2x+1$, rather than overwriting one branch with the other.

Correctness follows by reverse induction over this ordering:

1. the loss is seeded with its requested upstream derivative, normally one;
2. assume every processed child holds the sum of derivatives from all paths
   between that child and the loss;
3. each child applies the chain rule and adds its contribution to every parent;
4. when a parent is reached, all its children have been processed, so its
   accumulated adjoint is the full derivative of the loss with respect to it.

Graph-internal adjoints are cleared at the beginning of each reverse sweep.
Leaf adjoints intentionally accumulate across calls, matching established
autodiff semantics. Consequently, calling `backward` twice adds exactly two
fresh vector-Jacobian products; stale internal adjoints are never propagated a
second time. `zero_grad` clears a leaf when the caller wants a fresh total.

For $V$ graph nodes and $E$ parent edges, graph ordering costs $O(V+E)$. The
overall reverse pass costs $O(V+E)$ plus the numerical cost of the local kernels
and stores $O(V+E)$ graph metadata plus saved forward arrays. Central finite
differences,

$$
\frac{\partial L}{\partial\theta_j}\approx
\frac{L(\theta+\varepsilon e_j)-L(\theta-\varepsilon e_j)}{2\varepsilon},
$$

require two forward evaluations per checked parameter element, so they cost
$O(P)$ forwards for $P$ scalar parameters. They are a test oracle, not a
training algorithm. Perturbations are restored in `finally`, including when an
objective fails. With $\varepsilon=10^{-6}$, the reference frozen minibatches
showed maximum absolute disagreement $1.5634\times10^{-9}$.

## Multilayer perceptron from scratch

For layer widths $d_0,\ldots,d_L$, the classifier computes

$$
h_0=x,\qquad z_\ell=h_{\ell-1}W_\ell+b_\ell,
$$

$$
h_\ell=\phi(z_\ell)\quad (\ell<L),\qquad
s=z_L,
$$

where $\phi$ is ReLU or tanh and $s$ is the class-logit vector. For class index
$y_i$, the mean minibatch objective is

$$
\mathcal{L}=-\frac{1}{B}\sum_{i=1}^{B}
\log\frac{\exp s_{i,y_i}}{\sum_c\exp s_{i,c}}.
$$

The loss is formed from the autograd engine's stable log-softmax, so no raw
probability is logged. Integer-valued numerical class labels are sorted and
mapped to contiguous output indices; learned classes are returned through a
read-only defensive array.

Weights are sampled from a named initialization stream with

$$
W_{\ell,jk}\sim\mathcal{N}\left(0,
\frac{g}{d_{\ell-1}}\right),
\qquad
g=\begin{cases}2&\text{ReLU},\\1&\text{tanh},\end{cases}
$$

and biases begin at zero. A separately named stream shuffles minibatches. The
momentum update is

$$
v_t=\mu v_{t-1}-\eta\nabla_\theta\mathcal{L}_t,
\qquad
\theta_t=\theta_{t-1}+v_t.
$$

Gradients, candidate velocities, candidate parameters, and the full-data loss
recorded after each epoch must all remain finite. A refit snapshots the prior
model and restores every weight, bias, class, history, and feature-shape field
if validation or optimization fails. This prevents a failed refit from leaving
an object that claims to be fitted while holding incompatible partial state.

The planar benchmark uses independent named data, split, candidate, parameter,
and shuffle streams for XOR and two moons. Both candidates fit only their
training partitions. Mean validation accuracy selects between the nonlinear MLP
and linear logistic baseline; test accuracy and test log loss are reporting
fields only. Optimization curves show full-training cross-entropy after each
epoch, and decision-boundary panels show the learned class-1 probability field.

For $n$ examples and dense widths $d_0,\ldots,d_L$, one epoch costs

$$
O\left(n\sum_{\ell=1}^{L}d_{\ell-1}d_\ell\right),
$$

with $O(\sum_\ell d_{\ell-1}d_\ell)$ parameter and momentum state plus
minibatch activations. This implementation is intentionally small: it has no
GPU kernels, mixed precision, convolution, dropout, normalization layers,
automatic batching, or higher-order gradients.

## Shared PyTorch trainer

### Ownership and epoch contract

`Trainer` owns device placement, optimizer construction, deterministic loader
shuffling, train/evaluation mode transitions, validation measurement, early
stopping, checkpoint state, and best-weight restoration. A task implements only
the `Objective` protocol: a scalar mean minibatch loss and evaluation metrics.
Classification, sequence classification, and reconstruction objectives share
the loop without hiding their distinct batch arities.

Within a training epoch, the loop:

1. moves one tuple of aligned tensors to the resolved device;
2. clears optimizer gradients;
3. requires a finite scalar objective;
4. catches backward failures and rejects non-finite gradients;
5. optionally clips the gradient norm with non-finite checking;
6. applies Adam or momentum SGD and rejects non-finite model state; and
7. sample-weights minibatch means when forming the epoch loss.

Evaluation disables gradients, preserves the caller's prior train/eval mode,
uses deterministic unshuffled batches, sample-weights loss, and rejects
non-finite metrics. Tensor datasets must be non-empty, aligned on their first
dimension, finite, and non-scalar.

### Early stopping

After every completed epoch, validation loss and objective metrics are recorded
in an immutable `EpochRecord`. An epoch becomes the new best only when

$$
L_{\mathrm{best}}-L_{\mathrm{val},t}>\texttt{min_delta}.
$$

Otherwise the no-improvement counter advances. Training stops when it reaches
`patience`, and the returned model is restored to the saved best state rather
than left at the terminal epoch. `TrainingReport` exposes the complete history,
best epoch/loss, early-stop flag, and resume origin.

### Checkpoint and resume contract

Checkpoint schema `learning-atlas-trainer/2` captures:

- current and best model state;
- optimizer state;
- global Torch and loader-shuffle RNG state;
- every epoch record;
- best epoch/loss and no-improvement count;
- resume-critical trainer settings; and
- model key, shape, and dtype signature.

The terminal epoch budget is intentionally absent from the trainer signature,
so a compatible resume may extend `max_epochs`. Batch size, optimizer,
learning rate, regularization, patience, clipping, seed, and device policy must
remain compatible. Model signatures must match exactly. A checkpoint that has
already reached its configured patience boundary is terminal and cannot be
resumed merely by increasing the epoch budget.

Writes use an in-memory stable archive, a temporary file in the destination
directory, and atomic replacement. Loading uses `torch.load(...,
weights_only=True)` on CPU and then validates the exact schema, required and
unexpected fields, finite model and recursively nested optimizer tensor state,
contiguous history, best-state shape/dtype agreement, early-stopping invariants,
runtime-compatible RNG states, and signatures. Optimizer compatibility is
probed before the caller-owned model is changed; any later restoration failure
rolls model and RNG state back. Resume restores model, optimizer, both RNGs,
history, best state, and patience counter. On the same platform and dependency
set, interrupted-plus-resumed training is tested against uninterrupted training
for exact state and history equality.

The checkpoint protects continuity; it is not a model registry or a signed
artifact format. Resource exhaustion remains possible with hostile files even
under a restricted loader. Only locally produced or otherwise trusted
checkpoints should be loaded.

Checkpoint I/O is $O(P+S_{\mathrm{opt}})$ in time and storage, where $P$ is
model-state size and $S_{\mathrm{opt}}$ is optimizer state. Adam commonly adds
two parameter-sized moment buffers.

## Compact convolutional image classifier

The image study uses the network-independent scikit-learn 8×8 digits bundle,
normalizes intensities to $[0,1]$, and takes an exact-size stratified subsample.
The dense baseline receives the same `(n, 1, 8, 8)` tensors, split, Adam
settings, epoch ceiling, early-stopping rule, and 48-unit head.

For base channel count $C=12$, the compact CNN is:

1. 3×3 convolution $1\rightarrow C$, batch normalization, ReLU;
2. 3×3 convolution $C\rightarrow2C$, batch normalization, ReLU;
3. 2×2 max pooling;
4. 3×3 convolution $2C\rightarrow4C$, ReLU;
5. adaptive average pooling to 2×2; and
6. flatten, dense $16C\rightarrow48$, ReLU, dense $48\rightarrow10$.

The dense comparator flattens 64 pixels and applies $64\rightarrow48\rightarrow
10$ with ReLU. Both are selected by validation accuracy with a deterministic
preference for the CNN on an exact tie. Test accuracy, macro-F1, loss, and the
CNN-versus-MLP gap are reported after candidates are fixed.

Every vision candidate writes a resumable checkpoint. The reload audit compares
best-state tensors, logits, test loss, and test accuracy with the in-memory best
model; the reference deltas are exactly zero. Diagnostic plots include
train/validation loss, validation accuracy, the CNN held-out confusion matrix,
every remaining test error up to ten examples, and learned first-layer filters.

A dense convolution with output height $H$, width $W$, kernel width $k$, and
$C_{\mathrm{in}},C_{\mathrm{out}}$ channels costs
$O(HWk^2C_{\mathrm{in}}C_{\mathrm{out}})$ per example. The parameter count is
$k^2C_{\mathrm{in}}C_{\mathrm{out}}+C_{\mathrm{out}}$ before normalization
parameters. The Sprint 4 model is deliberately compact and makes no claim about
large-image accuracy, transfer learning, augmentation, calibration, or
adversarial robustness.

## Variable-length LSTM

Temporal XOR draws tokens from $\{-1,+1\}$. A sequence label is one exactly
when its first and true final token differ. Length varies from 8 to 32, making
the final informative position move within the zero-padded layout.

For input $x_t$, previous hidden state $h_{t-1}$, and cell $c_{t-1}$, the LSTM
implements the standard gates

$$
i_t=\sigma(W_{ii}x_t+W_{hi}h_{t-1}+b_i),\qquad
f_t=\sigma(W_{if}x_t+W_{hf}h_{t-1}+b_f),
$$

$$
g_t=\tanh(W_{ig}x_t+W_{hg}h_{t-1}+b_g),\qquad
o_t=\sigma(W_{io}x_t+W_{ho}h_{t-1}+b_o),
$$

$$
c_t=f_t\odot c_{t-1}+i_t\odot g_t,
\qquad
h_t=o_t\odot\tanh(c_t).
$$

`pack_padded_sequence` receives validated integer lengths and prevents padding
positions from entering recurrent updates. The classifier uses the final hidden
state at each sequence's true end. Appending eight additional zeros while
keeping the true lengths fixed changes reference logits by exactly 0.0; unit
tests also require zero input gradient at padding positions.

The comparator flattens the fixed padded layout into a 64-unit MLP and
intentionally receives no lengths. Its weights are tied to absolute positions,
so it cannot directly identify the moving final token. Both models otherwise
share the split, optimizer family, learning rate, batch size, epoch ceiling, and
patience. Validation accuracy selects the candidate. The evidence suite reports
train/validation loss, validation accuracy, and held-out accuracy in three
sequence-length buckets.

For input width $D$, hidden width $H$, and true length $T_i$, one LSTM layer has
exactly $4H(D+H+2)+(H+1)K$ trainable parameters for PyTorch's two LSTM bias
vectors and a $K$-class affine head, and costs
$O(\sum_i T_iH(D+H))$ over a batch of packed sequences. Training stores gate
activations across valid time steps. The observed comparison does not match
parameter counts and does not evaluate Transformers, bidirectional recurrence,
forecasting, noisy lengths, or sequence lengths outside the training range.

## Bottleneck autoencoder and anomaly scoring

For flattened digit $x\in\mathbb{R}^{64}$, the autoencoder is

$$
z=f_\theta(x),\qquad \hat x=g_\phi(z),
\qquad 64\rightarrow32\rightarrow8\rightarrow32\rightarrow64,
$$

with ReLU in the hidden encoder and decoder layers and a linear latent code and
reconstruction. It minimizes clean-only mean-squared error

$$
\mathcal{L}_{\mathrm{recon}}=
\frac{1}{n}\sum_{i=1}^{n}\frac{1}{64}
\lVert x_i-\hat x_i\rVert_2^2.
$$

At evaluation, the same per-sample mean squared error is the anomaly score.
Pixel permutations preserve each selected image's intensity multiset while
destroying its spatial arrangement. Validation and test anomalies are sampled
from disjoint partitions with distinct named random streams. The validation
AUROC selects between the autoencoder and a rank-8 PCA reconstruction baseline;
the independent test corruption is reporting-only.

The error separation ratio is

$$
\frac{\operatorname{mean}(s(x_{\mathrm{anomaly}}))}
{\max(\operatorname{mean}(s(x_{\mathrm{inlier}})),10^{-12})}.
$$

Class labels never enter autoencoder fitting, PCA fitting, anomaly generation,
scoring, or candidate selection. They are used only after selection to color a
two-dimensional latent visualization and calculate silhouette on the fixed
eight-dimensional test codes. The two-dimensional view itself is a PCA
projection of the latent codes when the bottleneck exceeds two dimensions.

For input width $d$, hidden width $h$, and latent width $z$, one dense
encoder-decoder pass costs $O(dh+hz)$ up to a constant factor of two and stores
$O(dh+hz)$ parameters. Rank-$z$ dense PCA fitting by SVD is approximately
$O(\min(nd^2,n^2d))$. The perfect AUROCs apply only to the declared pixel-
permutation mechanism; no threshold calibration, contamination study, natural
out-of-distribution dataset, or production anomaly prevalence is represented.

## Validation and untouched-test boundary

Every configurable profile value is fixed before the run. Source-defined design
choices—including topology, optimizer family, minimum improvement, corruption
mechanism, and deterministic tie preferences—are documented and versioned with
the implementation. Candidate selection receives a projection containing only
one explicitly named validation metric. The held-out phase starts only after
`_select_candidate` returns; test arrays, retrospective labels, and test
diagnostics are not evaluated before that boundary. Tests enforce both the
selector payload and event ordering.

| Study | Fit inputs | Selection dependency | Reporting-only evidence |
|---|---|---|---|
| Scratch MLP | task training features and labels | mean validation accuracy across XOR and moons | per-task/mean test accuracy and log loss; gradient check uses a frozen training minibatch |
| Vision | stratified digit training partition and labels | validation accuracy | test accuracy, macro-F1, loss, confusion matrix, mistakes, reload deltas |
| Sequence | stratified temporal-XOR training partition and labels | validation accuracy | test accuracy/loss, length buckets, padding-invariance delta |
| Autoencoder | clean digit training features only | validation anomaly AUROC from independently pixel-permuted validation examples | test anomaly AUROC/errors; class-colored latent view and silhouette are retrospective |

The deterministic preference is the deep candidate (`autograd_mlp`,
`compact_cnn`, `sequence_lstm`, or `bottleneck_autoencoder`) only when the
declared validation metric ties exactly. Test evidence never breaks a tie. This
is holdout discipline for a fixed demonstration suite, not nested model
selection or an estimate with repeated-split uncertainty.

## Reproducibility and publication

The root seed is 42. SHA-256-backed named namespaces derive data, split,
candidate, trainer, initialization, minibatch-shuffle, and anomaly streams.
Adding or reordering candidates therefore does not perturb an existing stream.
NumPy code uses explicit `Generator` instances and never mutates the process-
global RNG. PyTorch initialization is seeded before model construction; the
trainer requires deterministic algorithms, seeds its loader generator, uses
zero worker processes, and records both Torch and shuffle RNG state.

The reference profile pins `device: cpu`. `auto` can select CUDA, MPS, or CPU,
but exact equivalence is promised only on the same platform, device, dependency
versions, and deterministic-kernel availability. This sprint is single-process
and single-device. It establishes checkpoint and runtime seams that a later
distributed sprint can adapt; it does not claim distributed throughput,
elasticity, or cross-device bit identity.

Run the complete profile with:

```bash
uv sync --locked --all-groups
make demo-sprint4
```

The output directory must be empty or absent. Four experiment directories are
built beneath a sibling staging root. Each run validates its `RunResult`, writes
resolved configuration and environment metadata, hashes its files, and
publishes atomically. The root workflow then writes `comparison.json`,
`comparison.csv`, `comparison.md`, and `benchmark_manifest.json`; any failure
removes the entire staging tree. A replay test compares every nonvolatile
artifact byte-for-byte and verifies overwrite refusal.

## Evidence inventory

The generated evidence is deliberately inspectable rather than notebook-only:

- Scratch MLP: candidate JSON/CSV, NumPy parameter archive, loss curves, and
  XOR/two-moons probability boundaries.
- Vision: candidate JSON/CSV, one resumable checkpoint per candidate, training
  and validation curves, held-out confusion matrix, misclassified-example
  gallery, and learned convolution filters.
- Sequence: candidate JSON/CSV, one state dictionary per candidate, training and
  validation curves, and held-out accuracy by true-length bucket.
- Autoencoder: candidate JSON/CSV, model state, reconstruction training curves,
  original/reconstruction/corruption panels, score histograms, validation/test
  ROC curves for both candidates, and the retrospective latent scatter.
- Suite root: machine-readable and human-readable comparisons plus a SHA-256
  manifest covering every published file.

The four studies produce at least 15 diagnostic plots: 2 scratch, 5 vision, 3
sequence, and 5 autoencoder. Titles, axes, legends, model context, seed, sample
budget, optimizer settings, and split context are carried by the reporting
layer. Checkpoints and generated `runs/` remain untracked; durable claims are
the source, tests, configuration, and documented observed values.

## Verification strategy

Unit and property tests cover:

- closed-form and finite-difference gradients, broadcasting, fan-out, repeated
  backward, explicit vector seeds, stable log-softmax, domain errors, overflow,
  and exception-safe perturbation restoration;
- MLP fitted-state and shape contracts, invalid hyperparameters, deterministic
  replay, global-RNG isolation, multiclass probabilities, nonlinear learning,
  serialization, full-epoch loss semantics, and failed-refit rollback;
- deterministic immutable generators and exact temporal-XOR/padding invariants;
- trainer configuration, CPU determinism, sample weighting, finite losses,
  gradients and state, early stopping, exact resume, stable checkpoint bytes,
  corrupted/incompatible checkpoint rejection, and task-objective arity;
- CNN, LSTM, padded MLP, and autoencoder shapes, gradients, state restoration,
  input validation, padding invariance, and reconstruction-score semantics; and
- import-boundary tests proving the from-scratch modules do not depend on
  PyTorch, scikit-learn, reporting, configuration, or experiment code.

Integration tests exercise CLI and registry discovery, schema exposure, all
four reference configurations, validation-isolated model selection, acceptance
thresholds, checkpoint reload equivalence, complete artifact declarations,
manifest hashes, byte-stable replay, overwrite refusal, and cleanup after an
injected suite failure. The repository-wide gate additionally runs formatting,
lint, strict mypy, an explicit raw branch-edge coverage floor, lock verification,
and wheel/sdist builds on the supported Python range.

## Security, reliability, and interpretation limits

- The profile uses synthetic data and the bundled digits dataset. It downloads
  nothing and contains no personal, employer, proprietary, or credentialed data.
- Configuration bounds reduce accidental resource exhaustion, but this is not a
  multi-tenant service and does not enforce quotas, authentication, or job
  isolation.
- Tensor, shape, dtype, finiteness, label, length, and fitted-state checks stop
  malformed inputs before plausible-looking outputs are published.
- The scratch MLP restores its prior fit after failure. The PyTorch trainer
  rejects numerical corruption and atomically checkpoints completed epochs, but
  a caller's in-memory model may already contain updates from earlier successful
  minibatches when a later minibatch fails. The transactional suite prevents
  that partial state from becoming published evidence.
- `weights_only=True`, schema checks, signatures, finite tensors, and manifest
  hashes narrow checkpoint risk; they do not authenticate origin or make hostile
  archives safe from every denial-of-service technique.
- Hashes establish integrity relative to the generated manifest, not identity,
  signing, or long-term artifact provenance.
- The benchmark is one seed and one split per study. It reports deterministic
  reproducibility, not sampling uncertainty, robustness across distributions,
  fairness, privacy, calibration, interpretability, or safety certification.
- The NumPy engine is first-order and dense. The PyTorch models are compact,
  single-process references. There is no mixed precision, accelerator benchmark,
  distributed execution, compiler capture, hyperparameter sweep, or production
  serving claim in Sprint 4.
