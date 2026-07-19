# Sprint 3 — From-scratch unsupervised systems

Sprint 3 implements the mathematical core of clustering and representation
learning with typed NumPy code, then evaluates it through a reproducible,
label-isolated experiment boundary. The from-scratch modules do not import
scikit-learn, SciPy, plotting, configuration, artifact, or CLI code. Established
libraries appear only as independent test oracles.

The observed reference metrics and inspected diagnostic gallery are in the
[results report](results.md#sprint-3--from-scratch-unsupervised-systems). The
causal boundary between feature-only selection and retrospective labels is an
accepted architecture decision in
[ADR 0005](adr/0005-label-free-unsupervised-selection.md).

## Work-item traceability

| Work item | Delivered surface |
|---|---|
| [#17 — k-means++ and Lloyd clustering](https://github.com/srgangaram-swe/learning-systems-atlas/issues/17) | D² initialization, deterministic Lloyd updates, empty-cluster repair, isolated restarts, convergence histories, and inductive assignment |
| [#18 — full-covariance Gaussian mixtures](https://github.com/srgangaram-swe/learning-systems-atlas/issues/18) | Stable EM, Cholesky log densities, regularized covariance estimates, restart evidence, AIC, and BIC |
| [#19 — DBSCAN](https://github.com/srgangaram-swe/learning-systems-atlas/issues/19) | Core/border/noise semantics, deterministic density-connected components, k-distance evidence, and explicit transductive behavior |
| [#20 — agglomerative clustering](https://github.com/srgangaram-swe/learning-systems-atlas/issues/20) | Single, complete, average, and Ward linkage; Lance–Williams updates; validated full merge histories; and exact cuts |
| [#21 — principal component analysis](https://github.com/srgangaram-swe/learning-systems-atlas/issues/21) | Thin-SVD PCA, deterministic sign orientation, whitening, inverse transformation, rank, variance, and reconstruction diagnostics |
| [#22 — exact t-SNE](https://github.com/srgangaram-swe/learning-systems-atlas/issues/22) | Per-row perplexity search, symmetric affinities, exact KL objective and gradient, early exaggeration, adaptive optimization, and convergence diagnostics |
| [#23 — clustering metrics](https://github.com/srgangaram-swe/learning-systems-atlas/issues/23) | Silhouette analysis, contingency tables, ARI, arithmetic-mean NMI, partition stability, degeneracy contracts, and allocation guards |
| [#93 — integrated unsupervised evidence](https://github.com/srgangaram-swe/learning-systems-atlas/issues/93) | Heterogeneous benchmark suites, label-free selection, causal-isolation tests, deterministic artifacts, failure diagnostics, and an atomic public workflow |

## Scientific boundary and notation

The public workflow runs two experiments:

- `scratch_clustering_benchmark` compares seven candidates on isotropic blobs,
  two moons, anisotropic blobs, and variable-density blobs. Every feature matrix
  is standardized before fitting. Generator labels are immutable retrospective
  evidence; they are not estimator inputs or selection-score inputs.
- `scratch_representation_benchmark` compares a two-dimensional PCA projection
  with a two-dimensional exact t-SNE embedding of standardized, six-dimensional
  Gaussian data. Neighborhood overlap and distance correlation select the
  representation. Truth is used only to audit a fixed representation with a
  separately seeded k-means fit and to color presentation plots.

For fixed-cluster-count methods, the clustering study treats the generator's
declared cluster count as a predeclared experimental-design parameter. It does
not read the label vector to infer that value, and it does not tune over possible
values of $k$. The resulting comparison is therefore conditional on a known
cluster-count hypothesis; it is not a claim to solve unknown-$k$ discovery.

The notation used below is:

- $X\in\mathbb{R}^{n\times d}$: $n$ observations with $d$ features;
- $K$: number of clusters or mixture components;
- $C_k$: the observations assigned to cluster $k$;
- $R$: independent initialization restarts;
- $T$: optimizer iterations; and
- $q$: retained or embedded dimension.

## Shared estimator and failure contracts

Every estimator accepts a real, finite, rectangular two-dimensional feature
array and rejects malformed shapes, non-finite values, invalid hyperparameters,
and incompatible inference dimensions before returning plausible-looking state.
Learned properties fail with a domain `NotFittedError` before `fit`. Public
arrays are defensive copies or immutable views, so callers cannot mutate the
model accidentally. Pickle/joblib round trips are exercised by tests.

Fitting is publish-after-success. Numerical work is completed in local state;
validation, convergence, overflow, singularity, or resource-limit failures do
not publish a partial fit. K-means and mixture refits additionally preserve the
previous fitted feature contract if a new fit fails. Numerical failures carry
actionable context such as restart, iteration, failing component or row, and a
scaling or configuration remedy.

The API does not force every unsupervised method into a supervised interface:

- k-means and Gaussian mixtures define out-of-sample `predict`;
- PCA defines `transform` and `inverse_transform`; and
- DBSCAN, agglomerative clustering, and t-SNE are deliberately transductive and
  expose no invented `predict` or `transform` method.

## k-means++ and Lloyd optimization

### Objective

For non-empty clusters $C_1,\ldots,C_K$ with centers
$\mu_1,\ldots,\mu_K$, k-means minimizes within-cluster sum of squares:

$$
J(C,\mu)=\sum_{k=1}^{K}\sum_{x_i\in C_k}\lVert x_i-\mu_k\rVert_2^2.
$$

Given centers, the assignment step chooses

$$
z_i=\arg\min_k\lVert x_i-\mu_k\rVert_2^2,
$$

and, given assignments, the update step uses the exact arithmetic mean

$$
\mu_k=\frac{1}{|C_k|}\sum_{x_i\in C_k}x_i.
$$

These alternating minimizers make an ordinary Lloyd step non-increasing in
$J$. The implementation checks that invariant up to a scale-aware floating-
point allowance and raises `KMeansNumericalError` if finite inputs violate it.

### Initialization, ties, and empty clusters

The first k-means++ center is a uniformly sampled observation. If
$D(x_i)^2$ is the squared distance to the closest center already selected, the
next center is sampled with

$$
\Pr(x_i\text{ selected})=
\frac{D(x_i)^2}{\sum_j D(x_j)^2}.
$$

The alternative `random` initializer samples distinct feature rows without
replacement. A request for more clusters than distinct observations fails
before optimization.

Assignment ties resolve to the lowest center index through stable `argmin`
semantics. If an assignment step empties a cluster, empty cluster identifiers are
repaired in ascending order. Each empty cluster receives the observation with
the largest current assigned-center squared distance whose donor cluster has
more than one member; equal distances resolve to the lowest row index. Repair is
part of the declared algorithm, not hidden post-processing.

### Convergence and restarts

A restart succeeds when either assignments are unchanged without a repair, or
the updated solution has no empty cluster and one of these tolerances is met:

$$
\max_k\lVert\mu_k^{(t)}-\mu_k^{(t-1)}\rVert_2\le \texttt{tol},
$$

or

$$
\frac{\max(0,J_{t-1}-J_t)}{\max(1,J_{t-1})}\le \texttt{tol}.
$$

Every requested restart has an independent named random stream. All restarts
must converge; one exhausted restart fails the fit with its inertia history
rather than silently reducing `n_init`. The published solution has the lowest
final inertia, with the lower restart index as the exact tie-break. Winning and
per-restart histories and terminal inertias remain available for audit.

`predict` applies the fitted nearest-center rule, and `transform` returns the
Euclidean distance to every center. For $R$ restarts and $T$ Lloyd steps, the
dominant dense cost is $O(RTnKd)$, with $O(nK+Kd)$ working memory. The method
still optimizes a non-convex objective, assumes a declared $K$, and generally
prefers compact clusters; restarts reduce but do not remove local-optimum risk.

## Full-covariance Gaussian mixtures

### Model and stable E-step

The mixture density is

$$
p(x_i\mid\theta)=\sum_{k=1}^{K}\pi_k
\,\mathcal{N}(x_i\mid\mu_k,\Sigma_k),
\qquad \pi_k>0,\quad\sum_k\pi_k=1.
$$

The E-step computes responsibilities

$$
\gamma_{ik}=
\frac{\pi_k\mathcal{N}(x_i\mid\mu_k,\Sigma_k)}
{\sum_{j=1}^{K}\pi_j\mathcal{N}(x_i\mid\mu_j,\Sigma_j)}.
$$

Component densities are evaluated in log space. For a Cholesky factor
$L_kL_k^\top=\Sigma_k$, the implementation obtains the Mahalanobis term with a
triangular solve and computes

$$
\log\mathcal{N}(x\mid\mu_k,\Sigma_k)=
-\frac{1}{2}\left[
d\log(2\pi)+2\sum_j\log (L_k)_{jj}
+\lVert L_k^{-1}(x-\mu_k)\rVert_2^2
\right].
$$

The marginal normalizer uses max-shifted log-sum-exp. No covariance inverse and
no product of small raw densities is formed.

### M-step, regularization, and convergence

With $N_k=\sum_i\gamma_{ik}$, the M-step is

$$
\pi_k=\frac{N_k}{n},\qquad
\mu_k=\frac{1}{N_k}\sum_i\gamma_{ik}x_i,
$$

$$
\Sigma_k=
\frac{1}{N_k}\sum_i\gamma_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top
+\lambda I,
$$

where λ is `reg_covar`. The empirical covariance is symmetrized before the
diagonal regularizer is added. A component with numerically negligible mass is
reported as collapsed; it is not assigned a fabricated weight or covariance.

The recorded objective is the observed-data log likelihood

$$
\ell(\theta;X)=\sum_{i=1}^{n}\log\sum_{k=1}^{K}
\pi_k\mathcal{N}(x_i\mid\mu_k,\Sigma_k).
$$

EM stops when the non-negative improvement per observation is at most `tol`.
A decrease beyond the declared scale-aware numerical allowance raises
`GaussianMixtureNumericalError`. History includes the initial E-step, so a fit
with $T$ M/E updates publishes $T+1$ likelihood values. The public
`lower_bound_` is the final mean observed-data log likelihood, ℓ/n.

Initialization is either a one-restart k-means++ partition or seeded random
distinct means with a shared empirical covariance. Restart streams are named
independently. Every requested restart must succeed, and the greatest final
mean log likelihood wins; exact ties favor the lower restart index. Component
numbers themselves have no semantic identity: tests and evaluation compare
partitions or align components rather than treating label switching as error.

### Information criteria

For $K$ full-covariance components in $d$ dimensions, the number of free
parameters is

$$
p=(K-1)+Kd+K\frac{d(d+1)}{2}.
$$

The fitted model reports, for the feature matrix supplied to each method,

$$
\operatorname{AIC}=2p-2\ell,
\qquad
\operatorname{BIC}=p\log n-2\ell.
$$

Lower values are preferred when these criteria are used, but the Sprint 3
cross-family selector does not use AIC or BIC; they remain model-specific
diagnostics. A dense EM update costs
$O(nKd^2+Kd^3)$, dominated by weighted full-covariance updates, Cholesky
factorization, and solves. Across restarts and iterations, the bound is
$O(RT(nKd^2+Kd^3))$, with $O(nK+Kd^2)$ working state. EM is non-convex,
Gaussian components are a modeling assumption, and regularization mitigates
rather than eliminates ill-conditioning.

## DBSCAN density connectivity

For the closed Euclidean neighborhood

$$
N_\varepsilon(x_i)=
\{x_j:\lVert x_i-x_j\rVert_2\le\varepsilon\},
$$

the observation itself is included. A point is **core** exactly when

$$
|N_\varepsilon(x_i)|\ge \texttt{min_samples}.
$$

A point is directly density-reachable from a core point when it lies in that
core point's closed neighborhood. Chaining this relation through core points
defines density reachability, and the connected components of the undirected
core-neighbor graph define the implementation's clusters. A non-core point
adjacent to at least one core component is **border**; an observation that is
neither core nor border is **noise** and receives label `-1`.

Core components are discovered and labeled in a deterministic geometry order.
When a border point touches more than one component, the closest adjacent core
point wins. An exact distance tie resolves by the component's lexicographically
smallest sorted feature-row representation, then by its deterministic component
identifier. The `eps` boundary is inclusive, and tests cover exact-boundary
points, duplicate rows, all-noise data, one-cluster data, and multi-component
border ties.

The fitted diagnostics include immutable labels, core indices and mask,
core/border/noise roles, closed-neighborhood counts, core feature rows, and the
`min_samples`-th neighbor distance including self (capped at the available
sample count). The implementation materializes the dense $n\times n$ distance
and neighborhood matrices. Its dominant cost is $O(n^2d)$ time and
$O(n^2)$ memory, with an explicit `max_pairwise_elements` guard before the
allocation. It intentionally provides no out-of-sample `predict`: assigning a
new point can change density connectivity and would require a separately
specified inductive approximation.

DBSCAN avoids a declared number of clusters and can recover non-convex
structure, but one global ε and density threshold can fail on variable-density
data. The benchmark therefore publishes ε sensitivity, k-distance, assigned
coverage, and noise diagnostics rather than presenting one fit as universally
stable.

## Agglomerative hierarchy and Lance–Williams updates

Starting from singleton clusters, the implementation repeatedly merges the
active pair with minimum dissimilarity. For clusters $A$ and $B$, the
supported Euclidean linkages are

$$
d_{\text{single}}(A,B)=
\min_{a\in A,b\in B}\lVert a-b\rVert_2,
$$

$$
d_{\text{complete}}(A,B)=
\max_{a\in A,b\in B}\lVert a-b\rVert_2,
$$

$$
d_{\text{average}}(A,B)=
\frac{1}{|A||B|}\sum_{a\in A}\sum_{b\in B}\lVert a-b\rVert_2,
$$

and Ward dissimilarity

$$
d_{\text{Ward}}(A,B)=
\sqrt{\frac{2|A||B|}{|A|+|B|}}\,
\lVert\bar{x}_A-\bar{x}_B\rVert_2,
$$

whose square is twice the increase in within-cluster sum of squares caused by
the merge.

Distances to a newly merged cluster $U=A\cup B$ are cached through the exact
Lance–Williams specializations:

$$
d_{\text{single}}(U,C)=\min\{d(A,C),d(B,C)\},
$$

$$
d_{\text{complete}}(U,C)=\max\{d(A,C),d(B,C)\},
$$

$$
d_{\text{average}}(U,C)=
\frac{|A|d(A,C)+|B|d(B,C)}{|A|+|B|},
$$

$$
d_{\text{Ward}}(U,C)=
\sqrt{
\frac{
(|C|+|A|)d(A,C)^2+(|C|+|B|)d(B,C)^2-|C|d(A,B)^2
}{|A|+|B|+|C|}
}.
$$

Roundoff-sized negative Ward radicands are clipped to zero; a materially
negative or non-finite update raises instead of corrupting the tree. Merge ties
resolve by the ordered active cluster identifiers after distance comparison.
Initial identifiers are row indices and merged identifiers are chronological,
so the rule is deterministic for a fixed input order. Untied partitions are
row-permutation invariant; no stronger invariance is claimed for exactly tied
geometries.

The complete history has SciPy-compatible rows
`[left_id, right_id, distance, merged_size]`. Validation requires shape
$(n-1,4)$, finite non-negative and nondecreasing merge distances, integer
identifiers that reference two distinct active clusters, the exact merged size,
and one final hierarchy. Cutting to $K$ clusters applies exactly $n-K$
chronological merges and labels the remaining clusters by their smallest
original row index. Thus every valid cut has exactly the requested cluster
count, including the canonical one-cluster and all-singleton cuts.

The implementation stores $O(n^2)$ pair distances. Repeated minimum scans and
pair-cache maintenance make the clarity-first fit $O(n^3)$ time after the
$O(n^2d)$ initial distance calculation. A cut is $O(n)$ in the size of the
stored tree. Merges are irreversible, the metric is Euclidean only, and the
estimator is transductive; no out-of-sample assignment is implied.

## PCA by thin singular-value decomposition

Let

$$
X_c=X-\mathbf{1}\bar{x}^{\top}
$$

and compute the thin SVD

$$
X_c=U\operatorname{diag}(s_1,\ldots,s_r)V^\top,
\qquad s_1\ge\cdots\ge s_r\ge0,
$$

where $r=\min(n,d)$ is the thin-SVD dimension, not necessarily the numerical
rank. The first $q\le r$ rows of $V^\top$ are the principal axes. The
implementation forms no covariance matrix.

The sample explained variance and total-variance ratio are

$$
\lambda_j=\frac{s_j^2}{n-1},
\qquad
\rho_j=\frac{s_j^2}{\sum_{\ell=1}^{r}s_\ell^2}.
$$

Ratios are evaluated from singular values scaled by $s_1$, which avoids an
unnecessary overflow. Constant data receives explicit zero variances and zero
ratios. Numerical rank counts singular values greater than

$$
\epsilon_{64}\max(n,d)s_1.
$$

SVD vectors have an arbitrary sign. For each component, the implementation
finds its largest-magnitude loading (the first column wins an exact tie) and
orients that loading non-negative. This makes ordinary replays deterministic;
it does not remove the freedom to rotate a basis within an exactly repeated
singular-value subspace.

Without whitening, projection and reconstruction are

$$
Z=X_cV_q,
\qquad
\widehat{X}=ZV_q^\top+\mathbf{1}\bar{x}^{\top}.
$$

The squared Frobenius reconstruction error equals the energy of omitted
singular directions in exact arithmetic and therefore cannot increase as $q$
grows. With whitening,

$$
Z_{\text{white},j}=\frac{Z_j}{\sqrt{\lambda_j}},
$$

so the retained coordinates have identity sample covariance. Inverse
transformation restores the standard deviations before applying the component
basis. Whitening is rejected if any requested component is numerically rank
deficient or its variance underflows to zero; projection without whitening
remains well-defined for rank-deficient data.

Thin dense SVD has the usual $O(nd\min(n,d))$ arithmetic order, subject to the
linked LAPACK implementation, and stores the dense input plus thin factors. PCA
is an inductive linear transform, but maximizing variance need not preserve
nonlinear neighborhoods or task-relevant structure.

## Exact t-SNE

### High-dimensional affinities and perplexity

For each row $i$, excluding self, t-SNE defines

$$
p_{j\mid i}(\beta_i)=
\frac{\exp(-\beta_i\lVert x_i-x_j\rVert_2^2)}
{\sum_{k\ne i}\exp(-\beta_i\lVert x_i-x_k\rVert_2^2)},
\qquad p_{i\mid i}=0.
$$

Max-shifting the logits stabilizes each row. The precision βᵢ is found by a
bounded binary search so that

$$
\operatorname{Perp}(P_i)=
\exp\!\left(-\sum_{j\ne i}p_{j\mid i}\log p_{j\mid i}\right)
$$

matches the configured perplexity in entropy space. The implementation records
every achieved perplexity and precision. Perplexity must lie in
$[1,n-1]$. A row of equal distances supports only the maximum perplexity; an
unattainable target or exhausted search raises `PerplexitySearchError` with the
failing row and closest result.

The symmetric high-dimensional probabilities are

$$
p_{ij}=\frac{p_{j\mid i}+p_{i\mid j}}{2n},
\qquad p_{ii}=0,
$$

followed by a defensive normalization to unit mass. The matrix must be finite,
non-negative, symmetric, zero-diagonal, and normalized before objective or
gradient helpers accept it.

### Low-dimensional objective and exact gradient

For embedded coordinates $y_i\in\mathbb{R}^q$, the Student-t affinities are

$$
q_{ij}=
\frac{(1+\lVert y_i-y_j\rVert_2^2)^{-1}}
{\sum_{a\ne b}(1+\lVert y_a-y_b\rVert_2^2)^{-1}},
\qquad q_{ii}=0.
$$

The optimized divergence is

$$
\operatorname{KL}(P\Vert Q)=
\sum_{i\ne j}p_{ij}\log\frac{p_{ij}}{q_{ij}},
$$

and the exact gradient, with early-exaggeration factor α, is

$$
\frac{\partial C}{\partial y_i}=
4\sum_{j\ne i}
(\alpha p_{ij}-q_{ij})
(1+\lVert y_i-y_j\rVert_2^2)^{-1}
(y_i-y_j).
$$

The implementation evaluates all pairs: there is no Barnes–Hut, nearest-neighbor,
FFT, or library approximation behind the result. A central finite-difference
test checks the analytical gradient.

### Initialization, optimizer, and convergence meaning

PCA initialization projects to $q$ dimensions and rescales the embedding so
the first coordinate has sample standard deviation $10^{-4}$ when nonzero.
It is deterministic and does not consume the configured seed. Random
initialization draws independent $N(0,10^{-8})$ coordinate values from the
explicit seed.

Each update uses momentum and per-coordinate gains. A gain increases by `0.2`
when the current gradient sign differs from the previous velocity sign,
otherwise it is multiplied by `0.8`, with a positive floor. The learning-rate
step is added to momentum, and the embedding is recentered to zero mean after
every update. The attraction term is multiplied by the configured early-
exaggeration factor during the declared prefix of iterations; reported KL
values remain the unexaggerated objective after each update.

`converged_` has a deliberately narrow meaning:

$$
\operatorname{KL}_{\text{final}}
<\operatorname{KL}_{\text{post-exaggeration start}}-\texttt{kl_tolerance}.
$$

It does **not** mean that KL decreased at every update or that the non-convex
global optimum was found. After exaggeration, optimization can stop at the
gradient-norm threshold, after the declared no-progress window, or at
`max_iter`; `stop_reason_`, gradient norms, the full KL history, and both
initial and post-exaggeration baselines make that distinction visible. Strict
mode raises `TSNEConvergenceError` if the final contract is not met; non-strict
mode publishes `converged_=False` rather than disguising the outcome.

Affinity construction costs $O(n^2d+n^2B)$, where $B$ is the maximum
perplexity-search iteration count. Each exact optimizer step costs
$O(n^2q)$, and dense affinities require $O(n^2)$ memory. The estimator is
therefore a CPU-capable mathematical reference for small data, not a scalable
t-SNE implementation. It is transductive, seed and hyperparameter sensitive,
and preserves local probability structure rather than interpretable global
distances or cluster sizes.

## Metrics, stability, and selection

### Silhouette with an explicit degeneracy contract

For an assigned, non-singleton observation $i$, let $a_i$ be the mean
distance to other members of its own cluster and $b_i$ the smallest mean
distance to any other cluster. Its silhouette is

$$
s_i=\frac{b_i-a_i}{\max(a_i,b_i)}.
$$

Singleton clusters contribute zero. If both distances are zero, the score is
also zero. Noise label `-1` is excluded by default, and

$$
\text{coverage}=\frac{\text{retained non-noise observations}}{n}
$$

is reported separately. Fewer than two retained clusters, or no more retained
samples than clusters, makes silhouette mathematically undefined. Because the
shared result schema forbids NaN, the implementation returns the finite sentinel
`-1.0` with `defined=False`, retained sample/cluster counts, and coverage. A
dense pairwise-distance budget guards its $O(n^2d)$-time,
$O(n^2)$-memory implementation.

### Adjusted Rand index

For contingency counts $n_{ij}$, row sums $a_i$, column sums $b_j$, and
$M={n\choose 2}$, define

$$
A=\sum_{ij}{n_{ij}\choose 2},\quad
B=\sum_i{a_i\choose 2},\quad
C=\sum_j{b_j\choose 2}.
$$

Then

$$
\operatorname{ARI}=
\frac{A-BC/M}{\tfrac12(B+C)-BC/M}.
$$

The implementation evaluates an algebraically equivalent expression with
Python integer counts, avoiding overflow in large contingency totals, and clips
roundoff to $[-1,1]$. It is invariant to row ordering and label renaming. The
declared degenerate convention returns `1.0` for fewer than two retained rows or
when the adjustment denominator is zero. Noise may be included as an ordinary
cluster or jointly excluded from both partitions; excluding every row is an
error.

### Arithmetic-mean normalized mutual information

With empirical joint probabilities $p_{ij}$, marginals $p_i,p_j$, and
natural logarithms,

$$
I(U;V)=\sum_{ij:p_{ij}>0}p_{ij}
\log\frac{p_{ij}}{p_ip_j},
$$

$$
\operatorname{NMI}_{\mathrm{arith}}(U,V)=
\frac{I(U;V)}{\tfrac12(H(U)+H(V))}.
$$

Changing the log base cancels in the ratio. Exact partition equivalence up to
label names returns `1.0`, as does the declared zero-entropy denominator case;
the general result is clipped to $[0,1]$. This is the arithmetic-mean
normalization, matching the independent differential-test oracle.

### Stability definitions

The metric library can form a symmetric stability matrix

$$
S_{ab}=\operatorname{ARI}(z^{(a)},z^{(b)})
$$

for repeated partitions and can average its strict upper triangle. The benchmark
uses a related but more diagnostic baseline-perturbation definition. For each
dataset and named trial, it adds independent isotropic Gaussian noise of the
configured scale to the standardized features, refits with the candidate's same
fit seed, and records

$$
r_t=\operatorname{ARI}(z_{\text{original}},z_{\text{perturbed},t}).
$$

The per-dataset stability input is the arithmetic mean of these $r_t$ values;
the aggregate report also exposes their dispersion. This measures robustness
to one declared perturbation distribution. It is not evidence of population
identifiability, and reusing the fit seed deliberately isolates feature
perturbation from initialization variation.

### Label-free clustering selection

For one candidate on one dataset, let $s$ be silhouette, $c$ assigned
coverage, and $r$ mean perturbation stability. The predeclared score is

$$
g=\max(s,0)c,
\qquad
\widetilde r=\frac{r+1}{2},
$$

$$
\operatorname{score}_{\text{cluster}}=
0.65g+0.35\widetilde r.
$$

The candidate score is the arithmetic mean across the four structures. The
highest finite score wins; an exact tie resolves alphabetically by candidate
name. Multiplying silhouette by coverage prevents a density method from
improving its geometry score merely by abstaining on difficult rows, while the
separate coverage output preserves the meaning of that abstention.

### Label-free representation selection

For each observation, the neighborhood score computes the overlap between its
$k$ nearest neighbors in $X$ and in $Z$, divides by $k$, and averages over
rows. The reference profile uses $k=10$:

$$
N_k(X,Z)=\frac1n\sum_{i=1}^{n}
\frac{|\mathcal N_k^X(i)\cap\mathcal N_k^Z(i)|}{k}.
$$

Distance correlation is the Pearson correlation between the strict upper
triangles of the Euclidean distance matrices of $X$ and $Z$. A constant
distance vector receives the declared finite value zero. The representation
score is

$$
\operatorname{score}_{\text{repr}}=
0.8N_{10}(X,Z)+0.2\max(\operatorname{corr}(D_X,D_Z),0).
$$

Again, the higher score wins with an alphabetical exact-tie rule. Neither term
reads class or cluster labels.

### Retrospective boundary

ARI and NMI compare final partitions with generator truth, and a separately
seeded k-means fit audits structure in each fixed representation. These values
and truth-colored plots are reporting outputs only: no selection function reads
them. Source fingerprints include labels so the complete evidence source is
auditable, while `target_used_for_fit=False` records their causal role.

The isolation contract is tested adversarially. Relabeling truth must change at
least some retrospective evidence while leaving standardized features, named
seeds, fitted model bytes, assignments, internal metrics, stability evidence,
and the selected candidate unchanged. Reversing candidate registry order must
also leave those quantities unchanged.

## Seed namespaces and replay

`derive_named_seed` hashes non-empty namespace components with SHA-256, combines
the digest words with the unsigned 32-bit root seed through NumPy
`SeedSequence`, and returns a deterministic unsigned 32-bit child seed. This is
order-independent: inserting or reordering candidates does not shift a shared
RNG cursor.

| Random operation | Namespace beneath the root seed |
|---|---|
| Clustering dataset | `unsupervised / data / <dataset>` |
| Clustering candidate fit | `<experiment> / candidate / <candidate> / <dataset> / fit` |
| k-means restart | `kmeans / <init> / restart / <index>` beneath the candidate seed |
| GMM restart | `gaussian-mixture / <init> / restart / <index>` beneath the candidate seed |
| Stability perturbation | `<experiment> / stability / <dataset> / trial-<index>` |
| Random-partition metric diagnostic | `<experiment> / metric-diagnostic / trial-<index>` |
| Representation data | `<experiment> / data` |
| PCA/t-SNE candidate record | `<experiment> / candidate / pca|tsne` |
| Retrospective representation k-means | `<experiment> / retrospective / <candidate>` |
| PCA-init t-SNE perplexity study | `<experiment> / sensitivity / pca-init-perplexity-study` (one shared seed) |
| Random-init t-SNE seed study | `<experiment> / sensitivity / random-init / trial-<index>` |
| Retrospective perplexity-study k-means | `<experiment> / retrospective / sensitivity / perplexity-<value>` |

PCA and PCA-initialized t-SNE are deterministic and do not consume their
recorded candidate seed. Sensitivity evidence therefore separates two causal
questions: three perplexities use PCA initialization and one shared named seed,
while three fixed-perplexity trials use random initialization and distinct named
seeds. Each row records initialization, study, seed, perplexity, convergence,
final KL, label-free neighborhood preservation, and retrospective ARI. Random
t-SNE initialization is also unit-tested for same-seed replay and changed-seed
behavior.

No module mutates NumPy's process-global RNG. Exact parameters and integer seeds
are serialized with each candidate record. The integration profile reruns the
workflow and compares all deterministic artifacts byte for byte; timestamped,
duration-bearing manifests are excluded from that byte-equality assertion.

## Artifact and publication contract

The CLI command is:

```bash
uv run learning-atlas benchmark-unsupervised \
  --config-dir configs/unsupervised/sprint-03 \
  --output-dir runs/sprint-03
```

Each experiment publishes:

- the resolved configuration, typed result, environment/source provenance, and
  a manifest of SHA-256 artifact digests;
- JSON and CSV candidate records with exact parameters, seeds, internal scores,
  and explicitly named retrospective metrics;
- convergence histories, restart outcomes, perturbation trials, affinity and
  perplexity diagnostics, and sensitivity records;
- compressed NumPy state for assignments, centers, responsibilities,
  covariances, hierarchy cuts, PCA factors, affinities, embeddings, and
  optimizer histories;
- serialized selected clustering models or representation models; and
- diagnostic plots for assignments, internal geometry, coverage, stability,
  restart objectives, EM/Lloyd/t-SNE optimization, GMM covariance density,
  DBSCAN k-distance and ε sensitivity, hierarchy and dendrogram structure, PCA
  variance/reconstruction, affinities, representation quality, and retrospective
  recovery.

The suite root adds aggregate JSON, CSV, and Markdown comparison reports and a
second checksum manifest covering the complete publication. Files are written
through run-root-confined atomic targets, and both child experiments are staged
before one atomic suite promotion. A non-empty destination is never overwritten.
Any exception removes the staging tree, so a failed candidate cannot leave a
directory that looks like a successful benchmark.

## Verification strategy

The verification suite exercises behavior at four levels:

1. **Mathematical fixtures and invariants.** Hand-derived distances, linkage
   merges, likelihoods, information criteria, contingency counts, gradients,
   covariance identities, monotone reconstruction error, normalized
   probabilities, and convergence histories are asserted directly.
2. **Metamorphic and differential tests.** Row and label permutations,
   translations, orthogonal rotations, bounded property-generated PCA cases,
   and independent NumPy/scikit-learn comparisons test equivalence without
   making the production implementation depend on those libraries.
3. **Adversarial contracts.** Tests cover duplicate and constant rows,
   singular/rank-deficient matrices, empty clusters, collapsed mixture
   components, all-noise partitions, exact ties, malformed linkage matrices,
   unattainable perplexity, allocation ceilings, non-finite intermediates,
   inference before fit, serialization, and failed-refit state preservation.
4. **End-to-end systems behavior.** Integration tests invoke the public CLI,
   verify both registry/config paths, candidate inventories, plots and model
   state, validate every manifest digest, replay stable artifacts, refuse an
   overwrite, and prove transaction cleanup after an injected suite failure.

The repository-level commands are:

```bash
make check

uv run pytest tests/unit/unsupervised
uv run pytest tests/integration/test_sprint03_benchmark.py

uv run learning-atlas benchmark-unsupervised \
  --config-dir configs/unsupervised/sprint-03 \
  --output-dir runs/sprint-03
```

## Complexity and interpretation limits

| Method | Dominant fit cost | Dominant memory | Deliberate limitation |
|---|---:|---:|---|
| k-means | $O(RTnKd)$ | $O(nK+Kd)$ | local optimum; declared $K$; compact-cluster bias |
| full-covariance GMM | $O(RT(nKd^2+Kd^3))$ | $O(nK+Kd^2)$ | local optimum; Gaussian assumption; covariance conditioning |
| DBSCAN | $O(n^2d)$ | $O(n^2)$ | one global density scale; transductive dense reference |
| agglomerative | $O(n^3+n^2d)$ | $O(n^2)$ | irreversible merges; Euclidean only; transductive dense reference |
| thin-SVD PCA | $O(nd\min(n,d))$ | dense input and thin factors | linear variance objective; repeated-subspace basis ambiguity |
| exact t-SNE | $O(n^2d+n^2B+Tn^2q)$ | $O(n^2+nq)$ | non-convex, transductive, local-geometry visualization |
| silhouette | $O(n^2d)$ | $O(n^2)$ | favors separation/compactness; explicit degenerate sentinel |
| ARI/NMI | $O(n)$ expected counting work | contingency counts | retrospective only; comparison depends on supplied partitions |

Here $B$ is the maximum per-row perplexity-search iteration count. These are
dense clarity-first bounds for the actual code, not idealized bounds for spatial
indexes, sparse graphs, distributed solvers, or approximate-neighbor methods.

The committed studies are deterministic synthetic CPU references. They show
that the mathematics, causal evaluation boundary, diagnostics, and publication
system are inspectable; they do not claim production-scale throughput,
population cluster identifiability, real-world external validity, or
state-of-the-art embedding quality. Internal and retrospective metrics can
legitimately disagree because they answer different questions. That disagreement
is preserved as evidence rather than tuned away.
