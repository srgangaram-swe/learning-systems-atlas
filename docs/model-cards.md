# Model-family cards and complexity

These cards describe research implementations, not deployed predictive services.
All families reject unsupported shapes, non-finite observations and invalid fitted
state at their documented boundaries. Numerical and convergence errors remain
explicit. Use the [API reference](api/index.md) for exact constructor, fit, inference,
checkpoint and artifact contracts. The cards apply to all classes in each named
family, including the established-library reference candidates.

Notation: n training rows, d features, q queries, K classes/clusters/arms, D tree
depth, T trees/stages, I optimizer iterations, P parameters, B batch size, E epochs,
S states and A actions. Bounds describe the implemented dense CPU mechanisms;
they exclude input/output storage where stated and are not measured speed claims.

| Family | Training/work upper bound | Inference/query bound | Principal fitted/working memory |
| --- | --- | --- | --- |
| Mean/prior baselines | O(nd) including validation | O(q) labels; O(qK) probabilities | O(K) |
| OLS; Ridge with alpha=0 | O(nd min(n,d)) dense least squares | O(qd) | O(nd+d²) conservative dense workspace |
| Ridge with alpha>0 | O(nd²+d³), Gram construction and dense solve | O(qd) | O(nd+d²) |
| Gradient linear/logistic models | O(Ind) | O(qd) | O(nd+d) |
| Lasso coordinate descent | O(Ind) | O(qd) | O(nd+d) |
| CART regression/classification | O(Ddn log n), conservative repeated-sort bound | O(qD) | O(nd+nK), including classification scores |
| Random forests | T times corresponding CART work | O(qTD) | O(TnK+nd) |
| Gradient boosting | O(TDdn log n) | O(qTD) | O(Tn+nd) |
| k-nearest neighbors | O(nd) validation/copy | O(q(nd+n log n)), full deterministic neighbor ordering | O(nd+nK) |
| Kernel SVM/SMO | O(n²d+In³), conservative bounded-sweep upper bound | O(qsd), s support vectors for simple kernels | O(n²+nd) |
| Gaussian/Multinomial NB | O(nd+Kd) | O(qKd) | O(Kd) plus input |
| k-means++/Lloyd | O(RInKd), R restarts | O(qKd) | O(nK+Kd+nd) |
| Full-covariance Gaussian mixture | O(RI(nKd²+Kd³)) | O(qKd²+Kd³) | O(nK+Kd²+nd) |
| Dense DBSCAN | O(n²d+n²) | No out-of-sample prediction contract | O(n²+nd), explicitly bounded |
| Agglomerative clustering | O(n²d+n³), conservative merge bound | No out-of-sample prediction contract | O(n²+nd) |
| PCA/full SVD | O(nd min(n,d)) | O(qdk), k retained components | O(nd+d²) |
| Exact t-SNE | O(n²d+In²k), plus bounded perplexity search | No out-of-sample transform | O(n²+nk), explicitly bounded |
| Autodiff/MLP/CNN/LSTM/autoencoder | O(EnM), M forward/backward arithmetic per example | O(qM) forward arithmetic | O(P+B·activation size), architecture-dependent |
| Stationary bandits | O(steps·K), including policy choice | O(K) per decision | O(K) |
| Dense discounted MDP planning | VI: O(IS²A); PI: O(I(S³+S²A)) | O(A) policy decision | O(S²A) |
| Q-learning/SARSA | O(steps·A), including action selection | O(A) per decision | O(SA) |
| REINFORCE/PPO | O(environment steps + gradient updates·BM) | O(M) per action | O(P+rollout storage) |
| DQN/replay/target network | O(environment steps·M + updates·(BM+R)), conservative replay capacity R bound | O(M) per action | O(P+R·transition size) |

## Mean and prior baselines

Estimate the training mean or empirical class prior. They establish whether a
complex model learns useful structure. They assume nothing about feature effects,
cannot adapt to conditional or temporal shifts, and must be fit on training data
only. Evidence: Sprint 1/2 held-out comparisons. Reference:
[scikit-learn dummy estimators](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators).

## Linear least squares and Ridge

Minimize squared residuals, with Ridge adding alpha times squared coefficient
norm. OLS and zero-penalty Ridge use dense least squares. Positive-penalty Ridge
forms the d-by-d regularized Gram matrix and solves its linear system; it does not
explicitly invert that matrix. This costs O(nd²+d³), including when d exceeds n,
and forming normal equations can worsen conditioning. The intercept is treated
separately. Collinearity affects identifiability; Ridge
trades bias for stability and depends on scale. Gradient descent has an explicit
convergence budget. Evidence: Sprint 2 and fixed-seed Sprint 6 oracle reports.
Reference: [linear models](https://scikit-learn.org/stable/modules/linear_model.html).

## Lasso

Coordinate descent applies soft thresholding for the L1 penalty. Sparsity is a
modeling choice, not evidence of causal importance. Correlated features can make
selected coefficients unstable; scaling and alpha matter. Non-convergence is
reported. Evidence: regularization paths and sklearn oracle comparisons.
Reference: [Lasso objective](https://scikit-learn.org/stable/modules/linear_model.html#lasso).

## Logistic regression

Optimize penalized binary cross-entropy with stable sigmoid/log-probability
arithmetic. The implementation's penalty convention must be reconciled before
comparing with a reference C parameter. Separation, imbalance and distribution
shift affect calibration; accuracy alone is insufficient. Evidence: Sprint 2
training/CV/held-out loss, ROC and convergence diagnostics. The breast-cancer
dataset is a software fixture, never a clinical validation.
Reference: [logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression).

## CART trees

Greedy impurity reduction over sorted thresholds yields piecewise constant
regression or class probabilities. Atlas chooses deterministic feature/threshold
ties. Different equally optimal training partitions can predict differently on
new rows; Sprint 6 retains two such oracle disagreements. Deep trees overfit and
are unstable under small perturbations. Evidence: boundary/differential tests and
held-out diagnostics. Reference: [decision trees](https://scikit-learn.org/stable/modules/tree.html).

## Random forests

Aggregate independently seeded trees over bootstrap/subfeature samples.
Out-of-bag evidence records coverage rather than assuming every row receives a
vote. Correlated trees, small samples and distribution shift limit the variance
reduction argument. The finite ensemble is not an uncertainty guarantee.
Evidence: Sprint 2 OOB and held-out ensemble plots. Reference:
[forest methods](https://scikit-learn.org/stable/modules/ensemble.html#forest).

## Gradient boosting

Sequential shallow-tree corrections optimize regression residuals or binary
classification loss. Learning rate, stage count and depth interact; early stopping
uses training-only validation. Noise and long stage sequences can overfit.
Evidence: learning curves and held-out comparisons in Sprint 2. Reference:
[gradient boosting](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting).

## k-nearest neighbors

Store training observations and order distances deterministically, including ties.
Uniform/distance weighting and exact-match handling are explicit. Scaling is
learned inside training folds. High dimension weakens local-distance meaning;
prediction cost and training-data retention can be large. Evidence: Sprint 2
boundaries and Sprint 6 exact oracle fixtures. Reference:
[nearest neighbors](https://scikit-learn.org/stable/modules/neighbors.html).

## Kernel support-vector classification

SMO solves the constrained dual using a dense kernel cache; inference sums
support-vector contributions. Kernel and scaling choices define geometry.
Inspect KKT residuals/convergence rather than treating a finite iteration stop as
an optimum. Gram storage is quadratic, so this is not a large-scale solver.
Evidence: adversarial/differential tests and Sprint 2 diagnostics. Reference:
[support vector machines](https://scikit-learn.org/stable/modules/svm.html).

## Gaussian and multinomial Naive Bayes

Combine class priors with conditionally independent feature likelihoods in log
space. Gaussian variance smoothing and multinomial pseudocounts prevent common
degeneracies. Multinomial inputs must be nonnegative; the independence assumption
often makes probabilities overconfident. Evidence: Sprint 2 and Sprint 6 oracle
reports. Reference: [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html).

## k-means

k-means++ initialization and Lloyd updates optimize within-cluster squared
Euclidean distance. Restarts are independently seeded; empty clusters and
non-convergence are explicit. Convex, similarly scaled clusters are favored;
non-convex recovery can disagree with silhouette. Evidence: Sprint 3 objective,
stability and label-isolated retrospective diagnostics. Reference:
[k-means](https://scikit-learn.org/stable/modules/clustering.html#k-means).

## Gaussian mixtures

Full-covariance EM alternates responsibilities and weighted sufficient-statistic
updates, using stable log densities and covariance regularization. Local optima,
singular components and small effective sample sizes remain risks. Likelihood
increase does not prove a globally correct clustering. Evidence: Sprint 3
likelihood paths, information criteria and restart stability. Reference:
[mixture models](https://scikit-learn.org/stable/modules/mixture.html).

## DBSCAN

Density connectivity expands core neighborhoods; noise remains unassigned.
Neighborhood radius and minimum samples determine which structures can be found.
Varying density and high-dimensional distances can fail; label metrics always
report assigned coverage. Dense pairwise storage has a hard resource ceiling.
Evidence: Sprint 3 coverage/parameter sensitivity. Reference:
[DBSCAN](https://scikit-learn.org/stable/modules/clustering.html#dbscan).

## Agglomerative clustering

Single, complete, average and Ward linkage merge clusters according to explicit
distance updates and deterministic ties. Ward assumes Euclidean variance
geometry; linkage choice changes sensitivity to chains/outliers. Pairwise state
is bounded. Evidence: independent reference, tie and dendrogram tests plus Sprint
3 diagnostics. Reference: [hierarchical clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering).

## Principal components

Centered dense SVD identifies orthogonal directions of maximum variance; optional
whitening changes scale. Signs and bases inside repeated singular subspaces are
not identifiable, so tests compare reconstruction/subspaces. Low variance need
not mean irrelevant signal. Evidence: Sprint 3 reconstruction and component
diagnostics. Reference: [PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca).

## Exact t-SNE

Bounded perplexity search forms high-dimensional affinities; gradient descent
minimizes KL divergence to a low-dimensional heavy-tailed affinity model.
Coordinates, global distances and cluster areas are not calibrated measurements.
Initialization/perplexity sensitivity and non-convergence remain visible. The
quadratic exact method is deliberately small-scale. Evidence: Sprint 3 KL and
neighborhood preservation. Reference: [t-SNE](https://scikit-learn.org/stable/modules/manifold.html#t-sne).

## Reverse-mode autodiff and NumPy MLP

A topologically ordered computation graph applies chain-rule vector-Jacobian
products; the MLP trains with explicit optimizer/seed state. Broadcasting,
accumulation and graph lifetime are tested. Saturation, poor scaling and unstable
steps remain failure modes. MLP operation count M is the sum of layer matrix
products, not merely parameter count. Evidence: Sprint 4 gradient checks and
PyTorch comparison. Reference: [automatic differentiation](https://pytorch.org/docs/stable/notes/autograd.html).

## CNN and packed recurrent networks

Convolution shares local filters; the packed LSTM processes only valid sequence
lengths. CPU training has deterministic seeds, finite losses and transactional
checkpoint recovery. Spatial/temporal assumptions, padding errors and class
imbalance can invalidate results. Evidence: Sprint 4 confusion, error examples,
length sensitivity and interrupted/resumed training. Reference:
[PyTorch neural modules](https://pytorch.org/docs/stable/nn.html).

## Autoencoder anomaly reconstruction

A bottleneck reconstructs observations; thresholds are selected on training-only
validation and compared with PCA. Reconstruction error is not a calibrated
anomaly probability; novel but compressible anomalies can be missed. Evidence:
Sprint 4 score distributions, ROC and latent diagnostics. Reference:
[PCA reconstruction baseline](https://scikit-learn.org/stable/modules/decomposition.html#pca).

## Stationary bandits

Epsilon-greedy/UCB and Bayesian posterior policies trade exploration for reward
under stationary arm assumptions. Regret depends on the simulated reward model;
posterior uncertainty is not valid under arbitrary nonstationarity. Evidence:
Sprint 5 Gaussian/Bernoulli regret and posterior plots. Reference:
[Sutton and Barto, Reinforcement Learning](http://incompleteideas.net/book/the-book-2nd.html).

## Discounted MDP planning and tabular control

Bellman operators support value/policy iteration with residual bounds. Q-learning
and SARSA use different off-policy/on-policy bootstrapped targets and explicit
terminal/truncation semantics. Tabular assumptions do not extend automatically
to continuous control. The [planning proof](proofs/discounted-mdp-planning.md)
states contraction assumptions and limits. Evidence: Sprint 5 residuals,
policy gaps, cliff risk and held-out seed returns. Reference: Sutton and Barto.

## REINFORCE, DQN and PPO

REINFORCE uses sampled policy gradients; DQN combines replay/target networks;
PPO uses clipped ratios and generalized advantage estimates. Each family has its
own objective, checkpoint and interaction contract. High variance, bootstrap
bias, stale replay and policy collapse are possible; clipped objectives do not
guarantee monotonic returns. Evidence: Sprint 5 independent train/validation/test
streams, ablations, TD loss, PPO optimization and held-out distributions.
References: [DQN](https://www.nature.com/articles/nature14236),
[PPO](https://arxiv.org/abs/1707.06347), [GAE](https://arxiv.org/abs/1506.02438).

## Shared data, contracts and reporting classes

Configuration, source metadata, run context/results, artifact stores, registry
records, dataset splits, fitted pipelines and diagnostic records support these
families rather than introducing new estimators. Their invariants are the
[architecture](architecture.md), [reproducibility](reproducibility.md), and
per-class API contracts: immutable identities, finite/shape-checked data,
training-only fitted state, named RNG streams and bounded transactional outputs.
Complexity follows the owned data size or called model; serialization is linear
in emitted bytes and hashing is linear in artifact bytes.
