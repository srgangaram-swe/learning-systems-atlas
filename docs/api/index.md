# API reference

Every public top-level class is indexed from source. Mathematical context,
assumptions, complexity and failure modes appear in the [family cards](../model-cards.md).

The common execution boundary remains `Experiment.run(RunContext) -> RunResult`.

- [learning_atlas.core.artifacts](learning_atlas-core-artifacts.md): `ArtifactStore`
- [learning_atlas.core.config](learning_atlas-core-config.md): `StrictConfig`, `BaseExperimentConfig`, `RegressionBenchmarkConfig`, `ClassificationBenchmarkConfig`, `ScratchRegressionBenchmarkConfig`, `ScratchClassificationBenchmarkConfig`, `ClusteringBenchmarkConfig`, `ScratchClusteringBenchmarkConfig`, `ScratchRepresentationBenchmarkConfig`, `ScratchMLPBenchmarkConfig`, `DeepVisionBenchmarkConfig`, `DeepSequenceBenchmarkConfig`, `DeepAutoencoderBenchmarkConfig`, `QLearningConfig`, `ReinforcementBenchmarkConfig`
- [learning_atlas.core.contracts](learning_atlas-core-contracts.md): `LearningParadigm`, `SourceKind`, `ContractModel`, `SourceMetadata`, `CandidateResult`, `RunResult`, `RunContext`, `Experiment`
- [learning_atlas.core.estimators](learning_atlas-core-estimators.md): `NotFittedError`, `Estimator`, `RegressorMixin`, `ClassifierMixin`
- [learning_atlas.core.registry](learning_atlas-core-registry.md): `ExperimentSpec`
- [learning_atlas.core.reproducibility](learning_atlas-core-reproducibility.md): `GitState`
- [learning_atlas.deep.autoencoder](learning_atlas-deep-autoencoder.md): `BottleneckAutoencoder`
- [learning_atlas.deep.autograd](learning_atlas-deep-autograd.md): `AutogradError`, `Tensor`
- [learning_atlas.deep.benchmarks](learning_atlas-deep-benchmarks.md): `ScratchMLPBenchmark`, `DeepVisionBenchmark`, `DeepSequenceBenchmark`, `DeepAutoencoderBenchmark`
- [learning_atlas.deep.datasets](learning_atlas-deep-datasets.md): `PlanarDataset`, `SequenceDataset`
- [learning_atlas.deep.mlp](learning_atlas-deep-mlp.md): `MLPTrainingError`, `MLPClassifier`
- [learning_atlas.deep.sequence](learning_atlas-deep-sequence.md): `SequenceLSTM`, `PaddedSequenceMLP`
- [learning_atlas.deep.training](learning_atlas-deep-training.md): `CheckpointPayload`, `DeepTrainingError`, `BatchObjective`, `Objective`, `TrainerConfig`, `EpochRecord`, `TrainingReport`, `Trainer`, `ClassificationObjective`, `SequenceClassificationObjective`, `ReconstructionObjective`
- [learning_atlas.deep.vision](learning_atlas-deep-vision.md): `VisionMLP`, `CompactCNN`
- [learning_atlas.reinforcement.bandits](learning_atlas-reinforcement-bandits.md): `BanditAgent`, `BanditTrace`
- [learning_atlas.reinforcement.benchmark](learning_atlas-reinforcement-benchmark.md): `LaboratoryEvidence`, `ReinforcementBenchmark`
- [learning_atlas.reinforcement.checkpoints](learning_atlas-reinforcement-checkpoints.md): `TensorRecord`, `AdamRecord`, `RandomRecord`, `ReplayRecord`, `DQNCheckpoint`, `Envelope`
- [learning_atlas.reinforcement.control](learning_atlas-reinforcement-control.md): `ControlReport`, `ValidationSelector`
- [learning_atlas.reinforcement.dqn](learning_atlas-reinforcement-dqn.md): `DQNSession`
- [learning_atlas.reinforcement.gridworld](learning_atlas-reinforcement-gridworld.md): `FiniteMDP`, `Gridworld`
- [learning_atlas.reinforcement.neural_config](learning_atlas-reinforcement-neural_config.md): `ControlConfig`
- [learning_atlas.reinforcement.objectives](learning_atlas-reinforcement-objectives.md): `ActorCritic`, `PolicyObjective`, `DQNObjective`
- [learning_atlas.reinforcement.planning](learning_atlas-reinforcement-planning.md): `PlanningResult`
- [learning_atlas.reinforcement.policy_gradient](learning_atlas-reinforcement-policy_gradient.md): `Rollout`
- [learning_atlas.reinforcement.q_learning](learning_atlas-reinforcement-q_learning.md): `QLearningAgent`, `PolicyEvaluation`, `FrozenLakeBenchmark`
- [learning_atlas.reinforcement.replay](learning_atlas-reinforcement-replay.md): `ReplayBuffer`
- [learning_atlas.reinforcement.tabular](learning_atlas-reinforcement-tabular.md): `TabularTrace`
- [learning_atlas.reinforcement.validation](learning_atlas-reinforcement-validation.md): `ReinforcementError`
- [learning_atlas.reporting.deep](learning_atlas-reporting-deep.md): `DecisionPanel`
- [learning_atlas.reporting.parity](learning_atlas-reporting-parity.md): `Predictor`, `ParityPair`, `ParityObservation`
- [learning_atlas.reporting.supervised](learning_atlas-reporting-supervised.md): `Predictor`
- [learning_atlas.supervised.baselines](learning_atlas-supervised-baselines.md): `MeanRegressor`, `PriorClassifier`
- [learning_atlas.supervised.classification](learning_atlas-supervised-classification.md): `ClassificationData`, `ClassificationBenchmark`
- [learning_atlas.supervised.comparison](learning_atlas-supervised-comparison.md): `CandidateExecutionError`, `ProbabilisticClassifier`, `FeatureTransformer`, `FittedPipeline`, `ScratchRegressionBenchmark`, `ScratchClassificationBenchmark`
- [learning_atlas.supervised.datasets](learning_atlas-supervised-datasets.md): `RegressionDataset`, `ClassificationDataset`, `BlobsDataset`
- [learning_atlas.supervised.ensemble](learning_atlas-supervised-ensemble.md): `RandomForestRegressor`, `RandomForestClassifier`, `GradientBoostingRegressor`, `GradientBoostingClassifier`
- [learning_atlas.supervised.linear_model](learning_atlas-supervised-linear_model.md): `ConvergenceError`, `LinearRegression`, `Ridge`, `Lasso`, `LogisticRegression`
- [learning_atlas.supervised.model_selection](learning_atlas-supervised-model_selection.md): `FoldIndices`
- [learning_atlas.supervised.naive_bayes](learning_atlas-supervised-naive_bayes.md): `GaussianNB`, `MultinomialNB`
- [learning_atlas.supervised.neighbors](learning_atlas-supervised-neighbors.md): `KNeighborsClassifier`, `KNeighborsRegressor`
- [learning_atlas.supervised.preprocessing](learning_atlas-supervised-preprocessing.md): `SplitData`, `StandardScaler`
- [learning_atlas.supervised.regression](learning_atlas-supervised-regression.md): `RegressionData`, `RegressionBenchmark`
- [learning_atlas.supervised.svm](learning_atlas-supervised-svm.md): `SVMConvergenceError`, `KernelSVMClassifier`
- [learning_atlas.supervised.tree](learning_atlas-supervised-tree.md): `TreeNode`, `DecisionTreeRegressor`, `DecisionTreeClassifier`
- [learning_atlas.unsupervised.base](learning_atlas-unsupervised-base.md): `NotFittedError`, `UnsupervisedModel`, `Clusterer`, `Transformer`
- [learning_atlas.unsupervised.clustering](learning_atlas-unsupervised-clustering.md): `ClusteringBenchmark`
- [learning_atlas.unsupervised.comparison](learning_atlas-unsupervised-comparison.md): `UnsupervisedCandidateError`, `ScratchClusteringBenchmark`, `ScratchRepresentationBenchmark`
- [learning_atlas.unsupervised.datasets](learning_atlas-unsupervised-datasets.md): `StructureDataset`
- [learning_atlas.unsupervised.decomposition](learning_atlas-unsupervised-decomposition.md): `DecompositionConvergenceError`, `PCA`
- [learning_atlas.unsupervised.density](learning_atlas-unsupervised-density.md): `DBSCAN`
- [learning_atlas.unsupervised.hierarchical](learning_atlas-unsupervised-hierarchical.md): `AgglomerativeClustering`
- [learning_atlas.unsupervised.kmeans](learning_atlas-unsupervised-kmeans.md): `KMeansConvergenceError`, `KMeansNumericalError`, `KMeans`
- [learning_atlas.unsupervised.manifold](learning_atlas-unsupervised-manifold.md): `PerplexitySearchError`, `TSNENumericalError`, `TSNEConvergenceError`, `AffinityResult`, `TSNE`
- [learning_atlas.unsupervised.metrics](learning_atlas-unsupervised-metrics.md): `SilhouetteResult`
- [learning_atlas.unsupervised.mixture](learning_atlas-unsupervised-mixture.md): `GaussianMixtureConvergenceError`, `GaussianMixtureNumericalError`, `GaussianMixture`
