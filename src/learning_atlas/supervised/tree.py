"""Deterministic, NumPy-only classification and regression trees.

The implementations in this module intentionally expose a small estimator API while
keeping the tree-building mathematics visible.  Splits are selected by exhaustive
greedy impurity minimization.  Equal-gain candidates are resolved by feature index and
then threshold, which makes a fixed seed replayable even when feature subsampling is
enabled.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Self, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from learning_atlas.core.estimators import ClassifierMixin, Estimator, RegressorMixin
from learning_atlas.core.validation import FloatArray

IntArray = NDArray[np.int64]
Criterion = Literal["gini", "entropy", "squared_error"]
MaxFeatures: TypeAlias = int | float | Literal["sqrt", "log2"] | None
RandomState: TypeAlias = int | np.random.Generator | None

_SPLIT_TOLERANCE = 1e-14


@dataclass(frozen=True, slots=True)
class TreeNode:
    """Immutable fitted node used for inspection and recursive inference."""

    prediction: float
    impurity: float
    n_samples: int
    feature_index: int | None = None
    threshold: float | None = None
    left: TreeNode | None = None
    right: TreeNode | None = None
    probabilities: tuple[float, ...] | None = None

    @property
    def is_leaf(self) -> bool:
        """Whether this node has no children."""

        return self.feature_index is None


@dataclass(frozen=True, slots=True)
class _Split:
    feature_index: int
    threshold: float
    left_indices: IntArray
    right_indices: IntArray
    gain: float


def _validate_integer(value: int, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        msg = f"{name} must be an integer greater than or equal to {minimum}"
        raise TypeError(msg)
    validated = int(value)
    if validated < minimum:
        msg = f"{name} must be greater than or equal to {minimum}"
        raise ValueError(msg)
    return validated


def _validate_random_state(random_state: RandomState) -> RandomState:
    if random_state is None or isinstance(random_state, np.random.Generator):
        return random_state
    if isinstance(random_state, bool) or not isinstance(random_state, (int, np.integer)):
        msg = "random_state must be a non-negative integer, Generator, or None"
        raise TypeError(msg)
    if int(random_state) < 0:
        msg = "random_state must be non-negative"
        raise ValueError(msg)
    return int(random_state)


def _local_generator(random_state: RandomState) -> np.random.Generator:
    """Return estimator-owned randomness without mutating a caller's generator."""

    if isinstance(random_state, np.random.Generator):
        return deepcopy(random_state)
    return np.random.default_rng(random_state)


def _validate_max_features(max_features: MaxFeatures) -> MaxFeatures:
    if max_features is None:
        return None
    if isinstance(max_features, str):
        if max_features not in {"sqrt", "log2"}:
            msg = "max_features must be None, 'sqrt', 'log2', a positive integer, or a float in (0, 1]"
            raise ValueError(msg)
        return max_features
    if isinstance(max_features, bool):
        msg = "max_features must not be boolean"
        raise TypeError(msg)
    if isinstance(max_features, (int, np.integer)):
        if int(max_features) < 1:
            msg = "integer max_features must be positive"
            raise ValueError(msg)
        return int(max_features)
    if isinstance(max_features, (float, np.floating)):
        value = float(max_features)
        if not np.isfinite(value) or not 0.0 < value <= 1.0:
            msg = "float max_features must be finite and in (0, 1]"
            raise ValueError(msg)
        return value
    msg = "max_features must be None, 'sqrt', 'log2', a positive integer, or a float in (0, 1]"
    raise TypeError(msg)


def _resolve_max_features(max_features: MaxFeatures, n_features: int) -> int:
    if max_features is None:
        return n_features
    if isinstance(max_features, str):
        if max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        return max(1, int(np.log2(n_features)))
    if isinstance(max_features, int):
        if max_features > n_features:
            msg = (
                f"integer max_features={max_features} exceeds the fitted feature count "
                f"of {n_features}"
            )
            raise ValueError(msg)
        return max_features
    return max(1, int(float(max_features) * n_features))


def _validate_tree_parameters(
    *,
    max_depth: int | None,
    min_samples_split: int,
    min_samples_leaf: int,
    max_features: MaxFeatures,
    min_impurity_decrease: float,
    random_state: RandomState,
) -> tuple[int | None, int, int, MaxFeatures, float, RandomState]:
    if max_depth is not None:
        max_depth = _validate_integer(max_depth, name="max_depth", minimum=0)
    min_samples_split = _validate_integer(min_samples_split, name="min_samples_split", minimum=2)
    min_samples_leaf = _validate_integer(min_samples_leaf, name="min_samples_leaf", minimum=1)
    max_features = _validate_max_features(max_features)
    if not np.isfinite(min_impurity_decrease) or min_impurity_decrease < 0.0:
        msg = "min_impurity_decrease must be finite and non-negative"
        raise ValueError(msg)
    random_state = _validate_random_state(random_state)
    return (
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features,
        float(min_impurity_decrease),
        random_state,
    )


class _TreeBuilder:
    """Own recursive construction state for one transactional fit."""

    def __init__(
        self,
        features: FloatArray,
        targets: FloatArray,
        *,
        class_indices: IntArray | None,
        classes: FloatArray | None,
        criterion: Criterion,
        max_depth: int | None,
        min_samples_split: int,
        min_samples_leaf: int,
        max_features: int,
        min_impurity_decrease: float,
        generator: np.random.Generator,
    ) -> None:
        self.features = features
        self.targets = targets
        self.class_indices = class_indices
        self.classes = classes
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.generator = generator
        self.importance_totals = np.zeros(features.shape[1], dtype=np.float64)

    def build(self) -> TreeNode:
        indices = np.arange(len(self.features), dtype=np.int64)
        return self._build_node(indices, depth=0)

    def _build_node(self, indices: IntArray, *, depth: int) -> TreeNode:
        impurity = self._impurity(indices)
        leaf = self._leaf(indices, impurity)
        if self._must_stop(indices, depth=depth, impurity=impurity):
            return leaf

        split = self._best_split(indices, parent_impurity=impurity)
        if split is None or split.gain <= self.min_impurity_decrease + _SPLIT_TOLERANCE:
            return leaf

        self.importance_totals[split.feature_index] += len(indices) * split.gain
        return TreeNode(
            prediction=leaf.prediction,
            impurity=impurity,
            n_samples=len(indices),
            feature_index=split.feature_index,
            threshold=split.threshold,
            left=self._build_node(split.left_indices, depth=depth + 1),
            right=self._build_node(split.right_indices, depth=depth + 1),
            probabilities=leaf.probabilities,
        )

    def _must_stop(self, indices: IntArray, *, depth: int, impurity: float) -> bool:
        return (
            impurity <= _SPLIT_TOLERANCE
            or (self.max_depth is not None and depth >= self.max_depth)
            or len(indices) < self.min_samples_split
            or len(indices) < 2 * self.min_samples_leaf
        )

    def _leaf(self, indices: IntArray, impurity: float) -> TreeNode:
        if self.class_indices is None:
            prediction = float(np.mean(self.targets[indices]))
            return TreeNode(prediction=prediction, impurity=impurity, n_samples=len(indices))

        assert self.classes is not None
        counts = np.bincount(self.class_indices[indices], minlength=len(self.classes))
        class_index = int(np.argmax(counts))
        probabilities = tuple(float(count / len(indices)) for count in counts)
        return TreeNode(
            prediction=float(self.classes[class_index]),
            impurity=impurity,
            n_samples=len(indices),
            probabilities=probabilities,
        )

    def _impurity(self, indices: IntArray) -> float:
        if self.class_indices is None:
            values = self.targets[indices]
            return float(np.mean((values - np.mean(values)) ** 2))

        assert self.classes is not None
        counts = np.bincount(self.class_indices[indices], minlength=len(self.classes))
        probabilities = counts[counts > 0] / len(indices)
        if self.criterion == "gini":
            return float(1.0 - np.sum(probabilities**2))
        return float(-np.sum(probabilities * np.log2(probabilities)))

    def _candidate_features(self) -> IntArray:
        n_features = self.features.shape[1]
        if self.max_features == n_features:
            return np.arange(n_features, dtype=np.int64)
        selected = self.generator.choice(n_features, size=self.max_features, replace=False)
        return np.sort(np.asarray(selected, dtype=np.int64))

    def _best_split(self, indices: IntArray, *, parent_impurity: float) -> _Split | None:
        best_score = np.inf
        best_feature: int | None = None
        best_threshold = 0.0
        n_samples = len(indices)

        for feature in self._candidate_features():
            feature_index = int(feature)
            order = np.argsort(self.features[indices, feature_index], kind="stable")
            sorted_indices = indices[order]
            sorted_values = self.features[sorted_indices, feature_index]
            if self.class_indices is None:
                scores = self._regression_split_scores(sorted_indices)
            else:
                scores = self._classification_split_scores(sorted_indices)

            for position in range(self.min_samples_leaf, n_samples - self.min_samples_leaf + 1):
                if sorted_values[position - 1] == sorted_values[position]:
                    continue
                score = float(scores[position])
                if score < best_score - _SPLIT_TOLERANCE:
                    best_score = score
                    best_feature = feature_index
                    lower = float(sorted_values[position - 1])
                    upper = float(sorted_values[position])
                    midpoint = lower / 2.0 + upper / 2.0
                    # Adjacent floats have no representable midpoint. Rounding
                    # to ``upper`` would violate the scored partition and can
                    # recursively reproduce the same node forever. ``lower``
                    # represents the identical <= / > split exactly.
                    best_threshold = lower if not lower <= midpoint < upper else midpoint

        if best_feature is None:
            return None
        left_mask = self.features[indices, best_feature] <= best_threshold
        left_indices = np.asarray(indices[left_mask], dtype=np.int64)
        right_indices = np.asarray(indices[~left_mask], dtype=np.int64)
        if len(left_indices) == 0 or len(right_indices) == 0:  # defensive numerical guard
            return None
        gain = max(0.0, parent_impurity - best_score)
        return _Split(best_feature, best_threshold, left_indices, right_indices, gain)

    def _regression_split_scores(self, sorted_indices: IntArray) -> FloatArray:
        values = self.targets[sorted_indices]
        count = len(values)
        prefix = np.concatenate((np.zeros(1), np.cumsum(values, dtype=np.float64)))
        prefix_squares = np.concatenate((np.zeros(1), np.cumsum(values * values, dtype=np.float64)))
        positions = np.arange(count + 1, dtype=np.float64)
        left_count = positions
        right_count = count - positions
        left_sse = np.zeros(count + 1, dtype=np.float64)
        right_sse = np.zeros(count + 1, dtype=np.float64)
        valid_left = left_count > 0
        valid_right = right_count > 0
        left_sse[valid_left] = prefix_squares[valid_left] - (
            prefix[valid_left] ** 2 / left_count[valid_left]
        )
        total = prefix[-1]
        total_squares = prefix_squares[-1]
        right_sum = total - prefix
        right_square_sum = total_squares - prefix_squares
        right_sse[valid_right] = right_square_sum[valid_right] - (
            right_sum[valid_right] ** 2 / right_count[valid_right]
        )
        # Roundoff can make a mathematically zero sum of squares slightly negative.
        return np.maximum(0.0, left_sse + right_sse) / count

    def _classification_split_scores(self, sorted_indices: IntArray) -> FloatArray:
        assert self.class_indices is not None
        assert self.classes is not None
        encoded = self.class_indices[sorted_indices]
        count = len(encoded)
        n_classes = len(self.classes)
        cumulative = np.zeros((count + 1, n_classes), dtype=np.float64)
        for position, class_index in enumerate(encoded, start=1):
            cumulative[position] = cumulative[position - 1]
            cumulative[position, int(class_index)] += 1.0
        totals = cumulative[-1]
        scores = np.full(count + 1, np.inf, dtype=np.float64)
        for position in range(1, count):
            left_counts = cumulative[position]
            right_counts = totals - left_counts
            if self.criterion == "gini":
                left_probability = left_counts / position
                right_probability = right_counts / (count - position)
                left_impurity = 1.0 - float(np.sum(left_probability**2))
                right_impurity = 1.0 - float(np.sum(right_probability**2))
            else:
                left_probability = left_counts[left_counts > 0] / position
                right_probability = right_counts[right_counts > 0] / (count - position)
                left_impurity = -float(np.sum(left_probability * np.log2(left_probability)))
                right_impurity = -float(np.sum(right_probability * np.log2(right_probability)))
            scores[position] = (
                position * left_impurity + (count - position) * right_impurity
            ) / count
        return scores


def _tree_depth(node: TreeNode) -> int:
    if node.is_leaf:
        return 0
    assert node.left is not None and node.right is not None
    return 1 + max(_tree_depth(node.left), _tree_depth(node.right))


def _leaf_count(node: TreeNode) -> int:
    if node.is_leaf:
        return 1
    assert node.left is not None and node.right is not None
    return _leaf_count(node.left) + _leaf_count(node.right)


def _normalized_importances(importances: FloatArray) -> FloatArray:
    total = float(np.sum(importances))
    if total == 0.0:
        return np.zeros_like(importances)
    return np.asarray(importances / total, dtype=np.float64)


def _terminal_node(root: TreeNode, row: FloatArray) -> TreeNode:
    node = root
    while not node.is_leaf:
        assert node.feature_index is not None
        assert node.threshold is not None
        assert node.left is not None and node.right is not None
        node = node.left if row[node.feature_index] <= node.threshold else node.right
    return node


class DecisionTreeRegressor(RegressorMixin, Estimator):
    """CART-style regression tree minimizing within-node squared error."""

    def __init__(
        self,
        *,
        criterion: Literal["squared_error"] = "squared_error",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: MaxFeatures = None,
        min_impurity_decrease: float = 0.0,
        random_state: RandomState = None,
    ) -> None:
        super().__init__()
        if criterion != "squared_error":
            msg = "criterion must be 'squared_error' for DecisionTreeRegressor"
            raise ValueError(msg)
        (
            self.max_depth,
            self.min_samples_split,
            self.min_samples_leaf,
            self.max_features,
            self.min_impurity_decrease,
            self.random_state,
        ) = _validate_tree_parameters(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            random_state=random_state,
        )
        self.criterion = criterion
        self._tree: TreeNode | None = None
        self._feature_importances: FloatArray | None = None

    @property
    def tree_(self) -> TreeNode:
        self._require_fitted()
        assert self._tree is not None
        return self._tree

    @property
    def depth_(self) -> int:
        return _tree_depth(self.tree_)

    @property
    def n_leaves_(self) -> int:
        return _leaf_count(self.tree_)

    @property
    def feature_importances_(self) -> FloatArray:
        self._require_fitted()
        assert self._feature_importances is not None
        return self._feature_importances.copy()

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        validated_features, validated_targets = self._validate_fit_data(features, targets)
        resolved_features = _resolve_max_features(self.max_features, validated_features.shape[1])
        builder = _TreeBuilder(
            validated_features,
            validated_targets,
            class_indices=None,
            classes=None,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=resolved_features,
            min_impurity_decrease=self.min_impurity_decrease,
            generator=_local_generator(self.random_state),
        )
        tree = builder.build()
        importances = _normalized_importances(builder.importance_totals)

        self._tree = tree
        self._feature_importances = importances
        self._mark_fitted(validated_features.shape[1])
        return self

    def predict(self, features: ArrayLike) -> FloatArray:
        validated = self._validate_predict_data(features)
        root = self.tree_
        return np.asarray(
            [_terminal_node(root, row).prediction for row in validated], dtype=np.float64
        )


class DecisionTreeClassifier(ClassifierMixin, Estimator):
    """CART-style classifier supporting Gini and Shannon-entropy criteria."""

    def __init__(
        self,
        *,
        criterion: Literal["gini", "entropy"] = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: MaxFeatures = None,
        min_impurity_decrease: float = 0.0,
        random_state: RandomState = None,
    ) -> None:
        super().__init__()
        if criterion not in {"gini", "entropy"}:
            msg = "criterion must be 'gini' or 'entropy' for DecisionTreeClassifier"
            raise ValueError(msg)
        (
            self.max_depth,
            self.min_samples_split,
            self.min_samples_leaf,
            self.max_features,
            self.min_impurity_decrease,
            self.random_state,
        ) = _validate_tree_parameters(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            random_state=random_state,
        )
        self.criterion = criterion
        self._tree: TreeNode | None = None
        self._classes: FloatArray | None = None
        self._feature_importances: FloatArray | None = None

    @property
    def tree_(self) -> TreeNode:
        self._require_fitted()
        assert self._tree is not None
        return self._tree

    @property
    def classes_(self) -> FloatArray:
        self._require_fitted()
        assert self._classes is not None
        return self._classes.copy()

    @property
    def depth_(self) -> int:
        return _tree_depth(self.tree_)

    @property
    def n_leaves_(self) -> int:
        return _leaf_count(self.tree_)

    @property
    def feature_importances_(self) -> FloatArray:
        self._require_fitted()
        assert self._feature_importances is not None
        return self._feature_importances.copy()

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        validated_features, validated_targets = self._validate_fit_data(features, targets)
        classes = np.unique(validated_targets)
        class_indices = np.asarray(np.searchsorted(classes, validated_targets), dtype=np.int64)
        resolved_features = _resolve_max_features(self.max_features, validated_features.shape[1])
        builder = _TreeBuilder(
            validated_features,
            validated_targets,
            class_indices=class_indices,
            classes=classes,
            criterion=cast(Criterion, self.criterion),
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=resolved_features,
            min_impurity_decrease=self.min_impurity_decrease,
            generator=_local_generator(self.random_state),
        )
        tree = builder.build()
        importances = _normalized_importances(builder.importance_totals)

        self._tree = tree
        self._classes = np.asarray(classes, dtype=np.float64)
        self._feature_importances = importances
        self._mark_fitted(validated_features.shape[1])
        return self

    def predict(self, features: ArrayLike) -> FloatArray:
        validated = self._validate_predict_data(features)
        root = self.tree_
        return np.asarray(
            [_terminal_node(root, row).prediction for row in validated], dtype=np.float64
        )

    def predict_proba(self, features: ArrayLike) -> FloatArray:
        """Return leaf class frequencies aligned to ``classes_``."""

        validated = self._validate_predict_data(features)
        root = self.tree_
        probabilities = []
        for row in validated:
            leaf = _terminal_node(root, row)
            assert leaf.probabilities is not None
            probabilities.append(leaf.probabilities)
        return np.asarray(probabilities, dtype=np.float64)
