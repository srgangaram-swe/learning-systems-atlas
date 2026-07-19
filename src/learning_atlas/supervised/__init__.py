"""Public supervised-learning experiments and from-scratch estimators."""

from learning_atlas.supervised.classification import ClassificationBenchmark
from learning_atlas.supervised.comparison import (
    ScratchClassificationBenchmark,
    ScratchRegressionBenchmark,
)
from learning_atlas.supervised.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from learning_atlas.supervised.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from learning_atlas.supervised.naive_bayes import GaussianNB, MultinomialNB
from learning_atlas.supervised.neighbors import KNeighborsClassifier, KNeighborsRegressor
from learning_atlas.supervised.regression import RegressionBenchmark
from learning_atlas.supervised.svm import KernelSVMClassifier
from learning_atlas.supervised.tree import DecisionTreeClassifier, DecisionTreeRegressor

__all__ = [
    "ClassificationBenchmark",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "GaussianNB",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "KernelSVMClassifier",
    "Lasso",
    "LinearRegression",
    "LogisticRegression",
    "MultinomialNB",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "RegressionBenchmark",
    "Ridge",
    "ScratchClassificationBenchmark",
    "ScratchRegressionBenchmark",
]
