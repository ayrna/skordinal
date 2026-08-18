"""Model selection and estimator loading utilities."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from skordinal.metrics import get_ordinal_scorer
from skordinal.model_selection._validation import (
    is_searchcv,
    prepare_param_grid,
)

_SKORDINAL_CLASSIFIERS = {
    "NNOP": "skordinal.classifiers._nnop.NNOP",
    "NNPOM": "skordinal.classifiers._nnpom.NNPOM",
    "OrdinalDecomposition": "skordinal.classifiers._ordinal_decomposition.OrdinalDecomposition",
    "REDSVM": "skordinal.classifiers._redsvm.REDSVM",
    "SVOREX": "skordinal.classifiers._svorex.SVOREX",
}

_SKLEARN_CLASSIFIERS = {
    "SVC": "sklearn.svm.SVC",
    "LogisticRegression": "sklearn.linear_model.LogisticRegression",
    "RandomForestClassifier": "sklearn.ensemble.RandomForestClassifier",
}

_CLASSIFIERS = {**_SKORDINAL_CLASSIFIERS, **_SKLEARN_CLASSIFIERS}


def get_classifier_by_name(classifier_name: str) -> type[BaseEstimator]:
    """Return a classifier not instantiated matching a given input name.

    Parameters
    ----------
    classifier_name : str
        Name of the classification algorithm being employed.

    Returns
    -------
    classifier : object
        Returns a classifier, either from a scikit-learn module, or from a
        module of this framework.

    Raises
    ------
    ValueError
        If an unknown classifier name is provided.

    Examples
    --------
    >>> get_classifier_by_name("SVOREX")
    <class 'skordinal.classifiers._svorex.SVOREX'>
    >>> get_classifier_by_name("REDSVM")
    <class 'skordinal.classifiers._redsvm.REDSVM'>
    >>> get_classifier_by_name("SVC")
    <class 'sklearn.svm._classes.SVC'>

    """
    if classifier_name not in _CLASSIFIERS:
        raise ValueError(f"Unknown classifier '{classifier_name}'.")

    module_path, class_name = _CLASSIFIERS[classifier_name].rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, class_name)


def load_classifier(
    classifier_name: str,
    random_state: int | np.random.RandomState | None = None,
    n_jobs: int = 1,
    cv_n_folds: int = 3,
    cv_metric: str | Callable[..., Any] = "mae",
    param_grid: dict[str, Any] | None = None,
) -> BaseEstimator | GridSearchCV:
    """Return a fully configured classifier, optionally with cross-validation.

    This function loads a classifier, configures its parameters, and optionally
    sets up cross-validation if multiple parameter values are provided.

    Parameters
    ----------
    classifier_name : str
        Name of the classification algorithm being employed.

    random_state : int, RandomState instance or None, optional (default=None)
        Seed for reproducible randomization in model training and probability
        estimation.

    n_jobs : int, optional (default=1)
        Number of parallel processing cores for computational tasks.

    cv_n_folds : int, optional (default=3)
        Number of folds for cross-validation (only used if applicable).

    cv_metric : str or callable, optional (default="mae")
        Evaluation metric for cross-validation performance assessment.

    param_grid : dict or None, optional (default=None)
        Hyperparameter grid. If multiple values are given, cross-validation will be applied.

    Returns
    -------
    classifier : object
        The initialized classifier object, optionally wrapped in GridSearchCV.

    Raises
    ------
    ValueError
        If an unknown classifier name is provided or if an invalid parameter
        is specified for the classifier.

    Examples
    --------
    >>> from skordinal.model_selection import load_classifier
    >>> clf = load_classifier("SVC", random_state=0)
    >>> clf
    SVC()
    >>> clf_cv = load_classifier("SVC", random_state=0, param_grid={"C": [0.1, 1.0]})
    >>> clf_cv.__class__.__name__
    'GridSearchCV'

    """

    # Check if the classifier_name is already an instance of BaseEstimator
    if isinstance(classifier_name, BaseEstimator):
        estimator = clone(classifier_name)
        if param_grid:
            estimator.set_params(**param_grid)
        return estimator

    classifier_cls = get_classifier_by_name(classifier_name)

    if param_grid is None:
        return classifier_cls()

    param_grid = prepare_param_grid(classifier_cls, param_grid, random_state)

    if is_searchcv(param_grid):
        scorer = (
            get_ordinal_scorer(cv_metric) if isinstance(cv_metric, str) else cv_metric
        )
        cv = StratifiedKFold(
            n_splits=cv_n_folds, shuffle=True, random_state=random_state
        )

        return GridSearchCV(
            estimator=classifier_cls(),
            param_grid=param_grid,
            scoring=scorer,
            n_jobs=n_jobs,
            cv=cv,
            error_score="raise",
        )

    try:
        classifier = classifier_cls(**param_grid)
        classifier.assigned_params_ = param_grid
        return classifier
    except TypeError as e:
        invalid_param = str(e).split("'")[1]
        raise ValueError(
            f"Invalid parameter '{invalid_param}' for classifier '{classifier_name}'."
        )
