"""Ordinal Projection-Based Ensemble (OPBE)."""

from __future__ import annotations

import warnings
from numbers import Integral, Real

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from scipy.special import log_expit
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    _fit_context,
    clone,
)
from sklearn.utils import get_tags
from sklearn.utils._param_validation import HasMethods, Interval
from sklearn.utils.extmath import softmax
from sklearn.utils.validation import check_is_fitted, validate_data

from skordinal.utils.validation import check_ordinal_targets

from ._svor import SVOR


class OPBE(MetaEstimatorMixin, ClassifierMixin, BaseEstimator):
    r"""Ordinal Projection-Based Ensemble.

    OPBE [1]_ fits one threshold model per class on the whole training set,
    with the target lower, equal or upper relative to that class. It rebuilds
    the group probabilities of every model from its projection and thresholds,
    fuses them with a product and predicts the class with the largest fused
    score.

    Parameters
    ----------
    estimator : estimator object, default=None
        Threshold model with ``fit``, ``predict_projection`` and non-decreasing
        ``thresholds_``, such as ``SVOR``, ``POM`` or ``KDLOR``, cloned once per
        class. If ``None``, ``SVOR()`` is used, as in the authors'
        implementation. Its own hyperparameters are reachable as
        ``estimator__<param>``.

    min_spread : float, default=4.0
        Minimum spread of the three-class submodels, the distance between their
        two thresholds. A submodel below it has its projection and thresholds
        multiplied by the factor that brings its spread to ``min_spread``, which
        acts as a temperature on its probabilities. The submodels of the first
        and last class have a single threshold and are never rescaled. The
        default 4 is that of the authors' implementation, and 0 disables the
        rescaling.

    n_jobs : int, default=None
        Number of jobs to run in parallel to fit the submodels. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means using
        all processors.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels in ascending order.

    n_features_in_ : int
        Number of features seen during fit.

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit. Defined only when X has feature
        names that are all strings.

    estimators_ : list of n_classes estimators
        Fitted submodels, one per class in the order of ``classes_``.

    fusion_weights_ : ndarray of shape (n_classes,)
        Weights ``w_j`` of the fusion in the Notes. They depend only on the
        number of classes.

    See Also
    --------
    OrdinalDecomposition : Ordinal classification from binary subproblems.

    Notes
    -----
    The submodel of class ``i`` is trained with the target 0 for the classes
    below ``i``, 1 for ``i`` and 2 for the classes above, so the submodels of
    the first and last class are binary. Its group probabilities are those of a
    cumulative logit model on its projection and thresholds, whatever its own
    ``predict_proba``, and the probability of a group is split equally among
    its classes. The fused score of class ``j`` is

    .. math::

        S_j = \frac{1}{w_j} \prod_{i} q_{ij}, \quad
        q_{ij} = \begin{cases}
            p_i(\text{lower}) / i & j < i \\
            p_i(\text{equal}) & j = i \\
            p_i(\text{upper}) / (K - 1 - i) & j > i
        \end{cases}

    with ``K`` classes, ``i`` and ``j`` starting at 0, and weights
    :math:`w_j = 1 + \sum_{i > j} 1 / i + \sum_{i < j} 1 / (K - 1 - i)`.

    A submodel with equal thresholds gives probability 0 to its class, which
    is then never predicted, so ``fit`` warns about it.

    ``predict_proba`` returns the fused scores normalised to sum 1, not
    calibrated probabilities.

    References
    ----------
    .. [1] M. Pérez-Ortiz, P. A. Gutiérrez, and C. Hervás-Martínez,
       "Projection-based ensemble learning for ordinal regression", IEEE
       Transactions on Cybernetics, vol. 44, no. 5, pp. 681-694, 2014.
       https://doi.org/10.1109/TCYB.2013.2266336

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.classifiers import OPBE, POM
    >>> rng = np.random.RandomState(0)
    >>> X = rng.randn(120, 2)
    >>> y = np.digitize(X[:, 0] + 0.3 * rng.randn(120), [-0.8, 0.0, 0.8])
    >>> clf = OPBE(estimator=POM()).fit(X, y)
    >>> clf.predict([[-2.0, 0.0], [-0.4, 0.0], [0.4, 0.0], [2.0, 0.0]])
    array([0, 1, 2, 3])
    >>> clf.predict_proba(X[:2]).shape
    (2, 4)
    """

    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit", "predict_projection"]), None],
        "min_spread": [Interval(Real, 0.0, None, closed="left")],
        "n_jobs": [None, Integral],
    }

    def __init__(
        self,
        estimator=None,
        *,
        min_spread: float = 4.0,
        n_jobs: int | None = None,
    ) -> None:
        self.estimator = estimator
        self.min_spread = min_spread
        self.n_jobs = n_jobs

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X: ArrayLike, y: ArrayLike) -> OPBE:
        """Fit one submodel per class.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like of shape (n_samples,)
            The ordinal target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = validate_data(
            self, X, y, accept_sparse=True, dtype=None, ensure_all_finite=False
        )
        self.classes_, y_enc = check_ordinal_targets(y)
        n_classes = len(self.classes_)
        estimator = self._get_estimator()

        # Target 0 for the lower classes, 1 for the class itself, 2 for the upper ones
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(clone(estimator).fit)(X, (y_enc >= i).astype(int) + (y_enc > i))
            for i in range(n_classes)
        )

        for label, est in zip(self.classes_, self.estimators_, strict=True):
            spread = np.diff(est.thresholds_)
            if np.any(spread < 0):
                raise ValueError(
                    f"The submodel of class {label} in {self.__class__.__name__} "
                    f"has decreasing thresholds {est.thresholds_}."
                )
            if np.any(spread == 0):
                warnings.warn(
                    f"The submodel of class {label} in {self.__class__.__name__} "
                    "has equal thresholds, so its class is never predicted.",
                    UserWarning,
                    stacklevel=2,
                )

        self.fusion_weights_ = np.ones(n_classes)
        for j in range(n_classes):
            self.fusion_weights_[j] += sum(1 / i for i in range(j + 1, n_classes))
            self.fusion_weights_[j] += sum(1 / (n_classes - 1 - i) for i in range(j))
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict the class with the largest fused score.

        Ties go to the lower class.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Sparse matrices are accepted only if they are
            supported by the base estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted classes.
        """
        check_is_fitted(self)
        return self.classes_[np.argmax(self._log_scores(X), axis=1)]

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return the fused scores normalised to sum 1.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Sparse matrices are accepted only if they are
            supported by the base estimator.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Normalised scores, with classes ordered as in ``classes_``.
        """
        check_is_fitted(self)
        return softmax(self._log_scores(X), copy=False)

    def _log_scores(self, X: ArrayLike) -> np.ndarray:
        """Compute the logarithm of the fused score of every class."""
        X = validate_data(
            self,
            X,
            reset=False,
            accept_sparse=True,
            dtype=None,
            ensure_all_finite=False,
        )
        projections = np.column_stack(
            [est.predict_projection(X) for est in self.estimators_]
        )
        # Thresholds around the class of each submodel, infinite for an absent group
        lower_bound = np.array(
            [-np.inf] + [est.thresholds_[0] for est in self.estimators_[1:]]
        )
        upper_bound = np.array(
            [est.thresholds_[-1] for est in self.estimators_[:-1]] + [np.inf]
        )
        # Infinite edge spreads keep scale 1, and eps stands in for a zero spread
        eps = np.finfo(np.float64).eps
        scale = np.maximum(
            self.min_spread / np.maximum(upper_bound - lower_bound, eps), 1.0
        )
        projections = projections * scale
        lower_bound, upper_bound = lower_bound * scale, upper_bound * scale

        log_lower = log_expit(lower_bound - projections)
        log_upper = log_expit(projections - upper_bound)
        with np.errstate(divide="ignore"):
            # The sigmoid s satisfies s(b) - s(a) = s(b) s(-a) (1 - exp(a - b))
            log_equal = (
                log_expit(upper_bound - projections)
                + log_expit(projections - lower_bound)
                + np.log(-np.expm1(lower_bound - upper_bound))
            )

        # A group's probability is split equally among its classes
        n_lower = np.arange(len(self.estimators_))
        n_upper = n_lower[::-1]
        lower_terms = log_lower[:, 1:] - np.log(n_lower[1:])
        upper_terms = log_upper[:, :-1] - np.log(n_upper[:-1])
        # Class j sums the lower terms of submodels i > j and the upper terms of i < j
        log_scores = log_equal - np.log(self.fusion_weights_)
        log_scores[:, :-1] += np.cumsum(lower_terms[:, ::-1], axis=1)[:, ::-1]
        log_scores[:, 1:] += np.cumsum(upper_terms, axis=1)
        return log_scores

    def _get_estimator(self):
        """Return the user estimator or the default SVOR."""
        return SVOR() if self.estimator is None else self.estimator

    def __sklearn_tags__(self):
        """Take sparse and NaN support from the estimator."""
        tags = super().__sklearn_tags__()
        estimator_tags = get_tags(self._get_estimator())
        tags.input_tags.sparse = estimator_tags.input_tags.sparse
        tags.input_tags.allow_nan = estimator_tags.input_tags.allow_nan
        return tags
