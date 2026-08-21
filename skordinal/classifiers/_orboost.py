"""Ordinal boosting classifier (ORBoost)."""

from numbers import Integral

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from skordinal.utils.validation import check_ordinal_targets

from . import _orensemble as _orensemble_lib  # type: ignore[attr-defined]

# Backend "bag" codes: tens digit 3/4 = margins to the adjacent/all
# thresholds; units digit 1 adds a 1e-32 weight stabiliser, 0 does not
_BAG_CODES = {
    ("lr", False): 30,
    ("lr", True): 31,
    ("full", False): 40,
    ("full", True): 41,
}

# Backend weak-learner codes: 100 = stump, 200 = perceptron
_BASE_CODES = {
    "stump": 100,
    "perceptron": 200,
}


class ORBoost(ClassifierMixin, BaseEstimator):
    """ORBoost ordinal boosting classifier.

    ORBoost [1]_ fits a thresholded ensemble of ``n_estimators`` weak
    learners by minimising an exponential loss over ordinal ranks. Each
    boosting round adds one weak learner and re-estimates the ordered
    thresholds that convert the aggregated output into a rank.

    Parameters
    ----------
    n_estimators : int, default=200
        Number of boosting rounds. Must be >= 1.

    loss_form : {"lr", "full"}, default="lr"
        Loss formulation. ``"lr"`` penalises each sample's margins to
        the two thresholds adjacent to its rank; ``"full"`` penalises
        its margins to all thresholds.

    base_learner : {"stump", "perceptron"}, default="stump"
        Weak learner type. ``"stump"`` trains a decision stump;
        ``"perceptron"`` trains a linear perceptron by random
        coordinate descent.

    weight_reg : bool, default=True
        If ``True``, adds a small constant (``1e-32``) to both sides of
        the weak-learner weight computation, preventing division by
        zero and ``log(0)`` when one side of an ordinal split carries
        no sample mass.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique ordinal class labels seen during ``fit``, sorted in
        ascending order.

    n_features_in_ : int
        Number of features seen during ``fit``.

    feature_names_in_ : ndarray of shape (n_features_in_,), dtype object
        Feature names seen during ``fit``. Defined only when ``X`` is a
        ``pandas.DataFrame``.

    thresholds_ : ndarray of shape (n_classes - 1,)
        Fitted ordered thresholds partitioning the ensemble score into class
        regions.  ``predict`` assigns
        ``classes_[(predict_projection(X)[:, None] >= thresholds_).sum(axis=1)]``.

    model_ : dict
        Model state returned by the C++ backend after fitting, with
        keys ``'model'`` (learned parameters) and ``'params'``
        (training configuration). Treat as read-only; modifying this
        dict may corrupt subsequent calls to ``predict``.

    See Also
    --------
    REDSVM : SVM-based ordinal classifier with a C++ backend.

    Notes
    -----
    ``predict`` returns hard class labels; ORBoost exposes no
    ``predict_proba`` or ``decision_function``.

    ORBoost has no ``random_state`` parameter. Fitting with
    ``base_learner="stump"`` is deterministic. The perceptron learner
    draws from a process-global, unseeded random stream: the first fit
    in a fresh process is reproducible, but later fits in the same
    process may differ.

    Predictions with ``base_learner="perceptron"`` are sensitive to
    feature scale, so standardising features beforehand is
    recommended; the stump learner is invariant to monotone
    per-feature transformations.

    The confidence-rated weak learners of [1]_ are not exposed.

    References
    ----------
    .. [1] H.-T. Lin and L. Li, "Large-margin thresholded ensembles for
       ordinal regression: theory and practice," in Proc. Algorithmic
       Learning Theory (ALT), Lecture Notes in Computer Science,
       vol. 4264, pp. 319-333, Springer, 2006.
       https://doi.org/10.1007/11894841_26

    Examples
    --------
    >>> from skordinal.datasets import make_ordinal_classification
    >>> from skordinal.classifiers import ORBoost
    >>> X, y = make_ordinal_classification(n_samples=100, random_state=0)
    >>> clf = ORBoost(n_estimators=20).fit(X, y)
    >>> clf.predict(X[:3])  # doctest: +SKIP
    array([3, 4, 2])
    """

    _parameter_constraints: dict = {
        "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "loss_form": [StrOptions({"lr", "full"})],
        "base_learner": [StrOptions({"stump", "perceptron"})],
        "weight_reg": ["boolean"],
    }

    def __init__(
        self,
        n_estimators=200,
        loss_form="lr",
        base_learner="stump",
        weight_reg=True,
    ):
        self.n_estimators = n_estimators
        self.loss_form = loss_form
        self.base_learner = base_learner
        self.weight_reg = weight_reg

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit ORBoost on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training patterns.

        y : array-like of shape (n_samples,)
            Ordinal target labels.

        Returns
        -------
        self : ORBoost
            Fitted estimator.

        Raises
        ------
        ValueError
            If ``y`` contains fewer than 2 unique classes.
        """
        X, y = validate_data(self, X, y, dtype=np.float64)
        self.classes_, y_enc = check_ordinal_targets(y)
        K = len(self.classes_)

        bag_code = _BAG_CODES[(self.loss_form, bool(self.weight_reg))]
        base_code = _BASE_CODES[self.base_learner]

        # Backend expects 1-indexed float labels and plain Python lists
        self.model_ = _orensemble_lib.fit(
            X.tolist(),
            (y_enc + 1).astype(float).tolist(),
            bag_code,
            base_code,
            K,
            self.n_estimators,
        )
        # reported at the aggregation size predict uses
        self.thresholds_ = np.asarray(self.model_["thresholds"], dtype=np.float64)
        return self

    def predict(self, X):
        """Predict ordinal class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels drawn from ``self.classes_``.

        Raises
        ------
        NotFittedError
            If the model is not fitted yet.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        y_pred, _ = _orensemble_lib.predict(X.tolist(), self.model_)
        # Map the backend's 1-indexed ranks back to labels in classes_
        return self.classes_[np.array(y_pred, dtype=int) - 1]

    def predict_projection(self, X):
        """Return the raw latent projection for each sample.

        The aggregated ensemble score ``f(x)`` is the raw latent
        projection (ordinal-axis score) that ``thresholds_`` partitions
        into class regions, on the same scale.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        projection : ndarray of shape (n_samples,)
            Raw ensemble score for each sample.

        Raises
        ------
        NotFittedError
            If the model is not fitted yet.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        _, projection = _orensemble_lib.predict(X.tolist(), self.model_)
        return np.asarray(projection, dtype=np.float64)
