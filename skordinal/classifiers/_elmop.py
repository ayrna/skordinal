"""Extreme Learning Machine for Ordered Partitions (ELMOP)."""

from numbers import Integral

import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted

from skordinal.utils._sklearn_compat import validate_data
from skordinal.utils.extmath import losses_to_proba, proba_to_cumproba
from skordinal.utils.validation import check_ordinal_targets


class ELMOP(ClassifierMixin, BaseEstimator):
    """Extreme Learning Machine for Ordered Partitions.

    ELMOP [1]_ combines a single-hidden-layer network, whose input weights
    and biases are drawn randomly and then frozen, with the
    ordered-partitions coding of ordinal targets. The output layer is
    obtained in closed form via a single least-squares solve, so no
    iterative optimisation is required.

    Each of the ``K - 1`` output neurons solves one binary partition — the
    classes with rank above ``j`` vs. the rest — in ``{-1, +1}`` regression
    form. At prediction time the joint output is decoded to a single label
    by minimising an exponential loss against the ``(K, K - 1)`` code
    matrix.

    Parameters
    ----------
    n_hidden : int, default=50
        Number of neurons in the hidden layer. Must be >= 1.

    activation : {"sigmoid", "hardlim"}, default="sigmoid"
        Activation function applied to the pre-activation values.
        ``"sigmoid"`` is the logistic sigmoid; ``"hardlim"`` is the
        Heaviside step function (outputs 0 or 1).

    random_state : int, RandomState instance or None, default=None
        Controls the random initialisation of the input weights and biases.
        Pass an int for reproducible results across multiple calls.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique ordinal class labels seen during ``fit``, sorted in ascending
        order.

    n_features_in_ : int
        Number of features seen during ``fit``.

    feature_names_in_ : ndarray of shape (n_features_in_,), dtype object
        Feature names seen during ``fit``. Defined only when ``X`` is a
        ``pandas.DataFrame``.

    input_weights_ : ndarray of shape (n_hidden, n_features_in_)
        Random input-to-hidden weight matrix, drawn at ``fit`` and kept
        fixed. Each entry is uniform on ``[-1, 1]``.

    input_biases_ : ndarray of shape (n_hidden,)
        Random hidden-unit biases, drawn at ``fit`` and kept fixed. Each
        entry is uniform on ``[0, 1]``.

    output_weights_ : ndarray of shape (n_hidden, n_classes - 1)
        Output weights computed by least-squares regression of the hidden
        activations onto the ordered-partitions target matrix.

    Notes
    -----
    The random input biases are added to the pre-activations before either
    activation function is applied.

    When ``n_hidden`` exceeds the number of training samples the hidden
    matrix is rank-deficient and ``numpy.linalg.lstsq`` returns the
    minimum-norm solution.

    The formulation in [1]_ uses ``K`` output neurons, the first of which
    regresses a constant all-positive column. That column adds the same
    term to the decoding loss of every candidate class, so it cannot
    change the prediction; this implementation drops it and works with
    ``K - 1`` partitions.

    The input weights are drawn on ``[-1, 1]`` with no data-dependent
    scaling, so results are sensitive to feature scale. Standardising
    features (e.g. with ``sklearn.preprocessing.StandardScaler``) before
    fitting is strongly recommended.

    References
    ----------
    .. [1] W.-Y. Deng, Q.-H. Zheng, S. Lian, L. Chen, X. Wang,
       "Ordinal extreme learning machine", Neurocomputing, vol. 74,
       no. 1-3, pp. 447-456, 2010.
       https://doi.org/10.1016/j.neucom.2010.08.022

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.classifiers import ELMOP
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((80, 4))
    >>> y = np.repeat([1, 2, 3, 4], 20)
    >>> clf = ELMOP(n_hidden=50, random_state=0).fit(X, y)
    >>> clf.predict(X[:5])  # doctest: +SKIP
    array([1, 2, 3, 4, ...])
    """

    _parameter_constraints: dict = {
        "n_hidden": [Interval(Integral, 1, None, closed="left")],
        "activation": [StrOptions({"sigmoid", "hardlim"})],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        *,
        n_hidden=50,
        activation="sigmoid",
        random_state=None,
    ):
        self.n_hidden = n_hidden
        self.activation = activation
        self.random_state = random_state

    def _hidden(self, X):
        """Compute hidden-layer activations for pre-validated input X."""
        Z = X @ self.input_weights_.T + self.input_biases_
        if self.activation == "sigmoid":
            return expit(Z)
        return (Z >= 0.0).astype(np.float64)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit the ELMOP model to training data.

        Draws random input weights and biases, computes the hidden-layer
        activations, then solves for the output weights via least squares.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training patterns.

        y : array-like of shape (n_samples,)
            Ordinal target labels. Must contain at least 2 unique classes.

        Returns
        -------
        self : ELMOP
            Fitted estimator.

        Raises
        ------
        ValueError
            If ``y`` contains fewer than 2 unique classes.
        """
        X, y = validate_data(self, X, y, dtype=np.float64)
        self.classes_, y_enc = check_ordinal_targets(y)
        rng = check_random_state(self.random_state)

        d = X.shape[1]
        K = len(self.classes_)

        # Draw the frozen random feature map.
        self.input_weights_ = rng.uniform(-1.0, 1.0, size=(self.n_hidden, d))
        self.input_biases_ = rng.uniform(0.0, 1.0, size=(self.n_hidden,))

        # Build the ordered-partitions target matrix in {-1, +1}:
        # T[i, j] = +1 iff y_enc[i] >= j + 1.
        T = np.where(
            y_enc[:, np.newaxis] >= np.arange(1, K)[np.newaxis, :],
            1.0,
            -1.0,
        )  # (n, K-1)

        H = self._hidden(X)  # (n, n_hidden)
        self.output_weights_, *_ = np.linalg.lstsq(H, T, rcond=None)  # (n_hidden, K-1)

        return self

    def _losses(self, X):
        """Compute exponential decoding losses for pre-validated X."""
        K = len(self.classes_)
        H_test = self._hidden(X)  # (n_test, n_hidden)
        TY = H_test @ self.output_weights_  # (n_test, K-1)

        # Build the code matrix: C[k, j] = +1 if k >= j + 1 else -1.
        C = np.where(
            np.arange(K)[:, np.newaxis] >= np.arange(1, K)[np.newaxis, :],
            1.0,
            -1.0,
        )  # (K, K-1)

        # Sum the exponential loss of the outputs against each code row.
        losses = np.exp(-TY[:, np.newaxis, :] * C[np.newaxis, :, :]).sum(
            axis=2
        )  # (n_test, K)

        return losses

    def _proba(self, X):
        """Compute class probabilities from pre-validated X."""
        return losses_to_proba(self._losses(X))

    def predict(self, X):
        """Predict ordinal class labels for patterns in X.

        The predicted label is the class whose ordered-partitions code
        vector minimises the exponential decoding loss of the network
        outputs. Ties are broken toward the lowest class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input patterns.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels drawn from ``self.classes_``.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return self.classes_[self._losses(X).argmin(axis=1)]

    def predict_proba(self, X):
        """Return per-class probability estimates.

        Probabilities are derived from the per-class exponential decoding
        losses through a monotone decreasing transform, so smaller loss means
        higher probability and, apart from floating-point ties in the
        transformed scores, ``predict`` coincides with the argmax of
        ``predict_proba``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input patterns.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Non-negative class probabilities; each row sums to 1.0.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return self._proba(X)

    def predict_cumproba(self, X):
        """Cumulative class probabilities for each sample.

        Derived from ``predict_proba`` as the row-wise cumulative sum of
        all but the last column.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        cumproba : ndarray of shape (n_samples, n_classes - 1)
            Entry ``[i, k]`` is the estimated probability that sample ``i``
            belongs to class ``k`` or lower.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return proba_to_cumproba(self._proba(X))
