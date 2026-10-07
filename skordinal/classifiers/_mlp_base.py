"""Base class for Multi-Layer Perceptron classifiers."""

from abc import ABC, abstractmethod
from numbers import Integral, Real

import numpy as np
import scipy.optimize
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.utils import check_random_state, compute_sample_weight
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from skordinal.utils.validation import check_ordinal_targets


class MLPBaseClassifier(ClassifierMixin, BaseEstimator, ABC):
    """Base class for Multi-Layer Perceptron classifiers.

    Parameters
    ----------
    n_hidden_layers : int, default=1
        Number of hidden layers in the network.

    n_hidden_units : int, default=64
        Number of neurons per hidden layer.

    alpha : float, default=0.01
        L2 regularization parameter.

    class_weight : dict, "balanced", or None, default=None
        Weights associated with classes.

    max_iter : int, default=500
        Maximum number of iterations for the solver.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for weight initialization.

    verbose : bool, default=False
        Whether to print progress messages to stdout during training.
    """

    _parameter_constraints: dict = {
        "n_hidden_layers": [Interval(Integral, 1, None, closed="left")],
        "n_hidden_units": [Interval(Integral, 1, None, closed="left")],
        "alpha": [Interval(Real, 0.0, None, closed="left")],
        "class_weight": [None, dict, StrOptions({"balanced"})],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state", None],
        "verbose": ["boolean"],
    }

    def __init__(
        self,
        n_hidden_layers=1,
        n_hidden_units=64,
        alpha=0.01,
        class_weight=None,
        max_iter=500,
        random_state=None,
        verbose=False,
    ):
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_units = n_hidden_units
        self.alpha = alpha
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

    @abstractmethod
    def _initialize_parameters(self, rng: np.random.RandomState) -> np.ndarray:
        """Initialize and return all network weights as a single 1D array.

        Parameters
        ----------
        rng : RandomState instance
            Random number generator for reproducible weight initialization.
        """
        pass

    @abstractmethod
    def _unpack_parameters(self, params: np.ndarray):
        """Unpack the 1D parameter array into specific layer weights/biases."""
        pass

    @abstractmethod
    def _cost_and_grad(
        self,
        params: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        sample_weight: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Compute the cost and gradients.

        Parameters
        ----------
        params : ndarray
            1D array containing all unconstrained parameters.
        X : ndarray of shape (n_samples, n_features)
            Training data.
        Y : ndarray of shape (n_samples, n_classes)
            Target matrix.
        sample_weight : ndarray of shape (n_samples,)
            Sample weights.

        Returns
        -------
        J : float
            The scalar cost (loss + L2 penalty).
        grad : ndarray
            The gradient of the cost with respect to all parameters (1D array).
        """
        pass

    @abstractmethod
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute class probabilities using the optimized parameters (internal)."""
        pass

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit the model using L-BFGS-B optimizer.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        Y_target : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns a fitted instance of self.
        """
        X, y = validate_data(self, X=X, y=y, reset=True)

        self.classes_, y_encoded = check_ordinal_targets(y)

        self.num_classes_ = len(self.classes_)
        self.input_shape_ = X.shape[1]
        n_samples = X.shape[0]

        Y_target = np.zeros((n_samples, self.num_classes_))
        Y_target[np.arange(n_samples), y_encoded] = 1.0

        sample_weight = np.ones(n_samples)
        if self.class_weight is not None:
            sw = compute_sample_weight(class_weight=self.class_weight, y=y)
            sample_weight *= sw

        rng = check_random_state(self.random_state)

        params0 = self._initialize_parameters(rng)

        res = scipy.optimize.minimize(
            fun=self._cost_and_grad,
            x0=params0,
            args=(X, Y_target, sample_weight),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iter},
        )

        self.loss_ = res.fun
        self.n_iter_ = res.nit

        self._unpack_parameters(res.x)

        return self

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities.
        """
        check_is_fitted(self)
        X = validate_data(self, X=X, reset=False)
        return self._predict_proba(X)

    def predict(self, X):
        """Predict the class for the samples in X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted classes.
        """
        probas = self.predict_proba(X)
        indices = np.argmax(probas, axis=1)

        return self.classes_[indices]
