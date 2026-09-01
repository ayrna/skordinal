"""Neural Network with Ordered Partitions (NNOP)."""

from __future__ import annotations

import warnings
from numbers import Integral, Real

import numpy as np
import scipy
from numpy.typing import ArrayLike
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted, validate_data

from skordinal.utils.extmath import cumproba_to_proba, repair_cumproba
from skordinal.utils.validation import check_ordinal_targets


class NNOP(ClassifierMixin, BaseEstimator):
    """Neural Network with Ordered Partitions (NNOP).

    This model considers the OrderedPartitions coding scheme for the labels and a rule
    for decisions based on the first node whose output is higher than a predefined
    threshold (T=0.5, in our experiments). The model has one hidden layer with
    "n_hidden" neurons and one output layer with as many neurons as the number of
    classes minus one.

    The learning is based on iRProp+ algorithm and the implementation provided by
    Roberto Calandra in his toolbox Rprop Toolbox for MATLAB:
    http://www.ias.informatik.tu-darmstadt.de/Research/RpropToolbox

    The model is adjusted by minimizing mean squared error. A regularization parameter
    ``alpha`` is included based on L2, and the number of iterations is specified by the
    "iterations" parameter.

    Parameters
    ----------
    epsilon_init : float, default=0.5
        Range for initializing the weights.

    n_hidden : int, default=50
        Number of hidden neurons of the model.

    max_iter : int, default=500
        Maximum number of iterations. The solver iterates until convergence or this
        number of iterations.

    alpha : float, default=0.01
        Regularization parameter.

    random_state : int, RandomState instance, default=None
        Determines random number generation for weight initialization.
        Pass an int for reproducible results across multiple function calls.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels for each output.

    loss_ : float
        The current loss computed with the loss function.

    n_features_in_ : int
        Number of features seen during fit.

    n_iter_ : int
        The number of iterations the solver has run.

    n_layers_ : int
        Number of layers.

    n_outputs_ : int
        Number of outputs.

    out_activation_ : str
        Name of the output activation function.

    theta1_ : ndarray of shape (n_hidden, n_features + 1)
        Hidden layer weights (with bias).

    theta2_ : ndarray of shape (n_classes - 1, n_hidden + 1)
        Output layer weights.

    Notes
    -----
    If the L-BFGS-B solver stops before converging, a ``ConvergenceWarning``
    is raised and ``n_iter_`` reports the iterations actually run.

    ``predict`` uses the first-crossing rule on the raw cumulative outputs, so
    its output is not in general ``classes_[argmax(predict_proba(X), axis=1)]``;
    see ``predict`` for the rationale.

    This file is part of ORCA: https://github.com/ayrna/orca

    References
    ----------
    .. [1] J. Cheng, Z. Wang, and G. Pollastri, "A neural network approach to ordinal
           regression," in Proc. IEEE Int. Joint Conf. Neural Netw. (IEEE World Congr.
           Comput. Intell.), 2008, pp. 1279-1284.

    .. [2] P.A. Gutiérrez, M. Pérez-Ortiz, J. Sánchez-Monedero, F. Fernández-Navarro
           and C. Hervás-Martínez, "Ordinal regression methods: survey and
           experimental study", IEEE Transactions on Knowledge and Data
           Engineering, Vol. 28. Issue 1, 2016,
           http://dx.doi.org/10.1109/TKDE.2015.2457911

    Copyright
    ---------
    This software is released under the The GNU General Public License v3.0 licence
    available at http://www.gnu.org/licenses/gpl-3.0.html

    Authors
    -------
    Pedro Antonio Gutiérrez, María Pérez Ortiz, Javier Sánchez Monedero

    Citation
    --------
    If you use this code, please cite the associated paper
    http://www.uco.es/grupos/ayrna/orreview

    """

    _parameter_constraints: dict = {
        "epsilon_init": [Interval(Real, 0.0, None, closed="neither")],
        "n_hidden": [Interval(Integral, 1, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "alpha": [Interval(Real, 0.0, None, closed="neither")],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        epsilon_init: float = 0.5,
        n_hidden: int = 50,
        max_iter: int = 500,
        alpha: float = 0.01,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.epsilon_init = epsilon_init
        self.n_hidden = n_hidden
        self.max_iter = max_iter
        self.alpha = alpha
        self.random_state = random_state

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: ArrayLike, y: ArrayLike) -> NNOP:
        """Fit the model to data matrix X and target(s) y.

        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Fitted estimator.

        Raises
        ------
        ValueError
            If parameters are invalid or data has wrong format.

        """
        X, y = validate_data(self, X, y)
        self.classes_, y_encoded = check_ordinal_targets(y)
        rng = check_random_state(self.random_state)

        # Aux variables
        y_1indexed = (y_encoded + 1)[:, np.newaxis]
        n_classes = len(self.classes_)
        n_samples = X.shape[0]
        # n_features_in_ is set by validate_data; do not assign manually.

        # Recode y to Y using ordinalPartitions coding
        Y = 1 * (
            np.tile(y_1indexed, (1, n_classes))
            <= np.tile(np.arange(1, n_classes + 1)[np.newaxis, :], (n_samples, 1))
        )

        # Hidden layer weights (with bias)
        initial_theta1 = self._rand_initialize_weights(
            self.n_features_in_ + 1, self.n_hidden, rng
        )
        # Output layer weights
        initial_theta2 = self._rand_initialize_weights(
            self.n_hidden + 1, n_classes - 1, rng
        )

        # Pack parameters
        initial_nn_params = np.concatenate(
            (initial_theta1.flatten(order="F"), initial_theta2.flatten(order="F")),
            axis=0,
        )[:, np.newaxis]

        results_optimization = scipy.optimize.fmin_l_bfgs_b(
            func=self._objective,
            x0=initial_nn_params.ravel(),
            args=(
                self.n_features_in_,
                self.n_hidden,
                n_classes,
                X,
                Y,
                self.alpha,
            ),
            fprime=None,
            factr=1e3,
            maxiter=self.max_iter,
        )

        nn_params = results_optimization[0]
        self.loss_ = float(results_optimization[1])
        self.n_iter_ = int(results_optimization[2].get("nit", 0))

        if results_optimization[2].get("warnflag", 0) != 0:
            task = results_optimization[2].get("task", "")
            if isinstance(task, bytes):
                task = task.decode()
            warnings.warn(
                f"NNOP did not converge (stopped after {self.n_iter_} "
                f"iterations: {task}). Consider increasing max_iter, "
                "adjusting the regularization strength alpha, or "
                "standardizing X.",
                ConvergenceWarning,
                stacklevel=2,
            )

        # Unpack the parameters
        theta1, theta2 = self._unpack_parameters(
            nn_params, self.n_features_in_, self.n_hidden, n_classes
        )
        self.theta1_ = theta1
        self.theta2_ = theta2

        # Scikit-learn compatibility
        self.n_layers_ = 3
        self.n_outputs_ = n_classes - 1
        self.out_activation_ = "logistic"

        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Perform classification on samples in X.

        The predicted class is the first class whose raw cumulative
        estimate ``P(y <= k | x)`` exceeds 0.5 (the first-crossing
        rule). The isotonic repair applied by ``predict_proba`` is
        post-processing, not part of the model, so the result is not in
        general ``classes_[argmax(predict_proba(X), axis=1)]``; see the
        Notes on ``predict_proba``.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted classes.

        Raises
        ------
        NotFittedError
            If the model is not fitted yet.

        ValueError
            If input is invalid.

        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        cumproba = self._cumproba(X)
        a3 = np.multiply(
            np.where(np.append(cumproba, np.ones((n_samples, 1)), axis=1) > 0.5, 1, 0),
            np.tile(np.arange(1, n_classes + 1), (n_samples, 1)),
        )
        a3[np.where(a3 == 0)] = n_classes + 1
        y_pred = self.classes_[a3.min(axis=1).astype(int) - 1]

        return y_pred

    def predict_cumproba(self, X: ArrayLike) -> np.ndarray:
        """Cumulative class probabilities for each sample.

        The ``n_classes - 1`` output neurons independently estimate
        ``P(y <= k | x)`` under the ordered-partitions coding, so the
        raw outputs are not guaranteed to be non-decreasing across
        ``k``.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        cumproba : ndarray of shape (n_samples, n_classes - 1)
            Entry ``[i, k]`` is the estimated probability that sample
            ``i`` belongs to class ``k`` or lower. Isotonic repair
            enforces that each row is non-decreasing.

        Raises
        ------
        NotFittedError
            If the model is not fitted yet.

        ValueError
            If input is invalid.

        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return repair_cumproba(self._cumproba(X))

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Class probability estimates for each sample.

        Derives class probabilities from the repaired cumulative
        probability estimates via finite differencing.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Row-stochastic matrix of class probability estimates.

        Raises
        ------
        NotFittedError
            If the model is not fitted yet.

        ValueError
            If input is invalid.

        Notes
        -----
        ``predict`` applies the first-crossing rule to the raw cumulative
        estimates instead of taking the argmax of this method's output, so
        ``classes_[argmax(predict_proba(X), axis=1)]`` is not in general
        equal to ``predict(X)``. Users needing calibrated,
        argmax-consistent probabilities should wrap the estimator with
        ``sklearn.calibration.CalibratedClassifierCV``.

        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return cumproba_to_proba(self._cumproba(X), repair=True)

    def _cumproba(self, X: np.ndarray) -> np.ndarray:
        """Compute raw cumulative probabilities on pre-validated X."""
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        z2 = X_bias @ self.theta1_.T
        a2 = np.hstack([np.ones((z2.shape[0], 1)), expit(z2)])
        return expit(a2 @ self.theta2_.T)

    def _unpack_parameters(
        self,
        nn_params: np.ndarray,
        n_features: int,
        n_hidden: int,
        n_classes: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get theta1 and theta2 back from nn_params.

        Parameters
        ----------
        nn_params : ndarray of shape ((n_features+1)*n_hidden + n_hidden +
                                      (n_classes-1))
            Array that is a column vector. It stores the values of theta1, theta2 and
            thresholds_param, all of them together in an array in this order.

        n_features : int
            Number of nodes in the input layer of the neural network model.

        n_hidden : int
            Number of nodes in the hidden layer of the neural network model.

        n_classes : int
            Number of classes.

        Returns
        -------
        theta1 : ndarray of shape (n_hidden, n_features + 1)
            The weights between the input layer and the hidden layer (with biases
            included).

        theta2 : ndarray of shape (n_classes - 1, n_hidden + 1)
            The weights between the hidden layer and the output layer.

        """
        n_theta1 = n_hidden * (n_features + 1)
        theta1 = np.reshape(
            nn_params[0:n_theta1], (n_hidden, (n_features + 1)), order="F"
        )

        theta2 = np.reshape(
            nn_params[n_theta1:], (n_classes - 1, n_hidden + 1), order="F"
        )

        return theta1, theta2

    def _rand_initialize_weights(
        self,
        L_in: int,
        L_out: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Initialize layer weights randomly.

        Randomly initialize the weights of a layer with L_in incoming connections and
        L_out outgoing connections.

        Parameters
        ----------
        L_in : int
            Number of inputs of the layer.

        L_out : int
            Number of outputs of the layer.

        rng : numpy.random.RandomState
            Random number generator used for weight initialization.

        Returns
        -------
        W : ndarray of shape (L_out, L_in)
            Array with the weights of each synaptic relationship between nodes.

        """
        W = rng.rand(L_out, L_in) * 2 * self.epsilon_init - self.epsilon_init

        return W

    def _objective(
        self,
        nn_params: np.ndarray,
        n_features: int,
        n_hidden: int,
        n_classes: int,
        X: np.ndarray,
        Y: np.ndarray,
        alpha: float,
    ) -> tuple[float, np.ndarray]:
        """Implement the cost function and obtain the corresponding derivatives.

        Parameters
        ----------
        nn_params : ndarray of shape ((n_features+1)*n_hidden + n_hidden)
            Array that is a column vector. It stores the values of Theta1 and Theta2,
            all of them together in an array in this order.

        n_features : int
            Number of nodes in the input layer of the neural network model.

        n_hidden : int
            Number of nodes in the hidden layer of the neural network model.

        n_classes : int
            Number of classes.

        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training patterns array, where n_samples is the number of samples and
            n_features is the number of features

        Y : array-like of shape (n_samples,)
            Target vector relative to X

        alpha : float
            Regularization parameter.

        Returns
        -------
        J : float
            Matrix with cost function (updated weight matrix).

        grad : ndarray
            Array with the error gradient of each weight of each layer.

        """
        # Unroll all the parameters
        theta1, theta2 = self._unpack_parameters(
            nn_params, n_features, n_hidden, n_classes
        )

        # Setup some useful variables
        n_samples = np.size(X, 0)

        # Neural Network model
        a1 = np.append(np.ones((n_samples, 1)), X, axis=1)
        z2 = np.matmul(a1, theta1.T)
        a2 = np.append(np.ones((n_samples, 1)), expit(z2), axis=1)
        z3 = np.matmul(a2, theta2.T)
        h = np.append(expit(z3), np.ones((n_samples, 1)), axis=1)

        # Final output
        out = h

        # Calculate penalty (regularización L2)
        p = np.sum((theta1[:, 1:] ** 2).sum() + (theta2[:, 1:] ** 2).sum())

        # MSE
        J = np.sum((out - Y) ** 2).sum() / (2 * n_samples) + alpha * p / (2 * n_samples)

        # MSE
        error_der = out - Y

        # Calculate sigmas
        sigma3 = np.multiply(np.multiply(error_der, h), (1 - h))
        sigma3 = sigma3[:, :-1]

        sigma2 = np.multiply(np.multiply(np.matmul(sigma3, theta2), a2), (1 - a2))
        sigma2 = sigma2[:, 1:]

        # Accumulate gradients
        delta_1 = np.matmul(sigma2.T, a1)
        delta_2 = np.matmul(sigma3.T, a2)

        # Calculate regularized gradient
        p1 = (alpha / n_samples) * np.concatenate(
            (np.zeros((np.size(theta1, axis=0), 1)), theta1[:, 1:]), axis=1
        )
        p2 = (alpha / n_samples) * np.concatenate(
            (np.zeros((np.size(theta2, axis=0), 1)), theta2[:, 1:]), axis=1
        )
        theta1_grad = delta_1 / n_samples + p1
        theta2_grad = delta_2 / n_samples + p2

        # Unroll gradients
        grad = np.concatenate(
            (theta1_grad.flatten(order="F"), theta2_grad.flatten(order="F")), axis=0
        )

        return J, grad
