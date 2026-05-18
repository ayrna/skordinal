"""Neural Network based on Proportional Odd Model (NNPOM)."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
import scipy
from numpy.typing import ArrayLike
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted

from skordinal.utils._sklearn_compat import validate_data
from skordinal.utils.validation import check_ordinal_targets


class NNPOM(ClassifierMixin, BaseEstimator):
    """Neural Network based on Proportional Odd Model (NNPOM).

    This class implements a neural network model for ordinal regression. The model has
    one hidden layer with "n_hidden" neurons and one output layer with only one neuron
    but as many thresholds as the number of classes minus one. The standard POM model
    is applied in this neuron to have probabilistic outputs.

    The learning is based on iRProp+ algorithm and the implementation provided by
    Roberto Calandra in his toolbox Rprop Toolbox for MATLAB:
    http://www.ias.informatik.tu-darmstadt.de/Research/RpropToolbox

    The model is adjusted by minimizing cross entropy. A regularization parameter
    ``alpha`` is included based on L2, and the number of iterations is specified
    by the "max_iter" parameter.

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
        Hidden layer weights (with bias)

    theta2_ : ndarray of shape (1, n_hidden)
        Output layer weights (without bias, the biases will be the thresholds)

    thresholds_ : ndarray of shape (n_classes - 1, 1)
        Class thresholds parameters

    References
    ----------
    .. [1] P. McCullagh, "Regression models for ordinal data", Journal of the
           Royal Statistical Society. Series B (Methodological), vol. 42, no. 2,
           pp. 109-142, 1980.

    .. [2] M. J. Mathieson, "Ordinal models for neural networks", in Proc. 3rd Int.
           Conf. Neural Netw. Capital Markets, 1996, pp. 523-536.

    .. [3] P.A. Gutiérrez, M. Pérez-Ortiz, J. Sánchez-Monedero, F. Fernández-Navarro
           and C. Hervás-Martínez, "Ordinal regression methods: survey and experimental
           study", IEEE Transactions on Knowledge and Data Engineering, Vol. 28. Issue
           1, 2016,
           https://doi.org/10.1109/TKDE.2015.2457911

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
    def fit(self, X: ArrayLike, y: ArrayLike) -> NNPOM:
        """Fit the model to data matrix X and target(s) y.

        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.

        y : array-like of shape (n_samples,)
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

        # Recode y to Y using nominal coding
        Y = 1 * (
            np.tile(y_1indexed, (1, n_classes))
            == np.tile(np.arange(1, n_classes + 1)[np.newaxis, :], (n_samples, 1))
        )

        # Hidden layer weights (with bias)
        initial_theta1 = self._rand_initialize_weights(
            self.n_features_in_ + 1, self.n_hidden, rng
        )
        # Output layer weights (without bias, the biases will be the thresholds)
        initial_theta2 = self._rand_initialize_weights(self.n_hidden, 1, rng)
        # Class thresholds parameters
        initial_thresholds = self._rand_initialize_weights((n_classes - 1), 1, rng)

        # Pack parameters
        initial_nn_params = np.concatenate(
            (
                initial_theta1.flatten(order="F"),
                initial_theta2.flatten(order="F"),
                initial_thresholds.flatten(order="F"),
            ),
            axis=0,
        )[:, np.newaxis]

        results_optimization = scipy.optimize.fmin_l_bfgs_b(
            func=self._nnpom_cost_function,
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

        # Unpack the parameters
        theta1, theta2, thresholds_param = self._unpack_parameters(
            nn_params, self.n_features_in_, self.n_hidden, n_classes
        )

        self.theta1_ = theta1
        self.theta2_ = theta2
        self.thresholds_ = self._convert_thresholds(thresholds_param, n_classes)

        # Scikit-learn compatibility
        self.n_layers_ = 3
        self.n_outputs_ = n_classes - 1
        self.out_activation_ = "logistic"

        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Perform classification on samples in X.

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

        a1 = np.append(np.ones((n_samples, 1)), X, axis=1)
        z2 = np.matmul(a1, self.theta1_.T)
        a2 = expit(z2)
        projected = np.matmul(a2, self.theta2_.T)

        z3 = np.tile(self.thresholds_, (n_samples, 1)) - np.tile(
            projected, (1, n_classes - 1)
        )
        a3T = expit(z3)
        a3 = np.append(a3T, np.ones((n_samples, 1)), axis=1)
        a3[:, 1:] = a3[:, 1:] - a3[:, 0:-1]
        y_pred = self.classes_[a3.argmax(1)]

        return y_pred

    def _unpack_parameters(
        self,
        nn_params: np.ndarray,
        n_features: int,
        n_hidden: int,
        n_classes: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get theta1, theta2 and thresholds_param from nn_params.

        Parameters
        ----------
        nn_params : ndarray of shape ((n_features + 1) * n_hidden + n_hidden +
                                      (n_classes - 1))
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

        theta2 : ndarray of shape (1, n_hidden)
            The weights between the hidden layer and the output layer (biases are not
            included as they are the thresholds).

        thresholds_param : ndarray of shape (n_classes - 1, 1)
            Classification thresholds.

        """
        n_theta1 = n_hidden * (n_features + 1)
        theta1 = np.reshape(
            nn_params[0:n_theta1], (n_hidden, (n_features + 1)), order="F"
        )

        n_theta2 = n_hidden
        theta2 = np.reshape(
            nn_params[n_theta1 : (n_theta1 + n_theta2)], (1, n_hidden), order="F"
        )

        thresholds_param = np.reshape(
            nn_params[(n_theta1 + n_theta2) :], ((n_classes - 1), 1), order="F"
        )

        return theta1, theta2, thresholds_param

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

    def _convert_thresholds(
        self,
        thresholds_param: np.ndarray,
        n_classes: int,
    ) -> np.ndarray:
        """Transform thresholds to perform unconstrained optimization.

        thresholds(1) = thresholds_param(1)
        thresholds(2) = thresholds_param(1) + thresholds_param(2)**2
        thresholds(3) = thresholds_param(1) + thresholds_param(2)**2 +
                        thresholds_param(3)**2

        Parameters
        ----------
        thresholds_param : ndarray of shape (n_classes - 1, 1)
            Contains the original value of the thresholds between classes

        n_classes : int
            Number of classes.

        Returns
        -------
        thresholds : ndarray of shape (n_classes - 1, 1)
            Thresholds of the line

        """
        # Threshold ^2 element by element
        thresholds_pquad = thresholds_param**2

        # Gets row-array containing the thresholds
        thresholds = np.reshape(
            np.multiply(
                np.tile(
                    np.concatenate(
                        (thresholds_param[0:1], thresholds_pquad[1:]), axis=0
                    ),
                    (1, n_classes - 1),
                ).T,
                np.tril(np.ones((n_classes - 1, n_classes - 1))),
            ).sum(axis=1),
            (n_classes - 1, 1),
        ).T

        return thresholds

    def _nnpom_cost_function(
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
        nn_params : ndarray of shape ((n_features + 1) * n_hidden + n_hidden +
                                      (n_classes - 1))
            Array that is a column vector. It stores the values of theta1, theta2 and
            thresholds_param, all of them together in an array in this order.

        n_features : int
            Number of nodes in the input layer of the neural network model.

        n_hidden : int
            Number of nodes in the hidden layer of the neural network model.

        n_classes : int
            Number of classes.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training patterns array, where n_samples is the number of samples and
            n_features is the number of features.

        Y : array-like of shape (n_samples,)
            Target vector relative to X.

        alpha : float
            Regularization parameter.

        Returns
        -------
        J : float
            Cost function (updated weight matrix).

        grad : ndarray of shape ((n_features + 1) * n_hidden + n_hidden +
                                 (n_classes - 1))
            Error gradient of each weight of each layer.

        """
        # Unroll all the parameters
        nn_params = nn_params.reshape((nn_params.shape[0], 1))

        theta1, theta2, thresholds_param = self._unpack_parameters(
            nn_params, n_features, n_hidden, n_classes
        )

        # Convert thresholds
        thresholds = self._convert_thresholds(thresholds_param, n_classes)

        # Setup some useful variables
        n_samples = np.size(X, 0)

        # Neural Network model
        a1 = np.append(np.ones((n_samples, 1)), X, axis=1)
        z2 = np.matmul(a1, theta1.T)
        a2 = expit(z2)

        z3 = np.tile(thresholds, (n_samples, 1)) - np.tile(
            np.matmul(a2, theta2.T), (1, n_classes - 1)
        )
        a3T = expit(z3)
        a3 = np.append(a3T, np.ones((n_samples, 1)), axis=1)
        h = np.concatenate(
            (a3[:, 0].reshape((a3.shape[0], 1)), a3[:, 1:] - a3[:, 0:-1]), axis=1
        )

        # Guard against zero probabilities: log(0) and -1/0 would produce NaN.
        out = np.maximum(h, 1e-15)

        # Calculate penalty (L2 regularization)
        p = np.sum((theta1[:, 1:] ** 2).sum() + (theta2[:, 0:] ** 2).sum())

        # Cross entropy
        J = np.sum(-np.log(out[np.where(Y == 1)]), axis=0) / n_samples + alpha * p / (
            2 * n_samples
        )

        # Error derivative
        error_der = np.zeros(Y.shape)
        error_der[np.where(Y != 0)] = np.divide(
            -Y[np.where(Y != 0)], out[np.where(Y != 0)]
        )

        # Calculate sigmas
        f_gradients = np.multiply(a3T, (1 - a3T))
        g_gradients = np.multiply(
            error_der,
            np.concatenate(
                (
                    f_gradients[:, 0].reshape(-1, 1),
                    (f_gradients[:, 1:] - f_gradients[:, :-1]),
                    -f_gradients[:, -1].reshape(-1, 1),
                ),
                axis=1,
            ),
        )
        sigma3 = -np.sum(g_gradients, axis=1)[:, np.newaxis]
        sigma2 = np.multiply(np.multiply(np.matmul(sigma3, theta2), a2), (1 - a2))

        # Accumulate gradients
        delta_1 = np.matmul(sigma2.T, a1)
        delta_2 = np.matmul(sigma3.T, a2)

        # Calculate regularized gradient
        p1 = (alpha / n_samples) * np.concatenate(
            (np.zeros((np.size(theta1, axis=0), 1)), theta1[:, 1:]), axis=1
        )
        p2 = (alpha / n_samples) * theta2[:, 0:]
        theta1_grad = delta_1 / n_samples + p1
        theta2_grad = delta_2 / n_samples + p2

        # Threshold gradients
        thresh_grad_matrix = np.multiply(
            np.concatenate(
                (
                    np.triu(np.ones((n_classes - 1, n_classes - 1))),
                    np.ones((n_classes - 1, 1)),
                ),
                axis=1,
            ),
            np.tile(g_gradients.sum(axis=0), (n_classes - 1, 1)),
        )

        original_shape = thresh_grad_matrix.shape
        thresh_grad_matrix = thresh_grad_matrix.flatten(order="F")

        thresh_grad_matrix[(n_classes)::n_classes] = thresh_grad_matrix.flatten(
            order="F"
        )[(n_classes)::n_classes] + np.multiply(
            error_der[:, 1 : (n_classes - 1)], f_gradients[:, 0 : (n_classes - 2)]
        ).sum(axis=0)

        thresh_grad_matrix = np.reshape(
            thresh_grad_matrix[:, np.newaxis], original_shape, order="F"
        )

        threshold_grad = thresh_grad_matrix.sum(axis=1)[:, np.newaxis] / n_samples
        threshold_grad[1:] = 2 * np.multiply(threshold_grad[1:], thresholds_param[1:])

        # Unroll gradients
        grad = np.concatenate(
            (
                theta1_grad.flatten(order="F"),
                theta2_grad.flatten(order="F"),
                threshold_grad.flatten(order="F"),
            ),
            axis=0,
        )

        return J, grad
