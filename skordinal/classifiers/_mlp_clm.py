"""Multi-Layer Perceptron using a Cumulative Link Model output."""

from numbers import Real

import numpy as np
from scipy.special import expit
from sklearn.utils._param_validation import Interval, StrOptions

from ._clm import CLM
from ._mlp_base import MLPBaseClassifier

# Added to the probabilities inside the log-likelihood so the loss and its
# gradient stay finite when a sample is confidently misclassified. Unlike a
# hard floor, the smoothed loss keeps the gradient consistent with the loss
_PROBA_EPS = 1e-15


class MLPCLMClassifier(MLPBaseClassifier):
    """Multi-Layer Perceptron using a Cumulative Link Model output.

    The hidden layers project each sample onto a single latent scalar ``z``,
    which a :class:`CLM` output layer turns into the probabilities of the
    ordered classes through ``P(Y <= k | z) = F(theta_k - z)``. The network
    and the thresholds are trained jointly by minimising the weighted
    negative log-likelihood plus an L2 penalty on the weights.

    Parameters
    ----------
    n_hidden_layers : int, default=1
        Number of hidden layers in the network.

    n_hidden_units : int, default=4
        Number of neurons per hidden layer.

    alpha : float, default=0.01
        L2 regularization parameter applied to the weights. Biases and
        thresholds are not penalised.

    activation : {'sigmoid', 'tanh', 'relu'}, default='sigmoid'
        Activation function for the hidden layers.

    link_function : {'logit', 'probit', 'cloglog'}, default='logit'
        Link function of the CLM output layer.

    min_distance : float, default=0.0
        Minimum distance between consecutive thresholds.

    class_weight : dict, "balanced", or None, default=None
        Weights associated with classes.

    max_iter : int, default=500
        Maximum number of iterations of the L-BFGS-B solver.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for weight initialization.

    tol : float, default=1e-5
        Convergence tolerance forwarded to L-BFGS-B as ``gtol``. Must be
        strictly positive.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels, in ordinal order.

    n_classes_ : int
        Number of classes.

    n_features_in_ : int
        Number of features seen during fit.

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit. Defined only when ``X`` has
        feature names that are all strings.

    clm_ : CLM
        The output layer used by the network.

    W_ : list of ndarray
        Weight matrices of the hidden layers.

    b_ : list of ndarray
        Bias vectors of the hidden layers.

    w_out_ : ndarray of shape (n_hidden_units, 1)
        Weights of the projection onto the latent scalar.

    t_ : ndarray of shape (n_classes - 1,)
        Unconstrained threshold parameters of the CLM layer.

    thresholds_ : ndarray of shape (n_classes - 1,)
        Ordered thresholds of the CLM layer.

    loss_ : float
        Value of the objective at the end of training.

    n_iter_ : int
        Number of iterations run by the solver.

    References
    ----------
    .. [1] V. M. Vargas, P. A. Gutiérrez, and C. Hervás-Martínez,
       "Cumulative link models for deep ordinal classification,"
       Neurocomputing, vol. 401, pp. 48-58, 2020.

    .. [2] P. McCullagh, "Regression models for ordinal data",
       Journal of the Royal Statistical Society. Series B (Methodological),
       vol. 42, no. 2, pp. 109-142, 1980.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.classifiers import MLPCLMClassifier
    >>> rng = np.random.default_rng(42)
    >>> X = rng.standard_normal((100, 5))
    >>> y = np.sort(rng.integers(1, 5, 100))
    >>> clf = MLPCLMClassifier(n_hidden_units=8, max_iter=10, random_state=42)
    >>> clf.fit(X, y) # doctest: +SKIP
    MLPCLMClassifier(max_iter=10, n_hidden_units=8, random_state=42)
    >>> clf.predict(X[:3]) # doctest: +SKIP
    array([1, 1, 1])
    """

    _parameter_constraints = MLPBaseClassifier._parameter_constraints.copy()
    _parameter_constraints.update(
        {
            "activation": [StrOptions({"sigmoid", "tanh", "relu"})],
            "link_function": [StrOptions({"logit", "probit", "cloglog"})],
            "min_distance": [Interval(Real, 0.0, None, closed="left")],
        }
    )

    def __init__(
        self,
        n_hidden_layers=1,
        n_hidden_units=4,
        alpha=0.01,
        activation="sigmoid",
        link_function="logit",
        min_distance=0.0,
        class_weight=None,
        max_iter=500,
        random_state=None,
        tol=1e-5,
    ):
        self.activation = activation
        self.link_function = link_function
        self.min_distance = min_distance
        super().__init__(
            n_hidden_layers=n_hidden_layers,
            n_hidden_units=n_hidden_units,
            alpha=alpha,
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state,
            tol=tol,
        )

    def _initialize_parameters(self, rng: np.random.RandomState) -> np.ndarray:
        """Initialize hidden weights, projection weights and thresholds.

        Hidden weights use He-uniform initialization for ReLU and
        Glorot-uniform otherwise; biases start at zero and thresholds are
        spread evenly over ``[-2, 2]``.

        Parameters
        ----------
        rng : RandomState instance
            Random number generator passed by the base class.

        Returns
        -------
        params : ndarray
            1D array with the hidden weights and biases, the projection
            weights and the unconstrained thresholds, in that order.
        """
        self.clm_ = CLM(
            self.n_classes_,
            link_function=self.link_function,
            min_distance=self.min_distance,
        )

        params = []
        input_dim = self.n_features_in_
        for _ in range(self.n_hidden_layers):
            if self.activation == "relu":
                limit = np.sqrt(6 / input_dim)
            else:
                limit = np.sqrt(6 / (input_dim + self.n_hidden_units))
            W = rng.uniform(-limit, limit, (input_dim, self.n_hidden_units))
            params.extend([W.ravel(), np.zeros(self.n_hidden_units)])
            input_dim = self.n_hidden_units

        limit = np.sqrt(6 / (input_dim + 1))
        params.append(rng.uniform(-limit, limit, input_dim))
        params.append(self.clm_.init_params())

        return np.concatenate(params)

    def _split_parameters(self, params: np.ndarray):
        """Split the 1D parameter array into the layers' arrays.

        Parameters
        ----------
        params : ndarray
            1D array containing all unconstrained parameters of the model.

        Returns
        -------
        W : list of ndarray
            Weight matrices for each hidden layer.
        b : list of ndarray
            Bias vectors for each hidden layer.
        w_out : ndarray of shape (n_units_last_hidden, 1)
            Weights of the projection onto the latent scalar.
        t : ndarray of shape (n_classes - 1,)
            Unconstrained threshold parameters of the CLM layer.
        """
        idx = 0
        W = []
        b = []
        input_dim = self.n_features_in_
        for _ in range(self.n_hidden_layers):
            size_W = input_dim * self.n_hidden_units
            W.append(params[idx : idx + size_W].reshape(input_dim, self.n_hidden_units))
            idx += size_W
            b.append(params[idx : idx + self.n_hidden_units])
            idx += self.n_hidden_units
            input_dim = self.n_hidden_units

        w_out = params[idx : idx + input_dim].reshape(input_dim, 1)
        idx += input_dim

        t = params[idx:]
        if t.shape != (self.clm_.n_params,):
            raise ValueError(
                f"Expected {idx + self.clm_.n_params} parameters, got {params.size}."
            )

        return W, b, w_out, t

    def _unpack_parameters(self, params: np.ndarray):
        """Store the optimised parameters as fitted attributes."""
        self.W_, self.b_, self.w_out_, self.t_ = self._split_parameters(params)
        self.thresholds_ = self.clm_.thresholds(self.t_)

    def _activate(self, Z: np.ndarray) -> np.ndarray:
        if self.activation == "sigmoid":
            return expit(Z)
        if self.activation == "tanh":
            return np.tanh(Z)
        return np.maximum(0.0, Z)

    def _activation_derivative(self, A: np.ndarray) -> np.ndarray:
        """Derivative of the activation, expressed in terms of its output."""
        if self.activation == "sigmoid":
            return A * (1.0 - A)
        if self.activation == "tanh":
            return 1.0 - A**2
        return (A > 0.0).astype(float)

    def _forward(self, X, W, b, w_out):
        """Return the activations of every layer and the latent projection."""
        A = [X]
        for W_i, b_i in zip(W, b):
            A.append(self._activate(A[-1] @ W_i + b_i))
        return A, A[-1] @ w_out

    def _cost_and_grad(
        self,
        params: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        sample_weight: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Compute the CLM loss and its gradient by backpropagation.

        Parameters
        ----------
        params : ndarray
            1D array containing all unconstrained parameters.
        X : ndarray of shape (n_samples, n_features)
            Training data.
        Y : ndarray of shape (n_samples, n_classes)
            One-hot encoded targets.
        sample_weight : ndarray of shape (n_samples,)
            Sample weights.

        Returns
        -------
        J : float
            Weighted mean negative log-likelihood, with ``1e-15`` added to
            the probabilities for numerical stability, plus the L2 penalty.
        grad : ndarray
            Gradient of ``J`` with respect to ``params``.
        """
        weight_sum = np.sum(sample_weight)
        W, b, w_out, t = self._split_parameters(params)

        A, z = self._forward(X, W, b, w_out)
        P = self.clm_.forward(z, t) + _PROBA_EPS

        weighted_Y = sample_weight[:, np.newaxis] * Y
        l2_penalty = sum(np.sum(W_i**2) for W_i in W) + np.sum(w_out**2)
        J = -np.sum(weighted_Y * np.log(P)) / weight_sum
        J += (self.alpha / (2 * weight_sum)) * l2_penalty

        grad_z, grad_t = self.clm_.backward(z, t, -weighted_Y / P / weight_sum)

        grad_w_out = A[-1].T @ grad_z + (self.alpha / weight_sum) * w_out
        delta = (grad_z @ w_out.T) * self._activation_derivative(A[-1])

        grad_W: list[np.ndarray] = []
        grad_b: list[np.ndarray] = []
        for i in range(self.n_hidden_layers - 1, -1, -1):
            grad_W.insert(0, A[i].T @ delta + (self.alpha / weight_sum) * W[i])
            grad_b.insert(0, np.sum(delta, axis=0))
            if i > 0:
                delta = (delta @ W[i].T) * self._activation_derivative(A[i])

        grads = []
        for gW, gb in zip(grad_W, grad_b):
            grads.extend([gW.ravel(), gb])
        grads.extend([grad_w_out.ravel(), grad_t])

        return J, np.concatenate(grads)

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute class probabilities using the optimized parameters.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The validated input data.

        Returns
        -------
        probas : ndarray of shape (n_samples, n_classes)
            The predicted probabilities of each class for the samples in X.
        """
        _, z = self._forward(X, self.W_, self.b_, self.w_out_)
        return self.clm_.forward(z, self.t_)
