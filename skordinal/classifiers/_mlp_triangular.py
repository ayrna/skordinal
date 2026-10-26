"""Multi-Layer Perceptron trained with triangular soft labels."""

from numbers import Real

import numpy as np
from scipy.special import expit, softmax
from sklearn.utils._param_validation import Interval, StrOptions

from ._mlp_base import MLPBaseClassifier
from ._triangular_loss import TriangularLoss


class MLPTriangularClassifier(MLPBaseClassifier):
    """Multi-Layer Perceptron trained with triangular soft labels.

    A softmax network whose one-hot targets are replaced by the
    triangular soft labels of Vargas et al. [1]_: each class keeps most
    of the probability and leaves ``alpha2`` on each neighbouring class,
    so mistakes between adjacent classes are penalised less than distant
    ones. The objective is :class:`TriangularLoss` plus an L2 penalty on
    the weights.

    Parameters
    ----------
    n_hidden_layers : int, default=1
        Number of hidden layers in the network.

    n_hidden_units : int, default=4
        Number of neurons per hidden layer.

    alpha : float, default=0.01
        L2 regularization parameter applied to the weights. Biases are not
        penalised.

    activation : {'sigmoid', 'tanh', 'relu'}, default='sigmoid'
        Activation function for the hidden layers.

    alpha2 : float, default=0.05
        Probability the soft labels give to each neighbour of a middle
        class. Must be at most ``2/9`` when there are three or more
        classes; see :func:`get_triangular_soft_labels`.

    eta : float, default=1.0
        Weight of the soft labels in the targets, in ``[0, 1]``. ``1.0``
        trains on the soft labels alone and ``0.0`` on one-hot targets.

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

    loss_fn_ : TriangularLoss
        The loss used during training, which holds the soft labels in
        ``loss_fn_.cls_probs``.

    W_ : list of ndarray
        Weight matrices of the hidden layers.

    b_ : list of ndarray
        Bias vectors of the hidden layers.

    W_out_ : ndarray of shape (n_hidden_units, n_classes)
        Weights of the output layer.

    b_out_ : ndarray of shape (n_classes,)
        Biases of the output layer.

    loss_ : float
        Value of the objective at the end of training.

    n_iter_ : int
        Number of iterations run by the solver.

    References
    ----------
    .. [1] V. M. Vargas, P. A. Gutiérrez, J. Barbero-Gómez, and
       C. Hervás-Martínez, "Soft labelling based on triangular distributions
       for ordinal classification," Information Fusion, vol. 93,
       pp. 258-267, 2023.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.classifiers import MLPTriangularClassifier
    >>> rng = np.random.default_rng(42)
    >>> X = rng.standard_normal((100, 5))
    >>> y = np.sort(rng.integers(1, 5, 100))
    >>> clf = MLPTriangularClassifier(alpha2=0.1, max_iter=10, random_state=0)
    >>> clf.fit(X, y) # doctest: +SKIP
    MLPTriangularClassifier(alpha2=0.1, max_iter=10, random_state=0)
    >>> clf.predict(X[:3]) # doctest: +SKIP
    array([1, 1, 1])
    """

    _parameter_constraints = MLPBaseClassifier._parameter_constraints.copy()
    _parameter_constraints.update(
        {
            "activation": [StrOptions({"sigmoid", "tanh", "relu"})],
            # The upper bound depends on n_classes and is checked in fit
            "alpha2": [Interval(Real, 0.0, 1.0, closed="both")],
            "eta": [Interval(Real, 0.0, 1.0, closed="both")],
        }
    )

    def __init__(
        self,
        n_hidden_layers=1,
        n_hidden_units=4,
        alpha=0.01,
        activation="sigmoid",
        alpha2=0.05,
        eta=1.0,
        class_weight=None,
        max_iter=500,
        random_state=None,
        tol=1e-5,
    ):
        self.activation = activation
        self.alpha2 = alpha2
        self.eta = eta
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
        """Initialize hidden and output weights and biases.

        Hidden weights use He-uniform initialization for ReLU and
        Glorot-uniform otherwise; the output layer is Glorot-uniform and
        all biases start at zero.

        Parameters
        ----------
        rng : RandomState instance
            Random number generator passed by the base class.

        Returns
        -------
        params : ndarray
            1D array with the hidden weights and biases followed by the
            output weights and biases.
        """
        self.loss_fn_ = TriangularLoss(
            self.n_classes_, alpha2=self.alpha2, eta=self.eta
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

        limit = np.sqrt(6 / (input_dim + self.n_classes_))
        W_out = rng.uniform(-limit, limit, (input_dim, self.n_classes_))
        params.extend([W_out.ravel(), np.zeros(self.n_classes_)])

        return np.concatenate(params)

    def _split_parameters(self, params: np.ndarray):
        """Split the 1D parameter array into the layers' arrays.

        Parameters
        ----------
        params : ndarray
            1D array containing all parameters of the model.

        Returns
        -------
        W : list of ndarray
            Weight matrices for each hidden layer.
        b : list of ndarray
            Bias vectors for each hidden layer.
        W_out : ndarray of shape (n_units_last_hidden, n_classes)
            Weights of the output layer.
        b_out : ndarray of shape (n_classes,)
            Biases of the output layer.
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

        size_W_out = input_dim * self.n_classes_
        W_out = params[idx : idx + size_W_out].reshape(input_dim, self.n_classes_)
        idx += size_W_out

        b_out = params[idx:]
        if b_out.shape != (self.n_classes_,):
            raise ValueError(
                f"Expected {idx + self.n_classes_} parameters, got {params.size}."
            )

        return W, b, W_out, b_out

    def _unpack_parameters(self, params: np.ndarray):
        """Store the optimised parameters as fitted attributes."""
        self.W_, self.b_, self.W_out_, self.b_out_ = self._split_parameters(params)

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

    def _forward(self, X, W, b, W_out, b_out):
        """Return the activations of every hidden layer and the logits."""
        A = [X]
        for W_i, b_i in zip(W, b):
            A.append(self._activate(A[-1] @ W_i + b_i))
        return A, A[-1] @ W_out + b_out

    def _cost_and_grad(
        self,
        params: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        sample_weight: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Compute the triangular loss and its gradient by backpropagation.

        Parameters
        ----------
        params : ndarray
            1D array containing all parameters.
        X : ndarray of shape (n_samples, n_features)
            Training data.
        Y : ndarray of shape (n_samples, n_classes)
            One-hot encoded targets.
        sample_weight : ndarray of shape (n_samples,)
            Sample weights.

        Returns
        -------
        J : float
            Weighted mean cross-entropy against the soft targets plus the
            L2 penalty.
        grad : ndarray
            Gradient of ``J`` with respect to ``params``.
        """
        weight_sum = np.sum(sample_weight)
        W, b, W_out, b_out = self._split_parameters(params)

        A, logits = self._forward(X, W, b, W_out, b_out)
        J, grad_logits = self.loss_fn_.loss_and_grad(logits, Y, sample_weight)

        l2_penalty = sum(np.sum(W_i**2) for W_i in W) + np.sum(W_out**2)
        J += (self.alpha / (2 * weight_sum)) * l2_penalty

        grad_W_out = A[-1].T @ grad_logits + (self.alpha / weight_sum) * W_out
        grad_b_out = np.sum(grad_logits, axis=0)
        delta = (grad_logits @ W_out.T) * self._activation_derivative(A[-1])

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
        grads.extend([grad_W_out.ravel(), grad_b_out])

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
        _, logits = self._forward(X, self.W_, self.b_, self.W_out_, self.b_out_)
        return softmax(logits, axis=1)
