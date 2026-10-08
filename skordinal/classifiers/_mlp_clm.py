"""Multi-Layer Perceptron using a Cumulative Link Model output."""

import numpy as np
from scipy.special import expit
from sklearn.utils._param_validation import StrOptions

from skordinal.utils.extmath import (
    cumproba_to_proba,
    params_to_thresholds,
    thresholds_grad,
    thresholds_to_params,
)

from ._mlp_base import MLPBaseClassifier


class MLPCLMClassifier(MLPBaseClassifier):
    """Multi-Layer Perceptron using a Cumulative Link Model output.

    This neural network replaces the standard multiclass output with a
    Cumulative Link Model layer. It projects the features into a single
    latent scalar and estimates the cumulative probabilities of belonging
    to each ordinal category.

    Parameters
    ----------
    n_hidden_layers : int, default=1
        Number of hidden layers in the network.

    n_hidden_units : int, default=4
        Number of neurons per hidden layer.

    activation : {'sigmoid', 'tanh', 'relu'}, default='sigmoid'
        Activation function for the hidden layers.

    alpha : float, default=0.01
        L2 regularization parameter applied to the weights.

    class_weight : dict, "balanced", or None, default=None
        Weights associated with classes.

    max_iter : int, default=500
        Maximum number of iterations.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for weight initialization.

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
        }
    )

    def __init__(
        self,
        n_hidden_layers=1,
        n_hidden_units=4,
        alpha=0.01,
        activation="sigmoid",
        class_weight=None,
        max_iter=500,
        random_state=None,
    ):
        self.activation = activation
        super().__init__(
            n_hidden_layers=n_hidden_layers,
            n_hidden_units=n_hidden_units,
            alpha=alpha,
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state,
        )

    def _initialize_parameters(self, rng: np.random.RandomState) -> np.ndarray:
        """Initialize weights, biases, projection scalar, and thresholds.

        Parameters
        ----------
        rng : RandomState instance
            Random number generator passed by the base class.

        Returns
        -------
        params : ndarray
            A 1D array containing all initialized network parameters concatenated
            together (hidden weights, hidden biases, final projection weights,
            and unconstrained thresholds).
        """
        params = []
        input_dim = self.n_features_in_

        for _ in range(self.n_hidden_layers):
            limit = 0
            if self.activation == "relu":
                limit = np.sqrt(6 / input_dim)
            else:
                limit = np.sqrt(6 / (input_dim + self.n_hidden_units))

            W = rng.uniform(-limit, limit, (input_dim, self.n_hidden_units))
            b = np.zeros(self.n_hidden_units)
            params.extend([W.ravel(), b.ravel()])
            input_dim = self.n_hidden_units

        limit = np.sqrt(6 / (input_dim + 1))
        w_out = rng.uniform(-limit, limit, (input_dim, 1))
        params.append(w_out.ravel())

        initial_thresholds = np.linspace(-2.0, 2.0, self.n_classes_ - 1)
        t_params = thresholds_to_params(initial_thresholds)
        params.append(t_params.ravel())

        return np.concatenate(params)

    def _unpack_parameters(self, params: np.ndarray):
        """Extract the matrices W, vectors b, w_out, and thresholds t.

        Parameters
        ----------
        params : ndarray
            A 1D array containing all unconstrained parameters of the model.

        Returns
        -------
        W : list of ndarray
            Weight matrices for each hidden layer.
        b : list of ndarray
            Bias vectors for each hidden layer.
        w_out : ndarray of shape (n_units_last_hidden, 1)
            Weights for the final scalar projection layer.
        t : ndarray of shape (n_classes - 1,)
            Unconstrained threshold parameters for the Cumulative Link Model.
        """
        idx = 0
        W = []
        b = []
        input_dim = self.n_features_in_

        for _ in range(self.n_hidden_layers):
            size_W = input_dim * self.n_hidden_units
            W.append(params[idx : idx + size_W].reshape(input_dim, self.n_hidden_units))
            idx += size_W

            size_b = self.n_hidden_units
            b.append(params[idx : idx + size_b])
            idx += size_b

            input_dim = self.n_hidden_units

        self.w_out_ = params[idx : idx + input_dim].reshape(input_dim, 1)
        idx += input_dim

        self.t_ = params[idx:]

        self.W_ = W
        self.b_ = b

        return W, b, self.w_out_, self.t_

    def _cost_and_grad(
        self,
        params: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        sample_weight: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Compute the CLM loss and perform backpropagation.

        Parameters
        ----------
        params : ndarray
            1D array containing all unconstrained parameters.
        X : ndarray of shape (n_samples, n_features)
            Training data.
        Y : ndarray of shape (n_samples, n_classes)
            Target matrix (one-hot encoded representation of the classes).
        sample_weight : ndarray of shape (n_samples,)
            Sample weights.

        Returns
        -------
        J : float
            The scalar cost, which includes the Negative Log-Likelihood loss
            and the L2 regularization penalty.
        grads : ndarray
            1D array containing the gradient of the cost with respect to
            all parameters.
        """
        n_samples = X.shape[0]
        weight_sum = np.sum(sample_weight)
        W, b, w_out, t = self._unpack_parameters(params)

        # -----------------
        # Forwards pass
        # -----------------
        A = [X]
        for i in range(self.n_hidden_layers):
            Z = np.dot(A[-1], W[i]) + b[i]
            if self.activation == "sigmoid":
                A_next = expit(Z)
            elif self.activation == "tanh":
                A_next = np.tanh(Z)
            elif self.activation == "relu":
                A_next = np.maximum(0, Z)
            else:
                raise ValueError(f"Unsupported activation: {self.activation}")
            A.append(A_next)

        z_proj = np.dot(A[-1], w_out)

        theta = params_to_thresholds(t)

        # Cumulative link model formulation
        z_clm = theta[np.newaxis, :] - z_proj
        cumproba = expit(z_clm)

        # P = [ P(Y<=1), P(Y<=2)-P(Y<=1), ..., 1-P(Y<=K-1) ]
        a3 = np.hstack([cumproba, np.ones((n_samples, 1))])
        P = np.hstack([a3[:, 0:1], a3[:, 1:] - a3[:, :-1]])

        tiny = np.finfo(np.float64).tiny
        P = np.maximum(P, tiny)

        # -----------------
        # Loss + L2 penalty
        # -----------------
        l2_penalty = sum(np.sum(w**2) for w in W) + np.sum(w_out**2)

        J = -np.sum(sample_weight[:, np.newaxis] * Y * np.log(P)) / weight_sum

        J += (self.alpha / (2 * weight_sum)) * l2_penalty

        # -----------------
        # Backwards pass
        # -----------------

        P = np.clip(P, 1e-15, 1.0 - 1e-15)
        error_der = -(sample_weight[:, np.newaxis] * Y) / P

        f_grad = cumproba * (1 - cumproba)

        # Gradient pushing into the cumulative link layer
        g_grad = error_der * np.hstack(
            [f_grad[:, 0:1], f_grad[:, 1:] - f_grad[:, :-1], -f_grad[:, -1:]]
        )

        delta_z = -np.sum(g_grad, axis=1, keepdims=True)

        # Gradient with respect to the ordered thresholds
        raw_theta_grad = (
            np.sum(
                error_der[:, :-1] * f_grad - error_der[:, 1:] * f_grad,
                axis=0,
            )
            / weight_sum
        )

        # Push the gradient back through the cumsum-of-squares map
        grad_t = thresholds_grad(t, raw_theta_grad)
        grad_w_out = (
            np.dot(A[-1].T, delta_z) / weight_sum + (self.alpha / weight_sum) * w_out
        )

        # Propagate error back to the last hidden layer
        if self.activation == "sigmoid":
            deriv = A[-1] * (1 - A[-1])
        elif self.activation == "tanh":
            deriv = 1 - A[-1] ** 2
        elif self.activation == "relu":
            deriv = (A[-1] > 0).astype(float)

        delta = np.dot(delta_z, w_out.T) * deriv

        grad_W: list[np.ndarray] = []
        grad_b: list[np.ndarray] = []

        # Reverse loop through hidden layers
        for i in range(self.n_hidden_layers - 1, -1, -1):
            gW = np.dot(A[i].T, delta) / weight_sum + (self.alpha / weight_sum) * W[i]
            gb = np.sum(delta, axis=0) / weight_sum

            grad_W.insert(0, gW.ravel())
            grad_b.insert(0, gb.ravel())

            if i > 0:
                if self.activation == "sigmoid":
                    deriv = A[i] * (1 - A[i])
                elif self.activation == "tanh":
                    deriv = 1 - A[i] ** 2
                elif self.activation == "relu":
                    deriv = (A[i] > 0).astype(float)

                delta = np.dot(delta, W[i].T) * deriv

        # Pack all gradients
        grads = []
        for gW, gb in zip(grad_W, grad_b):
            grads.extend([gW, gb])
        grads.extend([grad_w_out.ravel(), grad_t.ravel()])

        return J, np.concatenate(grads)

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """ "Compute class probabilities using the optimized CLM parameters.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The validated input data for which to compute probabilities.

        Returns
        ----------
        probas : ndarray of shape (n_samples, n_classes)
            The predicted probabilities of each class for the samples in X.
        """
        A = X
        for W, b in zip(self.W_, self.b_):
            Z = np.dot(A, W) + b
            if self.activation == "sigmoid":
                A = expit(Z)
            elif self.activation == "tanh":
                A = np.tanh(Z)
            elif self.activation == "relu":
                A = np.maximum(0, Z)

        z_proj = np.dot(A, self.w_out_)
        theta = params_to_thresholds(self.t_)
        cumproba = expit(theta[np.newaxis, :] - z_proj)

        return cumproba_to_proba(cumproba, repair=True)
