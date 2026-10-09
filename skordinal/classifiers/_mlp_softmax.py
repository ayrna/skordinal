"""Multi-Layer Perceptron classifier (Softmax)."""

import numpy as np
from scipy.special import expit, logsumexp
from sklearn.utils._param_validation import StrOptions

from ._mlp_base import MLPBaseClassifier


class MLPSoftmaxClassifier(MLPBaseClassifier):
    """Multi-Layer Perceptron classifier (Softmax).

    This class implements a standard fully connected network with configurable
    hidden activations (relu, tanh, sigmoid) and a Softmax output layer, trained
    using categorical Cross-Entropy loss.

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
        Maximum number of iterations for the L-BFGS-B solver.

    tol: float, default=1e-4
        Tolerance for the optimization.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for weight initialization.
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
        activation="sigmoid",
        alpha=0.01,
        class_weight=None,
        max_iter=500,
        tol=1e-4,
        random_state=None,
    ):
        super().__init__(
            n_hidden_layers=n_hidden_layers,
            n_hidden_units=n_hidden_units,
            alpha=alpha,
            class_weight=class_weight,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        self.activation = activation

    def _initialize_parameters(self, rng: np.random.RandomState) -> np.ndarray:
        """Initialize weights, biases, output weights, and output biases.

        Parameters
        ----------
        rng : RandomState instance
            Random number generator.

        Returns
        -------
        params : ndarray
            A 1D array containing all initialized network parameters concatenated
            together (hidden weights, hidden biases, output weights, and output biases).
        """
        params = []
        input_dim = self.n_features_in_

        for _ in range(self.n_hidden_layers):
            limit = 0
            if self.activation == "relu":
                # He initialization
                limit = np.sqrt(6 / input_dim)
            else:
                # Xavier initialization
                limit = np.sqrt(6 / (input_dim + self.n_hidden_units))

            W = rng.uniform(-limit, limit, size=(input_dim, self.n_hidden_units))
            b = np.zeros(self.n_hidden_units)
            params.extend([W.ravel(), b.ravel()])
            input_dim = self.n_hidden_units

        limit = np.sqrt(6 / (input_dim + self.n_classes_))
        W_out = rng.uniform(-limit, limit, (input_dim, self.n_classes_))
        b_out = np.zeros(self.n_classes_)
        params.extend([W_out.ravel(), b_out.ravel()])

        return np.concatenate(params)

    def _unpack_parameters(self, params: np.ndarray):
        """Extract the list of matrices W, vectors b, W_out, and b_out.

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
        W_out : ndarray of shape (n_units_last_hidden, n_classes)
            Weight matrix for the final Softmax output layer.
        b_out : ndarray of shape (n_classes,)
            Bias vector for the final Softmax output layer.
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

        size_W_out = input_dim * self.n_classes_
        self.W_out_ = params[idx : idx + size_W_out].reshape(input_dim, self.n_classes_)
        idx += size_W_out

        size_b_out = self.n_classes_
        self.b_out_ = params[idx : idx + size_b_out]

        self.W_ = W
        self.b_ = b

        return W, b, self.W_out_, self.b_out_

    def _cost_and_grad(
        self,
        params: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        sample_weight: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Compute the categorical cross-entropy loss and gradients.

        Parameters
        ----------
        params : ndarray
            1D array containing all unconstrained parameters (hidden weights,
            hidden biases, output weights, and output biases).
        X : ndarray of shape (n_samples, n_features)
            Training data.
        Y : ndarray of shape (n_samples, n_classes)
            Target matrix (one-hot encoded representation of the classes).
        sample_weight : ndarray of shape (n_samples,)
            Sample weights.

        Returns
        -------
        J : float
            The scalar cost, which includes the categorical cross-entropy loss
            and the L2 regularization penalty.
        grads : ndarray
            1D array containing the gradient of the cost with respect to
            all parameters.
        """
        W, b, W_out, b_out = self._unpack_parameters(params)

        weight_sum = np.sum(sample_weight)

        # --------------
        # Forward pass
        # --------------
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

        # Output layer logits
        Z_out = np.dot(A[-1], W_out) + b_out

        # Numerically stable log-softmax
        log_prob = Z_out - logsumexp(
            Z_out,
            axis=1,
            keepdims=True,
        )
        P = np.exp(log_prob)

        # -------------------------
        # Loss + L2 regularization
        # -------------------------
        l2_penalty = sum(np.sum(w**2) for w in W) + np.sum(W_out**2)

        cross_entropy = (
            -np.sum(sample_weight[:, np.newaxis] * Y * log_prob) / weight_sum
        )

        J = cross_entropy + (self.alpha / (2 * weight_sum)) * l2_penalty

        # ----------------
        # Backward pass
        # ----------------

        # Derivative of cross-entropy with softmax simplifies to (P - Y)
        delta_out = (P - Y) * sample_weight[:, np.newaxis]

        # Gradients for the output layer
        grad_W_out = (
            np.dot(A[-1].T, delta_out) / weight_sum + (self.alpha / weight_sum) * W_out
        )

        grad_b_out = np.sum(delta_out, axis=0) / weight_sum

        if self.activation == "sigmoid":
            deriv = A[-1] * (1 - A[-1])
        elif self.activation == "tanh":
            deriv = 1 - A[-1] ** 2
        elif self.activation == "relu":
            deriv = (A[-1] > 0).astype(float)

        delta = np.dot(delta_out, W_out.T) * deriv

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

        # Concatenate all gradients
        grads = []
        for gW, gb in zip(grad_W, grad_b):
            grads.extend([gW, gb])
        grads.extend([grad_W_out.ravel(), grad_b_out.ravel()])

        return J, np.concatenate(grads)

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute class probabilities using the optimized parameters.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The validated input data for which to compute probabilities.

        Returns
        -------
        probas : ndarray of shape (n_samples, n_classes)
            The predicted probabilities of each class for the samples in X.
        """
        # Forward pass
        A = X
        for W, b in zip(self.W_, self.b_):
            Z = np.dot(A, W) + b
            if self.activation == "sigmoid":
                A = expit(Z)
            elif self.activation == "tanh":
                A = np.tanh(Z)
            elif self.activation == "relu":
                A = np.maximum(0, Z)

        Z_out = np.dot(A, self.W_out_) + self.b_out_

        # Apply softmax
        log_prob = Z_out - logsumexp(Z_out, axis=1, keepdims=True)
        return np.exp(log_prob)
