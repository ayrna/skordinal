"""OrdinalDecomposition ensemble."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    _fit_context,
    clone,
)
from sklearn.linear_model import LogisticRegression
from sklearn.utils._param_validation import HasMethods, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from skordinal.preprocessing import build_coding_matrix
from skordinal.utils.extmath import cumproba_to_proba, losses_to_proba
from skordinal.utils.validation import check_ordinal_targets


class OrdinalDecomposition(MetaEstimatorMixin, ClassifierMixin, BaseEstimator):
    """Ordinal decomposition ensemble classifier.

    This class implements an ensemble model where an ordinal problem is decomposed into
    several binary subproblems, each one of which will generate a different (binary)
    model, though all of them are clones of the same base estimator.

    There are 4 different ways to decompose the original problem based on how the
    coding matrix is built.

    Parameters
    ----------
    estimator : classifier instance or None, default=None
        Base binary classifier, cloned once per binary subproblem. Must implement
        ``fit`` and ``predict_proba``. If ``None``, a LogisticRegression with
        default hyperparameters is used. Its own hyperparameters are reachable
        as ``estimator__<param>``.

    decomposition : {'ordered_partitions', 'one_vs_next', 'one_vs_followers', 'one_vs_previous'}, \
            default='ordered_partitions'
        Type of decomposition used to build the coding matrix. Each row of the
        coding matrix corresponds to a class and each column to a binary subproblem.
        See :func:`~skordinal.preprocessing.build_coding_matrix`.

    decision_method : {'exponential_loss', 'hinge_loss', 'logarithmic_loss', 'frank_hall'}, \
            default='frank_hall'
        Method to aggregate the predictions of the binary estimators into class
        probabilities or labels.

    Attributes
    ----------
    estimators_ : list of length n_classes - 1
        Fitted clones of the base estimator, one per binary subproblem.

    classes_ : ndarray of shape (n_classes,)
        Class labels for each output.

    n_features_in_ : int
        Number of features seen during fit.

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit. Defined only when X has feature
        names that are all strings.

    coding_matrix_ : ndarray of shape (n_classes, n_classes - 1)
        Matrix that defines which classes will be used to build the model of each
        subproblem, and in which binary class they belong inside those new models.
        Built by :func:`~skordinal.preprocessing.build_coding_matrix`.

    References
    ----------
    .. [1] P.A. Gutiérrez, M. Pérez-Ortiz, J. Sánchez-Monedero, F. Fernández-Navarro
           and C. Hervás-Martínez, "Ordinal regression methods: survey and
           experimental study", IEEE Transactions on Knowledge and Data
           Engineering, Vol. 28. Issue 1, 2016,
           http://dx.doi.org/10.1109/TKDE.2015.2457911

    """

    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit", "predict_proba"]), None],
        "decomposition": [
            StrOptions(
                {
                    "ordered_partitions",
                    "one_vs_next",
                    "one_vs_followers",
                    "one_vs_previous",
                }
            )
        ],
        "decision_method": [
            StrOptions(
                {"exponential_loss", "hinge_loss", "logarithmic_loss", "frank_hall"}
            )
        ],
    }

    def __init__(
        self,
        estimator=None,
        decomposition: str = "ordered_partitions",
        decision_method: str = "frank_hall",
    ) -> None:
        self.estimator = estimator
        self.decomposition = decomposition
        self.decision_method = decision_method

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X: ArrayLike, y: ArrayLike) -> OrdinalDecomposition:
        """Fit underlying estimators to data matrix X and target(s) y.

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
        X, y = validate_data(
            self, X, y, accept_sparse=False, ensure_2d=True, dtype=None
        )

        self.classes_, y_encoded = check_ordinal_targets(y)

        decomposition = self.decomposition
        decision = self.decision_method
        if decision == "frank_hall" and decomposition != "ordered_partitions":
            raise ValueError(
                "When using Frank and Hall decision method, "
                "ordered_partitions must be used"
            )
        self._decision_method_ = decision

        # Give each train input its corresponding output label
        # for each binary classifier
        self.coding_matrix_ = build_coding_matrix(len(self.classes_), decomposition)
        class_labels = self.coding_matrix_[y_encoded, :]

        base = self.estimator if self.estimator is not None else LogisticRegression()
        self.estimators_ = []

        # Fitting n_classes - 1 classifiers
        for n in range(class_labels.shape[1]):
            est_n = clone(base)
            mask = class_labels[:, n] != 0
            est_n.fit(X[mask], class_labels[mask, n].ravel())
            self.estimators_.append(est_n)

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
        X = validate_data(self, X, reset=False, ensure_2d=True, dtype=None)

        return self.classes_[np.argmax(self._proba(X), axis=1)]

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Probability estimates.

        The returned estimates for all classes are ordered by label of classes.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            The probability of the sample for each class in the model, where classes are
            ordered as they are in self.classes_.

        Raises
        ------
        NotFittedError
            If the model is not fitted yet.

        ValueError
            If input is invalid.

        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, ensure_2d=True, dtype=None)

        return self._proba(X)

    def _proba(self, X: np.ndarray) -> np.ndarray:
        """Compute class probabilities from pre-validated X."""
        predictions = self._get_predictions(X)

        if self._decision_method_ == "frank_hall":
            return self._frank_hall_method(predictions)

        # Scaling predictions from [0, 1] range to [-1, 1]
        predictions = predictions * 2 - 1

        loss_fn = {
            "exponential_loss": self._exponential_loss,
            "hinge_loss": self._hinge_loss,
            "logarithmic_loss": self._logarithmic_loss,
        }[self._decision_method_]

        # Transforming from binary problems to the original problem
        return losses_to_proba(loss_fn(predictions))

    def _get_predictions(self, X: np.ndarray) -> np.ndarray:
        """Return the probability of positive class membership.

        For each pattern inside the dataset X, this method returns the probability for
        that pattern to belong to the positive class. There will be as many predictions
        (columns) as different binary classifiers have been fitted previously.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        predictions : array, shape (n_samples, n_classes - 1)
            Probability estimates or binary classification outcomes.

        """
        return np.column_stack([est.predict_proba(X)[:, 1] for est in self.estimators_])

    def _exponential_loss(self, predictions: np.ndarray) -> np.ndarray:
        """Compute the exponential losses for each label.

        Computation of the exponential losses for each label of the original ordinal
        multinomial problem. Transforms from n-1 binary subproblems to the original
        ordinal problem with n targets.

        Parameters
        ----------
        predictions : array, shape (n_samples, n_classes - 1)
            Probability estimates or binary classification outcomes.

        Returns
        -------
        e_losses : ndarray of shape (n_samples, n_classes)
            Exponential losses for each sample of dataset X. One different value for
            each class label.

        """
        C = self.coding_matrix_[None, :, :]
        M = predictions[:, None, :]
        e_losses = np.exp(-M * C).sum(axis=2)
        return e_losses

    def _hinge_loss(self, predictions: np.ndarray) -> np.ndarray:
        """Compute the Hinge losses for each label.

        Computation of the Hinge losses for each label of the original ordinal
        multinomial problem. Transforms from n-1 binary subproblems to the original
        ordinal problem with n targets.

        Parameters
        ----------
        predictions : array, shape (n_samples, n_classes - 1)
            Probability estimates or binary classification outcomes.

        Returns
        -------
        h_losses : ndarray of shape (n_samples, n_classes)
            Hinge losses for each sample of dataset X. One different value for each
            class label.

        """
        C = self.coding_matrix_[None, :, :]
        M = predictions[:, None, :]
        h_losses = np.maximum(0.0, 1.0 - C * M).sum(axis=2)
        return h_losses

    def _logarithmic_loss(self, predictions: np.ndarray) -> np.ndarray:
        """Compute the logarithmic losses for each label.

        Computation of the logarithmic losses for each label of the original ordinal
        multinomial problem. Transforms from n-1 binary subproblems to the original
        ordinal problem with n targets.

        Parameters
        ----------
        predictions : array, shape (n_samples, n_classes - 1)
            Probability estimates or binary classification outcomes.

        Returns
        -------
        l_losses : ndarray of shape (n_samples, n_classes)
            Logarithmic losses for each sample of dataset X. One different value for
            each class label.

        """
        C = self.coding_matrix_[None, :, :]
        M = predictions[:, None, :]
        l_losses = np.log1p(np.exp(-2.0 * C * M)).sum(axis=2)
        return l_losses

    def _frank_hall_method(self, predictions: np.ndarray) -> np.ndarray:
        """Calculate probability of each pattern belonging to each target.

        Returns the probability for each pattern of dataset to belong to each one of
        the original targets. Transforms from n-1 subproblems to the original ordinal
        problem with n targets. Non-monotonic binary outputs are repaired by isotonic
        regression before differencing.

        Parameters
        ----------
        predictions : array, shape (n_samples, n_classes - 1)
            Probability estimates or binary classification outcomes.

        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Class membership probabilities for each sample. Each row is non-negative
            and sums to one.

        """
        # Binary outputs are P(Y > k), the complement of the cumulative P(Y <= k)
        return cumproba_to_proba(1.0 - predictions, repair=True)
