"""EBANO (Ensemble BAsed on uNimodal Ordinal classifiers)."""

from __future__ import annotations

import ast
import os
from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted, validate_data

from skordinal.metrics import average_mean_absolute_error
from skordinal.utils.validation import check_ordinal_targets


class EBANO(ClassifierMixin, BaseEstimator):
    """
    EBANO (Ensemble BAsed on uNimodal Ordinal classifiers)

    An ensemble aggregator designed to operate in a two-phase workflow.
    It optimizes combination weights via random search to minimize the
    Average Mean Absolute Error (AMAE), and computes the final
    ensemble predictions using a weighted probability blending strategy.

    Parameters
    ----------
    models_saved_results_paths : list of str
        List of directory paths where each base classifier's saved results reside.

    dataset_name : str
        Name of the dataset being processed, used to resolve the file paths.

    weights_cv_n_iters : int, default=1000
        Number of iterations (I) for the random search to find the optimal
        probability blending weights during the fitting process.

    random_state : int, RandomState instance or None, default=None
        Controls the random seed for weight generation. Note: for resolving
        the file paths, the integer seed representation is required.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The unique ordinal classes observed during `fit`, sorted
        in ascending order.

    estimator_weights_ : ndarray of shape (n_estimators,)
        The optimal combination weights assigned to each base classifier's
        probability matrix, found after running `fit`.
    """

    _parameter_constraints: dict = {
        "models_saved_results_paths": [list],
        "dataset_name": [str],
        "weights_cv_n_iters": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state", None],
    }

    def __init__(
        self,
        models_saved_results_paths: list[str],
        dataset_name: str,
        weights_cv_n_iters: int = 1000,
        random_state: int | None = None,
    ) -> None:
        self.models_saved_results_paths = models_saved_results_paths
        self.dataset_name = dataset_name
        self.weights_cv_n_iters = weights_cv_n_iters
        self.random_state = random_state

    def _get_seed_for_path(self) -> str:
        """Extracts a valid string identifier for the directory path."""
        if isinstance(self.random_state, int):
            return str(self.random_state)
        elif self.random_state is None:
            return "None"
        else:
            raise TypeError(
                "To load predictions from disk, 'random_state' must be an integer "
                "so the directory path can be resolved correctly."
            )

    def _load_model_pred_probas(
        self, model_saved_results_path: str, split: str, expected_samples: int
    ) -> np.ndarray:
        """Method to read probability matrices from the CSV."""
        assert split in ["train", "test"], "split must be 'train' or 'test'"

        seed_str = self._get_seed_for_path()
        seed_results_path = os.path.join(
            model_saved_results_path,
            self.dataset_name,
            "predictions_by_seed",
            f"seed_{seed_str}",
            f"{split}_predictions.csv",
        )

        if not os.path.exists(seed_results_path):
            raise FileNotFoundError(
                f"Pretrained model not found at {seed_results_path}"
            )

        df = pd.read_csv(seed_results_path)

        if "Pattern ID" in df.columns:
            df = df.sort_values("Pattern ID").reset_index(drop=True)

        if len(df) != expected_samples:
            raise ValueError(
                f"Loaded {split} CSV has {len(df)} rows, but the input array X "
                f"has {expected_samples} rows. They must match exactly."
            )

        probas = np.array([ast.literal_eval(p) for p in df["Prediction probabilities"]])
        return probas

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: np.ndarray, y: np.ndarray) -> "EBANO":
        """Fit the EBANO ensemble model using pre-loaded probability matrices."""
        X, y = validate_data(self, X=X, y=y, reset=True)
        self.classes_, y_encoded = check_ordinal_targets(y)

        self._X_train_ = X

        if (
            not self.models_saved_results_paths
            or len(self.models_saved_results_paths) == 0
        ):
            raise ValueError("The 'models_saved_results_paths' list cannot be empty.")

        models_pred_probas = []
        for path in self.models_saved_results_paths:
            probas = self._load_model_pred_probas(
                path, split="train", expected_samples=X.shape[0]
            )
            models_pred_probas.append(probas)

        rng = check_random_state(self.random_state)
        num_estimators = len(models_pred_probas)

        best_score = float("inf")
        best_weights = np.zeros(num_estimators)

        for _ in range(self.weights_cv_n_iters):
            weights = rng.uniform(size=num_estimators)
            weights /= np.sum(weights)

            weighted_probas = np.zeros_like(models_pred_probas[0])
            for k, probas in enumerate(models_pred_probas):
                weighted_probas += probas * weights[k]

            preds = np.argmax(weighted_probas, axis=1)

            score = average_mean_absolute_error(y_encoded, preds)

            if score < best_score:
                best_score = score
                best_weights = weights

        self.estimator_weights_ = best_weights
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute the blended class probabilities loading the appropriate split."""
        check_is_fitted(self)
        X = validate_data(self, X=X, reset=False)

        is_train = X.shape == self._X_train_.shape and np.array_equal(X, self._X_train_)
        split = "train" if is_train else "test"

        models_pred_probas = []
        for path in self.models_saved_results_paths:
            probas = self._load_model_pred_probas(
                path, split=split, expected_samples=X.shape[0]
            )
            models_pred_probas.append(probas)

        weighted_probas = np.zeros_like(models_pred_probas[0])
        for k, probas in enumerate(models_pred_probas):
            weighted_probas += probas * self.estimator_weights_[k]

        return weighted_probas

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict ordinal class labels based on pre-computed probability matrices."""
        pred_probas = self.predict_proba(X)
        indices = np.argmax(pred_probas, axis=1)
        return self.classes_[indices]
