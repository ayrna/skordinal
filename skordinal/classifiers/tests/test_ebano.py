"""Tests for the EBANO probability-averaging classifier."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from skordinal.classifiers import EBANO


def _write_predictions(
    model_root: Path,
    dataset_name: str,
    seed: int,
    split: str,
    probabilities: np.ndarray,
    pattern_ids: np.ndarray | None = None,
) -> None:
    """Write one saved prediction CSV in the layout expected by EBANO."""
    if pattern_ids is None:
        pattern_ids = np.arange(probabilities.shape[0])
    seed_dir = (
        model_root
        / dataset_name
        / "predictions_by_seed"
        / f"seed_{seed}"
    )
    seed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Pattern ID": pattern_ids,
            "Prediction probabilities": [row.tolist() for row in probabilities],
        }
    ).to_csv(seed_dir / f"{split}_predictions.csv", index=False)


@pytest.fixture
def saved_model_predictions(tmp_path):
    """Create shuffled train probabilities and test probabilities for two models."""
    dataset_name = "toy"
    model_roots = [tmp_path / "model_a", tmp_path / "model_b"]

    train_probabilities = [
        np.array(
            [
                [0.05, 0.90, 0.05],
                [0.80, 0.15, 0.05],
                [0.05, 0.10, 0.85],
                [0.10, 0.20, 0.70],
            ]
        ),
        np.array(
            [
                [0.10, 0.80, 0.10],
                [0.70, 0.20, 0.10],
                [0.10, 0.15, 0.75],
                [0.15, 0.25, 0.60],
            ]
        ),
    ]
    test_probabilities = [
        np.array([[0.20, 0.70, 0.10], [0.10, 0.20, 0.70]]),
        np.array([[0.25, 0.60, 0.15], [0.15, 0.25, 0.60]]),
    ]
    train_order = np.array([2, 0, 3, 1])
    for root, train, test in zip(model_roots, train_probabilities, test_probabilities):
        _write_predictions(
            root,
            dataset_name,
            7,
            "train",
            train[train_order],
            pattern_ids=train_order,
        )
        _write_predictions(root, dataset_name, 7, "test", test)

    return dataset_name, model_roots, train_probabilities, test_probabilities


def test_fit_and_predict_use_sorted_saved_probabilities(saved_model_predictions):
    """fit, predict_proba and predict blend both saved splits in sample order."""
    dataset_name, model_roots, train_probas, test_probas = saved_model_predictions
    X_train = np.arange(4, dtype=float).reshape(-1, 1)
    y_train = np.array([0, 1, 2, 2])
    X_test = np.array([[10.0], [11.0]])

    model = EBANO(
        models_saved_results_paths=[str(path) for path in model_roots],
        dataset_name=dataset_name,
        weights_cv_n_iters=25,
        random_state=7,
    )

    assert model.fit(X_train, y_train) is model
    assert model.classes_.tolist() == [0, 1, 2]
    assert model.estimator_weights_.shape == (2,)
    assert np.all(model.estimator_weights_ >= 0)
    assert model.estimator_weights_.sum() == pytest.approx(1.0)

    train_expected = sum(
        weight * probabilities
        for weight, probabilities in zip(model.estimator_weights_, train_probas)
    )
    test_expected = sum(
        weight * probabilities
        for weight, probabilities in zip(model.estimator_weights_, test_probas)
    )

    np.testing.assert_allclose(model.predict_proba(X_train), train_expected)
    np.testing.assert_allclose(model.predict_proba(X_test), test_expected)
    np.testing.assert_array_equal(model.predict(X_test), np.argmax(test_expected, axis=1))


def test_predict_proba_before_fit_raises_not_fitted():
    """predict_proba follows the scikit-learn fitted-estimator contract."""
    model = EBANO(["unused"], dataset_name="toy", random_state=0)

    with pytest.raises(NotFittedError):
        model.predict_proba(np.zeros((2, 1)))


def test_fit_rejects_empty_model_paths():
    """At least one pretrained model probability source is required."""
    model = EBANO([], dataset_name="toy", random_state=0)

    with pytest.raises(ValueError, match="cannot be empty"):
        model.fit(np.zeros((2, 1)), np.array([0, 1]))


def test_fit_raises_for_missing_train_predictions(tmp_path):
    """Missing prediction files report the exact unresolved path."""
    model = EBANO([str(tmp_path / "missing")], dataset_name="toy", random_state=3)

    with pytest.raises(FileNotFoundError, match="train_predictions.csv"):
        model.fit(np.zeros((2, 1)), np.array([0, 1]))


def test_fit_rejects_prediction_row_count_mismatch(tmp_path):
    """Saved probabilities must have one row per input training sample."""
    model_root = tmp_path / "model"
    _write_predictions(
        model_root,
        "toy",
        2,
        "train",
        np.array([[0.8, 0.2], [0.2, 0.8]]),
    )
    model = EBANO([str(model_root)], dataset_name="toy", random_state=2)

    with pytest.raises(ValueError, match=r"Loaded train CSV has 2 rows.*3 rows"):
        model.fit(np.zeros((3, 1)), np.array([0, 1, 1]))


def test_predict_proba_raises_for_missing_test_predictions(tmp_path):
    """Test probabilities must also exist before test predictions can be made."""
    model_root = tmp_path / "model"
    _write_predictions(
        model_root,
        "toy",
        1,
        "train",
        np.array([[0.8, 0.2], [0.2, 0.8]]),
    )
    model = EBANO([str(model_root)], dataset_name="toy", random_state=1)
    X_train = np.array([[0.0], [1.0]])
    model.fit(X_train, np.array([0, 1]))

    with pytest.raises(FileNotFoundError, match="test_predictions.csv"):
        model.predict_proba(np.array([[2.0]]))


def test_predict_proba_loads_test_when_shapes_match_but_data_differs(tmp_path):
    """The model correctly identifies a test set even if it has the exact same shape as the train set."""
    model_root = tmp_path / "model"
    _write_predictions(model_root, "toy", 0, "train", np.array([[0.9, 0.1], [0.1, 0.9]]))
    _write_predictions(model_root, "toy", 0, "test", np.array([[0.5, 0.5], [0.5, 0.5]]))
    
    model = EBANO([str(model_root)], dataset_name="toy", random_state=0)
    
    X_train = np.array([[1.0], [2.0]])
    model.fit(X_train, np.array([0, 1]))
    
    X_test = np.array([[3.0], [4.0]])
    
    probas = model.predict_proba(X_test)
    np.testing.assert_allclose(probas, [[0.5, 0.5], [0.5, 0.5]])


def test_fit_rejects_random_state_instance():
    """A RandomState instance cannot be used because it cannot be parsed into a directory path."""
    model = EBANO(["unused"], dataset_name="toy", random_state=np.random.RandomState(42))
    
    with pytest.raises(TypeError, match="must be an integer"):
        model.fit(np.zeros((2, 1)), np.array([0, 1]))


@pytest.mark.parametrize(
    "param_name, invalid_value",
    [
        ("models_saved_results_paths", "not_a_list"),
        ("dataset_name", 123),
        ("weights_cv_n_iters", 0),
        ("weights_cv_n_iters", 10.5),
        ("random_state", "seed"),
    ]
)
def test_ebano_hyperparameter_validation(param_name, invalid_value):
    """Scikit-learn parameter constraints reject invalid types and values."""
    kwargs = {
        "models_saved_results_paths": ["path"],
        "dataset_name": "toy",
        "weights_cv_n_iters": 100,
        "random_state": 42
    }
    kwargs[param_name] = invalid_value
    model = EBANO(**kwargs)

    with pytest.raises(ValueError, match=rf"The '{param_name}' parameter.*"):
        model.fit(np.zeros((2, 1)), np.array([0, 1]))
