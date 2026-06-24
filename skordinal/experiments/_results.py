"""Results handling for storing and managing experiment results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


@dataclass(frozen=True)
class ExperimentResult:
    """Result of running a single classifier on one dataset partition.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.

    classifier_name : str
        Name of the classifier configuration.

    resample_id : int
        Partition identifier.

    train_predicted_y : ndarray of shape (n_train_samples,)
        Class predictions on the training partition.

    test_predicted_y : ndarray of shape (n_test_samples,) or None
        Class predictions on the test partition. ``None`` if no test partition
        was available.

    y_proba : ndarray of shape (n_test_samples, n_classes) or None
        Class probability estimates on the test partition. ``None`` if the
        estimator does not support ``predict_proba``.

    train_metrics : dict
        Metric values computed on the training partition, including timing.

    test_metrics : dict
        Metric values computed on the test partition, including timing.

    best_params : dict
        Best hyper-parameter values found during cross-validation.

    best_model : estimator
        Fitted estimator selected during cross-validation or direct fit.

    train_true_y : ndarray of shape (n_train_samples,) or None, default=None
        True class labels for the training partition. When provided, predictions
        CSV files include a ``y_true`` column alongside ``y_pred``.

    test_true_y : ndarray of shape (n_test_samples,) or None, default=None
        True class labels for the test partition. When provided, the test
        predictions CSV file includes a ``y_true`` column alongside ``y_pred``.

    """

    dataset_name: str
    classifier_name: str
    resample_id: int
    train_predicted_y: np.ndarray
    test_predicted_y: np.ndarray | None
    y_proba: np.ndarray | None
    train_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    best_params: dict[str, Any]
    best_model: BaseEstimator
    train_true_y: np.ndarray | None = None
    test_true_y: np.ndarray | None = None


class Results:
    """Handle all information from an experiment that needs to be saved.

    Parameters
    ----------
    output_folder : str or Path
        Directory where all results for this run will be stored. Used directly
        as the experiment root; no timestamp subfolder is created.

    Attributes
    ----------
    _experiment_folder : Path
        Path to the experiment folder.

    Notes
    -----
    On-disk layout under ``_experiment_folder``::

        train_summary.csv
        test_summary.csv
        <classifier_name>/
            <dataset_name>/
                report.csv
                params.json
                predictions/
                    train_<resample_id>.csv
                    test_<resample_id>.csv
                models/
                    <resample_id>.joblib

    The root-level ``train_summary.csv`` and ``test_summary.csv`` files are
    written by ``save_summary`` and are absent until it is called.

    """

    def __init__(self, output_folder: str | Path) -> None:
        self._experiment_folder = Path(output_folder)

    def save(
        self,
        result: ExperimentResult,
        *,
        save_model: bool = True,
    ) -> None:
        """Store information obtained from the run of one partition.

        Parameters
        ----------
        result : ExperimentResult
            All data produced by a single classifier run on one partition.

        save_model : bool, default=True
            Whether to persist the fitted model to disk with joblib.

        Raises
        ------
        OSError
            If the folder cannot be created.

        Examples
        --------
        >>> from skordinal.experiments import Results
        >>> results = Results("/path/to/my-run")  # doctest: +SKIP
        >>> results.save(result)  # doctest: +SKIP

        """
        base_dir, models_dir, pred_dir = self._ensure_dirs(
            result.classifier_name, result.dataset_name, save_model=save_model
        )

        # Write model and prediction CSVs
        if save_model:
            joblib.dump(result.best_model, models_dir / f"{result.resample_id}.joblib")
        train_df = pd.DataFrame({"y_pred": result.train_predicted_y})
        if result.train_true_y is not None:
            train_df.insert(0, "y_true", result.train_true_y)
        train_df.to_csv(pred_dir / f"train_{result.resample_id}.csv", index=False)
        if result.test_predicted_y is not None:
            test_df = pd.DataFrame({"y_pred": result.test_predicted_y})
            if result.test_true_y is not None:
                test_df.insert(0, "y_true", result.test_true_y)
            test_df.to_csv(pred_dir / f"test_{result.resample_id}.csv", index=False)

        self._append_report_row(result, base_dir)

        # Upsert params entry in params.json
        json_path = base_dir / "params.json"
        params: dict[str, Any] = {}
        if json_path.is_file():
            params = json.loads(json_path.read_text(encoding="utf-8"))
        params[str(result.resample_id)] = dict(result.best_params)
        json_path.write_text(json.dumps(params, indent=2), encoding="utf-8")

    def _ensure_dirs(
        self, classifier_name: str, dataset_name: str, *, save_model: bool
    ) -> tuple[Path, Path, Path]:
        """Create required sub-directories and return ``(base_dir, models_dir, pred_dir)``."""
        base = self._experiment_folder / classifier_name / dataset_name
        pred_dir = base / "predictions"
        models_dir = base / "models"
        try:
            pred_dir.mkdir(parents=True, exist_ok=True)
            if save_model:
                models_dir.mkdir(exist_ok=True)
        except OSError:
            raise OSError(
                f"Could not create folder {base} (or subfolders) to store results."
            )
        return base, models_dir, pred_dir

    def _append_report_row(self, result: ExperimentResult, base_dir: Path) -> None:
        """Append one metrics row to ``report.csv``."""
        row: dict[str, Any] = {**result.train_metrics, **result.test_metrics}

        csv_path = base_dir / "report.csv"
        df = pd.DataFrame([row], index=pd.Index([result.resample_id], dtype=str))
        if csv_path.is_file():
            existing = pd.read_csv(csv_path, index_col=0)
            existing.index = existing.index.astype(str)
            df = pd.concat([existing, df])
        df.to_csv(csv_path)

    @classmethod
    def load(cls, experiment_folder: str | Path) -> Results:
        """Load an existing experiment folder for post-hoc analysis.

        Parameters
        ----------
        experiment_folder : str or Path
            Path to an already-populated experiment folder. The folder does not
            need to exist at construction time; it is only accessed when a
            method such as ``exists`` is called.

        Returns
        -------
        Results
            A ``Results`` instance pointing at ``experiment_folder``.

        Examples
        --------
        >>> from pathlib import Path
        >>> from skordinal.experiments import Results
        >>> results = Results.load(Path("/path/to/my-run"))  # doctest: +SKIP

        """
        return cls(experiment_folder)

    def exists(
        self,
        classifier_name: str,
        dataset_name: str,
        resample_id: str,
    ) -> bool:
        """Return whether a partition result has already been saved.

        Parameters
        ----------
        classifier_name : str
            Name of the classifier configuration.

        dataset_name : str
            Name of the dataset.

        resample_id : str
            Partition identifier (the CSV row index).

        Returns
        -------
        bool
            ``True`` if the per-pair CSV exists **and** contains a row
            whose index equals ``resample_id``.

        Examples
        --------
        >>> from skordinal.experiments import Results
        >>> results = Results.load("/path/to/my-run")  # doctest: +SKIP
        >>> results.exists("SVC", "toy", "0")  # doctest: +SKIP
        False

        """
        csv_path = (
            self._experiment_folder / classifier_name / dataset_name / "report.csv"
        )
        if not csv_path.is_file():
            return False
        df = pd.read_csv(csv_path, index_col=0)
        return resample_id in df.index.astype(str)
