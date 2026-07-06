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
        Class probability estimates on the test partition, columns ordered
        by ``best_model.classes_``. ``None`` when no test partition is
        available or the estimator cannot provide probabilities.

    train_metrics : dict
        Metric values computed on the training partition, including timing.

    test_metrics : dict
        Metric values computed on the test partition, including timing.

    best_params : dict
        Best hyper-parameter values found during cross-validation.

    best_model : estimator
        Fitted estimator selected during cross-validation or direct fit.

    train_true_y : ndarray of shape (n_train_samples,) or None, default=None
        True class labels for the training partition, used to derive the
        ``Target`` column of the training predictions file.

    test_true_y : ndarray of shape (n_test_samples,) or None, default=None
        True class labels for the test partition, used to derive the
        ``Target`` column of the test predictions file.

    train_index : ndarray of shape (n_train_samples,) or None, default=None
        Zero-based positions of the training samples in the original,
        unsplit dataset array. Written as the ``Pattern ID`` column of the
        training predictions file. ``None`` falls back to a partition-local
        ``range(n_train_samples)`` at write time.

    test_index : ndarray of shape (n_test_samples,) or None, default=None
        Zero-based positions of the test samples in the original, unsplit
        dataset array. Written as the ``Pattern ID`` column of the test
        predictions file. ``None`` falls back to a partition-local
        ``range(n_test_samples)`` at write time.

    train_y_proba : ndarray of shape (n_train_samples, n_classes) or None, \
            default=None
        Class-probability estimates on the training partition, columns
        ordered by ``best_model.classes_``. ``None`` when the estimator
        cannot provide probabilities.

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
    train_index: np.ndarray | None = None
    test_index: np.ndarray | None = None
    train_y_proba: np.ndarray | None = None


def _format_proba_column(proba: np.ndarray) -> list[str]:
    """Render probability rows as single-line ``"[p0, p1, ...]"`` cells."""
    return [
        "[" + ", ".join(repr(float(p)) for p in row) + "]" for row in np.asarray(proba)
    ]


def _write_split_files(
    seed_dir: Path,
    split: str,
    *,
    index: np.ndarray | None,
    true_y: np.ndarray,
    predicted_y: np.ndarray,
    proba: np.ndarray | None,
    classes: np.ndarray,
) -> None:
    """Encode one split's labels and write its predictions file."""
    if not np.isin(true_y, classes).all():
        raise ValueError(
            f"'{split}' true labels contain classes unknown to the fitted model."
        )
    pattern_id = index if index is not None else np.arange(predicted_y.shape[0])
    target = np.searchsorted(classes, true_y)

    columns: dict[str, object] = {"Pattern ID": pattern_id, "Target": target}
    if proba is not None:
        if proba.shape != (true_y.shape[0], classes.size):
            raise ValueError(
                f"'{split}' probabilities have shape {proba.shape}; expected "
                f"({true_y.shape[0]}, {classes.size})."
            )
        columns["Prediction probabilities"] = _format_proba_column(proba)
        # Derive the prediction as the argmax index of the probabilities
        prediction = np.argmax(proba, axis=1)
    else:
        if not np.isin(predicted_y, classes).all():
            raise ValueError(
                f"'{split}' predicted labels contain classes unknown to the "
                "fitted model."
            )
        prediction = np.searchsorted(classes, predicted_y)
    columns["Prediction"] = prediction
    pd.DataFrame(columns).to_csv(seed_dir / f"{split}_predictions.csv", index=False)


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
                predictions_by_seed/
                    seed_<resample_id>/
                        train_predictions.csv
                        test_predictions.csv
                models/
                    <resample_id>.joblib

    Each ``*_predictions.csv`` has columns ``Pattern ID``, ``Target``, an
    optional ``Prediction probabilities`` column (present only when
    probability estimates are available), and ``Prediction``. ``Target``
    and ``Prediction`` are zero-based class indices into
    ``best_model.classes_``; ``Pattern ID`` is the sample's position in the
    original dataset array, or its position within the partition when no
    sample indices were recorded. When probabilities are present,
    ``Prediction`` is their argmax index, which may differ from the
    estimator's own decision rule reflected in ``report.csv``. The root-level
    ``train_summary.csv`` and ``test_summary.csv`` files are written by
    ``save_summary`` and are absent until it is called.

    """

    def __init__(self, output_folder: str | Path) -> None:
        self._experiment_folder = Path(output_folder)

    def save(
        self,
        result: ExperimentResult,
        *,
        save_model: bool = True,
    ) -> None:
        """Write one partition's per-seed predictions, report row and model.

        Parameters
        ----------
        result : ExperimentResult
            All data produced by a single classifier run on one partition.

        save_model : bool, default=True
            Whether to persist the fitted model to disk with joblib.

        Raises
        ------
        ValueError
            If ``result`` lacks the true labels required for the ``Target``
            column (``train_true_y``, or ``test_true_y`` when test
            predictions are present), if a true or predicted label is not
            one of ``best_model.classes_``, or if a probability matrix does
            not hold one row per sample and one column per class. Files
            already written for a preceding split are left in place;
            nothing else is recorded for the partition.

        OSError
            If the folder cannot be created.

        Examples
        --------
        >>> from skordinal.experiments import Results
        >>> results = Results("/path/to/my-run")  # doctest: +SKIP
        >>> results.save(result)  # doctest: +SKIP

        """
        if result.train_true_y is None:
            raise ValueError(
                "'result.train_true_y' is required to write the 'Target' "
                "column of the training predictions file."
            )
        if result.test_predicted_y is not None and result.test_true_y is None:
            raise ValueError(
                "'result.test_true_y' is required to write the 'Target' "
                "column of the test predictions file."
            )

        base_dir, models_dir, seed_dir = self._ensure_dirs(
            result.classifier_name,
            result.dataset_name,
            result.resample_id,
            save_model=save_model,
        )

        classes = np.asarray(result.best_model.classes_)
        _write_split_files(
            seed_dir,
            "train",
            index=result.train_index,
            true_y=result.train_true_y,
            predicted_y=result.train_predicted_y,
            proba=result.train_y_proba,
            classes=classes,
        )
        if result.test_predicted_y is not None:
            assert result.test_true_y is not None
            _write_split_files(
                seed_dir,
                "test",
                index=result.test_index,
                true_y=result.test_true_y,
                predicted_y=result.test_predicted_y,
                proba=result.y_proba,
                classes=classes,
            )

        if save_model:
            joblib.dump(result.best_model, models_dir / f"{result.resample_id}.joblib")

        self._append_report_row(result, base_dir)

        # Upsert params entry in params.json
        json_path = base_dir / "params.json"
        params: dict[str, Any] = {}
        if json_path.is_file():
            params = json.loads(json_path.read_text(encoding="utf-8"))
        params[str(result.resample_id)] = dict(result.best_params)
        json_path.write_text(json.dumps(params, indent=2), encoding="utf-8")

    def _ensure_dirs(
        self,
        classifier_name: str,
        dataset_name: str,
        resample_id: int,
        *,
        save_model: bool,
    ) -> tuple[Path, Path, Path]:
        """Create required sub-directories and return their paths."""
        base = self._experiment_folder / classifier_name / dataset_name
        seed_dir = base / "predictions_by_seed" / f"seed_{resample_id}"
        models_dir = base / "models"
        try:
            seed_dir.mkdir(parents=True, exist_ok=True)
            if save_model:
                models_dir.mkdir(exist_ok=True)
        except OSError:
            raise OSError(
                f"Could not create folder {base} (or subfolders) to store results."
            )
        return base, models_dir, seed_dir

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
