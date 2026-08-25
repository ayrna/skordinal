"""Results handling for storing and managing experiment results."""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix

from ._io import (
    _atomic_dump,
    _atomic_write,
    _check_path_component,
    _check_resample_id,
    _format_proba_column,
    _sweep_orphaned_temp_files,
)


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


def _write_split_files(
    seed_dir: Path,
    split: str,
    *,
    index: np.ndarray | None,
    true_y: np.ndarray,
    predicted_y: np.ndarray,
    proba: np.ndarray | None,
    classes: np.ndarray,
    resample_id: int,
) -> None:
    """Encode one split's labels and write its per-seed output files."""
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
    _atomic_write(
        seed_dir / f"{split}_predictions.csv",
        pd.DataFrame(columns).to_csv(index=False),
    )

    cm = confusion_matrix(target, prediction, labels=np.arange(classes.size))
    body = np.array2string(
        cm, separator=", ", threshold=cm.size, max_line_width=sys.maxsize
    )
    _atomic_write(
        seed_dir / f"{split}_confusion_matrix.txt",
        f"Seed {resample_id}\n{'=' * 21}\n{body}\n",
    )


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
                hyperparameter_configuration.csv
                predictions_by_seed/
                    seed_<resample_id>/
                        train_predictions.csv
                        test_predictions.csv
                        train_confusion_matrix.txt
                        test_confusion_matrix.txt
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
    estimator's own decision rule reflected in ``report.csv``. Each
    ``*_confusion_matrix.txt`` holds the confusion matrix of the same
    file's ``Target`` and ``Prediction`` columns.
    ``hyperparameter_configuration.csv`` records the best parameters per
    seed with the ``clf__`` pipeline prefix stripped; its ``Seed`` column
    always holds the resample identifier. The root-level
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
        """Write per-seed files, the model, hyperparameters and the report row.

        The report row is written last and acts as the commit marker read
        by ``exists``: a save interrupted at any earlier point leaves no
        row, so a rerun detects the partition as missing and rewrites it.
        Re-saving a partition first deletes its previous seed directory and
        model artefact, so no stale file outlives the row describing it.

        Parameters
        ----------
        result : ExperimentResult
            All data produced by a single classifier run on one partition.

        save_model : bool, default=True
            Whether to persist the fitted model to disk with joblib.

        Raises
        ------
        TypeError
            If ``result.classifier_name`` or ``result.dataset_name`` is not
            a string.

        ValueError
            If ``result.classifier_name``, ``result.dataset_name`` or a
            non-int ``result.resample_id`` is empty, a dot segment, or
            contains a path separator, if ``result`` lacks the true labels
            required for the ``Target`` column (``train_true_y``, or
            ``test_true_y`` when test predictions are present), if a true
            or predicted label is not one of ``best_model.classes_``, or if
            a probability matrix does not hold one row per sample and one
            column per class. Files already written for a preceding split
            are left in place; nothing else is recorded for the partition.

        OSError
            If a stale artefact cannot be removed or the folder cannot be
            created.

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
        _check_resample_id(result.resample_id)

        base_dir = self._pair_dir(result.classifier_name, result.dataset_name)
        seed_dir, model_path = self._resample_paths(base_dir, result.resample_id)
        # Clear any temp file left by a prior crash before writing new ones
        _sweep_orphaned_temp_files(base_dir)
        # Drop this resample's committed row so a crash mid-write cannot
        # leave a report row describing predictions no longer on disk
        self._uncommit_report_row(base_dir, result.resample_id)
        # Must run before the writes below recreate what it deletes
        self._remove_stale_artefacts(base_dir, result.resample_id)

        classes = np.asarray(result.best_model.classes_)
        _write_split_files(
            seed_dir,
            "train",
            index=result.train_index,
            true_y=result.train_true_y,
            predicted_y=result.train_predicted_y,
            proba=result.train_y_proba,
            classes=classes,
            resample_id=result.resample_id,
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
                resample_id=result.resample_id,
            )

        if save_model:
            _atomic_dump(model_path, result.best_model)

        self._upsert_hyperparameters(result, base_dir)
        # Write the report row last: it is the commit marker for exists()
        self._append_report_row(result, base_dir)

    def _pair_dir(self, classifier_name: str, dataset_name: str) -> Path:
        """Validate both components and return the classifier/dataset dir."""
        _check_path_component(classifier_name, "classifier_name")
        _check_path_component(dataset_name, "dataset_name")
        return self._experiment_folder / classifier_name / dataset_name

    def _resample_paths(self, base: Path, resample_id: int) -> tuple[Path, Path]:
        """Return this resample's predictions directory and model artefact."""
        return (
            base / "predictions_by_seed" / f"seed_{resample_id}",
            base / "models" / f"{resample_id}.joblib",
        )

    def _remove_stale_artefacts(self, base_dir: Path, resample_id: int) -> None:
        """Delete this resample's seed directory and model from a prior run."""
        seed_dir, model_path = self._resample_paths(base_dir, resample_id)
        try:
            if seed_dir.is_dir():
                shutil.rmtree(seed_dir)
            model_path.unlink(missing_ok=True)
        except OSError as exc:
            raise OSError(f"Could not remove stale results under {base_dir}.") from exc

    def _uncommit_report_row(self, base_dir: Path, resample_id: int) -> None:
        """Drop this resample's row from report.csv before rewriting it."""
        csv_path = base_dir / "report.csv"
        if not csv_path.is_file():
            return
        df = pd.read_csv(csv_path, index_col=0, float_precision="round_trip")
        df.index = df.index.astype(str)
        if str(resample_id) not in df.index:
            return
        df = df.drop(index=str(resample_id))
        if df.empty:
            # Remove the commit marker entirely so exists() reports False
            csv_path.unlink()
            return
        _atomic_write(csv_path, df.to_csv())

    def _append_report_row(self, result: ExperimentResult, base_dir: Path) -> None:
        """Append one metrics row to report.csv."""
        row: dict[str, Any] = {**result.train_metrics, **result.test_metrics}

        csv_path = base_dir / "report.csv"
        df = pd.DataFrame([row], index=pd.Index([result.resample_id], dtype=str))
        if csv_path.is_file():
            existing = pd.read_csv(csv_path, index_col=0, float_precision="round_trip")
            existing.index = existing.index.astype(str)
            df = pd.concat([existing, df])
        _atomic_write(csv_path, df.to_csv())

    def _upsert_hyperparameters(self, result: ExperimentResult, base_dir: Path) -> None:
        """Upsert one seed's row in the hyperparameter configuration CSV."""
        row: dict[str, Any] = {
            k.removeprefix("clf__"): v for k, v in result.best_params.items()
        }
        # Set Seed last so no same-named parameter can shadow the upsert key
        row["Seed"] = result.resample_id

        csv_path = base_dir / "hyperparameter_configuration.csv"
        df = pd.DataFrame([row])
        if csv_path.is_file():
            existing = pd.read_csv(csv_path, float_precision="round_trip")
            existing = existing[existing["Seed"] != result.resample_id]
            df = pd.concat([existing, df], ignore_index=True)

        columns = ["Seed"] + sorted(c for c in df.columns if c != "Seed")
        df = df[columns].sort_values("Seed").reset_index(drop=True)
        # Restore integer dtypes upcast to float by the NaN column union
        _atomic_write(csv_path, df.convert_dtypes().to_csv(index=False))

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

        Raises
        ------
        TypeError
            If ``classifier_name`` or ``dataset_name`` is not a string.

        ValueError
            If ``classifier_name`` or ``dataset_name`` is empty, a dot
            segment, or contains a path separator.

        Examples
        --------
        >>> from skordinal.experiments import Results
        >>> results = Results.load("/path/to/my-run")  # doctest: +SKIP
        >>> results.exists("SVC", "toy", "0")  # doctest: +SKIP
        False

        """
        csv_path = self._pair_dir(classifier_name, dataset_name) / "report.csv"
        if not csv_path.is_file():
            return False
        df = pd.read_csv(csv_path, index_col=0)
        return resample_id in df.index.astype(str)
