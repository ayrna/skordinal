"""Persistence and read-back layer for experiment results."""

from __future__ import annotations

import shutil
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from ._base import _check_path_component, _check_resample_id, _check_split
from ._io import (
    _atomic_dump,
    _atomic_write,
    _parse_proba_column,
    _sweep_orphaned_temp_files,
    _write_split_files,
)


@dataclass(frozen=True, kw_only=True)
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

    scaler : transformer or None, default=None
        Fitted scaler that produced the inputs ``best_model`` was fitted on,
        or ``None`` when no preprocessing ran. ``best_model`` stays bare;
        only the persisted artefact composes the two.
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
    scaler: BaseEstimator | None = None


class Results:
    """Persist experiment results under ``path`` and read them back.

    Parameters
    ----------
    path : str or Path
        Directory where all results for this run will be stored. Used directly
        as the experiment root; no timestamp subfolder is created.

    Attributes
    ----------
    path : Path
        Expanded, absolute path to the experiment folder.

    Notes
    -----
    On-disk layout under ``path``::

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
    sample indices were recorded. ``Prediction`` always reflects the
    estimator's own ``predict`` decision, even when probability
    estimates are stored alongside it. Each
    ``*_confusion_matrix.txt`` holds the confusion matrix of the same
    file's ``Target`` and ``Prediction`` columns.
    Each ``models/<resample_id>.joblib`` holds the fitted estimator, or a
    ``Pipeline`` of the run's scaler and that estimator when the inputs were
    scaled, so the artefact always accepts raw features.
    ``hyperparameter_configuration.csv`` records the best parameters per
    seed with the ``clf__`` pipeline prefix stripped; its ``Seed`` column
    always holds the resample identifier. ``report.csv`` is indexed by
    ``resample_id``, in numeric order. The root-level ``train_summary.csv``
    and ``test_summary.csv`` files are written by ``save_summary`` and are
    absent until it is called.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()

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
            # Wrap the scaler in so the artefact accepts raw features
            artefact = (
                Pipeline([("scaler", result.scaler), ("clf", result.best_model)])
                if result.scaler is not None
                else result.best_model
            )
            _atomic_dump(model_path, artefact)

        self._upsert_hyperparameters(result, base_dir)
        # Write the report row last: it is the commit marker for exists()
        self._append_report_row(result, base_dir)

    def _pair_dir(self, classifier_name: str, dataset_name: str) -> Path:
        """Validate both components and return the classifier/dataset dir."""
        _check_path_component(classifier_name, "classifier_name")
        _check_path_component(dataset_name, "dataset_name")
        return self.path / classifier_name / dataset_name

    def _resample_paths(self, base: Path, resample_id: int | str) -> tuple[Path, Path]:
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
        _atomic_write(csv_path, df.to_csv(index_label="resample_id"))

    def _append_report_row(self, result: ExperimentResult, base_dir: Path) -> None:
        """Append one metrics row to report.csv."""
        row: dict[str, Any] = {**result.train_metrics, **result.test_metrics}

        csv_path = base_dir / "report.csv"
        df = pd.DataFrame([row], index=pd.Index([str(result.resample_id)]))
        if csv_path.is_file():
            existing = pd.read_csv(csv_path, index_col=0, float_precision="round_trip")
            existing.index = existing.index.astype(str)
            df = pd.concat([existing, df])
        try:
            df = df.sort_index(key=lambda index: index.map(int))
        except ValueError:
            # Fall back to lexicographic order for non-integer ids
            df = df.sort_index()
        _atomic_write(csv_path, df.to_csv(index_label="resample_id"))

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
    def load(cls, path: str | Path) -> Results:
        """Load an existing experiment folder for post-hoc analysis.

        Parameters
        ----------
        path : str or Path
            Path to an already-populated experiment folder. The folder does not
            need to exist at construction time; it is only accessed when a
            method such as ``exists`` is called.

        Returns
        -------
        Results
            A ``Results`` instance pointing at ``path``.

        Examples
        --------
        >>> from pathlib import Path
        >>> from skordinal.experiments import Results
        >>> results = Results.load(Path("/path/to/my-run"))  # doctest: +SKIP
        """
        return cls(path)

    def _readable_seed_files(
        self, classifier_name: str, dataset_name: str, split: str
    ) -> list[tuple[int | str, Path]]:
        """Return sorted ``(resample_id, path)`` pairs of one split's readable seeds.

        Where a ``report.csv`` exists it is the commit marker, so a seed
        directory with no row in it was left behind by a crashed save and is
        skipped, and a row that does carry ``{split}`` metrics but has no
        predictions file is corruption, not a train-only run, and warns.

        Where the pair has none there is no marker to consult, so every seed
        holding the split's predictions file is read on the strength of that
        file, which ``_atomic_write`` leaves either complete or absent. A save
        that crashed before writing any report row falls here too.

        Ids sort like ``report.csv``'s index.
        """
        seeds_dir = (
            self._pair_dir(classifier_name, dataset_name) / "predictions_by_seed"
        )
        if not seeds_dir.is_dir():
            raise FileNotFoundError(f"No predictions directory at {seeds_dir}.")
        report: pd.DataFrame | None
        try:
            report = self.report(classifier_name, dataset_name)
        except FileNotFoundError:
            report = None
        else:
            report.index = report.index.astype(str)
        split_cols = (
            [c for c in report.columns if c.endswith(f"_{split}")]
            if report is not None
            else []
        )

        files: list[tuple[int | str, Path]] = []
        for seed_dir in seeds_dir.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            str_id = seed_dir.name.removeprefix("seed_")
            # Only report.csv membership decides: non-integer ids are legal
            if report is not None and str_id not in report.index:
                continue
            try:
                resample_id: int | str = int(str_id)
            except ValueError:
                resample_id = str_id
            path = seed_dir / f"{split}_predictions.csv"
            if path.is_file():
                files.append((resample_id, path))
            elif (
                report is not None
                and split_cols
                and report.loc[str_id, split_cols].notna().any()
            ):
                warnings.warn(
                    f"{classifier_name}/{dataset_name} seed {resample_id} is "
                    f"committed to report.csv with {split} metrics, but {path} "
                    "is missing.",
                    RuntimeWarning,
                    # Frames: here <- evaluate <- caller
                    stacklevel=3,
                )
        try:
            files.sort(key=lambda item: int(item[0]))
        except ValueError:
            # Fall back to lexicographic order for non-integer ids
            files.sort(key=lambda item: str(item[0]))
        return files

    def report(self, classifier_name: str, dataset_name: str) -> pd.DataFrame:
        """Return the stored per-seed metrics of one classifier/dataset pair.

        Parameters
        ----------
        classifier_name : str
            Name of the classifier configuration.

        dataset_name : str
            Name of the dataset.

        Returns
        -------
        pd.DataFrame
            Contents of ``report.csv``, indexed by resample id.

        Raises
        ------
        TypeError
            If ``classifier_name`` or ``dataset_name`` is not a string.

        ValueError
            If ``classifier_name`` or ``dataset_name`` is empty, a dot
            segment, or contains a path separator.

        FileNotFoundError
            If the pair has no ``report.csv``.

        Examples
        --------
        >>> from skordinal.experiments import Results
        >>> results = Results.load("/path/to/my-run")  # doctest: +SKIP
        >>> results.report("SVC", "toy")  # doctest: +SKIP
        """
        path = self._pair_dir(classifier_name, dataset_name) / "report.csv"
        if not path.is_file():
            raise FileNotFoundError(f"No report found at {path}.")
        return pd.read_csv(path, index_col=0, float_precision="round_trip")

    def hyperparameters(self, classifier_name: str, dataset_name: str) -> pd.DataFrame:
        """Return the per-seed best parameters of one classifier/dataset pair.

        Parameters
        ----------
        classifier_name : str
            Name of the classifier configuration.

        dataset_name : str
            Name of the dataset.

        Returns
        -------
        pd.DataFrame
            Contents of ``hyperparameter_configuration.csv``, one row per
            resample, with the resample id in the ``Seed`` column.

        Raises
        ------
        TypeError
            If ``classifier_name`` or ``dataset_name`` is not a string.

        ValueError
            If ``classifier_name`` or ``dataset_name`` is empty, a dot
            segment, or contains a path separator.

        FileNotFoundError
            If the pair has no ``hyperparameter_configuration.csv``.

        Examples
        --------
        >>> from skordinal.experiments import Results
        >>> results = Results.load("/path/to/my-run")  # doctest: +SKIP
        >>> results.hyperparameters("SVC", "toy")  # doctest: +SKIP
        """
        path = (
            self._pair_dir(classifier_name, dataset_name)
            / "hyperparameter_configuration.csv"
        )
        if not path.is_file():
            raise FileNotFoundError(f"No hyperparameter configuration at {path}.")
        return pd.read_csv(path, float_precision="round_trip")

    def predictions(
        self,
        classifier_name: str,
        dataset_name: str,
        resample_id: int | str,
        *,
        split: str = "test",
        parse_proba: bool = False,
    ) -> pd.DataFrame:
        """Return one resample's stored predictions for a single split.

        Parameters
        ----------
        classifier_name : str
            Name of the classifier configuration.

        dataset_name : str
            Name of the dataset.

        resample_id : int or str
            Partition identifier naming the seed directory.

        split : {"test", "train"}, default="test"
            Which of the two per-seed predictions files to read.

        parse_proba : bool, default=False
            Whether to expand the ``Prediction probabilities`` cells into
            ``ndarray`` rows instead of leaving them as stored strings.

        Returns
        -------
        pd.DataFrame
            Contents of ``{split}_predictions.csv``, with the columns
            described in the class ``Notes``.

        Raises
        ------
        TypeError
            If ``classifier_name`` or ``dataset_name`` is not a string.

        ValueError
            If ``classifier_name``, ``dataset_name`` or a non-int
            ``resample_id`` is empty, a dot segment, or contains a path
            separator, or if ``split`` is not ``"test"`` or ``"train"``.

        FileNotFoundError
            If the resample has no predictions file for ``split``.

        Examples
        --------
        >>> from skordinal.experiments import Results
        >>> results = Results.load("/path/to/my-run")  # doctest: +SKIP
        >>> results.predictions("SVC", "toy", 0, parse_proba=True)  # doctest: +SKIP
        """
        _check_resample_id(resample_id)
        _check_split(split, allow_both=False)
        seed_dir, _ = self._resample_paths(
            self._pair_dir(classifier_name, dataset_name), resample_id
        )
        path = seed_dir / f"{split}_predictions.csv"
        if not path.is_file():
            raise FileNotFoundError(f"No predictions file at {path}.")
        df = pd.read_csv(path)
        if parse_proba and "Prediction probabilities" in df.columns:
            df["Prediction probabilities"] = list(
                _parse_proba_column(df["Prediction probabilities"])
            )
        return df

    def model(
        self,
        classifier_name: str,
        dataset_name: str,
        resample_id: int | str,
    ) -> BaseEstimator:
        """Load one resample's persisted model artefact.

        Parameters
        ----------
        classifier_name : str
            Name of the classifier configuration.

        dataset_name : str
            Name of the dataset.

        resample_id : int or str
            Partition identifier naming the model artefact.

        Returns
        -------
        estimator
            The joblib-loaded artefact: the bare fitted estimator, or a
            ``Pipeline`` chaining the run's scaler and that estimator when
            the partition was scaled.

        Raises
        ------
        TypeError
            If ``classifier_name`` or ``dataset_name`` is not a string.

        ValueError
            If ``classifier_name``, ``dataset_name`` or a non-int
            ``resample_id`` is empty, a dot segment, or contains a path
            separator.

        FileNotFoundError
            If the resample was saved with ``save_model=False`` or its
            artefact has since been removed.

        Examples
        --------
        >>> from skordinal.experiments import Results
        >>> results = Results.load("/path/to/my-run")  # doctest: +SKIP
        >>> results.model("SVC", "toy", 0)  # doctest: +SKIP
        """
        _check_resample_id(resample_id)
        _, model_path = self._resample_paths(
            self._pair_dir(classifier_name, dataset_name), resample_id
        )
        if not model_path.is_file():
            raise FileNotFoundError(f"No model artefact at {model_path}.")
        return joblib.load(model_path)

    def exists(
        self,
        classifier_name: str,
        dataset_name: str,
        resample_id: int | str,
    ) -> bool:
        """Return whether a partition result has already been saved.

        Parameters
        ----------
        classifier_name : str
            Name of the classifier configuration.

        dataset_name : str
            Name of the dataset.

        resample_id : int or str
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
            If ``classifier_name``, ``dataset_name`` or a non-int
            ``resample_id`` is empty, a dot segment, or contains a path
            separator.

        Examples
        --------
        >>> from skordinal.experiments import Results
        >>> results = Results.load("/path/to/my-run")  # doctest: +SKIP
        >>> results.exists("SVC", "toy", 0)  # doctest: +SKIP
        False
        """
        _check_resample_id(resample_id)
        csv_path = self._pair_dir(classifier_name, dataset_name) / "report.csv"
        if not csv_path.is_file():
            return False
        df = pd.read_csv(csv_path, index_col=0)
        return str(resample_id) in df.index.astype(str)

    def iter_experiments(self) -> Iterator[tuple[str, str]]:
        """Yield every readable pair as ``(classifier_name, dataset_name)``.

        A pair qualifies on a ``report.csv``, the commit marker, or failing
        that on a ``predictions_by_seed`` directory, which is how a tree
        written by another tool is discovered. Read a yielded pair with
        ``report``, ``hyperparameters``, ``predictions`` or ``model``; each
        raises ``FileNotFoundError`` when its own file is absent, so only
        ``report`` is guaranteed to fail for a pair carrying no
        ``report.csv``.

        Yields
        ------
        tuple of (str, str)
            Classifier name and dataset name of a readable pair.

        Examples
        --------
        >>> from skordinal.experiments import Results
        >>> results = Results.load("/path/to/my-run")  # doctest: +SKIP
        >>> list(results.iter_experiments())  # doctest: +SKIP
        """
        if not self.path.is_dir():
            return
        for clf_dir in sorted(p for p in self.path.iterdir() if p.is_dir()):
            for ds_dir in sorted(p for p in clf_dir.iterdir() if p.is_dir()):
                if (ds_dir / "report.csv").is_file() or (
                    ds_dir / "predictions_by_seed"
                ).is_dir():
                    yield clf_dir.name, ds_dir.name
