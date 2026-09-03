"""Per-algorithm configuration: estimator binding and hyper-parameter grid."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone

from ._base import _set_nested_random_state


@dataclass(frozen=True)
class ModelConfig:
    """Bind a sklearn estimator to an optional hyper-parameter grid.

    An immutable description of what to run, carrying no evaluation
    protocol (cv folds, metric, seed): those live on the orchestrator.

    Parameters
    ----------
    estimator : BaseEstimator
        Untrained sklearn-compatible estimator instance.

    param_grid : dict[str, Any] or None, keyword-only, default=None
        Hyper-parameter grid mapping parameter names to a value or a
        sequence of candidate values (list, tuple or array).  ``None``
        means the estimator is fitted as-is.

    Raises
    ------
    TypeError
        If ``estimator`` is not a ``BaseEstimator`` instance, or if
        ``param_grid`` is neither a ``dict`` nor ``None``.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> cfg = ModelConfig(LogisticRegression())
    >>> cfg.param_grid is None
    True
    >>> cfg_grid = ModelConfig(
    ...     LogisticRegression(),
    ...     param_grid={"C": [0.1, 1.0, 10.0]},
    ... )
    >>> cfg_grid.needs_search
    True
    """

    estimator: BaseEstimator
    param_grid: dict[str, Any] | None = field(kw_only=True, default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.estimator, BaseEstimator):
            raise TypeError(
                "estimator must be a sklearn-compatible BaseEstimator instance."
            )
        if self.param_grid is not None and not isinstance(self.param_grid, dict):
            raise TypeError("param_grid must be a dict or None.")

    @staticmethod
    def _is_search_sequence(value: Any) -> bool:
        """Report whether a grid value enumerates candidates, as ParameterGrid does."""
        if isinstance(value, np.ndarray):
            # A 0-d array holds one value, and has no len() to enumerate
            return value.ndim > 0
        return isinstance(value, Sequence) and not isinstance(value, str)

    @property
    def needs_search(self) -> bool:
        """Return ``True`` iff the grid has at least one multi-value entry.

        Returns
        -------
        needs_search : bool
            ``True`` when a grid value is a non-string sequence (list, tuple
            or array) holding more than one element; ``False`` otherwise,
            which includes a ``None`` or empty grid.
        """
        if not self.param_grid:
            return False
        return any(
            self._is_search_sequence(v) and len(v) > 1 for v in self.param_grid.values()
        )

    def fixed_params(self) -> dict[str, Any]:
        """Return a flat parameter dict, unwrapping singleton sequences.

        Returns
        -------
        params : dict[str, Any]
            Flat parameter dict ready to pass to ``set_params``: a singleton
            sequence unwrapped to its element, an empty one skipped, a scalar
            unchanged.  Empty when ``param_grid`` is ``None``.
        """
        if self.param_grid is None:
            return {}
        out: dict[str, Any] = {}
        for key, value in self.param_grid.items():
            if self._is_search_sequence(value):
                if len(value) == 0:
                    continue
                out[key] = value[0]
            else:
                out[key] = value
        return out

    def search_grid(self) -> dict[str, Any]:
        """Return the grid with scalar values wrapped for ``GridSearchCV``.

        Returns
        -------
        grid : dict[str, Any]
            Parameter grid ready to pass to ``GridSearchCV``: sequences pass
            through, and a fixed scalar becomes the singleton list it requires.
        """
        if self.param_grid is None:
            return {}
        return {
            key: value if self._is_search_sequence(value) else [value]
            for key, value in self.param_grid.items()
        }

    def build(self, random_state: int | None = None) -> BaseEstimator:
        """Return a fresh clone of the estimator, optionally seeded.

        Clones via ``sklearn.base.clone``, so ``self.estimator`` is never
        mutated, then forwards a non-``None`` seed to every parameter named
        ``random_state`` or ending in ``__random_state``, including those of
        nested estimators and Pipeline steps.

        Parameters
        ----------
        random_state : int or None, default=None
            Seed to forward to the cloned estimator.

        Returns
        -------
        estimator : BaseEstimator
            An unfitted clone ready for ``fit``.
        """
        est = clone(self.estimator)
        _set_nested_random_state(est, random_state)
        return est
