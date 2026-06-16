"""Per-algorithm configuration: estimator binding and hyper-parameter grid."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sklearn.base import BaseEstimator, clone


@dataclass(frozen=True)
class ModelConfig:
    """Bind a sklearn estimator to an optional hyper-parameter grid.

    ``ModelConfig`` is a pure, immutable description of what to run.
    It carries no evaluation protocol (cv folds, metric, seed) — those
    live on the orchestrator that consumes it.

    Parameters
    ----------
    estimator : BaseEstimator
        Untrained sklearn-compatible estimator instance.

    param_grid : dict[str, Any] or None, keyword-only, default=None
        Hyper-parameter grid mapping parameter names to a value or list
        of values.  ``None`` means the estimator is fitted as-is.

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

    @property
    def needs_search(self) -> bool:
        """Return ``True`` iff the grid has at least one multi-value entry.

        A multi-value entry is a list or tuple with more than one element.
        When ``param_grid`` is ``None`` or empty, or every value is a scalar
        or singleton list/tuple, returns ``False``.

        Returns
        -------
        needs_search : bool
            ``True`` when at least one grid value is a list or tuple
            with more than one element; ``False`` otherwise.
        """
        if not self.param_grid:
            return False
        return any(
            isinstance(v, (list, tuple)) and len(v) > 1
            for v in self.param_grid.values()
        )

    def fixed_params(self) -> dict[str, Any]:
        """Return a flat parameter dict, unwrapping singleton lists/tuples.

        Singleton list or tuple values (``len == 1``) are unwrapped to their
        scalar element.  Empty list/tuple values are skipped.  Scalar values
        are passed through unchanged.  When ``param_grid`` is ``None``,
        an empty dict is returned.

        Returns
        -------
        params : dict[str, Any]
            Flat parameter dict ready to pass to ``set_params``.
        """
        if self.param_grid is None:
            return {}
        out: dict[str, Any] = {}
        for key, value in self.param_grid.items():
            if isinstance(value, (list, tuple)):
                if len(value) == 0:
                    continue
                out[key] = value[0]
            else:
                out[key] = value
        return out

    def build(self, random_state: int | None = None) -> BaseEstimator:
        """Return a fresh clone of the estimator, optionally seeded.

        Creates a clone via ``sklearn.base.clone`` so ``self.estimator``
        is never mutated.  When ``random_state`` is not ``None``, the
        seed is forwarded to the clone, using the first match:

        1. If the clone exposes ``random_state`` directly, it is set.
        2. Else if the clone exposes ``clf__random_state`` (Pipeline
           with a ``clf`` step), that is set.
        3. Otherwise the seed is silently ignored.

        Parameters
        ----------
        random_state : int or None, default=None
            Seed to forward to the cloned estimator.  Passing ``None``
            leaves the estimator's own default unchanged.

        Returns
        -------
        estimator : BaseEstimator
            An unfitted clone ready for ``fit``.
        """
        est = clone(self.estimator)
        if random_state is not None:
            params = est.get_params(deep=True)
            if "random_state" in params:
                est.set_params(random_state=random_state)
            elif "clf__random_state" in params:
                est.set_params(clf__random_state=random_state)
        return est
