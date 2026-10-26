"""Optimization utilities shared by the estimators."""

import warnings

from sklearn.exceptions import ConvergenceWarning


def _check_optimize_result(solver, result, max_iter=None):
    """Warn as scikit-learn if ``result`` did not converge and return its ``nit``.

    Copy of ``sklearn.utils.optimize._check_optimize_result`` for the
    ``OptimizeResult`` of ``scipy.optimize.minimize``, with three deviations.
    ``solver`` takes any name, where scikit-learn's accepts only ``"lbfgs"``.
    ``nit`` is not clipped to ``max_iter``, as SciPy already bounds it. There
    is no ``extra_warning_msg``, which no estimator needs. ``max_iter``, when
    given, is only used to advise raising it if the solver used it up.
    """
    if result.status != 0:
        warning_msg = (
            f"{solver} failed to converge after {result.nit} iteration(s) "
            f"(status={result.status}):\n"
            f"{result.message}\n"
        )
        # Advise more iterations only when the solver used them all
        if max_iter is not None and result.nit == max_iter:
            warning_msg += (
                "\nIncrease the number of iterations to improve the "
                f"convergence (max_iter={max_iter})."
            )
        warning_msg += (
            "\nYou might also want to scale the data as shown in:\n"
            "    https://scikit-learn.org/stable/modules/preprocessing.html"
        )
        warnings.warn(warning_msg, ConvergenceWarning, stacklevel=2)

    return result.nit
