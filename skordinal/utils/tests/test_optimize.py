"""Tests for the optimization utilities."""

import warnings

import pytest
from scipy.optimize import OptimizeResult
from sklearn.exceptions import ConvergenceWarning

from skordinal.utils._optimize import _check_optimize_result


def test_cor_converged_returns_nit_without_warning():
    """A converged result returns its iteration count and does not warn."""
    result = OptimizeResult(nit=3, status=0, message="CONVERGENCE")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        assert _check_optimize_result("lbfgs", result, 10) == 3


@pytest.mark.parametrize("solver", ["lbfgs", "newton-cg"])
def test_cor_warning_names_solver_iterations_and_reason(solver):
    """The warning names the solver, its iterations, its status and SciPy's reason."""
    result = OptimizeResult(nit=2, status=2, message="ABNORMAL")
    expected = (
        rf"^{solver} failed to converge after 2 iteration\(s\) \(status=2\):\n"
        "ABNORMAL"
    )
    with pytest.warns(ConvergenceWarning, match=expected):
        assert _check_optimize_result(solver, result, 10) == 2


@pytest.mark.parametrize(
    "nit, max_iter, advised",
    [(1, 1, True), (1, None, False), (2, 10, False)],
    ids=["used-up", "no-max-iter", "stopped-early"],
)
def test_cor_advises_max_iter_only_when_used_up(nit, max_iter, advised):
    """Raising max_iter is advised only when the solver used all its iterations."""
    result = OptimizeResult(nit=nit, status=1, message="STOP")
    with pytest.warns(ConvergenceWarning) as record:
        _check_optimize_result("lbfgs", result, max_iter)

    message = str(record[0].message)
    assert ("Increase the number of iterations" in message) is advised
    assert "scale the data" in message
