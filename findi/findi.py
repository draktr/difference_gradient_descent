from findi import _python_findi
from findi import _numba_findi


def descent(numba=False):
    if not numba:
        outputs, parameters = _python_findi._python_descent()
    elif numba:
        outputs, parameters = _numba_findi._numba_descent()

    return outputs, parameters


def partial_descent(numba=False):
    if not numba:
        outputs, parameters = _python_findi._python_partial_descent()
    elif numba:
        outputs, parameters = _numba_findi._numba_partial_descent()

    return outputs, parameters


def partially_partial_descent(numba=False):
    if not numba:
        outputs, parameters = _python_findi._python_partially_partial_descent()
    elif numba:
        outputs, parameters = _numba_findi._numba_partially_partial_descent()

    return outputs, parameters


def values_out(numba=False):
    if not numba:
        values = _python_findi.values_out()
    elif numba:
        values = _numba_findi.values_out()

    return values
