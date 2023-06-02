"""
The ``findi`` contains public functions that a user will use to
for optimization. They wrap functions from `_numba_findi` and
`_python_findi` modules allowing increased performance and
flexibility. The functions optimize the `objective` via Gradient
Descent Algorithm variation that uses finite difference instead
of infinitesimal differential for computing derivatives. This
approach allows for the application of Gradient Descent on
non-differentiable functions, functions without analytic form or
any other function, as long as it can be evaluated. `descent`
function performs regular finite difference gradient descent
algorithm, while `partial_descent` function allow a version of
finite difference gradient descent algorithm where only a random
subset of gradients is used in each epoch. `partially_partial_descent`
function performs `partial_descent` algorithm for the first
`partial_epochs` number of epochs and `descent` for the rest of
the epochs.Parallel computing for performance benefits is supported
in all of these functions. If `numba=True` parallelization is done
by `Numba`, otherwise it is done by `joblib` library. Objective
functions with multiple outputs are supported (only the first one
is taken as objective value to be minimized), as well as
objective function metaparameters that are held constant
throughout the epochs. Furthermore, `values_out` function is
included for compactly exporting values of outputs, parameters
and metaparameters for each epoch.
"""

from findi.findi import (
    descent,
    partial_descent,
    partially_partial_descent,
    values_out,
)

__all__ = [s for s in dir() if not s.startswith("_")]

__version__ = "0.1.0"
__author__ = "draktr"
