"""
The ``findi`` houses public functions that a user will use to
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
is taken as objective value to be minimized), as well as constant
objective function quasi-hyperparameters that are held constant
throughout the epochs. Furthermore, `values_out` function is
included for compactly exporting values of outputs, parameters
and constants for each epoch.
"""

from findi import _python_findi, _numba_findi, _checks


def descent(
    objective, initial, h, l, epochs, constants=None, momentum=0, threads=1, numba=False
):
    """
    Performs Gradient Descent Algorithm by using finite difference instead
    of infinitesimal differential. Allows for the implementation of gradient
    descent algorithm on variety of non-standard functions.

    :param objective: Objective function to be minimized. If `numba=True`,
                      `objective` has to be a `Numba` function
    :type objective: callable
    :param initial: Initial values of objective function parameters
    :type initial: int, float, list, nb.typed.List or ndarray
    :param h: Small change(s) in `x`. Can be a sequence or a number
              in which case constant change is used
    :type h: int, float, list, nb.typed.List or ndarray
    :param l: Learning rate(s). Can be a sequence or a number in which
              case constant learning rate is used
    :type l: int, float, list, nb.typed.List or ndarray
    :param epochs: Number of epochs
    :type epochs: int
    :param constants: Constants values used for the evaluation of the objective function
                      that aren't adjusted in the process of gradient descent, defaults to None
    :type constants: list, nb.typed.List or ndarray, optional
    :param momentum: Hyperparameter that dampens oscillations.
                     `momentum=0` implies vanilla algorithm, defaults to 0
    :type momentum: int or float, optional
    :param threads: Number of CPU threads used by `joblib` for computation.
                    Argument only used when `numba=False`, as in the other
                    case `Numba` takes case of parallelization, defaults to 1
    :type threads: int, optional
    :param numba: Whether to use `Numba`'s just-in-time compiler for performance
                  improvements. If `numba=True` the function provided in argument
                  `objective` has to be a `Numba` function (i.e. it has to be
                  decorated with one of the relevant `Numba` decorators such as
                  `@numba.njit`). For more information refer to `Numba documentation
                  <https://numba.pydata.org/numba-doc/dev/user/5minguide.html>`__,
                  defaults to False
    :type numba: bool, optional
    :return: Objective function outputs and parameters for each epoch
    :rtype: ndarray
    """

    _checks._check_numba(numba=numba)

    if not numba:
        outputs, parameters = _python_findi._python_descent(
            objective,
            initial,
            h,
            l,
            epochs,
            constants,
            momentum,
            threads,
        )
    elif numba:
        outputs, parameters = _numba_findi._numba_descent(
            objective,
            initial,
            h,
            l,
            epochs,
            constants,
            momentum,
        )

    return outputs, parameters


def partial_descent(
    objective,
    initial,
    h,
    l,
    epochs,
    parameters_used,
    constants=None,
    momentum=0,
    threads=1,
    rng_seed=88,
    numba=False,
):
    """
    Performs Gradient Descent Algorithm by computing derivatives on only
    specified number of randomly selected parameters in each epoch and
    by using finite difference instead of infinitesimal differential.
    Allows for the implementation of gradient descent algorithm on
    variety of non-standard functions.

    :param objective: Objective function to be minimized. If `numba=True`,
                      `objective` has to be a `Numba` function
    :type objective: callable
    :param initial: Initial values of objective function parameters
    :type initial: int, float, list, nb.typed.List or ndarray
    :param h: Small change(s) in `x`. Can be a sequence or a number
              in which case constant change is used
    :type h: int, float, list, nb.typed.List or ndarray
    :param l: Learning rate(s). Can be a sequence or a number in
              which case constant learning rate is used
    :type l: int, float, list, nb.typed.List or ndarray
    :param epochs: Number of epochs
    :type epochs: int
    :param parameters_used: Number of parameters used in each epoch
                            for computation of gradients
    :type parameters_used: int
    :param constants: Constants values used for the evaluation of the objective function
                      that aren't adjusted in the process of gradient descent, defaults to None
    :type constants: list, nb.typed.List or ndarray, optional
    :param momentum: Hyperparameter that dampens oscillations.
                     `momentum=0` implies vanilla algorithm, defaults to 0
    :type momentum: int or float, optional
    :param threads: Number of CPU threads used by `joblib` for computation.
                    Argument only used when `numba=False`, as in the other
                    case `Numba` takes case of parallelization, defaults to 1
    :type threads: int, optional
    :param rng_seed: Seed for the random number generator used for
                     determining which parameters are used in each
                     epoch for computation of gradients, defaults to 88
    :type rng_seed: int, optional
    :param numba: Whether to use `Numba`'s just-in-time compiler for performance
                  improvements. If `numba=True` the function provided in argument
                  `objective` has to be a `Numba` function (i.e. it has to be
                  decorated with one of the relevant `Numba` decorators such as
                  `@numba.njit`). For more information refer to `Numba documentation
                  <https://numba.pydata.org/numba-doc/dev/user/5minguide.html>`__,
                  defaults to False
    :type numba: bool, optional
    :return: Objective function outputs and parameters for each epoch
    :rtype: ndarray
    """

    _checks._check_numba(numba=numba)

    if not numba:
        outputs, parameters = _python_findi._python_partial_descent(
            objective,
            initial,
            h,
            l,
            epochs,
            parameters_used,
            constants,
            momentum,
            threads,
            rng_seed,
        )
    elif numba:
        outputs, parameters = _numba_findi._numba_partial_descent(
            objective,
            initial,
            h,
            l,
            epochs,
            parameters_used,
            constants,
            momentum,
            rng_seed,
        )

    return outputs, parameters


def partially_partial_descent(
    objective,
    initial,
    h,
    l,
    partial_epochs,
    total_epochs,
    parameters_used,
    constants=None,
    momentum=0,
    threads=1,
    rng_seed=88,
    numba=False,
):
    """
    Performs Partial Gradient Descent Algorithm for the first `partial_epochs`
    epochs and regular Finite Difference Gradient Descent for the rest of the
    epochs (i.e. `total_epochs`-`partial_epochs`).

    :param objective: Objective function to be minimized. If `numba=True`,
                      `objective` has to be a `Numba` function
    :type objective: callable
    :param initial: Initial values of objective function parameters
    :type initial: int, float, list, nb.typed.List or ndarray
    :param h: Small change(s) in `x`. Can be a sequence or a number
              in which case constant change is used
    :type h: int, float, list, nb.typed.List or ndarray
    :param l: Learning rate(s). Can be a sequence or a number in which
              case constant learning rate is used
    :type l: int, float, list, nb.typed.List or ndarray
    :param partial_epochs: Number of epochs for Partial Gradient Descent
    :type partial_epochs: int
    :param total_epochs: Total number of epochs including both for partial
                         and regular algorithms. Implies that the number of
                         epochs for the regular algorithm is given as
                         `total_epochs`-`partial_epochs`
    :type total_epochs: int
    :param parameters_used: Number of parameters used in each epoch for
                            computation of gradients
    :type parameters_used: int
    :param constants: Constants values used for the evaluation of the objective function
                      that aren't adjusted in the process of gradient descent, defaults to None
    :type constants: list, nb.typed.List or ndarray, optional
    :param momentum: Hyperparameter that dampens oscillations.
                     `momentum=0` implies vanilla algorithm, defaults to 0
    :type momentum: int or float, optional
    :param threads: Number of CPU threads used by `joblib` for computation.
                    Argument only used when `numba=False`, as in the other
                    case `Numba` takes case of parallelization, defaults to 1
    :type threads: int, optional
    :param rng_seed: Seed for the random number generator used for determining
                     which parameters are used in each epoch for computation
                     of gradients, defaults to 88
    :type rng_seed: int, optional
    :param numba: Whether to use `Numba`'s just-in-time compiler for performance
                  improvements. If `numba=True` the function provided in argument
                  `objective` has to be a `Numba` function (i.e. it has to be
                  decorated with one of the relevant `Numba` decorators such as
                  `@numba.njit`). For more information refer to `Numba documentation
                  <https://numba.pydata.org/numba-doc/dev/user/5minguide.html>`__,
                  defaults to False
    :type numba: bool, optional
    :return: Objective function outputs and parameters for each epoch
    :rtype: ndarray
    """

    if not numba:
        outputs, parameters = _python_findi._python_partially_partial_descent(
            objective,
            initial,
            h,
            l,
            partial_epochs,
            total_epochs,
            parameters_used,
            constants,
            momentum,
            threads,
            rng_seed,
        )
    elif numba:
        outputs, parameters = _numba_findi._numba_partially_partial_descent(
            objective,
            initial,
            h,
            l,
            partial_epochs,
            total_epochs,
            parameters_used,
            constants,
            momentum,
            rng_seed,
        )

    return outputs, parameters


def values_out(outputs, parameters, constants=None, columns=None):
    """
    Produces a Pandas DataFrame of objective function outputs, parameter
    values and constants values for each epoch of the algorithm.

    :param outputs: Objective function outputs throughout epochs
    :type outputs: list or ndarray
    :param parameters: Objective function parameter values throughout epochs
    :type parameters: list or ndarray
    :param constants: Constants values used for the evaluation of the objective function
                      that aren't adjusted in the process of gradient descent, defaults to None
    :type constants: list, nb.typed.List or ndarray, optional
    :param columns: Column names of outputs and parameters, defaults to None
    :type columns: list or ndarray, optional
    :return: Dataframe of all the values of inputs and outputs of
             the objective function for each epoch
    :rtype: pd.DataFrame
    """

    values = _python_findi.values_out(outputs, parameters, constants, columns)

    return values
