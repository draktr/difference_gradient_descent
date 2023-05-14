from findi import _python_findi
from findi import _numba_findi


def descent(
    objective,
    initial,
    h,
    l,
    epochs,
    momentum=0,
    threads=1,
    numba=False,
    *numba_constants,
    **constants,
):
    """
    Performs Gradient Descent Algorithm by using finite difference instead
    of infinitesimal differential. Allows for the implementation of gradient
    descent algorithm on variety of non-standard functions.

    :param objective: Objective function to minimize
    :type objective: callable
    :param initial: Initial values of objective function parameters
    :type initial: int, float, list or ndarray
    :param h: Small change(s) in `x`. Can be a sequence or a number
              in which case constant change is used
    :type h: int, float, list or ndarray
    :param l: Learning rate(s). Can be a sequence or a number in which
              case constant learning rate is used
    :type l: int, float, list or ndarray
    :param epochs: Number of epochs
    :type epochs: int
    :param momentum: Hyperparameter that dampens oscillations.
                     `momentum=0` implies vanilla algorithm, defaults to 0
    :type momentum: int or float, optional
    :param threads: Number of CPU threads used by `joblib` for computation, defaults to 1
    :type threads: int, optional
    :return: Objective function outputs and parameters for each epoch
    :rtype: ndarray
    """

    if not numba:
        outputs, parameters = _python_findi._python_descent(
            objective,
            initial,
            h,
            l,
            epochs,
            momentum,
            threads,
            **constants,
        )
    elif numba:
        outputs, parameters = _numba_findi._numba_descent(
            objective,
            initial,
            h,
            l,
            epochs,
            momentum,
            *numba_constants,
        )

    return outputs, parameters


def partial_descent(
    objective,
    initial,
    h,
    l,
    epochs,
    parameters_used,
    momentum=0,
    threads=1,
    rng_seed=88,
    numba=False,
    *numba_constants,
    **constants,
):
    """
    Performs Gradient Descent Algorithm by computing derivatives on only
    specified number of randomly selected parameters in each epoch and
    by using finite difference instead of infinitesimal differential.
    Allows for the implementation of gradient descent algorithm on
    variety of non-standard functions.

    :param objective: Objective function to minimize
    :type objective: callable
    :param initial: Initial values of objective function parameters
    :type initial: int, float, list or ndarray
    :param h: Small change(s) in `x`. Can be a sequence or a number
              in which case constant change is used
    :type h: int, float, list or ndarray
    :param l: Learning rate(s). Can be a sequence or a number in
              which case constant learning rate is used
    :type l: int, float, list or ndarray
    :param epochs: Number of epochs
    :type epochs: int
    :param parameters_used: Number of parameters used in each epoch
                            for computation of gradients
    :type parameters_used: int
    :param momentum: Hyperparameter that dampens oscillations.
                     `momentum=0` implies vanilla algorithm, defaults to 0
    :type momentum: int or float, optional
    :param threads: Number of CPU threads used by `joblib` for computation, defaults to 1
    :type threads: int, optional
    :param rng_seed: Seed for the random number generator used for
                     determining which parameters are used in each
                     epoch for computation of gradients, defaults to 88
    :type rng_seed: int, optional
    :return: Objective function outputs and parameters for each epoch
    :rtype: ndarray
    """

    if not numba:
        outputs, parameters = _python_findi._python_partial_descent(
            objective,
            initial,
            h,
            l,
            epochs,
            parameters_used,
            momentum,
            threads,
            rng_seed,
            **constants,
        )
    elif numba:
        outputs, parameters = _numba_findi._numba_partial_descent(
            objective,
            initial,
            h,
            l,
            epochs,
            parameters_used,
            momentum,
            rng_seed,
            *numba_constants,
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
    momentum=0,
    threads=1,
    rng_seed=88,
    numba=False,
    *numba_constants,
    **constants,
):
    """
    Performs Partial Gradient Descent Algorithm for the first `partial_epochs`
    epochs and regular Finite Difference Gradient Descent for the rest of the
    epochs (i.e. `total_epochs`-`partial_epochs`).

    :param objective: Objective function to minimize
    :type objective: callable
    :param initial: Initial values of objective function parameters
    :type initial: int, float, list or ndarray
    :param h: Small change(s) in `x`. Can be a sequence or a number
              in which case constant change is used
    :type h: int, float, list or ndarray
    :param l: Learning rate(s). Can be a sequence or a number in which
              case constant learning rate is used
    :type l: int, float, list or ndarray
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
    :param momentum: Hyperparameter that dampens oscillations.
                     `momentum=0` implies vanilla algorithm, defaults to 0
    :type momentum: int or float, optional
    :param threads: Number of CPU threads used by `joblib` for computation, defaults to 1
    :type threads: int, optional
    :param rng_seed: Seed for the random number generator used for determining
                     which parameters are used in each epoch for computation
                     of gradients, defaults to 88
    :type rng_seed: int, optional
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
            momentum,
            threads,
            rng_seed,
            **constants,
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
            momentum,
            rng_seed,
            *numba_constants,
        )

    return outputs, parameters


def values_out(outputs, parameters, columns=None, **constants):
    """
    Produces a Pandas DataFrame of objective function outputs, parameter
    values and constants values for each epoch of the algorithm.

    :param outputs: Objective function outputs throughout epochs
    :type outputs: list or ndarray
    :param parameters: Objective function parameter values throughout epochs
    :type parameters: list or ndarray
    :param columns: Column names of outputs and parameters, defaults to None
    :type columns: list or ndarray, optional
    :return: Dataframe of all the values of inputs and outputs of
             the objective function for each epoch
    :rtype: pd.DataFrame
    """
    values = _python_findi.values_out(outputs, parameters, columns, **constants)

    return values
