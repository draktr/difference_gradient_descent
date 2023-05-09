"""
The ``findi`` houses functions that optimizes objective
functions via Gradient Descent Algorithm variation that uses
finite difference instead of infinitesimal differential for
computing derivatives. This approach allows for the application
of Gradient Descent on non-differentiable functions, functions
without analytic form or any other function, as long as it can
be evaluated. `descent` function performs regular finite difference
gradient descent algorithm, while `partial_descent` function allow
a version of finite difference gradient descent algorithm where
only a random subset of gradients is used in each epoch.
`partially_partial_descent` function performs `partial_descent`
algorithm for the first `partial_epochs` number of epochs and `descent`
for the rest of the epochs. Parallel computing for performance benefits
is supported in all of these functions. Furthermore, objective functions
with multiple outputs are supported (only the first one is taken as
objective value to be minimized), as well as constant objective
function quasi-hyperparameters that are held constant throughout
the epochs.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import findi._checks


def _update(
    rate,
    difference_objective,
    outputs,
    difference,
    momentum,
    velocity,
    epoch,
    parameters,
):
    velocity = (
        momentum * velocity
        - rate * (difference_objective - outputs[epoch, 0]) / difference
    )
    updated_parameters = parameters[epoch] + velocity

    return updated_parameters


def descent(
    objective,
    initial,
    h,
    l,
    epochs,
    momentum=0,
    threads=1,
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

    findi._checks._check_objective(objective)
    (h, l, epochs) = findi._checks._check_iterables(h, l, epochs)
    initial = findi._checks._check_arguments(initial, momentum, threads)

    n_parameters = initial.shape[0]
    n_outputs = len(objective(initial, **constants))
    outputs = np.zeros([epochs, n_outputs])
    parameters = np.zeros([epochs + 1, n_parameters])
    parameters[0] = initial
    difference_objective = np.zeros(n_parameters)
    velocity = 0

    if threads == 1:
        for epoch, (rate, difference) in enumerate(zip(l, h)):
            # Evaluating the objective function that will count as
            # the "official" one for this epoch
            outputs[epoch] = objective(parameters[epoch], **constants)

            # Objective function is evaluated for every (differentiated) parameter
            # because we need it to calculate partial derivatives
            for parameter in range(n_parameters):
                current_parameters = parameters[epoch]
                current_parameters[parameter] = (
                    current_parameters[parameter] + difference
                )

                difference_objective[parameter] = objective(
                    current_parameters, **constants
                )[0]

            # These parameters will be used for the evaluation in the next epoch
            parameters[epoch + 1] = _update(
                rate,
                difference_objective,
                outputs,
                difference,
                momentum,
                velocity,
                epoch,
                parameters,
            )

    elif threads > 1:
        findi._checks._check_threads(threads, parameters)

        # One set of parameters is needed for each partial derivative,
        # and one is needed for the base case
        current_parameters = np.zeros([n_parameters + 1, n_parameters])

        for epoch, (rate, difference) in enumerate(zip(l, h)):
            current_parameters[0] = parameters[epoch]
            for parameter in range(n_parameters):
                current_parameters[parameter + 1] = parameters[epoch]
                current_parameters[parameter + 1, parameter] = (
                    current_parameters[0, parameter] + difference
                )

            parallel_outputs = Parallel(n_jobs=threads)(
                delayed(objective)(i, **constants) for i in current_parameters
            )

            # This objective function evaluation will be used as the
            # "official" one for this epoch
            outputs[epoch] = parallel_outputs[0]
            difference_objective = np.array(
                [parallel_outputs[i][0] for i in range(1, n_parameters + 1)]
            )

            # These parameters will be used for the evaluation in the next epoch
            parameters[epoch + 1] = _update(
                rate,
                difference_objective,
                outputs,
                difference,
                momentum,
                velocity,
                epoch,
                parameters,
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

    findi._checks._check_objective(objective)
    (h, l, epochs) = findi._checks._check_iterables(h, l, epochs)
    initial = findi._checks._check_arguments(
        initial=initial,
        parameters_used=parameters_used,
        momentum=momentum,
        threads=threads,
        rng_seed=rng_seed,
    )

    n_parameters = initial.shape[0]
    n_outputs = len(objective(initial, **constants))
    outputs = np.zeros([epochs, n_outputs])
    parameters = np.zeros([epochs + 1, n_parameters])
    parameters[0] = initial
    difference_objective = np.zeros(n_parameters)
    rng = np.random.default_rng(rng_seed)
    velocity = 0

    if threads == 1:
        for epoch, (rate, difference) in enumerate(zip(l, h)):
            param_idx = rng.integers(low=0, high=n_parameters, size=parameters_used)

            # Evaluating the objective function that will count as
            # the "official" one for this epoch
            outputs[epoch] = objective(parameters[epoch], **constants)

            # Objective function is evaluated only for random parameters because we need it
            # to calculate partial derivatives, while limiting computational expense
            for parameter in range(n_parameters):
                if parameter in param_idx:
                    current_parameters = parameters[epoch]
                    current_parameters[parameter] = (
                        current_parameters[parameter] + difference
                    )

                    difference_objective[parameter] = objective(
                        current_parameters, **constants
                    )[0]
                else:
                    # Difference objective value is still recorded (as base
                    # evaluation value) for non-differenced parameters
                    # (in current epoch) for consistency and convenience
                    difference_objective[parameter] = outputs[epoch, 0]

            # These parameters will be used for the evaluation in the next epoch
            parameters[epoch + 1] = _update(
                rate,
                difference_objective,
                outputs,
                difference,
                momentum,
                velocity,
                epoch,
                parameters,
            )

    elif threads > 1:
        findi._checks._check_threads(threads, parameters_used)

        # One set of parameters is needed for each partial derivative used,
        # and one is needed for the base case
        current_parameters = np.zeros([n_parameters + 1, n_parameters])

        for epoch, (rate, difference) in enumerate(zip(l, h)):
            param_idx = rng.integers(low=0, high=n_parameters, size=parameters_used)

            # Objective function is evaluated only for random parameters because we need it
            # to calculate partial derivatives, while limiting computational expense
            current_parameters[0] = parameters[epoch]
            for parameter in range(n_parameters):
                current_parameters[parameter + 1] = parameters[epoch]
                if parameter in param_idx:
                    current_parameters[parameter + 1, parameter] = (
                        current_parameters[0, parameter] + difference
                    )

            parallel_outputs = Parallel(n_jobs=threads)(
                delayed(objective)(i, **constants)
                for i in current_parameters[
                    np.append(np.array([0]), np.add(param_idx, 1))
                ]
            )

            # This objective function evaluation will be used as the
            # "official" one for this epoch
            outputs[epoch] = parallel_outputs[0]
            # Difference objective value is still recorded (as base
            # evaluation value) for non-differenced parameters
            # (in current epoch) for consistency and convenience
            difference_objective = np.full(n_parameters, parallel_outputs[0][0])
            difference_objective[param_idx] = np.array(
                [parallel_outputs[i][0] for i in range(1, parameters_used + 1)]
            )

            # These parameters will be used for the evaluation in the next epoch
            parameters[epoch + 1] = _update(
                rate,
                difference_objective,
                outputs,
                difference,
                momentum,
                velocity,
                epoch,
                parameters,
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

    (h, l, total_epochs) = findi._checks._check_iterables(h, l, total_epochs)
    initial = findi._checks._check_arguments(
        initial=initial,
        partial_epochs=partial_epochs,
        total_epochs=total_epochs,
        parameters_used=parameters_used,
        momentum=momentum,
        threads=threads,
        rng_seed=rng_seed,
    )

    outputs_p, parameters_p = partial_descent(
        objective,
        initial,
        h[:partial_epochs],
        l[:partial_epochs],
        partial_epochs,
        parameters_used,
        momentum,
        threads,
        rng_seed,
        **constants,
    )

    outputs_r, parameters_r = descent(
        objective=objective,
        initial=parameters_p[-1],
        h=h[partial_epochs:],
        l=l[partial_epochs:],
        epochs=(total_epochs - partial_epochs),
        momentum=momentum,
        threads=threads,
        **constants,
    )

    outputs = np.append(outputs_p, outputs_r)
    parameters = np.append(parameters_p, parameters_r)
    outputs = np.reshape(outputs, newshape=[-1, 1])
    parameters = np.reshape(parameters, newshape=[-1, 1])

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

    findi._checks._check_arguments(
        outputs=outputs,
        parameters=parameters,
        columns=columns,
    )
    constants_values = np.array(list(constants.values()))
    constants_keys = np.array(list(constants.keys()))
    if columns is None:
        columns = np.array([], dtype=str)

    if len(constants_values) == 0:
        inputs = parameters
    else:
        inputs = np.concatenate(
            [
                parameters,
                np.full(
                    (len(parameters), len(constants_values)),
                    constants_values,
                ),
            ],
            axis=1,
        )
        columns = np.concatenate((columns, constants_keys), axis=0)

    values = pd.DataFrame(
        np.column_stack((outputs, inputs)),
        columns=columns,
    )

    return values
