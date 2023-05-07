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
import numba as nb
import findi._checks


@nb.njit
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


@nb.njit(parallel=True)
def _descent_evaluate(
    objective,
    epoch,
    difference,
    parameters,
    difference_objective,
    n_parameters,
    constants,
):
    # Objective function is evaluated for every (differentiated) parameter
    # because we need it to calculate partial derivatives
    for parameter in nb.prange(n_parameters):
        current_parameters = parameters[epoch]
        current_parameters[parameter] = current_parameters[parameter] + difference

        difference_objective[parameter] = objective(current_parameters, constants)[0]

    return difference_objective


@nb.njit(parallel=True)
def _partial_evaluate(
    objective,
    epoch,
    difference,
    parameters,
    difference_objective,
    param_idx,
    constants,
):
    # Objective function is evaluated only for random parameters because we need it
    # to calculate partial derivatives, while limiting computational expense
    for parameter in nb.prange(param_idx.shape[0]):
        current_parameters = parameters[epoch]
        current_parameters[parameter] = current_parameters[parameter] + difference

        difference_objective[parameter] = objective(current_parameters, constants)[0]

    return difference_objective


@nb.njit
def _inner_descent(
    objective,
    epoch,
    rate,
    difference,
    outputs,
    parameters,
    difference_objective,
    momentum,
    velocity,
    n_parameters,
    constants,
):
    # Evaluating the objective function that will count as
    # the "official" one for this epoch
    outputs[epoch] = objective(parameters[epoch], constants)

    difference_objective = _descent_evaluate(
        objective,
        epoch,
        difference,
        parameters,
        difference_objective,
        n_parameters,
        constants,
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


@nb.njit
def _inner_partial(
    objective,
    epoch,
    rate,
    difference,
    outputs,
    parameters,
    difference_objective,
    parameters_used,
    momentum,
    velocity,
    n_parameters,
    constants,
):
    param_idx = np.zeros(parameters_used, dtype=np.int_)
    for i in range(parameters_used):
        param_idx[i] = np.random.randint(low=0, high=n_parameters)

    # Evaluating the objective function that will count as
    # the "official" one for this epoch
    outputs[epoch] = objective(parameters[epoch], constants)

    # Difference objective value is still recorded (as base
    # evaluation value) for non-differenced parameters
    # (in current epoch) for consistency and convenience
    difference_objective = np.repeat(outputs[epoch, 0], n_parameters)
    difference_objective = _partial_evaluate(
        objective,
        epoch,
        difference,
        parameters,
        difference_objective,
        param_idx,
        constants,
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


def descent(
    objective,
    initial,
    h,
    l,
    epochs,
    momentum=0,
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
    :return: Objective function outputs and parameters for each epoch
    :rtype: ndarray
    """

    findi._checks._check_objective(objective)
    (h, l, epochs) = findi._checks._check_iterables(h, l, epochs)
    initial = findi._checks._check_arguments(initial, momentum)

    constants = np.array(list(constants.values()))
    n_parameters = initial.shape[0]
    n_outputs = len(objective(initial, constants))
    outputs = np.zeros([epochs, n_outputs])
    parameters = np.zeros([epochs + 1, n_parameters])
    parameters[0] = initial
    difference_objective = np.zeros(n_parameters)
    velocity = 0

    for epoch, (rate, difference) in enumerate(zip(l, h)):
        outputs, parameters = _inner_descent(
            objective,
            epoch,
            rate,
            difference,
            outputs,
            parameters,
            difference_objective,
            momentum,
            velocity,
            n_parameters,
            constants,
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
    :return: Objective function outputs and parameters for each epoch
    :rtype: ndarray
    """

    findi._checks._check_objective(objective)
    (h, l, epochs) = findi._checks._check_iterables(h, l, epochs)
    initial = findi._checks._check_arguments(
        initial=initial,
        parameters_used=parameters_used,
        momentum=momentum,
    )

    constants = np.array(list(constants.values()))
    n_parameters = initial.shape[0]
    n_outputs = len(objective(initial, constants))
    outputs = np.zeros([epochs, n_outputs])
    parameters = np.zeros([epochs + 1, n_parameters])
    parameters[0] = initial
    difference_objective = np.zeros(n_parameters)
    velocity = 0

    for epoch, (rate, difference) in enumerate(zip(l, h)):
        outputs, parameters = _inner_partial(
            objective,
            epoch,
            rate,
            difference,
            outputs,
            parameters,
            difference_objective,
            parameters_used,
            momentum,
            velocity,
            n_parameters,
            constants,
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
    :return: Objective function outputs and parameters for each epoch
    :rtype: ndarray
    """

    (h, l, total_epochs) = findi._checks._check_iterables(h, l, total_epochs)
    initial = findi._checks._check_arguments(
        initial=initial,
        partial_epochs=partial_epochs,
        total_epochs=total_epochs,
    )

    outputs_p, parameters_p = partial_descent(
        objective,
        initial,
        h[:partial_epochs],
        l[:partial_epochs],
        partial_epochs,
        parameters_used,
        momentum,
        **constants,
    )

    outputs_r, parameters_r = descent(
        objective=objective,
        initial=parameters_p[-1],
        h=h[partial_epochs:],
        l=l[partial_epochs:],
        epochs=(total_epochs - partial_epochs),
        momentum=momentum,
        **constants,
    )

    outputs = np.append(outputs_p, outputs_r)
    parameters = np.append(parameters_p, parameters_r)
    outputs = np.reshape(outputs, newshape=[-1, 1])
    parameters = np.reshape(parameters, newshape=[-1, 1])

    return outputs, parameters


def values_out(outputs, parameters, constants_values=None, columns=None):
    """
    Produces a Pandas DataFrame of objective function outputs, parameter
    values and constants values for each epoch of the algorithm.

    :param outputs: Objective function outputs throughout epochs
    :type outputs: list or ndarray
    :param parameters: Objective function parameter values throughout epochs
    :type parameters: list or ndarray
    :param constants_values: Array with values of constants (for single epoch)
                             passed to the objective function individually as
                             keyword arguments to optimizers
    :type constants_values: list or ndarray
    :param columns: Column names for the DataFrame. Should include
                    names for all the outputs, parameters and
                    constants, defaults to None
    :type columns: list or ndarray, optional
    :return: Dataframe of all the values of inputs and outputs of
                the objective function for each epoch
    :rtype: pd.DataFrame
    """

    findi._checks._check_arguments(
        outputs=outputs,
        parameters=parameters,
        constants_values=constants_values,
        columns=columns,
    )

    if len(constants_values) == 0 or constants_values is None:
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

    values = pd.DataFrame(
        np.column_stack((outputs, inputs)),
        columns=columns,
    )

    return values
