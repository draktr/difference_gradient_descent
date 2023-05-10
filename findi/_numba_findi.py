"""
The ``findi`` houses functions that optimizes objective
functions via Gradient Descent Algorithm variation that uses
finite difference instead of infinitesimal differential for
computing derivatives. This approach allows for the application
of Gradient Descent on non-differentiable functions, functions
without analytic form or any other function, as long as it can
be evaluated. `_numba_descent` function performs regular finite difference
gradient descent algorithm, while `_numba_partial_descent` function allow
a version of finite difference gradient descent algorithm where
only a random subset of gradients is used in each epoch.
`partially_partial_descent` function performs `_numba_partial_descent`
algorithm for the first `partial_epochs` number of epochs and `_numba_descent`
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
    generator,
    constants,
):
    param_idx = np.zeros(parameters_used, dtype=np.int_)
    while np.unique(param_idx).shape[0] != param_idx.shape[0]:
        param_idx = generator.integers(
            low=0, high=n_parameters, size=parameters_used, dtype=np.int_
        )

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


def _numba_descent(
    objective,
    initial,
    h,
    l,
    epochs,
    momentum=0,
    **constants,
):
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

    return outputs, parameters[:-1]


def _numba_partial_descent(
    objective,
    initial,
    h,
    l,
    epochs,
    parameters_used,
    momentum=0,
    rng_seed=88,
    **constants,
):
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
    generator = np.random.default_rng(rng_seed)
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
            generator,
            constants,
        )

    return outputs, parameters[:-1]


def _numba_partially_partial_descent(
    objective,
    initial,
    h,
    l,
    partial_epochs,
    total_epochs,
    parameters_used,
    momentum=0,
    rng_seed=88,
    **constants,
):
    (h, l, total_epochs) = findi._checks._check_iterables(h, l, total_epochs)
    initial = findi._checks._check_arguments(
        initial=initial,
        partial_epochs=partial_epochs,
        total_epochs=total_epochs,
    )

    outputs_p, parameters_p = _numba_partial_descent(
        objective,
        initial,
        h[:partial_epochs],
        l[:partial_epochs],
        partial_epochs,
        parameters_used,
        momentum,
        rng_seed,
        **constants,
    )

    outputs_r, parameters_r = _numba_descent(
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
