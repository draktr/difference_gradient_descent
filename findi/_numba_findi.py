"""
Module `_numba_findi` stores functions that will be used for optimization
if the user chooses `numba=True` in public functions stored in `findi` module.
Functions here are optimized for `Numba`'s just-in-time compiler, and relevant
ones (i.e. `_descent_evaluate()` and `_partial_evaluate()`) are parallelized.
Their use requires the objective function to also be `Numba`-optimized, however,
it generally results in significant performance improvements. Detailed docstrings
are omitted, as they are provided in `findi` module.
"""

import numpy as np
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
    *constants,
):
    # Objective function is evaluated for every (differentiated) parameter
    # because we need it to calculate partial derivatives
    for parameter in nb.prange(n_parameters):
        current_parameters = parameters[epoch]
        current_parameters[parameter] = current_parameters[parameter] + difference

        difference_objective[parameter] = objective(current_parameters, *constants)[0]

    return difference_objective


@nb.njit(parallel=True)
def _partial_evaluate(
    objective,
    epoch,
    difference,
    parameters,
    difference_objective,
    param_idx,
    *constants,
):
    # Objective function is evaluated only for random parameters because we need it
    # to calculate partial derivatives, while limiting computational expense
    for parameter in nb.prange(param_idx.shape[0]):
        current_parameters = parameters[epoch]
        current_parameters[parameter] = current_parameters[parameter] + difference

        difference_objective[parameter] = objective(current_parameters, *constants)[0]

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
    *constants,
):
    # Evaluating the objective function that will count as
    # the "official" one for this epoch
    outputs[epoch] = objective(parameters[epoch], *constants)

    difference_objective = _descent_evaluate(
        objective,
        epoch,
        difference,
        parameters,
        difference_objective,
        n_parameters,
        *constants,
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
    *constants,
):
    param_idx = np.zeros(parameters_used, dtype=np.int_)
    while np.unique(param_idx).shape[0] != param_idx.shape[0]:
        param_idx = generator.integers(
            low=0, high=n_parameters, size=parameters_used, dtype=np.int_
        )

    # Evaluating the objective function that will count as
    # the "official" one for this epoch
    outputs[epoch] = objective(parameters[epoch], *constants)

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
        *constants,
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
    *constants,
):
    findi._checks._check_objective(objective)
    (h, l, epochs) = findi._checks._check_iterables(h, l, epochs)
    initial = findi._checks._check_arguments(initial, momentum)

    n_parameters = initial.shape[0]
    n_outputs = len(objective(initial, *constants))
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
            *constants,
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
    *constants,
):
    findi._checks._check_objective(objective)
    (h, l, epochs) = findi._checks._check_iterables(h, l, epochs)
    initial = findi._checks._check_arguments(
        initial=initial,
        parameters_used=parameters_used,
        momentum=momentum,
    )

    n_parameters = initial.shape[0]
    n_outputs = len(objective(initial, *constants))
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
            *constants,
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
    *constants,
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
        *constants,
    )

    outputs_r, parameters_r = _numba_descent(
        objective=objective,
        initial=parameters_p[-1],
        h=h[partial_epochs:],
        l=l[partial_epochs:],
        epochs=(total_epochs - partial_epochs),
        momentum=momentum,
        *constants,
    )

    outputs = np.append(outputs_p, outputs_r)
    parameters = np.append(parameters_p, parameters_r)
    outputs = np.reshape(outputs, newshape=[-1, 1])
    parameters = np.reshape(parameters, newshape=[-1, 1])

    return outputs, parameters
