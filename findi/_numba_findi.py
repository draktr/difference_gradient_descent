"""
Module `_numba_findi` stores functions that will be used for optimization
if the user chooses `numba=True` in public functions stored in `findi` module.
Functions here are optimized and parallelized for `Numba`'s just-in-time compiler.
Their use requires the objective function to also be `Numba`-optimized, however,
it generally results in significant performance improvements. Detailed docstrings
are omitted, as they are provided in `findi` module.
"""

import numpy as np
import numba as nb
import findi._checks


@nb.njit(parallel=True)
def _nmp_descent_epoch(
    objective,
    epoch,
    rate,
    difference,
    outputs,
    parameters,
    difference_outputs,
    momentum,
    velocity,
    n_parameters,
):
    # Evaluates one epoch of the regular Gradient Descent

    outputs[epoch] = objective(parameters[epoch])

    for parameter in nb.prange(n_parameters):
        current_parameters = parameters[epoch]
        current_parameters[parameter] = current_parameters[parameter] + difference

        difference_outputs[parameter] = objective(current_parameters)

    velocity = (
        momentum * velocity
        - rate * (difference_outputs[:, 0] - outputs[epoch, 0]) / difference
    )
    parameters[epoch + 1] = parameters[epoch] + velocity

    return outputs, parameters, velocity


@nb.njit(parallel=True)
def _nmp_partial_epoch(
    objective,
    epoch,
    rate,
    difference,
    outputs,
    parameters,
    difference_outputs,
    parameters_used,
    momentum,
    velocity,
    n_parameters,
    generator,
):
    # Evaluates one epoch of Partial Gradient Descent

    param_idx = np.zeros(parameters_used, dtype=np.int_)
    while np.unique(param_idx).shape[0] != param_idx.shape[0]:
        param_idx = generator.integers(
            low=0, high=n_parameters, size=parameters_used, dtype=np.int_
        )

    outputs[epoch] = objective(parameters[epoch])

    for i in nb.prange(n_parameters):
        difference_outputs[i] = outputs[epoch]

    for i in nb.prange(param_idx.shape[0]):
        parameter = param_idx[i]
        current_parameters = parameters[epoch]
        current_parameters[parameter] = current_parameters[parameter] + difference

        difference_outputs[parameter] = objective(current_parameters)

    velocity = (
        momentum * velocity
        - rate * (difference_outputs[:, 0] - outputs[epoch, 0]) / difference
    )
    parameters[epoch + 1] = parameters[epoch] + velocity

    return outputs, parameters, velocity


@nb.njit(parallel=True)
def _descent_epoch(
    objective,
    epoch,
    rate,
    difference,
    outputs,
    parameters,
    metaparameters,
    difference_outputs,
    momentum,
    velocity,
    n_parameters,
):
    # Evaluates one epoch of the regular Gradient Descent

    outputs[epoch] = objective(parameters[epoch], metaparameters)

    for parameter in nb.prange(n_parameters):
        current_parameters = parameters[epoch]
        current_parameters[parameter] = current_parameters[parameter] + difference

        difference_outputs[parameter] = objective(current_parameters, metaparameters)

    velocity = (
        momentum * velocity
        - rate * (difference_outputs[:, 0] - outputs[epoch, 0]) / difference
    )
    parameters[epoch + 1] = parameters[epoch] + velocity

    return outputs, parameters, velocity


@nb.njit(parallel=True)
def _partial_epoch(
    objective,
    epoch,
    rate,
    difference,
    outputs,
    parameters,
    metaparameters,
    difference_outputs,
    parameters_used,
    momentum,
    velocity,
    n_parameters,
    generator,
):
    # Evaluates one epoch of Partial Gradient Descent

    param_idx = np.zeros(parameters_used, dtype=np.int_)
    while np.unique(param_idx).shape[0] != param_idx.shape[0]:
        param_idx = generator.integers(
            low=0, high=n_parameters, size=parameters_used, dtype=np.int_
        )

    outputs[epoch] = objective(parameters[epoch], metaparameters)

    for i in nb.prange(n_parameters):
        difference_outputs[i] = outputs[epoch]

    for i in nb.prange(param_idx.shape[0]):
        parameter = param_idx[i]
        current_parameters = parameters[epoch]
        current_parameters[parameter] = current_parameters[parameter] + difference

        difference_outputs[parameter] = objective(current_parameters, metaparameters)

    velocity = (
        momentum * velocity
        - rate * (difference_outputs[:, 0] - outputs[epoch, 0]) / difference
    )
    parameters[epoch + 1] = parameters[epoch] + velocity

    return outputs, parameters, velocity


def _numba_descent(
    objective, initial, h, l, epochs, metaparameters=None, momentum=0, numba=True
):
    # Performs the regular Gradient Descent using Numba JIT compiler for evaluation

    initial, metaparameters = findi._checks._check_arguments(
        initial=initial,
        momentum=momentum,
        metaparameters=metaparameters,
        numba=numba,
    )
    n_outputs, output_is_number, no_metaparameters = findi._checks._check_objective(
        objective, initial, metaparameters, numba
    )
    (h, l, epochs) = findi._checks._check_iterables(h, l, epochs)

    n_parameters = initial.shape[0]
    outputs = np.zeros([epochs, n_outputs])
    parameters = np.zeros([epochs + 1, n_parameters])
    parameters[0] = initial
    difference_outputs = np.zeros((n_parameters, n_outputs))
    velocity = 0

    if no_metaparameters:
        for epoch, (rate, difference) in enumerate(zip(l, h)):
            outputs, parameters, velocity = _nmp_descent_epoch(
                objective,
                epoch,
                rate,
                difference,
                outputs,
                parameters,
                difference_outputs,
                momentum,
                velocity,
                n_parameters,
            )
    else:
        for epoch, (rate, difference) in enumerate(zip(l, h)):
            outputs, parameters, velocity = _descent_epoch(
                objective,
                epoch,
                rate,
                difference,
                outputs,
                parameters,
                metaparameters,
                difference_outputs,
                momentum,
                velocity,
                n_parameters,
            )

    return outputs, parameters[:-1]


def _numba_partial_descent(
    objective,
    initial,
    h,
    l,
    epochs,
    parameters_used,
    metaparameters=None,
    momentum=0,
    rng_seed=88,
    numba=True,
):
    # Performs Partial Gradient Descent using Numba JIT compiler for evaluation

    initial, metaparameters = findi._checks._check_arguments(
        initial=initial,
        parameters_used=parameters_used,
        momentum=momentum,
        rng_seed=rng_seed,
        metaparameters=metaparameters,
        numba=numba,
    )
    n_outputs, output_is_number, no_metaparameters = findi._checks._check_objective(
        objective, initial, metaparameters, numba
    )
    (h, l, epochs) = findi._checks._check_iterables(h, l, epochs)

    n_parameters = initial.shape[0]
    outputs = np.zeros([epochs, n_outputs])
    parameters = np.zeros([epochs + 1, n_parameters])
    parameters[0] = initial
    difference_outputs = np.zeros((n_parameters, n_outputs))
    generator = np.random.default_rng(rng_seed)
    velocity = 0

    if no_metaparameters:
        for epoch, (rate, difference) in enumerate(zip(l, h)):
            outputs, parameters, velocity = _nmp_partial_epoch(
                objective,
                epoch,
                rate,
                difference,
                outputs,
                parameters,
                difference_outputs,
                parameters_used,
                momentum,
                velocity,
                n_parameters,
                generator,
            )
    else:
        for epoch, (rate, difference) in enumerate(zip(l, h)):
            outputs, parameters, velocity = _partial_epoch(
                objective,
                epoch,
                rate,
                difference,
                outputs,
                parameters,
                metaparameters,
                difference_outputs,
                parameters_used,
                momentum,
                velocity,
                n_parameters,
                generator,
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
    metaparameters=None,
    momentum=0,
    rng_seed=88,
):
    # Performs Partially Partial Gradient Descent using Numba JIT compiler for evaluation

    initial, metaparameters = findi._checks._check_arguments(
        initial=initial,
        partial_epochs=partial_epochs,
        total_epochs=total_epochs,
        metaparameters=metaparameters,
    )
    (h, l, total_epochs) = findi._checks._check_iterables(h, l, total_epochs)

    outputs_p, parameters_p = _numba_partial_descent(
        objective=objective,
        initial=initial,
        h=h[:partial_epochs],
        l=l[:partial_epochs],
        epochs=partial_epochs,
        parameters_used=parameters_used,
        metaparameters=metaparameters,
        momentum=momentum,
        rng_seed=rng_seed,
    )

    outputs_r, parameters_r = _numba_descent(
        objective=objective,
        initial=parameters_p[-1],
        h=h[partial_epochs:],
        l=l[partial_epochs:],
        epochs=(total_epochs - partial_epochs),
        metaparameters=metaparameters,
        momentum=momentum,
    )

    outputs = np.append(outputs_p, outputs_r)
    parameters = np.append(parameters_p, parameters_r)
    outputs = np.reshape(outputs, newshape=[-1, 1])
    parameters = np.reshape(parameters, newshape=[-1, 1])

    return outputs, parameters
