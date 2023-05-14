"""
Module `_python_findi` stores functions that will be used for optimization
if the user chooses `numba=False` in public functions stored in `findi` module.
These are regular `Python` functions that will use `Python` interpreter for
execution and do not require any particular formating of an objective function.
Parallelization is possible using the `joblib` library. Detailed docstrings
are omitted, as they are provided in `findi` module.
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


def _python_descent(
    objective,
    initial,
    h,
    l,
    epochs,
    momentum=0,
    threads=1,
    constants=None,
):
    findi._checks._check_objective(objective)
    (h, l, epochs) = findi._checks._check_iterables(h, l, epochs)
    initial = findi._checks._check_arguments(initial, momentum, threads)

    n_parameters = initial.shape[0]
    n_outputs = len(objective(initial, constants))
    outputs = np.zeros([epochs, n_outputs])
    parameters = np.zeros([epochs + 1, n_parameters])
    parameters[0] = initial
    difference_objective = np.zeros(n_parameters)
    velocity = 0

    if threads == 1:
        for epoch, (rate, difference) in enumerate(zip(l, h)):
            # Evaluating the objective function that will count as
            # the "official" one for this epoch
            outputs[epoch] = objective(parameters[epoch], constants)

            # Objective function is evaluated for every (differentiated) parameter
            # because we need it to calculate partial derivatives
            for parameter in range(n_parameters):
                current_parameters = parameters[epoch]
                current_parameters[parameter] = (
                    current_parameters[parameter] + difference
                )

                difference_objective[parameter] = objective(
                    current_parameters, constants
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
                delayed(objective)(i, constants) for i in current_parameters
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

    return outputs, parameters[:-1]


def _python_partial_descent(
    objective,
    initial,
    h,
    l,
    epochs,
    parameters_used,
    momentum=0,
    threads=1,
    rng_seed=88,
    constants=None,
):
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
    n_outputs = len(objective(initial, constants))
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
            outputs[epoch] = objective(parameters[epoch], constants)

            # Objective function is evaluated only for random parameters because we need it
            # to calculate partial derivatives, while limiting computational expense
            for parameter in range(n_parameters):
                if parameter in param_idx:
                    current_parameters = parameters[epoch]
                    current_parameters[parameter] = (
                        current_parameters[parameter] + difference
                    )

                    difference_objective[parameter] = objective(
                        current_parameters, constants
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
                delayed(objective)(i, constants)
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

    return outputs, parameters[:-1]


def _python_partially_partial_descent(
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
    constants=None,
):
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

    outputs_p, parameters_p = _python_partial_descent(
        objective,
        initial,
        h[:partial_epochs],
        l[:partial_epochs],
        partial_epochs,
        parameters_used,
        momentum,
        threads,
        rng_seed,
        constants,
    )

    outputs_r, parameters_r = _python_descent(
        objective=objective,
        initial=parameters_p[-1],
        h=h[partial_epochs:],
        l=l[partial_epochs:],
        epochs=(total_epochs - partial_epochs),
        momentum=momentum,
        threads=threads,
        constants=constants,
    )

    outputs = np.append(outputs_p, outputs_r)
    parameters = np.append(parameters_p, parameters_r)
    outputs = np.reshape(outputs, newshape=[-1, 1])
    parameters = np.reshape(parameters, newshape=[-1, 1])

    return outputs, parameters


def values_out(outputs, parameters, columns=None, constants=None):
    findi._checks._check_arguments(
        outputs=outputs,
        parameters=parameters,
        constants=constants,
        columns=columns,
    )

    if columns is None:
        columns = np.array([], dtype=np.str_)
    if constants is None:
        constants = np.array([])

    if len(constants) == 0:
        inputs = parameters
    else:
        inputs = np.concatenate(
            [
                parameters,
                np.full(
                    (len(parameters), len(constants)),
                    constants,
                ),
            ],
            axis=1,
            dtype=np.str_,
        )

    values = pd.DataFrame(
        np.concatenate((outputs, inputs), axis=1),
        columns=columns,
    )

    return values.convert_dtypes()
