import numpy as np
import warnings
from optschedule import Schedule


def _check_iterables(h, l, epochs):
    if not isinstance(epochs, int):
        raise ValueError("Number of epochs should be an integer")
    if epochs < 1:
        raise ValueError("Number of epochs should be positive")

    if isinstance(h, (int, float)):
        h = Schedule(epochs).constant(h)
    elif isinstance(h, list):
        h = np.asarray(h)
    elif isinstance(h, np.ndarray):
        pass
    else:
        raise ValueError(
            "Differences should be of type `int`, `float`, `list` or `np.ndarray`"
        )

    if isinstance(l, (int, float)):
        l = Schedule(epochs).constant(l)
    elif isinstance(l, list):
        l = np.asarray(l)
    elif isinstance(l, np.ndarray):
        pass
    else:
        raise ValueError(
            "Learning rates should be of type `int`, `float`, `list` or `np.ndarray`"
        )

    if h.shape[0] != l.shape[0]:
        raise ValueError("Number of differences and learning rates should be equal.")
    if epochs != h.shape[0]:
        raise ValueError(
            "Number of epochs, differences and learning rates given should be equal."
        )

    return h, l, epochs


def _check_objective(objective):
    if not callable(objective):
        raise ValueError(
            f"Objective function should be a callable. Current objective function type is:{type(objective)}"
        )


def _check_threads(threads, parameters):
    if len(parameters[0]) + 1 != threads:
        raise ValueError(
            "Each parameter should have only one CPU thread, along with one for the base evaluation."
        )


def _check_arguments(
    initial=None,
    momentum=None,
    threads=None,
    parameters_used=None,
    rng_seed=None,
    partial_epochs=None,
    total_epochs=None,
    outputs=None,
    parameters=None,
    columns=None,
):
    if isinstance(initial, (int, float)):
        initial = np.array([initial])
    elif isinstance(initial, list):
        initial = np.asarray(initial)
    elif isinstance(initial, (np.ndarray, type(None))):
        pass
    else:
        raise ValueError(
            "Initial parameters should expressed as either `int`, `float`, `list` or `np.ndarray`"
        )
    if not isinstance(momentum, (int, float, type(None))):
        raise ValueError("Momentum should be an `int` or a `float`")
    if momentum is not None:
        if momentum < 0:
            raise ValueError("Momentum should be non-negative")
    if not isinstance(threads, (int, type(None))):
        raise ValueError("Number of threads should be a positive integer")
    if threads is not None:
        if threads < 1:
            raise ValueError("Number of threads should be a positive integer")
    if not isinstance(parameters_used, (int, type(None))):
        raise ValueError("Number of parameters used should be a positive integer")
    if parameters_used is not None:
        if parameters_used < 1:
            raise ValueError("Number of parameters used should be a positive integer")
    if not isinstance(rng_seed, (int, type(None))):
        raise ValueError("RNG seed should be a non-negative integer")
    if rng_seed is not None:
        if rng_seed < 0:
            raise ValueError("RNG seed should be a non-negative integer")
    if not isinstance(partial_epochs, (int, type(None))):
        raise ValueError("Number of partial epochs should be non-negative integer")
    if partial_epochs is not None:
        if partial_epochs < 0:
            raise ValueError("Number of partial epochs should be non-negative integer")
    if partial_epochs is not None:
        if partial_epochs == 0:
            warnings.warn(
                "Number of partial epochs is 0 (zero). All epochs will be run with regular algorithm",
                UserWarning,
            )
    if not isinstance(total_epochs, (int, type(None))):
        raise ValueError("Number of total epochs should be a positive integer")
    if total_epochs is not None:
        if total_epochs < 1:
            raise ValueError("Number of total epochs should be a positive integer")
    if not isinstance(parameters, (np.ndarray, type(None))):
        raise ValueError("Parameters should be of type `np.ndarray`")
    if not isinstance(columns, (np.ndarray, list, type(None))):
        raise ValueError("Columns should be either a list or `np.ndarray`")
    if outputs is not None and parameters is not None and columns is not None:
        if (outputs.shape[1] + parameters.shape[1]) != len(columns):
            raise ValueError(
                "Number of column names given in `columns` doesn't match the combined number of outputs and parameters"
            )

    return initial
