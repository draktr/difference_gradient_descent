import numpy as np
import numba as nb
from inspect import signature
import warnings
from optschedule import Schedule


def _check_iterables(h, l, epochs):
    if not isinstance(epochs, int):
        raise ValueError("Number of epochs should be an integer")
    if epochs < 1:
        raise ValueError("Number of epochs should be positive")

    if isinstance(h, (int, float)):
        h = Schedule(epochs).constant(h)
    elif isinstance(h, (list, nb.typed.List)):
        h = np.asarray(h)
    elif isinstance(h, np.ndarray):
        pass
    else:
        raise ValueError(
            "Differences should be of type `int`, `float`, `list`, `nb.typed.List` or `np.ndarray`"
        )

    if isinstance(l, (int, float)):
        l = Schedule(epochs).constant(l)
    elif isinstance(l, (list, nb.typed.List)):
        l = np.asarray(l)
    elif isinstance(l, np.ndarray):
        pass
    else:
        raise ValueError(
            "Learning rates should be of type `int`, `float`, `list`, `nb.typed.List` or `np.ndarray`"
        )

    if h.shape[0] != l.shape[0]:
        raise ValueError("Number of differences and learning rates should be equal.")
    if epochs != h.shape[0]:
        raise ValueError(
            "Number of epochs, differences and learning rates given should be equal."
        )

    return h, l, epochs


def _check_objective(objective, parameters, metaparameters, numba):
    if not callable(objective):
        raise ValueError(
            f"Objective function should be a callable. Current objective function type is:{type(objective)}"
        )

    if numba and not isinstance(objective, nb.core.dispatcher.Dispatcher):
        raise ValueError(
            "`numba=True`, but the objective is not wrapped by one of the `Numba` `@jit` decorators, (e.g. `numba.jit`, `numba.njit`). If you wish to use `numba=True`, make sure to wrap the ***`Numba` compatible objective function*** by one of the `Numba` `@jit` decorators"
        )

    n_arguments = len(signature(objective).parameters)
    if n_arguments != 2:
        raise ValueError(
            "Objective function should take exactly 2 arguments, one for `parameters` and one for `metaparameters` even if metaparameters aren't used in the evaluation"
        )

    outputs = objective(parameters, metaparameters)
    try:
        n_outputs = len(outputs)
    except TypeError:
        raise ValueError(
            "Objective function should return a subscriptable array-like structure (e.g. `np.ndarray`) even if one value is only returned"
        )

    return n_outputs


def _check_threads(threads, parameters):
    if isinstance(parameters, int):
        if parameters + 1 != threads:
            raise ValueError(
                "Each parameter should have only one CPU thread, along with one for the base evaluation."
            )
    elif len(parameters[0]) + 1 != threads:
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
    metaparameters=None,
    columns=None,
    numba=None,
):
    if isinstance(initial, (int, float)):
        initial = np.array([initial])
    elif isinstance(initial, (list, nb.typed.List)):
        initial = np.asarray(initial)
    elif isinstance(initial, (np.ndarray, type(None))):
        pass
    else:
        raise ValueError(
            "Initial parameters should expressed as either `int`, `float`, `list`, `nb.typed.List` or `np.ndarray`"
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

    if not isinstance(outputs, (list, np.ndarray, type(None))):
        raise ValueError("Outputs should be of type `list` or `np.ndarray`")
    try:
        len_outputs = len(outputs[1])
    except TypeError:
        len_outputs = 1

    if not isinstance(parameters, (list, np.ndarray, type(None))):
        raise ValueError("Parameters should be of type `list` or `np.ndarray`")
    try:
        len_parameters = len(parameters[0])
    except TypeError:
        len_parameters = 1

    if not isinstance(metaparameters, (list, nb.typed.List, np.ndarray, type(None))):
        raise ValueError(
            "metaparameters should be of type `list`, `nb.typed.List` or `np.ndarray`"
        )
    if numba and isinstance(metaparameters, list):
        dt = list()
        for i, value in enumerate(metaparameters):
            dt.append((str(value), str(type(value))[8:-2]))
        metaparameters = np.array(metaparameters, dtype=dt)
    elif not numba and isinstance(metaparameters, list):
        metaparameters = np.asarray(metaparameters)
    elif isinstance(metaparameters, nb.typed.List):
        metaparameters = np.asarray(metaparameters)
    if isinstance(metaparameters, type(None)):
        len_metaparameters = 0
    elif isinstance(metaparameters, (list, nb.typed.List, np.ndarray)):
        len_metaparameters = len(metaparameters)

    if not isinstance(columns, (list, np.ndarray, type(None))):
        raise ValueError("Columns should be either a `list` or `np.ndarray`")

    if outputs is not None and parameters is not None and columns is not None:
        if (len_outputs + len_parameters + len_metaparameters) != len(columns):
            raise ValueError(
                "Number of column names given in `columns` doesn't match the combined number of outputs, parameters and columns"
            )

    return initial, metaparameters


def _check_numba(numba):
    if not isinstance(numba, bool):
        raise ValueError(
            "`numba` argument specifies whether the algorithm will use `Numba` JIT compiler or Python interpreter, and should be of type `bool`"
        )
