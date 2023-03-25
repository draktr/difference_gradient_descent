import numpy as np
import warnings
from optschedule import Schedule


def _check_iterables(differences, learning_rates, epochs):
    if not isinstance(epochs, int):
        raise ValueError("Number of epochs should be an integer")
    if epochs < 1:
        raise ValueError("Number of epochs should be positive")

    if isinstance(differences, (int, float)):
        differences = Schedule().constant(differences[0])
    elif isinstance(differences, list):
        differences = np.asarray(differences)
    elif isinstance(differences, np.ndarray):
        pass
    else:
        raise ValueError(
            "Differences should be of type `int`, `float`, `list` or `np.ndarray`"
        )

    if isinstance(learning_rates, (int, float)):
        learning_rates = Schedule(epochs).constant(learning_rates[0])
    elif isinstance(learning_rates, list):
        learning_rates = np.asarray(learning_rates)
    elif isinstance(learning_rates, np.ndarray):
        pass
    else:
        raise ValueError(
            "Learning rates should be of type `int`, `float`, `list` or `np.ndarray`"
        )

    if differences.shape[0] != learning_rates.shape[0]:
        raise ValueError("Number of differences and learning rates should be equal.")
    if epochs != differences.shape[0]:
        raise ValueError(
            "Number of epochs, differences and learning rates given should be equal."
        )

    return differences, learning_rates, epochs


def _check_objective(objective_function):
    if not callable(objective_function):
        raise ValueError(
            f"Objective function should be a callable. Current objective function type is:{type(objective_function)}"
        )


def _check_threads(threads, parameters):
    if len(parameters[0]) + 1 != threads:
        raise ValueError(
            "Each parameter should have only one CPU thread, along with one for the base evaluation."
        )


def _check_arguments(
    initial_parameters=None,
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
    if isinstance(initial_parameters, (int, float)):
        initial_parameters = np.array([initial_parameters])
    elif isinstance(initial_parameters, list):
        initial_parameters = np.asarray(initial_parameters)
    elif isinstance(initial_parameters, (np.ndarray, type(None))):
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
    if not isinstance(outputs, (np.ndarray, type(None))):
        raise ValueError("Outputs should be of type `np.ndarray`")
    if not isinstance(parameters, (np.ndarray, type(None))):
        raise ValueError("Parameters should be of type `np.ndarray`")
    if not isinstance(columns, (np.ndarray, list, type(None))):
        raise ValueError("Columns should be either a list or `np.ndarray`")
    if parameters is not None and columns is not None:
        if parameters.shape[1] != columns.shape[0]:
            raise ValueError(
                "Number of parameter columns in `parameters` array doesn't match the number of column names given in `columns`"
            )
    if outputs is not None and columns is not None:
        if outputs.shape[0] != parameters.shape[0]:
            raise ValueError(
                "Number of epochs in `outputs` and `parameters` arrays doesn't match"
            )

    return initial_parameters
