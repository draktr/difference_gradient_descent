import pytest
import numpy as np
from findi import descent, partial_descent, values_out
from optschedule import Schedule


def foo(params):
    return [(params[0] + 2) ** 2]


def loo(params, permission):
    if permission:
        return [(params[0] + 2) ** 2]


def goo(params):
    return [
        (params[0] + 2) ** 2 + (params[1] + 3) ** 2 + (params[2] + 1) ** 2,
        params[0] + params[1] + params[2],
    ]


@pytest.fixture
def scheduler():
    scheduler = Schedule(n_steps=1000)

    return scheduler


@pytest.fixture
def differences(scheduler):
    differences = scheduler.exponential_decay(initial_value=0.01, decay_rate=0.0005)

    return differences


@pytest.fixture
def rates(scheduler):
    rates = scheduler.exponential_decay(initial_value=0.01, decay_rate=0.5)

    return rates


def test_momentum(differences, rates):
    outputs, parameters = descent(
        objective=foo,
        initial=[5],
        h=differences,
        l=rates,
        epochs=1000,
        momentum=0.9,
    )

    assert outputs[-1] <= 0.1


def test_rng_seed(differences, rates):
    outputs, parameters = partial_descent(
        objective=foo,
        initial=[5],
        h=differences,
        l=rates,
        epochs=1000,
        parameters_used=1,
        rng_seed=2,
    )

    assert outputs[-1] <= 0.1


def test_values_out(differences, rates):
    outputs, parameters = descent(
        objective=foo,
        initial=[5],
        h=differences,
        l=rates,
        epochs=1000,
    )

    values = values_out(
        outputs=outputs,
        parameters=parameters,
        columns=["objective_value", "x_variable"],
    )

    assert (
        outputs[-1] <= 0.1
        and values.columns[0] == "objective_value"
        and values.columns[1] == "x_variable"
        and not np.all(np.isnan(values))
        and not np.all(np.isinf(values))
    )


def test_values_out_constants(differences, rates):
    outputs, parameters = descent(
        objective=loo,
        initial=[5],
        h=differences,
        l=rates,
        epochs=1000,
        permission=True,
    )

    values = values_out(
        outputs=outputs,
        parameters=parameters,
        columns=["objective_value", "x_variable"],
        permission=True,
    )

    assert (
        outputs[-1] <= 0.1
        and values.columns[0] == "objective_value"
        and values.columns[1] == "x_variable"
        and values.columns[2] == "permission"
        and not np.all(np.isnan(values))
        and not np.all(np.isinf(values))
    )


def test_values_out_multiple_outputs(differences, rates):
    outputs, parameters = descent(
        objective=goo,
        initial=[5, 5, 5],
        h=differences,
        l=rates,
        epochs=1000,
    )

    values = values_out(
        outputs=outputs,
        parameters=parameters,
        columns=[
            "objective_value",
            "additional_output",
            "x_variable",
            "y_variable",
            "z_variable",
        ],
    )

    assert (
        outputs[-1][0] <= 0.1
        and abs(outputs[-1][1] - (-6)) <= 10**-1
        and values.columns[0] == "objective_value"
        and values.columns[1] == "additional_output"
        and values.columns[2] == "x_variable"
        and values.columns[3] == "y_variable"
        and values.columns[4] == "z_variable"
        and not np.all(np.isnan(values))
        and not np.all(np.isinf(values))
    )
