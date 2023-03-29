import pytest
import numpy as np
from fdgd import FDGD
from optschedule import Schedule


@pytest.fixture
def optimizer():
    def foo(params):
        return [(params[0] + 2) ** 2]

    optimizer = FDGD(objective=foo)

    return optimizer


@pytest.fixture
def constants_optimizer():
    def loo(params, permission):
        if permission:
            return [(params[0] + 2) ** 2]

    constants_optimizer = FDGD(objective=loo)

    return constants_optimizer


@pytest.fixture
def outputs_optimizer():
    def goo(params):
        return [
            (params[0] + 2) ** 2 + (params[1] + 3) ** 2 + (params[2] + 1) ** 2,
            params[0] + params[1] + params[2],
        ]

    outputs_optimizer = FDGD(objective=goo)

    return outputs_optimizer


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


def test_momentum(optimizer, differences, rates):
    outputs, parameters = optimizer.descent(
        initial=[5],
        h=differences,
        l=rates,
        epochs=1000,
        momentum=0.9,
    )

    assert outputs[-1] <= 0.1


def test_rng_seed(optimizer, differences, rates):
    outputs, parameters = optimizer.partial_descent(
        initial=[5],
        h=differences,
        l=rates,
        epochs=1000,
        parameters_used=1,
        rng_seed=2,
    )

    assert outputs[-1] <= 0.1


def test_values_out(optimizer, differences, rates):
    outputs, parameters = optimizer.descent(
        initial=[5],
        h=differences,
        l=rates,
        epochs=1000,
    )

    values = optimizer.values_out(["objective_value", "x_variable"])

    assert (
        outputs[-1] <= 0.1
        and values.columns[0] == "objective_value"
        and values.columns[1] == "x_variable"
        and not np.all(np.isnan(values))
        and not np.all(np.isinf(values))
    )


def test_values_out_constants(constants_optimizer, differences, rates):

    outputs, parameters = constants_optimizer.descent(
        initial=[5],
        h=differences,
        l=rates,
        epochs=1000,
        permission=True,
    )

    values = constants_optimizer.values_out(
        ["objective_value", "x_variable", "permission"]
    )

    assert (
        outputs[-1] <= 0.1
        and values.columns[0] == "objective_value"
        and values.columns[1] == "x_variable"
        and values.columns[2] == "permission"
        and not np.all(np.isnan(values))
        and not np.all(np.isinf(values))
    )


def test_values_out_multiple_outputs(outputs_optimizer, differences, rates):
    outputs, parameters = outputs_optimizer.descent(
        initial=[5, 5, 5],
        h=differences,
        l=rates,
        epochs=1000,
    )

    values = outputs_optimizer.values_out(
        [
            "objective_value",
            "additional_output",
            "x_variable",
            "y_variable",
            "z_variable",
        ]
    )

    assert (
        outputs[-1][0] <= 0.1
        and abs(outputs[-1][1] - (-6)) <= 10**-1
        and values.columns[0] == "objective_value"
        and values.columns[1] == "additional_output"
        and values.columns[2] == "x_variable"
        and values.columns[2] == "y_variable"
        and values.columns[2] == "z_variable"
        and not np.all(np.isnan(values))
        and not np.all(np.isinf(values))
    )
