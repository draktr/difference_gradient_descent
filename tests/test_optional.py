import pytest
import numpy as np
from difference_gradient_descent import DifferenceGradientDescent
from optschedule import Schedule


@pytest.fixture
def optimizer():
    def foo(params):
        return [(params[0] + 2) ** 2]

    optimizer = DifferenceGradientDescent(objective_function=foo)

    return optimizer


@pytest.fixture
def constants_optimizer():
    def loo(params, permission):
        if permission:
            return [(params[0] + 2) ** 2]

    constants_optimizer = DifferenceGradientDescent(objective_function=loo)

    return constants_optimizer


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
    outputs, parameters = optimizer.difference_gradient_descent(
        initial_parameters=[5],
        differences=differences,
        learning_rates=rates,
        epochs=1000,
        momentum=0.9,
    )

    assert outputs[-1] <= 0.1


def test_rng_seed(optimizer, differences, rates):
    outputs, parameters = optimizer.partial_gradient_descent(
        initial_parameters=[5],
        differences=differences,
        learning_rates=rates,
        epochs=1000,
        parameters_used=1,
        rng_seed=2,
    )

    assert outputs[-1] <= 0.1


def test_values_out(optimizer, differences, rates):
    outputs, parameters = optimizer.difference_gradient_descent(
        initial_parameters=[5],
        differences=differences,
        learning_rates=rates,
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

    outputs, parameters = constants_optimizer.difference_gradient_descent(
        initial_parameters=[5],
        differences=differences,
        learning_rates=rates,
        epochs=1000,
        permission=True,
    )

    assert outputs[-1] <= 0.1
