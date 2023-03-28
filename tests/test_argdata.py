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


def test_ints(optimizer, differences, rates):
    outputs, parameters = optimizer.difference_gradient_descent(
        initial_parameters=5,
        differences=0.0001,
        learning_rates=0.01,
        epochs=1000,
        momentum=0,
    )

    assert outputs[-1] <= 0.1


def test_floats(optimizer, differences, rates):
    outputs, parameters = optimizer.difference_gradient_descent(
        initial_parameters=5.2,
        differences=0.0001,
        learning_rates=0.01,
        epochs=1000,
        momentum=0.2,
    )

    assert outputs[-1] <= 0.1


def test_lists(optimizer, differences, rates):
    outputs, parameters = optimizer.difference_gradient_descent(
        initial_parameters=[5.2],
        differences=list(differences),
        learning_rates=list(rates),
        epochs=1000,
    )

    assert outputs[-1] <= 0.1


def test_arrays(optimizer, differences, rates):
    outputs, parameters = optimizer.difference_gradient_descent(
        initial_parameters=np.array([5.2]),
        differences=differences,
        learning_rates=rates,
        epochs=1000,
    )

    assert outputs[-1] <= 0.1
