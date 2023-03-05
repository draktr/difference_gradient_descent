import pytest
from difference_gradient_descent import DifferenceGradientDescent
from schedules import Schedules


@pytest.fixture
def optimizer():
    def foo(inputs):
        return [(inputs[0] + 2 + inputs[1]) ** 2]

    optimizer = DifferenceGradientDescent(objective_function=foo)

    return optimizer


@pytest.fixture
def scheduler():
    scheduler = Schedules(n_steps=1000)

    return scheduler


@pytest.fixture
def differences(scheduler):

    differences = scheduler.exponential_decay(initial_value=0.01, decay_rate=0.0005)

    return differences


@pytest.fixture
def rates(scheduler):

    rates = scheduler.exponential_decay(initial_value=0.01, decay_rate=0.5)

    return rates


def test_one_thread(optimizer, differences, rates):

    outputs, parameters = optimizer.difference_gradient_descent(
        initial_parameters=[5],
        differences=differences,
        learning_rates=rates,
        epochs=1000,
        constants=[3],
    )

    assert outputs[-1] <= 0.1


def test_multithread(optimizer, differences, rates):

    outputs, parameters = optimizer.difference_gradient_descent(
        initial_parameters=[5],
        differences=differences,
        learning_rates=rates,
        epochs=1000,
        constants=[3],
        threads=2,
    )

    assert outputs[-1] <= 0.1


def test_partial_one_thread(optimizer, differences, rates):

    outputs, parameters = optimizer.partial_gradient_descent(
        initial_parameters=[5],
        differences=differences,
        learning_rates=rates,
        epochs=1000,
        parameters_used=1,
        constants=[3],
    )

    assert outputs[-1] <= 0.1


def test_partial_multithread(optimizer, differences, rates):

    outputs, parameters = optimizer.partial_gradient_descent(
        initial_parameters=[5],
        differences=differences,
        learning_rates=rates,
        epochs=1000,
        parameters_used=1,
        constants=[3],
        threads=2,
    )

    assert outputs[-1] <= 0.1
