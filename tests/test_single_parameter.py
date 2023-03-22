import pytest
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


def test_one_thread(optimizer, differences, rates):
    outputs, parameters = optimizer.difference_gradient_descent(
        initial_parameters=[5],
        differences=differences,
        learning_rates=rates,
        epochs=1000,
    )

    assert outputs[-1] <= 0.1


def test_multithread(optimizer, differences, rates):
    outputs, parameters = optimizer.difference_gradient_descent(
        initial_parameters=[5],
        differences=differences,
        learning_rates=rates,
        epochs=1000,
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
    )

    assert outputs[-1] <= 0.1


def test_partial_multithread(optimizer, differences, rates):
    outputs, parameters = optimizer.partial_gradient_descent(
        initial_parameters=[5],
        differences=differences,
        learning_rates=rates,
        epochs=1000,
        parameters_used=1,
        threads=2,
    )

    assert outputs[-1] <= 0.1


def test_partially_partial_one_thread(optimizer, differences, rates):
    outputs, parameters = optimizer.partially_partial_gradient_descent(
        initial_parameters=[5],
        differences=differences,
        learning_rates=rates,
        partial_epochs=300,
        total_epochs=1000,
        parameters_used=1,
        threads=2,
    )

    assert outputs[-1] <= 0.1
