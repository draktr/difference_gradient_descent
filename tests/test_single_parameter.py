import pytest
from findi import GradientDescent
from optschedule import Schedule


@pytest.fixture
def optimizer():
    def foo(params):
        return [(params[0] + 2) ** 2]

    optimizer = GradientDescent(objective=foo)

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
    outputs, parameters = optimizer.descent(
        initial=[5],
        h=differences,
        l=rates,
        epochs=1000,
    )

    assert outputs[-1] <= 0.1


def test_multithread(optimizer, differences, rates):
    outputs, parameters = optimizer.descent(
        initial=[5],
        h=differences,
        l=rates,
        epochs=1000,
        threads=2,
    )

    assert outputs[-1] <= 0.1


def test_partial_one_thread(optimizer, differences, rates):
    outputs, parameters = optimizer.partial_descent(
        initial=[5],
        h=differences,
        l=rates,
        epochs=1000,
        parameters_used=1,
    )

    assert outputs[-1] <= 0.1


def test_partial_multithread(optimizer, differences, rates):
    outputs, parameters = optimizer.partial_descent(
        initial=[5],
        h=differences,
        l=rates,
        epochs=1000,
        parameters_used=1,
        threads=2,
    )

    assert outputs[-1] <= 0.1


def test_partially_partial_multithread(optimizer, differences, rates):
    outputs, parameters = optimizer.partially_partial_descent(
        initial=[5],
        h=differences,
        l=rates,
        partial_epochs=300,
        total_epochs=1000,
        parameters_used=1,
        threads=2,
    )

    assert outputs[-1] <= 0.1
