import pytest
from findi import descent, partial_descent, partially_partial_descent
from optschedule import Schedule


def foo(params):
    return [(params[0] + 2) ** 2]


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


def test_one_thread(differences, rates):
    outputs, parameters = descent(
        objective=foo,
        initial=[5],
        h=differences,
        l=rates,
        epochs=1000,
    )

    assert outputs[-1] <= 0.1


def test_multithread(differences, rates):
    outputs, parameters = descent(
        objective=foo,
        initial=[5],
        h=differences,
        l=rates,
        epochs=1000,
        threads=2,
    )

    assert outputs[-1] <= 0.1


def test_partial_one_thread(differences, rates):
    outputs, parameters = partial_descent(
        objective=foo,
        initial=[5],
        h=differences,
        l=rates,
        epochs=1000,
        parameters_used=1,
    )

    assert outputs[-1] <= 0.1


def test_partial_multithread(differences, rates):
    outputs, parameters = partial_descent(
        objective=foo,
        initial=[5],
        h=differences,
        l=rates,
        epochs=1000,
        parameters_used=1,
        threads=2,
    )

    assert outputs[-1] <= 0.1


def test_partially_partial_multithread(differences, rates):
    outputs, parameters = partially_partial_descent(
        objective=foo,
        initial=[5],
        h=differences,
        l=rates,
        partial_epochs=300,
        total_epochs=1000,
        parameters_used=1,
        threads=2,
    )

    assert outputs[-1] <= 0.1
