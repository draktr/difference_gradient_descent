import pytest
import numpy as np
from findi import descent, partial_descent
from optschedule import Schedule


def foo(params, constants):
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


def test_ints():
    outputs, parameters = descent(
        objective=foo,
        initial=5,
        h=0.0001,
        l=0.01,
        epochs=1000,
        momentum=0,
    )

    assert outputs[-1] <= 0.1


def test_floats():
    outputs, parameters = descent(
        objective=foo,
        initial=5.2,
        h=0.0001,
        l=0.01,
        epochs=1000,
        momentum=0.2,
    )

    assert outputs[-1] <= 0.1


def test_lists(differences, rates):
    outputs, parameters = descent(
        objective=foo,
        initial=[5.2],
        h=list(differences),
        l=list(rates),
        epochs=1000,
    )

    assert outputs[-1] <= 0.1


def test_arrays(differences, rates):
    outputs, parameters = descent(
        objective=foo,
        initial=np.array([5.2]),
        h=differences,
        l=rates,
        epochs=1000,
    )

    assert outputs[-1] <= 0.1
