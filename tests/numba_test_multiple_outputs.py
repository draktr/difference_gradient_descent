import pytest
import numba as nb
from findi import descent, partial_descent
from optschedule import Schedule


@nb.njit
def foo(params, metaparameters):
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


def test_descent(differences, rates):
    outputs, parameters = descent(
        objective=foo,
        initial=[5, 5, 5],
        h=differences,
        l=rates,
        epochs=1000,
        numba=True,
    )

    assert outputs[-1][0] <= 0.1 and abs(outputs[-1][1] - (-6)) <= 0.1


def test_partial(differences, rates):
    outputs, parameters = partial_descent(
        objective=foo,
        initial=[5, 5, 5],
        h=differences,
        l=rates,
        epochs=1000,
        parameters_used=2,
        numba=True,
    )

    assert outputs[-1][0] <= 0.1 and abs(outputs[-1][1] - (-6)) <= 0.1
