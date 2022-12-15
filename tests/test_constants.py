import pytest
from difference_gradient_descent import DifferenceGradientDescent
from schedules import Schedules


@pytest.fixture
def optimizer():
    def foo(params, const):
        return[(params[0]+2+const[0])**2]

    optimizer = DifferenceGradientDescent(objective_function = foo)

    return optimizer

@pytest.fixture
def scheduler():
    scheduler = Schedules(n_steps = 10000)

    return scheduler

def test_one_thread(optimizer, scheduler):
    outputs, parameters = optimizer.difference_gradient_descent(initial_parameters=[5],
                                                                differences = scheduler.constant(0.001),
                                                                learning_rates = scheduler.constant(0.01),
                                                                epochs = 1000,
                                                                constants = [3])

    assert abs(parameters[-1] - (-3)) <= 10**-2

def test_multithread(optimizer, scheduler):
    outputs, parameters = optimizer.difference_gradient_descent(initial_parameters=[5],
                                                                differences = scheduler.constant(0.001),
                                                                learning_rates = scheduler.constant(0.01),
                                                                epochs = 1000,
                                                                constants = [3],
                                                                threads = 2)

    assert abs(parameters[-1] - (-3)) <= 10**-2

def test_partial_one_thread(optimizer, scheduler):
    outputs, parameters = optimizer.partial_gradient_descent(initial_parameters=[5],
                                                             differences = scheduler.constant(0.001),
                                                             learning_rates = scheduler.constant(0.01),
                                                             epochs = 10000,
                                                             parameters_used = 1,
                                                             constants = [3])

    assert abs(parameters[-1] - (-3)) <= 10**-2

def test_partial_multithread(optimizer, scheduler):
    outputs, parameters = optimizer.partial_gradient_descent(initial_parameters=[5],
                                                             differences = scheduler.constant(0.001),
                                                             learning_rates = scheduler.constant(0.01),
                                                             epochs = 10000,
                                                             parameters_used = 1,
                                                             constants = [3],
                                                             threads = 2)

    assert abs(parameters[-1] - (-3)) <= 10**-2
