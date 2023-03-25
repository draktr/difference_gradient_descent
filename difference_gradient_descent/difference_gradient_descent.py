"""
The ``difference_gradient_descent`` houses ``DifferenceGradientDescent`` class that optimizes objective functions
via Gradient Descent Algorithm variation that uses specified difference instead of infinitesimal differential for
computing gradients. This approach allows for the application of Gradient Descent on non-differentiable or any other
function as long as it can be evaluated. Methods in this class allow for computing only a random subset of gradients,
as well as parallel computing for performance benefits.

:raises ValueError: If the number of epochs, learning rates and differences do not match
:raises ValueError: If the number of threads specified doesn't match the number of processes
:raises ValueError: If negative value of threads is given
:return: Objective function outputs and parameters for each epoch
:rtype: ndarray
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import difference_gradient_descent._checks


class DifferenceGradientDescent:
    def __init__(self, objective_function):
        """
        Initializes DifferenceGradientDescent optimizer.

        Object properties:

        - ``objective_function`` - function to be optimized

        :param objective_function: Objective function to minimize
        :type objective_function: Function
        """

        self.objective_function = objective_function

    def _update(
        self,
        rate,
        difference_objective,
        outputs,
        difference,
        momentum,
        change,
        epoch,
        parameters,
    ):

        change = (
            momentum * change
            - rate * (difference_objective - outputs[epoch, 0]) / difference
        )
        updated_parameters = parameters[epoch] + change

        return updated_parameters

    def difference_gradient_descent(
        self,
        initial_parameters,
        differences,
        learning_rates,
        epochs,
        momentum=0,
        threads=1,
        **constants
    ):
        """
        Performs Gradient Descent Algorithm by using difference instead of infinitesimal differential.
        Allows for the application of the algorithm on the non-differentiable functions.

        :param initial_parameters: Starting parameter values for the algorithm
        :type initial_parameters: list
        :param differences: Sequence of difference values, one for each epoch.
        :type differences: ndarray
        :param learning_rates: Sequence of learning rates, one for each epoch
        :type learning_rates: ndarray
        :param epochs: Number of epochs
        :type epochs: int
        :param threads: Number of CPU threads used for computation, defaults to 1
        :type threads: int, optional
        :raises ValueError: If the number of epochs, learning rates and differences do not match
        :raises ValueError: If the number of threads specified doesn't match the number of processes
        :raises ValueError: If negative value of threads is given
        :return: Objective function outputs and parameters for each epoch
        :rtype: ndarray
        """

        (
            differences,
            learning_rates,
            epochs,
        ) = difference_gradient_descent._checks._check_iterables(
            differences, learning_rates, epochs
        )

        n_parameters = len(initial_parameters)
        n_outputs = len(self.objective_function(initial_parameters, **constants))
        outputs = np.zeros([epochs, n_outputs])
        parameters = np.zeros([epochs + 1, n_parameters])
        parameters[0] = initial_parameters
        difference_objective = np.zeros(n_parameters)
        change = 0

        if threads == 1:
            for epoch, (rate, difference) in enumerate(
                zip(learning_rates, differences)
            ):

                # Evaluating the objective function that will count as "official" one for this epoch
                outputs[epoch] = self.objective_function(parameters[epoch], **constants)

                # Objective function is evaluated for every (differentiated) parameter
                # because we need it to calculate partial derivatives
                for parameter in range(n_parameters):
                    current_parameters = parameters[epoch]
                    current_parameters[parameter] = (
                        current_parameters[parameter] + difference
                    )

                    difference_objective[parameter] = self.objective_function(
                        current_parameters, **constants
                    )[0]

                # These parameters will be used for the evaluation in the next epoch
                parameters[epoch + 1] = self._update(
                    rate,
                    difference_objective,
                    outputs,
                    difference,
                    momentum,
                    change,
                    epoch,
                    parameters,
                )

        elif threads > 1:
            if len(parameters[0]) + 1 != threads:
                raise ValueError(
                    "Each parameter should have only one CPU thread, along with one for the base evaluation."
                )

            # One set of parameters is needed for each partial derivative, and one is needed for the base case
            current_parameters = np.zeros([n_parameters + 1, n_parameters])

            for epoch, (rate, difference) in enumerate(
                zip(learning_rates, differences)
            ):
                current_parameters[0] = parameters[epoch]
                for parameter in range(n_parameters):
                    current_parameters[parameter + 1] = parameters[epoch]
                    current_parameters[parameter + 1, parameter] = (
                        current_parameters[0, parameter] + difference
                    )

                parallel_outputs = Parallel(n_jobs=threads)(
                    delayed(self.objective_function)(i, **constants)
                    for i in current_parameters
                )

                # This objective function evaluation will be used as the "official" one for this epoch
                outputs[epoch] = parallel_outputs[0]
                difference_objective = np.array(
                    [parallel_outputs[i][0] for i in range(1, n_parameters + 1)]
                )

                # These parameters will be used for the evaluation in the next epoch
                parameters[epoch + 1] = self._update(
                    rate,
                    difference_objective,
                    outputs,
                    difference,
                    momentum,
                    change,
                    epoch,
                    parameters,
                )

        else:
            raise ValueError("Number of threads should be positive.")

        return outputs, parameters

    def partial_gradient_descent(
        self,
        initial_parameters,
        differences,
        learning_rates,
        epochs,
        parameters_used,
        momentum=0,
        threads=1,
        rng_seed=88,
        **constants
    ):
        """
        Performs Gradient Descent Algorithm by computing gradients on only specified number of randomly selected parameters
        in each epoch and by using difference instead of infinitesimal differential. Allows for the application of the algorithm
        on the non-differentiable functions and decreases computational expense.

        :param initial_parameters: Starting parameter values for the algorithm
        :type initial_parameters: list
        :param differences: Sequence of difference values, one for each epoch.
        :type differences: ndarray
        :param learning_rates: Sequence of learning rates, one for each epoch
        :type learning_rates: ndarray
        :param epochs: Number of epochs
        :type epochs: int
        :param parameters_used: Number of parameters used in each epoch for computation of gradients
        :type parameters_used: int
        :param threads: Number of CPU threads used for computation, defaults to 1
        :type threads: int, optional
        :param rng_seed: Seed for the random number generator used for determining which parameters
                         are used in each epoch for computation of gradients, defaults to 88
        :type rng_seed: int, optional
        :raises ValueError: If the number of epochs, learning rates and differences do not match
        :raises ValueError: If the number of threads specified doesn't match the number of processes
        :raises ValueError: If negative value of threads is given
        :return: Objective function outputs and parameters for each epoch
        :rtype: ndarray
        """

        (
            differences,
            learning_rates,
            epochs,
        ) = difference_gradient_descent._checks._check_iterables(
            differences, learning_rates, epochs
        )

        n_parameters = len(initial_parameters)
        n_outputs = len(self.objective_function(initial_parameters, **constants))
        outputs = np.zeros([epochs, n_outputs])
        parameters = np.zeros([epochs + 1, n_parameters])
        parameters[0] = initial_parameters
        difference_objective = np.zeros(n_parameters)
        rng = np.random.default_rng(rng_seed)
        change = 0

        if threads == 1:
            for epoch, (rate, difference) in enumerate(
                zip(learning_rates, differences)
            ):
                param_idx = rng.integers(low=0, high=n_parameters, size=parameters_used)

                # Evaluating the objective function that will count as "official" one for this epoch
                outputs[epoch] = self.objective_function(parameters[epoch], **constants)

                # Objective function is evaluated only for random parameters because we need it
                # to calculate partial derivatives, while limiting computational expense
                for parameter in range(n_parameters):
                    if parameter in param_idx:
                        current_parameters = parameters[epoch]
                        current_parameters[parameter] = (
                            current_parameters[parameter] + difference
                        )

                        difference_objective[parameter] = self.objective_function(
                            current_parameters, **constants
                        )[0]
                    else:
                        # Difference objective value is still recorded (as base evaluation value)
                        # for non-differenced parameters (in current epoch) for consistency and convenience
                        difference_objective[parameter] = outputs[epoch, 0]

                # These parameters will be used for the evaluation in the next epoch
                parameters[epoch + 1] = self._update(
                    rate,
                    difference_objective,
                    outputs,
                    difference,
                    momentum,
                    change,
                    epoch,
                    parameters,
                )

        elif threads > 1:
            if parameters_used + 1 != threads:
                raise ValueError(
                    "Each parameter should have only one CPU thread, along with one for the base evaluation."
                )

            # One set of parameters is needed for each partial derivative used, and one is needed for the base case
            current_parameters = np.zeros([parameters_used + 1, n_parameters])

            for epoch, (rate, difference) in enumerate(
                zip(learning_rates, differences)
            ):
                param_idx = rng.integers(low=0, high=n_parameters, size=parameters_used)

                # Objective function is evaluated only for random parameters because we need it
                # to calculate partial derivatives, while limiting computational expense
                current_parameters[0] = parameters[epoch]
                for parameter in range(parameters_used):
                    current_parameters[parameter + 1] = parameters[epoch]
                    if parameter in param_idx:
                        current_parameters[parameter + 1, parameter] = (
                            current_parameters[0, parameter] + difference
                        )

                parallel_outputs = Parallel(n_jobs=threads)(
                    delayed(self.objective_function)(i, **constants)
                    for i in current_parameters
                )

                # This objective function evaluation will be used as the "official" one for this epoch
                outputs[epoch] = parallel_outputs[0]
                # Difference objective value is still recorded (as base evaluation value)
                # for non-differenced parameters (in current epoch) for consistency and convenience
                difference_objective = np.full(n_parameters, parallel_outputs[0][0])
                difference_objective[param_idx] = np.array(
                    [parallel_outputs[i][0] for i in range(1, parameters_used + 1)]
                )

                # These parameters will be used for the evaluation in the next epoch
                parameters[epoch + 1] = self._update(
                    rate,
                    difference_objective,
                    outputs,
                    difference,
                    momentum,
                    change,
                    epoch,
                    parameters,
                )

        else:
            raise ValueError("Number of threads should be positive.")

        return outputs, parameters

    def partially_partial_gradient_descent(
        self,
        initial_parameters,
        differences,
        learning_rates,
        partial_epochs,
        total_epochs,
        parameters_used,
        momentum=0.0,
        threads=1,
        rng_seed=88,
        **constants
    ):
        """
        Performs Partial Gradient Descent Algorithm for the first `partial_epochs` epochs and
        regular Difference Gradient Descent for the rest of the epochs (i.e. `total_epochs`-`partial_epochs`).

        :param initial_parameters: Starting parameter values for the algorithm
        :type initial_parameters: list
        :param differences: Sequence of difference values, one for each epoch.
        :type differences: ndarray
        :param learning_rates: Sequence of learning rates, one for each epoch
        :type learning_rates: ndarray
        :param partial_epochs: Number of epochs for Partial Gradient Descent
        :type partial_epochs: int
        :param total_epochs: Total number of epochs including both for partial and regular algorithms.
                             Implies that the number of epochs for the regular algorithm is given as
                             `total_epochs`-`partial_epochs`
        :type total_epochs: int
        :param parameters_used: Number of parameters used in each epoch for computation of gradients
        :type parameters_used: int
        :param momentum: Momentum turn for stabilizing the rate of learning when moving towards the global optimum, defaults to 0.0
        :type momentum: float, optional
        :param threads: Number of CPU threads used for computation, defaults to 1
        :type threads: int, optional
        :param rng_seed: Seed for the random number generator used for determining which parameters
                         are used in each epoch for computation of gradients, defaults to 88
        :type rng_seed: int, optional
        :return: Objective function outputs and parameters for each epoch
        :rtype: ndarray
        """

        (
            differences,
            learning_rates,
            total_epochs,
        ) = difference_gradient_descent._checks._check_iterables(
            differences, learning_rates, total_epochs
        )

        outputs_p, parameters_p = self.partial_gradient_descent(
            initial_parameters,
            differences[:partial_epochs],
            learning_rates[:partial_epochs],
            partial_epochs,
            parameters_used,
            momentum,
            threads,
            rng_seed,
            **constants
        )

        outputs_r, parameters_r = self.difference_gradient_descent(
            initial_parameters=parameters_p[-1],
            differences=differences[partial_epochs:],
            learning_rates=learning_rates[partial_epochs:],
            epochs=(total_epochs - partial_epochs),
            momentum=momentum,
            threads=threads,
            **constants
        )

        outputs = np.append(outputs_p, outputs_r)
        parameters = np.append(parameters_p, parameters_r)
        outputs = np.reshape(outputs, newshape=[-1, 1])
        parameters = np.reshape(parameters, newshape=[-1, 1])

        return outputs, parameters

    def values_out(self, outputs, parameters, columns, **constants):
        """
        Produces a Pandas DataFrame of objective function outputs and parameter values for each epoch of the algorithm.

        :param outputs: Objective function outputs in each of the epochs.
                        Should be ``epochs`` by ``n_outputs`` shaped ndarray
        :type outputs: ndarray
        :param parameters: Parameters values in each of the epochs. Should be ``epochs`` by ``n_parameters`` shaped ndarray
        :type parameters: ndarray
        :param columns: Column names for the DataFrame. Should include names for all the outputs, parameters and constants
        :type columns: list
        :return: Dataframe of all the values of inputs and outputs of the objective function for each epoch
        :rtype: pd.DataFrame
        """

        inputs = np.concatenate(
            [
                parameters,
                np.full(
                    (len(parameters), len(constants.values())), list(constants.values())
                ),
            ],
            axis=1,
        )

        values = pd.DataFrame(
            [pd.DataFrame(outputs), pd.DataFrame(inputs)], columns=columns
        )

        return values
