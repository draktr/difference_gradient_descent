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


class DifferenceGradientDescent():
    """
    DifferenceGradientDescent class.

    Instance variables:

    - ``objective_function`` - function
    - ``n_additional_steps`` - int
    """

    def __init__(self,
                 objective_function):
        """
        Initializes DifferenceGradientDescent optimizer.

        :param objective_function: Objective function to minimize
        :type objective_function: Function
        """

        self.objective_function = objective_function

    def _set_up_variables(self,
                          epochs,
                          learning_rates,
                          differences,
                          initial_parameters,
                          constants):
        """
        Sets up variables and data structures for algorithm.

        :param epochs: Number of epochs
        :type epochs: int
        :param learning_rates: Sequence of learning rates, one for each epoch
        :type learning_rates: ndarray
        :param differences: Sequence of difference values, one for each epoch.
        :type differences: ndarray
        :param initial_parameters: Starting parameter values for the algorithm
        :type initial_parameters: list
        :param constants: Constants values required for the evaluation of the objective function
                          that aren't adjusted in the process of gradient descent, defaults to None
        :type constants: list, optional
        :raises ValueError: If the number of epochs, learning rates and differences do not match
        :return: Number of parameters, outputs array, parameters array, difference objective array, change variable
        :rtype: int, ndarray, ndarray, ndarray, float
        """

        if epochs != len(learning_rates) or epochs != len(differences) or len(learning_rates) != len(differences):
            raise ValueError("Number of epochs, learning rates and differences given should be equal.")

        n_parameters = len(initial_parameters)
        if constants is None:
            n_outputs = len(self.objective_function(initial_parameters))
        else:
            n_outputs = len(self.objective_function(np.append(initial_parameters, constants)))
        outputs = np.zeros([epochs, n_outputs])
        parameters = np.zeros([epochs + 1, n_parameters])
        parameters[0] = initial_parameters
        difference_objective = np.zeros(n_parameters)
        change = 0.0

        return n_parameters, outputs, parameters, difference_objective, change

    def _check_constants(self,
                         constants,
                         parameters,
                         epoch):
        """
        Checks if objective function requires constants in which case it appends
        it to the parameters array for passing to the obejctive function for evaluation.

        :param constants: Constants values required for the evaluation of the objective function
                          that aren't adjusted in the process of gradient descent, defaults to None
        :type constants: list, optional
        :param parameters: Parameters of the objective function that are being optimized
        :type parameters: ndarray
        :param epoch: Current epoch
        :type epoch: int
        :return: Current parameters array containig all the neccesary inputs of the objective function
        :rtype: ndarray
        """

        if constants is None:
            current_parameters = parameters[epoch]
        else:
            current_parameters = np.append(parameters[epoch], constants)

        return current_parameters

    def _set_difference_objective(self,
                                  parameters,
                                  epoch,
                                  parameter,
                                  difference,
                                  constants):
        """
        Does neccesary operations for computation of the difference objective
        for a particular parameter and computes the difference objective value.

        :param parameters: Parameters of the objective function that are being optimized
        :type parameters: ndarray
        :param epoch: Current epoch
        :type epoch: int
        :param parameter: Current parameter
        :type parameter: int
        :param difference: Current value for the difference
        :type difference: float
        :param constants: Constants values required for the evaluation of the objective function
                          that aren't adjusted in the process of gradient descent, defaults to None
        :type constants: list, optional
        :return: Difference objective value for the particular parameter
        :rtype: float
        """

        current_parameters = parameters[epoch]
        current_parameters[parameter] = current_parameters[parameter] + difference
        if constants is not None:
            current_parameters = np.append(current_parameters, constants)
        parameter_objective = self.objective_function(current_parameters)[0]

        return parameter_objective

    def _update(self,
                rate,
                difference_objective,
                outputs,
                difference,
                momentum,
                change,
                epoch,
                parameters):
        """
        Updates parameter values by performing the gradient descent operation.

        :param rate: Current learning rate
        :type rate: float
        :param difference_objective: Difference objective values array
        :type difference_objective: ndarray
        :param outputs: Objective function outputs
        :type outputs: ndarray
        :param difference: Current difference
        :type difference: float
        :param momentum: Momentum value
        :type momentum: float
        :param change: Current change value
        :type change: float
        :param epoch: Current epoch
        :type epoch: int
        :param parameters: Parameters array of all parameters for all epochs
        :type parameters: ndarray
        :return: Updated parameter values
        :rtype: ndarray
        """

        change = rate * (difference_objective - outputs[epoch, 0]) / difference + momentum * change
        parameters = parameters[epoch] - change

        return parameters

    def difference_gradient_descent(self,
                                    initial_parameters,
                                    differences,
                                    learning_rates,
                                    epochs,
                                    constants = None,
                                    momentum = 0.0,
                                    threads = 1):
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
        :param constants: Constants values required for the evaluation of the objective function
                          that aren't adjusted in the process of gradient descent, defaults to None
        :type constants: list, optional
        :param momentum: Momentum turn for stabilizing the rate of learning when moving towards the global optimum, defaults to 0.0
        :type momentum: float, optional
        :param threads: Number of CPU threads used for computation, defaults to 1
        :type threads: int, optional
        :raises ValueError: If the number of threads specified doesn't match the number of processes
        :raises ValueError: If negative value of threads is given
        :return: Objective function outputs and parameters for each epoch
        :rtype: ndarray
        """

        n_parameters, outputs, parameters, difference_objective, change = self._set_up_variables(epochs,
                                                                                                 learning_rates,
                                                                                                 differences,
                                                                                                 initial_parameters,
                                                                                                 constants)

        if threads == 1:
            for epoch, rate, difference in zip(range(epochs), learning_rates, differences):
                current_parameters = self._check_constants(constants, parameters, epoch)

                # Evaluating the objective function that will count as "official" one for this epoch
                outputs[epoch] = self.objective_function(current_parameters)

                # Objective function is evaluated for every (differentiated) parameter
                # because we need it to calculate partial derivatives
                for parameter in range(n_parameters):
                    difference_objective[parameter] = self._set_difference_objective(parameters,
                                                                                     epoch,
                                                                                     parameter,
                                                                                     difference,
                                                                                     constants)

                # These parameters will be used for the evaluation in the next epoch
                parameters[epoch+1] = self._update(rate, difference_objective, outputs, difference, momentum, change, epoch, parameters)

        elif threads > 1:
            if len(parameters[0]) + 1 != threads:
                raise ValueError("Each parameter should have only one CPU thread, along with one for the base evaluation.")

            for epoch, rate, difference in zip(range(epochs), learning_rates, differences):
                # One set of parameters is needed for each partial derivative, and one is needed for the base case
                current_parameters = np.zeros([n_parameters + 1, n_parameters])
                current_parameters[0] = parameters[epoch]
                for parameter in range(n_parameters):
                    current_parameters[parameter + 1] = parameters[epoch]
                    current_parameters[parameter + 1, parameter] = current_parameters[0, parameter] + difference

                if constants is not None:
                    current_parameters = np.concatenate([current_parameters,
                                                         np.full((n_parameters + 1, len(constants)), constants)],
                                                         axis=1)

                parallel_outputs = Parallel(n_jobs=threads)(delayed(self.objective_function)(i) for i in current_parameters)

                # This objective function evaluation will be used as the "official" one for this epoch
                outputs[epoch] = parallel_outputs[0]
                difference_objective = np.array([parallel_outputs[i][0] for i in range(1, n_parameters+1)])

                # These parameters will be used for the evaluation in the next epoch
                parameters[epoch+1] = self._update(rate, difference_objective, outputs, difference, momentum, change, epoch, parameters)

        else:
            raise ValueError("Number of threads should be positive.")

        return outputs, parameters

    def partial_gradient_descent(self,
                                 initial_parameters,
                                 differences,
                                 learning_rates,
                                 epochs,
                                 parameters_used,
                                 constants = None,
                                 momentum = 0.0,
                                 threads = 1,
                                 rng_seed = 88):
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
        :param constants: Constants values required for the evaluation of the objective function
                          that aren't adjusted in the process of gradient descent, defaults to None
        :type constants: list, optional
        :param momentum: Momentum turn for stabilizing the rate of learning when moving towards the global optimum, defaults to 0.0
        :type momentum: float, optional
        :param threads: Number of CPU threads used for computation, defaults to 1
        :type threads: int, optional
        :param rng_seed: Seed for the random number generator used for determining which parameters
                         are used in each epoch for computation of gradients, defaults to 88
        :type rng_seed: int, optional
        :raises ValueError: If the number of threads specified doesn't match the number of processes
        :raises ValueError: If negative value of threads is given
        :return: Objective function outputs and parameters for each epoch
        :rtype: ndarray
        """

        n_parameters, outputs, parameters, difference_objective, change = self._set_up_variables(epochs,
                                                                                                 learning_rates,
                                                                                                 differences,
                                                                                                 initial_parameters,
                                                                                                 constants)
        rng = np.random.default_rng(rng_seed)

        if threads == 1:
            for epoch, rate, difference in zip(range(epochs), learning_rates, differences):
                param_idx = rng.integers(low=0, high=n_parameters, size=parameters_used)
                current_parameters = self._check_constants(constants, parameters, epoch)

                # Evaluating the objective function that will count as "official" one for this epoch
                outputs[epoch] = self.objective_function(current_parameters)

                # Objective function is evaluated only for random parameters because we need it
                # to calculate partial derivatives, while limiting computational expense
                for parameter in range(n_parameters):
                    if parameter in param_idx:
                        difference_objective[parameter] = self._set_difference_objective(parameters,
                                                                                         epoch,
                                                                                         parameter,
                                                                                         difference,
                                                                                         constants)
                    else:
                        # Difference objective value is still recorded (as base evaluation value)
                        # for non-differenced parameters (in current epoch) for consistency and convenience
                        difference_objective[parameter] = outputs[epoch, 0]

                # These parameters will be used for the evaluation in the next epoch
                parameters[epoch+1] = self._update(rate, difference_objective, outputs, difference, momentum, change, epoch, parameters)

        elif threads > 1:
            if parameters_used + 1 != threads:
                raise ValueError("Each parameter should have only one CPU thread, along with one for the base evaluation.")

            for epoch, rate, difference in zip(range(epochs), learning_rates, differences):
                param_idx = rng.integers(low=0, high=n_parameters, size=parameters_used)

                # Objective function is evaluated only for random parameters because we need it
                # to calculate partial derivatives, while limiting computational expense
                current_parameters = np.zeros([parameters_used + 1, n_parameters])
                current_parameters[0] = parameters[epoch]
                for parameter in range(parameters_used):
                    current_parameters[parameter + 1] = parameters[epoch]
                    if parameter in param_idx:
                        current_parameters[parameter + 1, parameter] = current_parameters[0, parameter] + difference

                if constants is not None:
                    current_parameters = np.concatenate([current_parameters,
                                                         np.full((n_parameters + 1, len(constants)), constants)],
                                                         axis=1)

                parallel_outputs = Parallel(n_jobs=threads)(delayed(self.objective_function)(i) for i in current_parameters)

                # This objective function evaluation will be used as the "official" one for this epoch
                outputs[epoch] = parallel_outputs[0]
                # Difference objective value is still recorded (as base evaluation value)
                # for non-differenced parameters (in current epoch) for consistency and convenience
                difference_objective = np.full(n_parameters, parallel_outputs[0][0])
                difference_objective[param_idx] = np.array([parallel_outputs[i][0] for i in range(1, parameters_used + 1)])

                # These parameters will be used for the evaluation in the next epoch
                parameters[epoch+1] = self._update(rate, difference_objective, outputs, difference, momentum, change, epoch, parameters)

        else:
            raise ValueError("Number of threads should be positive.")

        return outputs, parameters

    def values_out(self,
                   outputs,
                   parameters,
                   constants,
                   columns):
        """
        Produces a Pandas DataFrame of objective function outputs and parameter values for each epoch of the algorithm.

        :param outputs: Objective function outputs in each of the epochs.
                        Should be ``epochs`` by ``n_outputs`` shaped ndarray
        :type outputs: ndarray
        :param parameters: Parameters values in each of the epochs. Should be ``epochs`` by ``n_parameters`` shaped ndarray
        :type parameters: ndarray
        :param constants: Constant values for only one of the epochs. Shouls be 1 by ``len(constants)`` shaped ndarray
        :type constants: list
        :param columns: Column names for the DataFrame. Should include names for all the outputs, parameters and constants
        :type columns: list
        :return: Dataframe of all the values of inputs and outputs of the objective function for each epoch
        :rtype: pd.DataFrame
        """

        inputs = np.concatenate([parameters, np.full((len(parameters), len(constants)), constants)], axis = 1)

        values = pd.DataFrame([pd.DataFrame(outputs),
                               pd.DataFrame(inputs)],
                               columns=columns)

        return values
