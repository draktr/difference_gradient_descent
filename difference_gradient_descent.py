import numpy as np
import pandas as pd
from inspect import signature
from joblib import Parallel, delayed


class GradientDescent():

    def __init__(self,
                 objective_function):

        self.objective_function = objective_function
        self.n_additional_outputs = len(signature(self.objective_function).parameters)-1

    def gradient_descent(self,
                         initial_parameters,
                         difference,
                         learning_rate,
                         epochs,
                         constants = None,
                         threads = 1):

        n_parameters = len(initial_parameters)
        outputs = np.zeros([epochs, self.n_additional_outputs+1])
        parameters = np.zeros([epochs + 1, n_parameters])
        parameters[0] = initial_parameters
        difference_objective = np.zeros(n_parameters)

        if threads == 1:
            for epoch in range(epochs):
                if constants is None:
                    current_parameters = parameters[epoch]
                else:
                    current_parameters = np.append(parameters[epoch], constants)

                # Evaluating the objective function that will count as "official" one for this epoch
                outputs[epoch] = self.objective_function(current_parameters)

                # Objective function is evaluated for every (differentiated) parameter because we need it to calculate partial derivatives
                for parameter in range(n_parameters):
                    current_parameters = parameters[epoch]
                    current_parameters[parameter] = current_parameters[parameter] + difference

                    if constants is not None:
                        current_parameters = np.append(current_parameters, constants)

                    difference_objective[parameter] = self.objective_function(current_parameters)[0]

                # These parameters will be used for the evaluation in the next epoch
                parameters[epoch+1] = parameters[epoch] - learning_rate * (difference_objective - outputs[epoch, 0]) / difference

        elif threads > 1:
            for epoch in range(epochs):
                # One set of parameters is needed for each partial derivative, and one is needed for the base case
                current_parameters = np.zeros([n_parameters + 1, n_parameters])
                current_parameters[0] = parameters[epoch]
                for parameter in range(n_parameters + 1):
                    current_parameters[parameter + 1] = parameters[epoch]
                    current_parameters[parameter + 1, parameter] = current_parameters[0, parameter] + difference

                if constants is not None:
                    current_parameters = np.concatenate([current_parameters,
                                                         np.full((n_parameters, len(constants)), constants)],
                                                         axis=1)

                parallel_outputs = Parallel(n_jobs=threads)(delayed(self.objective_function)(i) for i in current_parameters)

                # This objective function evaluation will be used as the "official" one for this epoch
                outputs[epoch] = parallel_outputs[0]
                difference_objective = np.array([parallel_outputs[i][0] for i in range(1, n_parameters+1)])

                # These parameters will be used for the evaluation in the next epoch
                parameters[epoch+1] = parameters[epoch] - learning_rate * (difference_objective - outputs[epoch, 0]) / difference

        else:
            raise ValueError("Number of threads should be positive.")

        return outputs, parameters

    def partial_gradient_descent(self,
                                 initial_parameters,
                                 difference,
                                 learning_rate,
                                 epochs,
                                 parameters_used,
                                 constants = None,
                                 threads = 1,
                                 rng_seed = 88):

        n_parameters = len(initial_parameters)
        outputs = np.zeros([epochs, self.n_additional_outputs+1])
        parameters = np.zeros([epochs, n_parameters])
        parameters[0] = initial_parameters
        difference_objective = np.zeros(n_parameters)
        rng = np.random.default_rng(rng_seed)

        if threads == 1:
            for epoch in range(epochs):
                param_idx = rng.integers(low=0, high=n_parameters, size=parameters_used)
                # Evaluating the objective function that will count as "official" one for this epoch
                outputs[epoch] = self.objective_function(np.append(parameters[epoch], constants))

                # Objective function is evaluated for random parameters because we need it to calculate partial derivatives
                for parameter in range(n_parameters):
                    if parameter in param_idx:
                        current_parameters = parameters[epoch]
                        current_parameters[parameter] = current_parameters[parameter] + difference

                        if constants is not None:
                            current_parameters = np.append(current_parameters, constants)

                        difference_objective[parameter] = self.objective_function(current_parameters, constants)[0]
                    else:
                        difference_objective[parameter] = outputs[epoch, 0]

                # These parameters will be used for the evaluation in the next epoch
                parameters[epoch+1] = parameters[epoch] - learning_rate * (difference_objective - outputs[epoch, 0]) / difference

        elif threads > 1:
            for epoch in range(epochs):
                param_idx = rng.integers(low=0, high=n_parameters, size=parameters_used)

                current_parameters = np.zeros([n_parameters, n_parameters])
                for parameter in range(n_parameters):
                    current_parameters[parameter] = parameters[epoch]
                    if parameter in param_idx:
                        current_parameters[parameter, parameter] = current_parameters[parameter, parameter] + difference

                if constants is not None:
                    current_parameters = np.concatenate([current_parameters,
                                                         np.full((n_parameters, len(constants)), constants)],
                                                         axis=1)

                parallel_outputs = Parallel(n_jobs=threads)(delayed(self.objective_function)(i) for i in current_parameters)

                difference_objective = np.full(n_parameters, parallel_outputs[0][0])
                difference_objective[param_idx] = np.array([parallel_outputs[i][0] for i in range(1, parameters_used + 1)])

                # These parameters will be used for the evaluation in the next epoch
                parameters[epoch+1] = parameters[epoch] - learning_rate * (difference_objective - outputs[epoch, 0]) / difference

        else:
            raise ValueError("Number of threads should be positive.")

        return outputs, parameters

    def values_out(self,
                   outputs,
                   parameters,
                   columns):

        values = pd.DataFrame([pd.DataFrame(outputs),
                               pd.DataFrame(parameters)],
                               columns=columns)

        return values
