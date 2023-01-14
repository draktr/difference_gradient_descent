"""
Wrapper function for Tensorflow Stochastic Gradient Descent (SGD) optimizer. This optimizer requires the analytical
form of the problem to be known, thus this isn't an example of Difference Gradient Descent optimizer. This module is
included in Difference Gradient Descent for convenience.

:return: An array of objective values and an array of parameter values, one for each epoch
:rtype: ndarray, ndarray
"""

import numpy as np
import tensorflow as tf


def tensorflow_sgd(loss_function,
                   loss_minimize,
                   initial_parameters,
                   learning_rate,
                   epochs = 1000):
    """
    Wrapper function for Stochastic Gradient Descent from Tensorflow.

    :param loss_function: Loss function formated to take in argument. Used for objective value evaluation.
                          Has to have the same expression as `loss_minimize`
    :type loss_function: function
    :param loss_minimize: Loss function formated for `tf.keras.optimizers.SGD.minimize()` method, doesn't
                          take any arguments. Has to have the same expression as `loss_function`
    :type loss_minimize: function
    :param initial_parameters: Initial set of parameters for the optimizer to start from
    :type initial_parameters: ndarray
    :param learning_rate: Stochastic Gradient Descent learning rate
    :type learning_rate: ndarray
    :param epochs: Number of iterations of the Stochastic Gradient Descent algorithm, defaults to 1000
    :type epochs: int, optional
    :return: An array of objective values and an array of parameter values, one for each epoch
    :rtype: ndarray, ndarray
    """

    objective_value = np.zeros(epochs)
    parameters = np.zeros([epochs, len(initial_parameters)])
    parameters[0] = initial_parameters

    opt = tf.keras.optimizers.SGD(learning_rate = learning_rate)
    var = tf.Variable(initial_value=initial_parameters,
                      name="variable")

    for epoch in range(epochs):
        opt.minimize(loss=loss_minimize,
                     var_list=[var])

        parameters[epoch] = var.numpy()

        objective_value[epoch] = loss_function(var.numpy())

    return objective_value, parameters
