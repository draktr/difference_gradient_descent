"""
The ``learning_rates`` module houses ``LearningRates`` class that produces learning rates
for gradient descent and other optimizers that utilize them.

:raises ValueError: Error if there is more or less than exactly one more element of `values` that `boundaries`
:return: Array of learning rates with each element being a learning rate for each epoch
:rtype: ndarray
"""

import numpy as np


class LearningRates():
    """
    LearningRates class.

    Instance variables:

    - ``decay_steps`` - int
    - ``steps`` - ndarray
    """

    def __init__(self,
                 decay_steps) -> None:
        """
        Initializes Learning Rates Generator Object.

        :param decay_steps: Number of decay steps. Must be equal to the number of epochs of the algorithm
        :type decay_steps: int
        """

        self.decay_steps = decay_steps
        self.steps = np.linspace(0, decay_steps, decay_steps)

    def exponential_decay(self,
                          initial_learning_rate,
                          decay_rate,
                          staircase = False):
        """
        Learning rate with exponential decay.

        :param initial_learning_rate: Initial learning rate
        :type initial_learning_rate: float
        :param decay_rate: Rate of decay
        :type decay_rate: float
        :param staircase: If True decay the learning rate at discrete intervals, defaults to False
        :type staircase: bool, optional
        :return: Array of learning rates with each element being a learning rate for each epoch
        :rtype: ndarray
        """

        if staircase is True:
            learning_rates = initial_learning_rate*np.power(decay_rate, np.floor(np.divide(self.steps, self.decay_steps)))
        else:
            learning_rates = initial_learning_rate*np.power(decay_rate, np.divide(self.steps, self.decay_steps))

        return learning_rates

    def cosine_decay(self,
                     initial_learning_rate,
                     alpha):
        """
        Learning rate with cosine decay.

        :param initial_learning_rate: Initial learning rate
        :type initial_learning_rate: float
        :param alpha: Minimum learning rate value as a fraction of initial_learning_rate
        :type alpha: float
        :return: Array of learning rates with each element being a learning rate for each epoch
        :rtype: ndarray
        """

        steps = np.minimum(self.steps, self.decay_steps)
        cosine_decay = 0.5 * (1 + np.cos(np.multiply(np.pi, np.divide(steps, self.decay_steps))))
        decayed = (1 - alpha) * cosine_decay + alpha

        learning_rates = initial_learning_rate * decayed

        return learning_rates

    def inverse_time_decay(self,
                           initial_learning_rate,
                           decay_rate,
                           staircase = False):
        """
        Learning rate with inverse time decay

        :param initial_learning_rate: Initial learning rate
        :type initial_learning_rate: float
        :param decay_rate: Rate of decay
        :type decay_rate: float
        :param staircase: If True decay the learning rate at discrete intervals, defaults to False
        :type staircase: bool, optional
        :return: Array of learning rates with each element being a learning rate for each epoch
        :rtype: ndarray
        """

        if staircase is True:
            learning_rates = np.divide(initial_learning_rate,
                                      (1 + np.multiply(decay_rate, np.floor(np.divide(self.steps, self.decay_steps)))))
        else:
            learning_rates = np.divide(initial_learning_rate,
                                      (1 + np.multiply(decay_rate, np.divide(self.steps, self.decay_steps))))

        return learning_rates

    def polynomial_decay(self,
                         initial_learning_rate,
                         end_learning_rate,
                         power,
                         cycle = False):
        """
        Learning rate withpolynomial decay.

        :param initial_learning_rate: Initial learning rate
        :type initial_learning_rate: float
        :param end_learning_rate: The minimal end learning rate
        :type end_learning_rate: float
        :param power: The power of the polynomial
        :type power: float
        :param cycle: Whether or not it should cycle beyond self.decay_steps, defaults to False
        :type cycle: bool, optional
        :return: Array of learning rates with each element being a learning rate for each epoch
        :rtype: ndarray
        """

        if cycle is True:
            decay_steps = np.multiply(self.decay_steps, np.ceil(np.divide(self.steps, self.decay_steps)))
            learning_rates = np.multiply((initial_learning_rate - end_learning_rate),
                                         (1 - np.power(np.divide(self.steps, decay_steps), (power))
                                         )) + end_learning_rate
        else:
            steps = np.minimum(self.steps, self.decay_steps)
            learning_rates = np.multiply((initial_learning_rate - end_learning_rate),
                                         (1 - np.power(np.divide(steps, self.decay_steps), (power))
                                         )) + end_learning_rate

        return learning_rates

    def piecewise_constant_decay(self,
                                 boundaries,
                                 values):
        """
        Learning rate with piecewise constant decay.

        :param boundaries: Boundaries of the pieces
        :type boundaries: float
        :param values: values of learning rates in each of the pieces
        :type values: float
        :raises ValueError: Error if there is more or less than exactly one more element of `values` that `boundaries`
        :return: Array of learning rates with each element being a learning rate for each epoch
        :rtype: ndarray
        """

        if len(boundaries)+1 != len(values):
            raise ValueError("There should be only one value for each piece of array, \
                              i.e. there should be exactly one more element of `values` that `boundaries`")

        learning_rates = np.zeros(len(self.steps))
        for boundary in range(len(boundaries)):
            learning_rates[boundaries[boundary]:boundaries[boundary+1]] = np.full(boundaries[boundary+1]-boundaries[boundary],
                                                                                  values[boundary])

        return learning_rates

    def constant(self,
                 learning_rate):
        """
        Learning rate that stays constant

        :param learning_rate: Learning rate for each epoch
        :type learning_rate: float
        :return: Array of learning rates with each element being a learning rate for each epoch
        :rtype: ndarray
        """

        learning_rates = np.full(len(self.steps), learning_rate)

        return learning_rates
