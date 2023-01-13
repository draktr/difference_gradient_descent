"""
The ``schedules`` module houses ``Schedules`` class that produces sequences used in
gradient descent and other optimizers for variable learning rates and other hyperparameters

:raises ValueError: Error if there is more or less than exactly one more element of `values` that `boundaries`
:return: Sequence of values with each element being a value (e.g. learning rate or difference) for each epoch
:rtype: ndarray
"""

import numpy as np


class Schedules():
    """
    Schedules class.

    Instance variables:

    - ``n_steps`` - int
    - ``steps`` - ndarray
    """

    def __init__(self,
                 n_steps) -> None:
        """
        Initializes Schedules Object.

        :param n_steps: Number of decay steps. Must be equal to the number of epochs of the algorithm
        :type n_steps: int
        """

        self.n_steps = n_steps
        self.steps = np.linspace(0, n_steps, n_steps)

    def exponential_decay(self,
                          initial_value,
                          decay_rate,
                          staircase = False):
        """
        Sequence with exponential decay.

        :param initial_value: Initial value of the sequence
        :type initial_value: float
        :param decay_rate: Rate of decay
        :type decay_rate: float
        :param staircase: If True decay the sequence at discrete intervals, defaults to False
        :type staircase: bool, optional
        :return: Sequence of values with each element being a value (e.g. learning rate or difference) for each epoch
        :rtype: ndarray
        """

        if staircase is True:
            sequence = initial_value*np.power(decay_rate, np.floor(np.divide(self.steps, self.n_steps)))
        else:
            sequence = initial_value*np.power(decay_rate, np.divide(self.steps, self.n_steps))

        return sequence

    def cosine_decay(self,
                     initial_value,
                     alpha):
        """
        Sequence with cosine decay.

        :param initial_value: Initial value of the sequence
        :type initial_value: float
        :param alpha: Minimum sequence value as a fraction of initial_value
        :type alpha: float
        :return: Sequence of values with each element being a value (e.g. learning rate or difference) for each epoch
        :rtype: ndarray
        """

        steps = np.minimum(self.steps, self.n_steps)
        cosine_decay = 0.5 * (1 + np.cos(np.multiply(np.pi, np.divide(steps, self.n_steps))))
        decayed = (1 - alpha) * cosine_decay + alpha

        sequence = initial_value * decayed

        return sequence

    def inverse_time_decay(self,
                           initial_value,
                           decay_rate,
                           staircase = False):
        """
        Sequence with inverse time decay

        :param initial_value: Initial value of the sequence
        :type initial_value: float
        :param decay_rate: Rate of decay
        :type decay_rate: float
        :param staircase: If True decay the sequence at discrete intervals, defaults to False
        :type staircase: bool, optional
        :return: Sequence of values with each element being a value (e.g. learning rate or difference) for each epoch
        :rtype: ndarray
        """

        if staircase is True:
            sequence = np.divide(initial_value,
                                      (1 + np.multiply(decay_rate, np.floor(np.divide(self.steps, self.n_steps)))))
        else:
            sequence = np.divide(initial_value,
                                      (1 + np.multiply(decay_rate, np.divide(self.steps, self.n_steps))))

        return sequence

    def polynomial_decay(self,
                         initial_value,
                         end_value,
                         power,
                         cycle = False):
        """
        Sequence with polynomial decay.

        :param initial_value: Initial value of the sequence
        :type initial_value: float
        :param end_value: The minimal end sequence value
        :type end_value: float
        :param power: The power of the polynomial
        :type power: float
        :param cycle: Whether or not it should cycle beyond self.n_steps, defaults to False
        :type cycle: bool, optional
        :return: Sequence of values with each element being a value (e.g. learning rate or difference) for each epoch
        :rtype: ndarray
        """

        if cycle is True:
            n_steps = np.multiply(self.n_steps, np.ceil(np.divide(self.steps, self.n_steps)))
            sequence = np.multiply((initial_value - end_value),
                                         (np.power(1 - np.divide(self.steps, n_steps), power)
                                         )) + end_value
        else:
            steps = np.minimum(self.steps, self.n_steps)
            sequence = np.multiply((initial_value - end_value),
                                         (np.power(1 - np.divide(steps, self.n_steps), power)
                                         )) + end_value

        return sequence

    def piecewise_constant_decay(self,
                                 boundaries,
                                 values):
        """
        Sequence with piecewise constant decay.

        :param boundaries: Boundaries of the pieces
        :type boundaries: list
        :param values: list of values in sequence in each of the pieces
        :type values: list
        :raises ValueError: Error if there is more or less than exactly one more element of `values` that `boundaries`
        :return: Sequence of values with each element being a value (e.g. learning rate or difference) for each epoch
        :rtype: ndarray
        """

        if len(boundaries)+1 != len(values):
            raise ValueError("There should be only one value for each piece of array, \
                              i.e. there should be exactly one more element of `values` that `boundaries`")

        boundaries = np.append(0, boundaries)
        boundaries = np.append(boundaries, self.n_steps)

        sequence = np.zeros(self.n_steps)
        for value in range(len(values)):
            sequence[boundaries[value]:boundaries[value+1]] = np.full(boundaries[value+1]-boundaries[value],
                                                                            values[value])

        return sequence

    def constant(self,
                 value):
        """
        Constant sequence

        :param value: Value for each epoch
        :type value: float
        :return: Sequence of values with each element being a value (e.g. learning rate or difference) for each epoch
        :rtype: ndarray
        """

        sequence = np.full(len(self.steps), value)

        return sequence
