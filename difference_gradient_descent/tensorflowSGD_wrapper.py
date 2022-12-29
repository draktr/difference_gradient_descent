import numpy as np
import tensorflow as tf

def loss_minimize():
    return (var-[3, 2])**2

def loss(var):
    return (var-[3, 2])**2

def tensorflow_sgd(initial_parameters,
                   learning_rate,
                   epochs = 1000):

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

        objective_value[epoch] = loss(var.numpy())

    return objective_value, parameters
