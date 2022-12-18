# Difference Gradient Descent

## Introduction

Difference Gradient Descent (DGD) expands Gradient Descent method for convex optimization to allow for objective functions where analytically gradient cannot be found. It does so by approximating the derivatives as $\frac{\partial F}{\partial X} \approx \frac{\Delta F}{\Delta X}$. Where $F$ is the objective function and $X$ is the parameter vector to be estimated. In n-dimensional space, the approximated gradient is given by
$$
    \begin{bmatrix}
        \frac{\partial F}{\partial x_1} \\
        \frac{\partial F}{\partial x_2} \\
        \vdots                          \\
        \frac{\partial F}{\partial x_n} \\
    \end{bmatrix}
    \approx
    \begin{bmatrix}
        \frac{\Delta F}{\Delta x_1} \\
        \frac{\Delta F}{\Delta x_2} \\
        \vdots                          \\
        \frac{\Delta F}{\Delta x_n} \\
    \end{bmatrix}
$$
In the implementation $\Delta X := X_{t} - X_{t-1}$ is specified by `difference` argument and $\Delta F$ is computed by evaluating the objective function as: $\Delta F := F(X_{t}) - F(X_{t-1})$.

## A Quick Example

```python
from difference_gradient_descent import DifferenceGradientDescent
from schedules import Schedules

# Defining the objective function
def foo(params):
    return [(params[0]+2)**2]

optimizer = DifferenceGradientDescent(objective_function = foo)

# Creating decay schedules for difference values and learning rates
scheduler = Schedules(n_steps = 1000)
differences = scheduler.exponential_decay(initial_value=0.01,
                                          decay_rate=0.0005)

rates = scheduler.exponential_decay(initial_value=0.01,
                                    decay_rate=0.5)

# Running the algorithm
outputs, parameters = optimizer.difference_gradient_descent(initial_parameters=[5],
                                                            differences = differences,
                                                            learning_rates = rates,
                                                            epochs = 1000)

print("Solutions: ", parameters[-1])
print("Objective value: ", outputs[-1])

# Saves values of outputs and parameters as Pandas DataFrame...
values = optimizer.values_out(outputs,
                              parameters,
                              ["x"])
# ...and stores them as a CSV file
values.to_csv("values.csv")
```

## Features and Advantages

1) Can optimize loss functions that cannot be solved analytically or where we cannot find analytical gradient
2) Supports parallelization via `joblib` library
3) Object-oriented approach makes it convenient for trying out different learning rates, differences values and algorithms
4) Partial Gradient Descent makes high-dimensional, simple problems less computationally expensive to solve
5) Built-in support for variable learning rates and differences
